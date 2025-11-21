import torch
import os, sys
from torch import nn
import torch.nn.functional as F
from functorch.einops import rearrange

from common_fusions import *


# -------------------------- PMDF核心组件 --------------------------
class GDFN(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(out_c, out_c * 2, 1, 1, 0)
        self.dwconv = nn.Conv2d(out_c * 2, out_c * 2, 3, 1, 1, groups=out_c * 2)
        self.conv2 = nn.Conv2d(out_c * 2, out_c, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class ModalityPromptEmbed(nn.Module):
    def __init__(self, modal_types, embed_dims, out_c, target_h=16, target_w=16):
        super().__init__()
        self.modal_prompts = nn.ModuleDict()  # 键为'text'或'image'
        self.out_c = out_c
        self.target_h = target_h
        self.target_w = target_w
        
        for modal, dim in zip(modal_types, embed_dims):
            if modal == "image":
                # 图像模态处理：适应单/多通道输入（3或6通道）
                self.modal_prompts[modal] = nn.Sequential(
                    nn.Conv2d(dim, out_c, 3, 1, 1),  # dim=3（单图）或6（双图）
                    nn.GELU(),
                    nn.Conv2d(out_c, out_c, 3, 2, 1),  # 下采样到8x8
                    nn.GELU(),
                    nn.Conv2d(out_c, out_c, 3, 2, 1)   # 下采样到4x4（最终会插值到16x16）
                )
            elif modal == "text":
                # 文本模态处理
                self.modal_prompts[modal] = nn.Sequential(
                    nn.Linear(dim, out_c * target_h * target_w),  # 映射到16x16特征图
                    nn.GELU(),
                    nn.Unflatten(1, (out_c, target_h, target_w))
                )
        self.global_prompt = nn.Parameter(torch.randn(1, out_c, 1, 1))  # 全局提示

    def forward(self, modal_features):
        prompt_feats = {}
        for modal, feat in modal_features.items():
            # 模态专属嵌入（确保modal为'text'或'image'）
            embed_feat = self.modal_prompts[modal](feat)
            # 强制对齐到目标尺寸（16x16）
            embed_feat = F.interpolate(
                embed_feat, 
                size=(self.target_h, self.target_w), 
                mode='bilinear', 
                align_corners=True
            )
            # 加入全局提示
            prompt_feats[modal] = embed_feat + self.global_prompt
        return prompt_feats


class PromptGuidedDynamicWeighter(nn.Module):
    def __init__(self, out_c, num_modals):
        super().__init__()
        self.ms_cam = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Sigmoid()
        )
        self.weight_generator = nn.Sequential(
            nn.Conv2d(out_c, out_c//2, 1, 1, 0),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_c//2, 1)
        )
        self.num_modals = num_modals

    def forward(self, prompt_feats):
        if not prompt_feats:
            raise ValueError("prompt_feats cannot be empty")
        
        modal_weights = []
        for feat in prompt_feats:
            cam_feat = self.ms_cam(feat) * feat
            weight = self.weight_generator(cam_feat)
            modal_weights.append(weight)
        
        dynamic_weights = torch.stack(modal_weights, dim=1)
        dynamic_weights = F.softmax(dynamic_weights, dim=1).unsqueeze(-1).unsqueeze(-1)
        weighted_feats = [feat * weight for feat, weight in zip(prompt_feats, dynamic_weights.unbind(1))]
        return dynamic_weights, weighted_feats


class PromptMDTA(nn.Module):
    def __init__(self, out_c, num_modals):
        super().__init__()
        self.num_modals = num_modals
        self.out_c = out_c
        self.q_conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.GELU()
        ) for _ in range(num_modals)])
        self.k_conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.GELU()
        ) for _ in range(num_modals)])
        self.v_conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.GELU()
        ) for _ in range(num_modals)])
        self.conv4 = None  # 动态初始化

    def forward(self, weighted_feats):
        if not weighted_feats:
            raise ValueError("weighted_feats cannot be empty")
        
        actual_num_modals = len(weighted_feats)
        b, c, h, w = weighted_feats[0].shape
        spatial_size = h * w
        
        q_list, k_list, v_list = [], [], []
        for i in range(actual_num_modals):
            if i < len(self.q_conv):
                q_conv = self.q_conv[i]
                k_conv = self.k_conv[i]
                v_conv = self.v_conv[i]
            else:
                q_conv = self.q_conv[-1]
                k_conv = self.k_conv[-1]
                v_conv = self.v_conv[-1]
            
            feat = weighted_feats[i]
            if feat.shape[1] != c:
                feat = F.conv2d(feat, torch.eye(c, device=feat.device).unsqueeze(-1).unsqueeze(-1), padding=0)
            
            q = q_conv(feat)
            q = rearrange(q, 'b c h w -> b (h w) c')
            k = k_conv(feat)
            k = rearrange(k, 'b c h w -> b c (h w)')
            v = v_conv(feat)
            v = rearrange(v, 'b c h w -> b (h w) c')
            
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)
        
        aligned_v_list = []
        for i in range(actual_num_modals):
            cross_attention = 0.0
            for j in range(actual_num_modals):
                q = q_list[i]
                k = k_list[j]
                v = v_list[j]
                
                min_dim = min(q.shape[-1], k.shape[1])
                if q.shape[-1] != k.shape[1]:
                    q = q[..., :min_dim]
                    k = k[:, :min_dim, :]
                    v = v[..., :min_dim]
                
                A = torch.matmul(k, q)
                A = F.softmax(A / torch.sqrt(torch.tensor(min_dim, dtype=torch.float32, device=A.device)), dim=1)
                cross_attention += torch.matmul(v, A)
            
            aligned_v_list.append(cross_attention)
        
        aligned_v = torch.cat(aligned_v_list, dim=-1)
        aligned_v = rearrange(aligned_v, 'b (h w) (c nm) -> b (c nm) h w', 
                             h=h, w=w, c=c, nm=actual_num_modals)
        
        # 动态创建卷积层
        input_channels = c * actual_num_modals
        if self.conv4 is None or self.conv4.in_channels != input_channels:
            self.conv4 = nn.Conv2d(input_channels, self.out_c, 1, 1, 0).to(aligned_v.device)
        
        return self.conv4(aligned_v)


class CrossModalAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, prompt_feats_list):
        if len(prompt_feats_list) < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        alignment_loss = 0.0
        for i in range(len(prompt_feats_list)):
            for j in range(i+1, len(prompt_feats_list)):
                feat_i = F.adaptive_avg_pool2d(prompt_feats_list[i], 1).flatten(1)
                feat_j = F.adaptive_avg_pool2d(prompt_feats_list[j], 1).flatten(1)
                sim = self.cos_sim(feat_i, feat_j)
                alignment_loss += (1 - sim).mean()
        return alignment_loss / (len(prompt_feats_list) * (len(prompt_feats_list)-1) / 2)


# -------------------------- 原有融合模型 --------------------------
class fusion_tr(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = nn.Conv1d(768, 512, kernel_size=1, padding=0, bias=False) 
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        
        self.linear = self.face_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x.permute([0, 2, 1])) 
        x = x.permute([0, 2, 1])
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        return self.linear(x)


class fusion_lr(nn.Module):
    def __init__(self, n):
        super().__init__()
        input_dims = [768] * n 
        self.fusion = LowRankTensorFusion(input_dims, 32, 1) 
    def forward(self, x):
        x = list(torch.unbind(x, dim=1)) 
        x = self.fusion(x)
        return x


# -------------------------- 主模型（支持PMDF和传统融合） --------------------------
class Model(nn.Module):
    def __init__(self, modality=[], fusion=""):
        super().__init__()
        self.modality = modality
        self.fusion_type = fusion  # 记录融合类型
        self._modal_features = None  # 存储模态特征（用于损失计算）
        
        # 文本模态（ask）处理
        self.ask_fc = nn.Sequential(
            nn.Linear(43, 64),  # 适配最新数据维度43
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 图像模态（face/tongue）处理
        self.img_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32)
        )
        
        # 根据融合类型初始化组件
        if fusion == "pmdf":
            # PMDF配置：模态类型映射（ask→text，face/tongue→image）
            modal_dims = {
                'ask': 43,                # 文本特征维度
                'face': 3,                 # 单图像通道数
                'tongue': 3,               # 单图像通道数
                'face+tongue': 3 + 3       # 双图像拼接后通道数
            }
            self.modal_types = []
            self.embed_dims = []
            
            # 确定图像模态数量
            img_modal_count = sum(1 for m in modality if m in ['face', 'tongue'])
            if img_modal_count > 0:
                self.modal_types.append('image')
                self.embed_dims.append(3 * img_modal_count)  # 3通道×图像数量
            
            # 添加文本模态
            if 'ask' in modality:
                self.modal_types.append('text')
                self.embed_dims.append(modal_dims['ask'])
            
            # 初始化PMDF组件
            self.prompt_embed = ModalityPromptEmbed(
                modal_types=self.modal_types,
                embed_dims=self.embed_dims,
                out_c=64
            )
            self.dynamic_weighter = PromptGuidedDynamicWeighter(
                out_c=64,
                num_modals=len(self.modal_types)
            )
            self.prompt_mdta = PromptMDTA(
                out_c=64,
                num_modals=len(self.modal_types)
            )
            self.gdfn = GDFN(out_c=64)
            self.fusion_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 32)
            )
            cls_input_dim = 32
        else:
            # 原有融合方法（lr/tr）
            if len(modality) > 1:
                self.fusion = fusion_lr(len(modality)) if fusion == "lr" else fusion_tr(len(modality))
                cls_input_dim = 32 + 32
            elif len(modality) == 1:
                cls_input_dim = 32
            else:
                cls_input_dim = 32
        
        # 分类头（统一输出3类）
        self.cls = nn.Sequential(
            nn.Linear(cls_input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 3),
        )

    def forward(self, x1, x2=None, x3=None, return_weights=False, return_prompt_feats=False):
        if self.fusion_type == "pmdf":
            # -------------------------- PMDF融合逻辑（键统一为'text'和'image'） --------------------------
            modal_features = {}
            
            # 文本模态（ask）：键固定为'text'
            if 'ask' in self.modality and x1 is not None:
                modal_features['text'] = x1  # x1形状：(batch, 43)
            
            # 图像模态（face/tongue）：键固定为'image'，多图像拼接
            img_feats = []
            if 'face' in self.modality and x2 is not None:
                b = x2.shape[0]
                img_feats.append(x2.view(b, 3, 16, 16))  # 768 → 3×16×16
            if 'tongue' in self.modality and x3 is not None:
                b = x3.shape[0]
                img_feats.append(x3.view(b, 3, 16, 16))  # 768 → 3×16×16
            
            # 拼接多图像模态（若存在）
            if img_feats:
                modal_features['image'] = torch.cat(img_feats, dim=1) if len(img_feats) > 1 else img_feats[0]
            
            self._modal_features = modal_features  # 保存用于损失计算
            
            # 提示嵌入（键为'text'或'image'，与ModalityPromptEmbed匹配）
            prompt_feats = self.prompt_embed(modal_features)
            prompt_feats_list = list(prompt_feats.values())
            
            # 动态加权
            dynamic_weights, weighted_feats = self.dynamic_weighter(prompt_feats_list)
            
            # 跨模态对齐
            aligned_feat = self.prompt_mdta(weighted_feats)
            fused_feat = self.gdfn(aligned_feat)
            
            # 特征融合
            fusion_out = self.fusion_head(fused_feat)
            out = self.cls(fusion_out)
            
            # 按需返回额外信息
            if return_prompt_feats and return_weights:
                return out, dynamic_weights, prompt_feats_list
            elif return_weights:
                return out, dynamic_weights
            elif return_prompt_feats:
                return out, prompt_feats_list
            return out
        
        else:
            # -------------------------- 传统融合逻辑 --------------------------
            x = self.ask_fc(x1) if 'ask' in self.modality and x1 is not None else None
            img = []
            if 'face' in self.modality and x2 is not None:
                img.append(x2)
            if 'tongue' in self.modality and x3 is not None:
                img.append(x3)
            
            if len(self.modality) > 1 and img:
                img = torch.stack(img, dim=1)
                img = self.fusion(img)
                x = torch.cat([x, img], dim=1) if x is not None else img
            elif len(self.modality) == 1 and img:
                img = self.img_fc(img[0])
                x = img
            
            return self.cls(x)


