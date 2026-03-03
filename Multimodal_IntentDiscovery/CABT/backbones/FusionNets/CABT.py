import torch
import torch.nn.functional as F
# from losses import loss_map
from ..SubNets.FeatureNets import BERTEncoder, SubNet, RoBERTaEncoder, AuViSubNet
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from .sampler import ConvexSampler
from torch import nn
from ..SubNets.AlignNets import AlignSubNet
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertLayer
from ..SubNets.FeatureNets import BERTEncoderSDIF, BertCrossEncoder

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
__all__ = ['cabt']

class CABT(nn.Module):
    
    def __init__(self, args):

        super(CABT, self).__init__()
        
        self.args = args
        base_dim = args.base_dim
        self.device = args.device
        
        self.num_heads = args.nheads
        self.attn_dropout = args.attn_dropout

        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.attn_mask = args.attn_mask
        self.layers_self = args.n_levels_self 
        self.self_num_heads = args.self_num_heads
        
        self.text_embedding = BERTEncoder(args)
        
        self.text_layer = nn.Linear(args.text_feat_dim, base_dim)
        self.video_layer = nn.Linear(args.video_feat_dim, base_dim)
        self.audio_layer = nn.Linear(args.audio_feat_dim, base_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=base_dim, nhead=self.self_num_heads)
        self.self_att = nn.TransformerEncoder(encoder_layer, num_layers=self.layers_self)

        self.video2text_cross = BertCrossEncoder(
            args.cross_num_heads,
            base_dim,
            args.cross_dp_rate,
            n_layers=args.n_levels_cross
        )
        self.audio2text_cross = BertCrossEncoder(
            args.cross_num_heads,
            base_dim,
            args.cross_dp_rate,
            n_layers=args.n_levels_cross
        )
         
        self.v_encoder = self.get_transformer_encoder(base_dim, args.encoder_layers_1)
        self.a_encoder = self.get_transformer_encoder(base_dim, args.encoder_layers_1)

        self.deeplinear = nn.Linear(base_dim * 6, base_dim * 3)
        
        self.shared_embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(base_dim, base_dim),
        )

        self.fusion_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(args.hidden_dropout_prob),    #nn.ReLu
            nn.Linear(3 * base_dim, base_dim),
        )

        self.mlp_project =  nn.Sequential(
                nn.Linear(base_dim, base_dim),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)  
           
    def forward(self, text_feats, video_feats, audio_feats): 

        
        bert_sent_mask = text_feats[:,1]

        video = video_feats.float()
        audio = audio_feats.float()
        text = self.text_embedding(text_feats) #torch.Size([128, 30, 768])
      
        text_rep = text[:, 0]#torch.Size([128, 768])   
        text_seq = self.text_layer(text) #torch.Size([128, 30, 256])
        text_rep = self.text_layer(text_rep)

        video_seq = self.video_layer(video)#torch.Size([128, 230, 256])
        video_rep = video_seq.permute(1, 0, 2)
        video_rep= self.v_encoder(video_rep)[-1]#torch.Size([128, 256])
        
        
        audio_seq = self.audio_layer(audio)#torch.Size([128, 480, 256])
        audio_rep = audio_seq.permute(1, 0, 2)
        audio_rep = self.a_encoder(audio_rep)[-1] # torch.Size([128, 256])

        # 修改视频掩码处理
        video_mask = (torch.sum(video, dim=-1) != 0).int()  # [128, 230]
        extended_video_mask = video_mask.unsqueeze(1).unsqueeze(1)  # [128,1,1,230]
        extended_video_mask = extended_video_mask.to(dtype=next(self.parameters()).dtype)
        
        # 修改音频掩码处理
        audio_mask = (torch.sum(audio, dim=-1) != 0).int()  # [128, 480]
        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(1)  # [128,1,1,480]
        extended_audio_mask = extended_audio_mask.to(dtype=next(self.parameters()).dtype)

        video2text_seq = self.video2text_cross(text_seq, video_seq, extended_video_mask)
        audio2text_seq = self.audio2text_cross(text_seq, audio_seq, extended_audio_mask)

        shallow_seq = self.mlp_project(torch.cat([audio2text_seq, text_seq, video2text_seq], dim=1))

        text_mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True) 

        video2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), video2text_seq)
        video2text_rep = torch.sum(video2text_masked_output, dim=1, keepdim=False) / text_mask_len

        audio2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), audio2text_seq)
        audio2text_rep = torch.sum(audio2text_masked_output, dim=1, keepdim=False) / text_mask_len

        # Deep Interaction
        tri_cat_mask = torch.cat([bert_sent_mask, bert_sent_mask, bert_sent_mask], dim=-1)

        tri_mask_len = torch.sum(tri_cat_mask, dim=1, keepdim=True) 
        shallow_masked_output = torch.mul(tri_cat_mask.unsqueeze(2), shallow_seq)
        shallow_rep = torch.sum(shallow_masked_output, dim=1, keepdim=False) / tri_mask_len

        text_rep = text_rep.to(video2text_rep.dtype)# [128, 256]
        video_rep = video_rep.to(video2text_rep.dtype)# [128, 256]
        audio_rep = audio_rep.to(video2text_rep.dtype)# [128, 256]
        audio2text_rep = audio2text_rep.to(video2text_rep.dtype)# [128, 256]

        all_reps = torch.stack((text_rep, video_rep, audio_rep, video2text_rep, audio2text_rep, shallow_rep), dim=0)
        all_hiddens = self.self_att(all_reps)#torch.Size([6, 128, 256])
        deep_rep = torch.cat((all_hiddens[0], all_hiddens[1], all_hiddens[2], all_hiddens[3], all_hiddens[4], all_hiddens[5]), dim=1)#torch.Size([128, 1536])
       
        deep_rep = self.deeplinear(deep_rep)
        features = self.fusion_layer(deep_rep)# [128, 256]
        
        return features
        