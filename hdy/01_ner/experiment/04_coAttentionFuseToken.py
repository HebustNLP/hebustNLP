import json
import torch
import torch.nn as nn

# ============================================================
# 融合模块定义 (Co-Attention)
# ============================================================
class TokenSemanticCoAttention(nn.Module):
    def __init__(self, token_dim=512, sem_dim=256, hidden_dim=512, num_heads=8):
        super().__init__()
        # 把语义矩阵投影到 token_dim
        self.sem_proj = nn.Linear(sem_dim, token_dim)

        # 两个 cross-attention
        self.attn_token_to_sem = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)
        self.attn_sem_to_token = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)

        # 融合线性层
        self.fusion = nn.Linear(token_dim * 2, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, token_vecs, sem_matrix):
        """
        token_vecs: [seq_len, token_dim]
        sem_matrix: [sem_len, sem_dim]
        """
        # 1. 投影语义矩阵到 token 维度
        sem_proj = self.sem_proj(sem_matrix)  # [sem_len, token_dim]

        # 2. token -> sem (token 关注语义)
        token_attn, _ = self.attn_token_to_sem(
            token_vecs.unsqueeze(0),   # Q
            sem_proj.unsqueeze(0),     # K
            sem_proj.unsqueeze(0)      # V
        )
        token_attn = token_attn.squeeze(0)  # [seq_len, token_dim]

        # 3. sem -> token (语义关注 token)
        sem_attn, _ = self.attn_sem_to_token(
            sem_proj.unsqueeze(0),     # Q
            token_vecs.unsqueeze(0),   # K
            token_vecs.unsqueeze(0)    # V
        )
        sem_attn = sem_attn.mean(dim=1)     # [1, token_dim] → 全局语义向量
        sem_attn = sem_attn.expand(token_vecs.size(0), -1)  # broadcast → [seq_len, token_dim]

        # 4. 融合 (拼接 + 非线性映射)
        fused = torch.cat([token_attn, sem_attn], dim=-1)  # [seq_len, 2*token_dim]
        fused = self.tanh(self.fusion(fused))              # [seq_len, hidden_dim]

        return fused


# ============================================================
# 读取 JSON 或 JSONL 文件
# ============================================================
def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":  
            # JSON 数组
            return json.load(f)
        else:
            # JSONL，每行一个 JSON 对象
            return [json.loads(line) for line in f if line.strip()]


# ============================================================
# 主处理流程
# ============================================================
def process_json_list(json_path, save_path, hidden_dim=512):
    data_list = load_json_or_jsonl(json_path)

    for sample in data_list:
        token_vecs = torch.tensor(sample['token_vectors'], dtype=torch.float32)  # [seq_len, token_dim]

        if 'semantic_matrix' in sample:
            sem_matrix = torch.tensor(sample['semantic_matrix'], dtype=torch.float32)  # [sem_len, sem_dim]
        else:
            sem_matrix = token_vecs.mean(dim=0, keepdim=True)  # 没有语义矩阵就用平均值代替

        token_dim = token_vecs.size(1)
        sem_dim = sem_matrix.size(1)

        fusion_model = TokenSemanticCoAttention(token_dim, sem_dim, hidden_dim)
        fused_output = fusion_model(token_vecs, sem_matrix)   # [seq_len, hidden_dim]

        # 如果有标签，保证长度对齐
        if "label" in sample:
            seq_len, hidden_dim_out = fused_output.size()
            label_len = len(sample["label"])

            if seq_len > label_len:
                fused_output = fused_output[:label_len, :]
            elif seq_len < label_len:
                pad = torch.zeros(label_len - seq_len, hidden_dim_out)
                fused_output = torch.cat([fused_output, pad], dim=0)

            assert fused_output.size(0) == len(sample["label"]), \
                f"长度对齐失败：{fused_output.size(0)} vs {len(sample['label'])}"

        # 覆盖 token_vectors
        sample["token_vectors"] = fused_output.detach().cpu().tolist()

    # 保存为 JSON 数组（更通用）
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)



# -----------------------------
# 测试运行
# -----------------------------
if __name__ == "__main__":
    json_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train_03_tokens.json"
    save_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train_04_fused.json"
    process_json_list(json_path, save_path)
    print(f"Fused vectors saved to {save_path}")
