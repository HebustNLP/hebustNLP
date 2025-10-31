# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import precision_recall_fscore_support


# # =========================================================
# # 数据集：加载 JSON 格式的 token 向量 + BIO 标签
# # =========================================================
# class NERDataset(Dataset):
#     def __init__(self, json_path, label2id):
#         with open(json_path, "r", encoding="utf-8") as f:
#             self.data = json.load(f)
#         self.label2id = label2id
#         self.num_labels = len(label2id)

#     def __len__(self):
#         return len(self.data)

#     def bio_to_span(self, bio_labels):
#         seq_len = len(bio_labels)
#         span_matrix = torch.zeros((seq_len, seq_len), dtype=torch.long)

#         i = 0
#         while i < seq_len:
#             if bio_labels[i].startswith("B-"):
#                 ent_type = bio_labels[i][2:]       # "game"
#                 j = i
#                 while j + 1 < seq_len and bio_labels[j + 1] == f"I-{ent_type}":
#                     j += 1
#                 # 用 B-xxx 的 id 来标记 span
#                 b_label = f"B-{ent_type}"
#                 if b_label in self.label2id:
#                     label_id = self.label2id[b_label]
#                     span_matrix[i, j] = label_id
#                 i = j + 1
#             else:
#                 i += 1
#         return span_matrix

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         token_vecs = torch.tensor(sample["token_vectors"], dtype=torch.float32)
#         bio_labels = sample["label"]
#         span_labels = self.bio_to_span(bio_labels)
#         return token_vecs, span_labels


# def collate_fn(batch):
#     token_vecs, span_labels = zip(*batch)
#     max_len = max(t.size(0) for t in token_vecs)

#     padded_vecs, padded_spans, masks = [], [], []
#     for vecs, spans in zip(token_vecs, span_labels):
#         seq_len, hidden_dim = vecs.size()
#         pad_len = max_len - seq_len

#         # pad token 向量
#         pad_vecs = torch.cat([vecs, torch.zeros(pad_len, hidden_dim)], dim=0)
#         padded_vecs.append(pad_vecs)

#         # pad span 矩阵
#         pad_spans = torch.zeros((max_len, max_len), dtype=torch.long)
#         pad_spans[:seq_len, :seq_len] = spans
#         padded_spans.append(pad_spans)

#         # mask
#         mask = torch.zeros(max_len, dtype=torch.bool)
#         mask[:seq_len] = 1
#         masks.append(mask)

#     return torch.stack(padded_vecs), torch.stack(padded_spans), torch.stack(masks)


# # =========================================================
# # 模型定义：MLP + Biaffine
# # =========================================================
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = F.relu(self.linear(x))
#         x = self.dropout(x)
#         return self.out(x)


# class Biaffine(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.U = nn.Parameter(torch.Tensor(out_dim, in_dim + 1, in_dim + 1))
#         nn.init.xavier_uniform_(self.U)

#     def forward(self, head, tail):
#         batch, seq_len, in_dim = head.size()
#         ones = head.new_ones(batch, seq_len, 1)
#         head = torch.cat([head, ones], dim=-1)
#         tail = torch.cat([tail, ones], dim=-1)

#         head_U = torch.einsum("bxi,oij->bxoj", head, self.U)
#         scores = torch.einsum("bxoj,byj->bxyo", head_U, tail)
#         return scores


# class SpanNER(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_labels):
#         super().__init__()
#         self.mlp_head = MLP(input_dim, hidden_dim, hidden_dim)
#         self.mlp_tail = MLP(input_dim, hidden_dim, hidden_dim)
#         self.biaffine = Biaffine(hidden_dim, num_labels)

#     def forward(self, token_vecs):
#         head = self.mlp_head(token_vecs)
#         tail = self.mlp_tail(token_vecs)
#         scores = self.biaffine(head, tail)
#         return scores


# # =========================================================
# # Loss & Evaluate
# # =========================================================
# def compute_loss(scores, gold_spans, mask):
#     batch, seq_len, _, num_labels = scores.size()
#     scores = scores.reshape(-1, num_labels)   # ⚡ reshape 替代 view
#     gold = gold_spans.reshape(-1)
#     loss = F.cross_entropy(scores, gold, ignore_index=0)
#     return loss


# def evaluate(model, dataloader, device="cpu"):
#     model.eval()
#     all_preds, all_golds = [], []
#     with torch.no_grad():
#         for token_vecs, span_labels, mask in dataloader:
#             token_vecs, span_labels, mask = token_vecs.to(device), span_labels.to(device), mask.to(device)
#             scores = model(token_vecs)
#             preds = torch.argmax(scores, dim=-1)

#             all_preds.extend(preds.reshape(-1).cpu().numpy())
#             all_golds.extend(span_labels.reshape(-1).cpu().numpy())

#     # 去掉 "O" (0)
#     y_true = [g for g in all_golds if g != 0]
#     y_pred = [p for g, p in zip(all_golds, all_preds) if g != 0]

#     if len(y_true) == 0:
#         return 0, 0, 0

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_true, y_pred, average="micro"
#     )
#     return precision, recall, f1


# # =========================================================
# # 主程序：训练 + 评估
# # =========================================================
# if __name__ == "__main__":
#     # 只保留实体类型，不要 BIO 前缀
#     # unique_labels = [
#     #     'organization', 'company', 'game', 'name', 'address', 'movie',
#     #     'position', 'government', 'scene', 'book', 'O'
#     # ]
#     unique_labels = [
#         'I-organization', 'I-company', 'I-game', 'B-name', 'I-address', 'I-movie',
#         'I-position', 'I-government', 'B-movie', 'B-organization', 'I-scene',
#         'I-book', 'B-game', 'B-book', 'B-government', 'I-name', 'B-company',
#         'B-position', 'B-address', 'B-scene', 'O'
#     ]
#     label2id = {label: idx for idx, label in enumerate(unique_labels)}
#     id2label = {idx: label for label, idx in label2id.items()}

#     json_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train_04_fused.json"
#     dataset = NERDataset(json_path, label2id)
#     dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

#     input_dim = len(dataset[0][0][0])
#     model = SpanNER(input_dim=input_dim, hidden_dim=128, num_labels=len(label2id))

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     # ====== 训练循环 ======
#     for epoch in range(100):
#         model.train()
#         total_loss = 0
#         for token_vecs, span_labels, mask in dataloader:
#             token_vecs, span_labels, mask = token_vecs.to(device), span_labels.to(device), mask.to(device)

#             scores = model(token_vecs)
#             loss = compute_loss(scores, span_labels, mask)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         precision, recall, f1 = evaluate(model, dataloader, device)
#         print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | "
#               f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.metrics import precision_recall_fscore_support
# from tqdm import tqdm
# import os

# # =========================================================
# # 数据集
# # =========================================================
# class NERDataset(Dataset):
#     def __init__(self, json_path, label2id):
#         with open(json_path, "r", encoding="utf-8") as f:
#             self.data = json.load(f)
#         self.label2id = label2id
#         self.num_labels = len(label2id)

#     def __len__(self):
#         return len(self.data)
#     def spans_dict_to_matrix(self, label_dict, seq_len):
#         span_matrix = torch.zeros((seq_len, seq_len), dtype=torch.long)
#         for ent_type, ents in label_dict.items():
#             for _, spans in ents.items():
#                 for start, end in spans:
#                     # 越界处理
#                     if start >= seq_len:
#                         start = seq_len - 1
#                     if end >= seq_len:
#                         end = seq_len - 1
#                     b_label = f"B-{ent_type}"
#                     if b_label in self.label2id:
#                         span_matrix[start, end] = self.label2id[b_label]
#         return span_matrix


#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         token_vecs = torch.tensor(sample["token_vectors"], dtype=torch.float32)
#         seq_len = token_vecs.size(0)
#         span_labels = self.spans_dict_to_matrix(sample["label"], seq_len)
#         return token_vecs, span_labels


# def collate_fn(batch):
#     token_vecs, span_labels = zip(*batch)
#     max_len = max(t.size(0) for t in token_vecs)

#     padded_vecs, padded_spans, masks = [], [], []
#     for vecs, spans in zip(token_vecs, span_labels):
#         seq_len, hidden_dim = vecs.size()
#         pad_len = max_len - seq_len

#         pad_vecs = torch.cat([vecs, torch.zeros(pad_len, hidden_dim)], dim=0)
#         padded_vecs.append(pad_vecs)

#         pad_spans = torch.zeros((max_len, max_len), dtype=torch.long)
#         pad_spans[:seq_len, :seq_len] = spans
#         padded_spans.append(pad_spans)

#         mask = torch.zeros(max_len, dtype=torch.bool)
#         mask[:seq_len] = 1
#         masks.append(mask)

#     return torch.stack(padded_vecs), torch.stack(padded_spans), torch.stack(masks)

# # =========================================================
# # 模型定义
# # =========================================================
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = F.relu(self.linear(x))
#         x = self.dropout(x)
#         return self.out(x)


# class Biaffine(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.U = nn.Parameter(torch.Tensor(out_dim, in_dim + 1, in_dim + 1))
#         nn.init.xavier_uniform_(self.U)

#     def forward(self, head, tail):
#         batch, seq_len, in_dim = head.size()
#         ones = head.new_ones(batch, seq_len, 1)
#         head = torch.cat([head, ones], dim=-1)
#         tail = torch.cat([tail, ones], dim=-1)

#         head_U = torch.einsum("bxi,oij->bxoj", head, self.U)
#         scores = torch.einsum("bxoj,byj->bxyo", head_U, tail)
#         return scores


# class SpanNER(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_labels):
#         super().__init__()
#         self.mlp_head = MLP(input_dim, hidden_dim, hidden_dim)
#         self.mlp_tail = MLP(input_dim, hidden_dim, hidden_dim)
#         self.biaffine = Biaffine(hidden_dim, num_labels)

#     def forward(self, token_vecs):
#         head = self.mlp_head(token_vecs)
#         tail = self.mlp_tail(token_vecs)
#         scores = self.biaffine(head, tail)
#         return scores

# # =========================================================
# # Loss & Evaluate
# # =========================================================
# def compute_loss(scores, gold_spans):
#     batch, seq_len, _, num_labels = scores.size()
#     scores = scores.reshape(-1, num_labels)
#     gold = gold_spans.reshape(-1)
#     loss = F.cross_entropy(scores, gold, ignore_index=0)
#     return loss


# def evaluate(model, dataloader, device="cpu"):
#     model.eval()
#     all_preds, all_golds = [], []
#     with torch.no_grad():
#         for token_vecs, span_labels, mask in dataloader:
#             token_vecs, span_labels = token_vecs.to(device), span_labels.to(device)
#             scores = model(token_vecs)
#             preds = torch.argmax(scores, dim=-1)

#             all_preds.extend(preds.reshape(-1).cpu().numpy())
#             all_golds.extend(span_labels.reshape(-1).cpu().numpy())

#     y_true = [g for g in all_golds if g != 0]
#     y_pred = [p for g, p in zip(all_golds, all_preds) if g != 0]

#     if len(y_true) == 0:
#         return 0, 0, 0

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_true, y_pred, average="micro"
#     )
#     return precision, recall, f1

# # =========================================================
# # 主程序：训练 + 验证 + tqdm
# # =========================================================
# if __name__ == "__main__":
#     unique_labels = [
#         'I-organization', 'I-company', 'I-game', 'B-name', 'I-address', 'I-movie',
#         'I-position', 'I-government', 'B-movie', 'B-organization', 'I-scene',
#         'I-book', 'B-game', 'B-book', 'B-government', 'I-name', 'B-company',
#         'B-position', 'B-address', 'B-scene', 'O'
#     ]
#     label2id = {label: idx for idx, label in enumerate(unique_labels)}

#     json_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train_04_fused.json"
#     dataset = NERDataset(json_path, label2id)

#     # 划分训练/验证/测试集 (8:1:1)
#     total_size = len(dataset)
#     train_size = int(0.8 * total_size)
#     valid_size = int(0.1 * total_size)
#     test_size = total_size - train_size - valid_size
#     train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
#     valid_loader = DataLoader(valid_dataset, batch_size=4, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

#     input_dim = len(dataset[0][0][0])
#     model = SpanNER(input_dim=input_dim, hidden_dim=128, num_labels=len(label2id))

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     best_f1 = 0.0
#     save_path = "/root/autodl-tmp/shiyan/code/save/best_span_ner_model.pth"

#     for epoch in range(100):
#         model.train()
#         total_loss = 0
#         # ===== tqdm 显示训练进度 =====
#         loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=120, leave=False)
#         for token_vecs, span_labels, _ in loop:
#             token_vecs, span_labels = token_vecs.to(device), span_labels.to(device)
#             scores = model(token_vecs)
#             loss = compute_loss(scores, span_labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#             # 实时更新 tqdm
#             loop.set_postfix(loss=total_loss/ (loop.n+1))

#         # 每个 epoch 结束后评估验证集
#         precision, recall, f1 = evaluate(model, valid_loader, device)
#         print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | "
#               f"Val Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

#         if f1 > best_f1:
#             best_f1 = f1
#             torch.save(model.state_dict(), save_path)
#             print(f"--> Saved Best Model with F1: {best_f1:.4f}")

#     # ====== 测试集评估 ======
#     model.load_state_dict(torch.load(save_path))
#     test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
#     print(f"Test Set --> Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm
import os

# =========================================================
# 数据集
# =========================================================
class NERDataset(Dataset):
    def __init__(self, json_path, label2id):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.label2id = label2id
        self.num_labels = len(label2id)
        
        # 创建实体类型到ID的映射（忽略B/I前缀）
        self.entity_type_to_id = {}
        for label, idx in label2id.items():
            if label == 'O':
                self.entity_type_to_id['O'] = idx
            elif '-' in label:
                entity_type = label.split('-', 1)[1]  # 移除B-或I-前缀
                if entity_type not in self.entity_type_to_id:
                    self.entity_type_to_id[entity_type] = idx
            else:
                # 处理没有前缀的标签
                self.entity_type_to_id[label] = idx

    def __len__(self):
        return len(self.data)
    
    def spans_dict_to_matrix(self, label_dict, seq_len):
        span_matrix = torch.zeros((seq_len, seq_len), dtype=torch.long)
        for ent_type, ents in label_dict.items():
            for _, spans in ents.items():
                for start, end in spans:
                    # 越界处理
                    if start >= seq_len:
                        start = seq_len - 1
                    if end >= seq_len:
                        end = seq_len - 1
                    # 使用实体类型而不是B/I标签
                    if ent_type in self.entity_type_to_id:
                        span_matrix[start, end] = self.entity_type_to_id[ent_type]
        return span_matrix

    def __getitem__(self, idx):
        sample = self.data[idx]
        token_vecs = torch.tensor(sample["token_vectors"], dtype=torch.float32)
        seq_len = token_vecs.size(0)
        span_labels = self.spans_dict_to_matrix(sample["label"], seq_len)
        return token_vecs, span_labels


def collate_fn(batch):
    token_vecs, span_labels = zip(*batch)
    max_len = max(t.size(0) for t in token_vecs)

    padded_vecs, padded_spans, masks = [], [], []
    for vecs, spans in zip(token_vecs, span_labels):
        seq_len, hidden_dim = vecs.size()
        pad_len = max_len - seq_len

        pad_vecs = torch.cat([vecs, torch.zeros(pad_len, hidden_dim)], dim=0)
        padded_vecs.append(pad_vecs)

        pad_spans = torch.zeros((max_len, max_len), dtype=torch.long)
        pad_spans[:seq_len, :seq_len] = spans
        padded_spans.append(pad_spans)

        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:seq_len] = 1
        masks.append(mask)

    return torch.stack(padded_vecs), torch.stack(padded_spans), torch.stack(masks)

# =========================================================
# 模型定义
# =========================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.dropout(x)
        return self.out(x)


class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.U = nn.Parameter(torch.Tensor(out_dim, in_dim + 1, in_dim + 1))
        nn.init.xavier_uniform_(self.U)

    def forward(self, head, tail):
        batch, seq_len, in_dim = head.size()
        ones = head.new_ones(batch, seq_len, 1)
        head = torch.cat([head, ones], dim=-1)
        tail = torch.cat([tail, ones], dim=-1)

        head_U = torch.einsum("bxi,oij->bxoj", head, self.U)
        scores = torch.einsum("bxoj,byj->bxyo", head_U, tail)
        return scores


class SpanNER(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super().__init__()
        self.mlp_head = MLP(input_dim, hidden_dim, hidden_dim)
        self.mlp_tail = MLP(input_dim, hidden_dim, hidden_dim)
        self.biaffine = Biaffine(hidden_dim, num_labels)

    def forward(self, token_vecs):
        head = self.mlp_head(token_vecs)
        tail = self.mlp_tail(token_vecs)
        scores = self.biaffine(head, tail)
        return scores

# =========================================================
# Loss & Evaluate
# =========================================================
def compute_loss(scores, gold_spans):
    batch, seq_len, _, num_labels = scores.size()
    scores = scores.reshape(-1, num_labels)
    gold = gold_spans.reshape(-1)
    loss = F.cross_entropy(scores, gold, ignore_index=0)  # 忽略'O'标签
    return loss


def evaluate(model, dataloader, device="cpu", id2label=None):
    model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for token_vecs, span_labels, mask in dataloader:
            token_vecs, span_labels = token_vecs.to(device), span_labels.to(device)
            scores = model(token_vecs)
            preds = torch.argmax(scores, dim=-1)

            # 应用mask，只考虑有效位置
            valid_mask = (span_labels != 0)  # 只考虑非'O'标签
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_golds.extend(span_labels[valid_mask].cpu().numpy())

    if len(all_golds) == 0:
        return 0, 0, 0, None

    # 计算每个类别的指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_golds, all_preds, average='macro', zero_division=0
    )
    
    # 生成详细的分类报告
    if id2label:
        target_names = [id2label[i] for i in sorted(id2label.keys()) if id2label[i] != 'O']
        report = classification_report(
            all_golds, all_preds, 
            target_names=target_names,
            zero_division=0
        )
    else:
        report = None
        
    return precision, recall, f1, report

# =========================================================
# 主程序：训练 + 验证 + tqdm
# =========================================================
if __name__ == "__main__":
    # 提取所有实体类型（忽略B/I前缀）
    unique_entity_types = set()
    for label in [
        'I-organization', 'I-company', 'I-game', 'B-name', 'I-address', 'I-movie',
        'I-position', 'I-government', 'B-movie', 'B-organization', 'I-scene',
        'I-book', 'B-game', 'B-book', 'B-government', 'I-name', 'B-company',
        'B-position', 'B-address', 'B-scene', 'O'
    ]:
        if label == 'O':
            unique_entity_types.add('O')
        elif '-' in label:
            entity_type = label.split('-', 1)[1]
            unique_entity_types.add(entity_type)

    # 转换为排序列表
    unique_entity_types = sorted(unique_entity_types)
    label2id = {label: idx for idx, label in enumerate(unique_entity_types)}
    id2label = {idx: label for label, idx in label2id.items()}

    json_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train_04_fused.json"
    dataset = NERDataset(json_path, label2id)

    # 划分训练/验证/测试集 (8:1:1)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)
    test_size = total_size - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # 增加批量大小
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    input_dim = len(dataset[0][0][0])
    model = SpanNER(input_dim=input_dim, hidden_dim=128, num_labels=len(label2id))

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # 降低学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f1 = 0.0
    # 使用相对路径或可配置路径
    save_dir = "save"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_span_ner_model.pth")

    for epoch in range(100):
        model.train()
        total_loss = 0
        # ===== tqdm 显示训练进度 =====
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=120, leave=False)
        for token_vecs, span_labels, _ in loop:
            token_vecs, span_labels = token_vecs.to(device), span_labels.to(device)
            scores = model(token_vecs)
            loss = compute_loss(scores, span_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 实时更新 tqdm
            loop.set_postfix(loss=loss.item())

        # 每个 epoch 结束后评估验证集
        precision, recall, f1, report = evaluate(model, valid_loader, device, id2label)
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        if report:
            print("详细分类报告:")
            print(report)

        # 学习率调度
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            # 保存模型时处理DataParallel情况
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"--> Saved Best Model with F1: {best_f1:.4f}")

    # ====== 测试集评估 ======
    # 加载模型时处理DataParallel情况
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(save_path))
    else:
        model.load_state_dict(torch.load(save_path))
        
    test_precision, test_recall, test_f1, test_report = evaluate(model, test_loader, device, id2label)
    print(f"Test Set --> Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
    if test_report:
        print("测试集详细分类报告:")
        print(test_report)