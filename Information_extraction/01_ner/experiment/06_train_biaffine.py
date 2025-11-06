import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# ===============================
# 1. 数据集定义
# ===============================
class SpanNERDataset(Dataset):
    def __init__(self, json_path, label2id=None, max_span_len=10):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # 构建标签映射
        if label2id is None:
            labels = set()
            for item in self.data:
                for lab in item["label"]:
                    if lab == "O":
                        labels.add("O")
                    else:
                        labels.add(lab.split("-")[-1])  # 去掉B- I-
            self.label2id = {l: i for i, l in enumerate(sorted(labels))}
        else:
            self.label2id = label2id
        self.id2label = {i: l for l, i in self.label2id.items()}

        self.max_span_len = max_span_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["text"]
        vectors = torch.tensor(item["token_vectors"], dtype=torch.float)
        labels = item["label"]  # BIO 序列

        seq_len = len(tokens)
        num_labels = len(self.label2id)

        # span 标签矩阵
        span_labels = torch.zeros((seq_len, seq_len), dtype=torch.long)
        for i in range(seq_len):
            if labels[i].startswith("B-"):
                ent_type = labels[i][2:]
                j = i
                while j + 1 < seq_len and labels[j + 1].startswith("I-"):
                    j += 1
                if j - i + 1 <= self.max_span_len:
                    span_labels[i, j] = self.label2id[ent_type]

        return vectors, span_labels


def collate_fn(batch):
    vectors, span_labels = zip(*batch)
    lengths = [len(x) for x in vectors]
    vectors = nn.utils.rnn.pad_sequence(vectors, batch_first=True)
    max_len = vectors.size(1)
    span_matrix = torch.zeros(len(batch), max_len, max_len, dtype=torch.long)
    for i, sl in enumerate(span_labels):
        span_matrix[i, :sl.size(0), :sl.size(1)] = sl
    mask = torch.arange(max_len)[None, :] < torch.tensor(lengths)[:, None]
    return vectors, span_matrix, mask, lengths


# ===============================
# 2. Span-based Biaffine NER Model
# ===============================
class SpanBiaffineNER(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, max_span_len=10):
        super().__init__()
        self.start_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.end_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.biaffine = nn.Bilinear(hidden_dim, hidden_dim, num_labels)
        self.max_span_len = max_span_len

    def forward(self, x, lengths):
        start_h = self.start_mlp(x)
        end_h = self.end_mlp(x)

        batch_size, seq_len, hidden_dim = start_h.size()
        num_labels = self.biaffine.out_features

        scores = torch.zeros(batch_size, seq_len, seq_len, num_labels, device=x.device)

        for b in range(batch_size):
            L = lengths[b]
            for i in range(L):
                j_max = min(i + self.max_span_len, L)
                for j in range(i, j_max):
                    scores[b, i, j, :] = self.biaffine(start_h[b, i, :], end_h[b, j, :])
        return scores


# ===============================
# 3. 训练
# ===============================
def train_model(train_loader, dev_loader, input_dim, num_labels, epochs=30, hidden_dim=128, lr=1e-4, device="cpu"):
    model = SpanBiaffineNER(input_dim, hidden_dim, num_labels).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 0 表示 O 类

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for vectors, span_matrix, mask, lengths in train_loader:
            vectors, span_matrix = vectors.to(device), span_matrix.to(device)
            optimizer.zero_grad()
            scores = model(vectors, lengths)  # (B, L, L, C)
            loss = loss_fn(scores.view(-1, num_labels), span_matrix.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}")
        evaluate_span(model, dev_loader, device)
    return model


# ===============================
# 4. 评估
# ===============================
def evaluate_span(model, loader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for vectors, span_matrix, mask, lengths in loader:
            vectors, span_matrix = vectors.to(device), span_matrix.to(device)
            scores = model(vectors, lengths)  # (B, L, L, C)
            preds = scores.argmax(dim=-1)

            for b in range(len(lengths)):
                L = lengths[b]
                for i in range(L):
                    for j in range(i, min(i + model.max_span_len, L)):
                        all_labels.append(span_matrix[b, i, j].item())
                        all_preds.append(preds[b, i, j].item())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", labels=list(range(1, model.biaffine.out_features))
    )
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


# ===============================
# 5. Run
# ===============================
if __name__ == "__main__":
    json_path = "/root/autodl-tmp/shiyan/dataset/result/fused.json"
    dataset = SpanNERDataset(json_path, max_span_len=10)
    train_size = int(0.8 * len(dataset))
    dev_size = len(dataset) - train_size
    train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    input_dim = len(dataset[0][0][0])  # token 向量维度
    num_labels = len(dataset.label2id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_model(train_loader, dev_loader, input_dim, num_labels, device=device)
