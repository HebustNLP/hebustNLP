# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel

# # ===============================
# # 1. Dataset
# # ===============================
# class TextDataset(Dataset):
#     def __init__(self, json_path, tokenizer, max_len=128):
#         with open(json_path, "r", encoding="utf-8") as f:
#             self.data = json.load(f)
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         text = item["text"]
#         label = item.get("label", None)  # 保留原始 label

#         encoding = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_len
#         )
#         return {
#             "text": text,
#             "label": label,
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0)
#         }

# # ===============================
# # 2. BERT + BiLSTM
# # ===============================
# class BertBiLSTM(nn.Module):
#     def __init__(self, bert_model, hidden_dim=256, output_dim=512):
#         super(BertBiLSTM, self).__init__()
#         self.bert = bert_model
#         self.lstm = nn.LSTM(
#             input_size=self.bert.config.hidden_size,
#             hidden_size=hidden_dim,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True
#         )
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
#         lstm_out, _ = self.lstm(last_hidden_state)    # (batch, seq_len, hidden_dim*2)
#         token_vecs = self.fc(lstm_out)                # (batch, seq_len, output_dim)
#         return token_vecs

# # ===============================
# # 3. Collate function
# # ===============================
# def my_collate_fn(batch):
#     texts = [x["text"] for x in batch]
#     labels = [x["label"] for x in batch]  # 原始 list，不转 tensor
#     input_ids = torch.stack([x["input_ids"] for x in batch])
#     attention_mask = torch.stack([x["attention_mask"] for x in batch])
#     return {
#         "text": texts,
#         "label": labels,
#         "input_ids": input_ids,
#         "attention_mask": attention_mask
#     }

# # ===============================
# # 4. 向量生成 + 保存  会生成一个512维度的上下文词序向量
# # ===============================
# if __name__ == "__main__":
#     pretrained_model = "/root/autodl-tmp/shiyan/code/bert-base-chinese"
#     tokenizer = BertTokenizer.from_pretrained(pretrained_model)
#     bert = BertModel.from_pretrained(pretrained_model)

#     dataset = TextDataset("/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train_02_semanticsMatrix.json", tokenizer, max_len=128)
#     dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=my_collate_fn)

#     model = BertBiLSTM(bert)
#     model.eval()

#     results = []
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch["input_ids"]
#             attention_mask = batch["attention_mask"]
#             texts = batch["text"]
#             labels = batch["label"]

#             token_vecs = model(input_ids, attention_mask)  # (batch, seq_len, 512)
#             token_vecs = token_vecs.cpu().numpy()

#             for text, label, vecs, mask in zip(texts, labels, token_vecs, attention_mask):
#                 valid_len = mask.sum().item()  # 去掉 padding
#                 results.append({
#                     "text": text,
#                     "label": label,
#                     "token_vectors": vecs[:valid_len].tolist()  # 每个 token 的 512 维向量
#                 })

#     # 保存新 JSON 文件
#     with open("/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train_03_tokens.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     print("✅ Token 级向量已保存到 new_token_vectors.json")


import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# ===============================
# 1. Dataset (支持 JSONL)
# ===============================
class TextDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 按行读取 jsonl
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item.get("label", None)  # 有些样本可能没有 label

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        return {
            "text": text,
            "label": label,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

# ===============================
# 2. BERT + BiLSTM
# ===============================
class BertBiLSTM(nn.Module):
    def __init__(self, bert_model, hidden_dim=256, output_dim=512):
        super(BertBiLSTM, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(last_hidden_state)    # (batch, seq_len, hidden_dim*2)
        token_vecs = self.fc(lstm_out)                # (batch, seq_len, output_dim)
        return token_vecs

# ===============================
# 3. Collate function
# ===============================
def my_collate_fn(batch):
    texts = [x["text"] for x in batch]
    labels = [x["label"] for x in batch]
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    return {
        "text": texts,
        "label": labels,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

# ===============================
# 4. 向量生成 + 保存
# ===============================
if __name__ == "__main__":
    pretrained_model = "/root/autodl-tmp/shiyan/code/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    bert = BertModel.from_pretrained(pretrained_model)

    dataset = TextDataset(
        "/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train_02_semanticsMatrix.jsonl",
        tokenizer,
        max_len=128
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=my_collate_fn)

    model = BertBiLSTM(bert)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            texts = batch["text"]
            labels = batch["label"]

            token_vecs = model(input_ids, attention_mask)  # (batch, seq_len, 512)
            token_vecs = token_vecs.cpu().numpy()

            for text, label, vecs, mask in zip(texts, labels, token_vecs, attention_mask):
                valid_len = mask.sum().item()  # 去掉 padding 部分
                results.append({
                    "text": text,
                    "label": label,
                    "token_vectors": vecs[:valid_len].tolist()
                })

    # 保存为 JSONL
    save_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train_03_tokens.jsonl"
    with open(save_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Token 级向量已保存到 {save_path}")
