import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataProcess import *
from wordRep import *

# ---------------- 参数 ----------------
class Args:
    model_name_or_path = "/root/autodl-tmp/code/experiment/model/bert"  # 本地BERT
    max_seq_length = 64

args = Args()

# ---------------- 读取标签文件 ----------------
def read_label_file(path):
    """逐行读取txt，去掉空格和换行"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

intent_label_set = read_label_file("/root/autodl-tmp/code/experiment/datasets/MixATIS_clean/intent_labels.txt")
slot_label_set = read_label_file("/root/autodl-tmp/code/experiment/datasets/MixATIS_clean/slot_labels.txt")

# ---------------- 初始化 tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# ---------------- 构建数据集 ----------------
dataset = MyDataSet(
    args,
    file_path="/root/autodl-tmp/code/experiment/datasets/MixATIS_clean/dev_100.jsonl",
    intent_label_set=intent_label_set,
    slot_label_set=slot_label_set,
    tokenizer=tokenizer
)

# 直接使用从dataProcess导入的collate_fn函数
def collate_fn_local(batch):
    return collate_fn(batch, pad_id=tokenizer.pad_token_id, max_seq_len=args.max_seq_length)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_local)

# ---------------- 初始化 WordRep ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WordRep(args).to(device)
model.eval()

# ---------------- 获取一批样本 ----------------
batch = next(iter(loader))
input_ids, attention_mask, intent_label, slot_label, words_lengths, utterance_token_len = batch

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
words_lengths = words_lengths.to(device)
utterance_token_len = utterance_token_len.to(device)

# ---------------- 前向编码 ----------------
with torch.no_grad():
    cls_output, context_embedding_align = model(input_ids, attention_mask, words_lengths, utterance_token_len)

# ---------------- 打印维度 ----------------
print("=== 输入维度 ===")
print("input_ids:", input_ids.shape)
print("attention_mask:", attention_mask.shape)
print("words_lengths:", words_lengths.shape)
print("utterance_token_len:", utterance_token_len.shape)

print("\n=== 输出维度 ===")
print("cls_output:", cls_output.shape)               # [B, H]
print("context_embedding_align:", context_embedding_align.shape)  # [B, max_word, H]