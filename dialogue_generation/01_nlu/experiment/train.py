import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from dataProcess import MyDataSet, collate_fn
from jointModel import JointModel 
from argparse import Namespace

# ============================
# 1. 读取标签文件
# ============================
def read_labels(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


# ============================
# 2. 训练函数
# ============================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 读取标签 ---
    intent_label_set = read_labels(args.intent_label_path)
    slot_label_set = read_labels(args.slot_label_path)

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # --- 构建dataset & dataloader ---
    train_dataset = MyDataSet(args, args.train_file, intent_label_set, slot_label_set, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_id=tokenizer.pad_token_id, max_seq_len=args.max_seq_length)
    )

    # --- 模型 ---
    model = JointModel(args, num_intent_labels=len(intent_label_set), num_slot_labels=len(slot_label_set))
    model.to(device)

    # --- 优化器与学习率调度 ---
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # --- 损失函数 ---
    criterion_intent = nn.BCEWithLogitsLoss()  # 多标签意图
    criterion_slot = nn.CrossEntropyLoss(ignore_index=-100)  # 槽位忽略padding

    # --- 训练循环 ---
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss, total_intent_loss, total_slot_loss = 0, 0, 0

        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, intent_label, slot_label, words_lengths, utterance_token_len = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            intent_label, slot_label = intent_label.to(device), slot_label.to(device)
            words_lengths, utterance_token_len = words_lengths.to(device), utterance_token_len.to(device)

            optimizer.zero_grad()
            intent_logits, slot_logits = model(input_ids, attention_mask, words_lengths, utterance_token_len)

            # --- loss计算 ---  
            intent_loss = criterion_intent(intent_logits, intent_label.float())
            
            # 修复slot_loss计算，确保形状匹配
            aligned_logits = []
            aligned_labels = []
            for i in range(slot_logits.shape[0]):
                # 只取utterance部分的实际长度（不包含padding）
                actual_len = min(int(utterance_token_len[i].item()), slot_logits.shape[1], slot_label.shape[1])
                if actual_len > 0:
                    # 截取每个样本的实际长度部分
                    aligned_logits.append(slot_logits[i, :actual_len])
                    aligned_labels.append(slot_label[i, :actual_len])
            
            # 拼接所有样本的实际长度部分
            if aligned_logits:
                aligned_logits = torch.cat(aligned_logits, dim=0)
                aligned_labels = torch.cat(aligned_labels, dim=0)
                slot_loss = criterion_slot(aligned_logits.view(-1, aligned_logits.shape[-1]), aligned_labels.view(-1))
                # slot_loss = criterion_slot(
                #     slot_logits.view(-1, slot_logits.shape[-1]),
                #     slot_label.view(-1)
                # )
            else:
                slot_loss = torch.tensor(0.0, device=device)

            loss = intent_loss + slot_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_intent_loss += intent_loss.item()
            total_slot_loss += slot_loss.item()

            if step % 10 == 0:
                print(f"[Epoch {epoch+1}/{args.num_train_epochs}] Step {step}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Intent: {intent_loss.item():.4f} | Slot: {slot_loss.item():.4f}")

        print(f"Epoch {epoch+1} finished. Avg Loss={total_loss/len(train_loader):.4f}")

    # --- 保存模型 ---
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "joint_model.pt"))
    print("✅ 模型已保存至:", args.output_dir)


# ============================
# 3. 推理预测示例（softmax）
# ============================
def predict(model, tokenizer, sentence, explanation, intent_labels, slot_labels, device):
    model.eval()
    with torch.no_grad():
        tokens = sentence.split()
        inputs = tokenizer(
            f"{' '.join(tokens)} [SEP] {explanation}",
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        ).to(device)

        word_pieces = [tokenizer.tokenize(w) or [tokenizer.unk_token] for w in tokens]
        dummy_words_lengths = torch.tensor([[len(p) for p in word_pieces]], device=device)
        utter_len = torch.tensor([sum(len(p) for p in word_pieces) + 1], device=device)

        intent_logits, slot_logits = model(
            inputs["input_ids"], inputs["attention_mask"], dummy_words_lengths, utter_len
        )

         # --- Intent prediction ---
        intent_probs = torch.sigmoid(intent_logits).cpu().numpy()[0]
        intent_pred = [intent_labels[i] for i, p in enumerate(intent_probs) if p > 0.5]

        # --- Slot prediction ---
        slot_probs = F.softmax(slot_logits, dim=-1)
        slot_preds = torch.argmax(slot_probs, dim=-1)[0].cpu().numpy()

        utter_token_len = int(utter_len.cpu().numpy()[0])
        slot_pred_labels = [slot_labels[i] for i in slot_preds[:utter_token_len - 1]]

    print("🔹输入句子:", sentence)
    print("🔹预测意图:", intent_pred)
    print("🔹预测槽位标签:", slot_pred_labels)


# ============================
# 4. 主入口
# ============================
if __name__ == "__main__":
    args = Namespace(
        model_name_or_path="/root/autodl-tmp/code/experiment/model/bert",
        train_file="/root/autodl-tmp/code/experiment/datasets/MixATIS_clean/glm_explaind.jsonl",
        intent_label_path="/root/autodl-tmp/code/experiment/datasets/MixATIS_clean/intent_labels.txt",
        slot_label_path="/root/autodl-tmp/code/experiment/datasets/MixATIS_clean/slot_labels.txt",
        output_dir="checkpoints/",
        batch_size=8,
        learning_rate=3e-5,
        max_seq_length=64,
        hidden_dim_ffw=256,
        dropout_rate=0.3,
        num_train_epochs=3
    )

    train(args)

    # --- 推理演示 ---
    intent_labels = read_labels(args.intent_label_path)
    slot_labels = read_labels(args.slot_label_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = JointModel(args, num_intent_labels=len(intent_labels), num_slot_labels=len(slot_labels))
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "joint_model.pt")))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    predict(model, tokenizer,
             "show me all economy prices from dallas to baltimore and also i need information for ground transportation denver colorado",
             "The user wants to see all economy flight prices from Dallas to Baltimore. Additionally, they are requesting information about ground transportation options in Denver, Colorado. The user is looking for both flight and ground transportation details.",
               intent_labels, slot_labels, 
               device="cuda" if torch.cuda.is_available() else "cpu")
