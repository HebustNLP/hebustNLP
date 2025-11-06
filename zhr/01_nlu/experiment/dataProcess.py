import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json


class JSONLSample:
    """读取 JSONL 格式数据"""
    def __init__(self, file_path, max_length):
        self.file_path = file_path
        self.max_length = max_length

    def read_jsonl(self):
        samples = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                utterance = data["utterance"]
                if isinstance(utterance, str):
                    utterance = utterance.split()
                slots = data["slots"]
                intents = data["intents"]
                explanation = data.get("explanation", "")

                # 截断
                if len(utterance) > self.max_length - 2:
                    utterance = utterance[:self.max_length - 2]
                    slots = slots[:self.max_length - 2]

                samples.append({
                    "utterance": utterance,
                    "slot_label": slots,
                    "intent_label": intents,
                    "explanation": explanation
                })
        return samples


class MyDataSet(Dataset):
    """Intent + Slot Dataset"""

    def __init__(self, args, file_path, intent_label_set, slot_label_set, tokenizer):
        self.samples = JSONLSample(file_path, args.max_seq_length).read_jsonl()
        self.tokenizer = tokenizer
        self.intent_label_id = {w: i for i, w in enumerate(intent_label_set)}
        self.slot_label_id = {w: i for i, w in enumerate(slot_label_set)}
        self.max_seq_length = args.max_seq_length

    def __getitem__(self, idx):
        sample = self.samples[idx]
        words = sample["utterance"]
        slot_tags = sample["slot_label"]
        explanation = sample["explanation"]

        # ===================== Tokenize 全句（含 explanation） =====================
        utterance_text = " ".join(words)
        full_text = f"{utterance_text} [SEP] {explanation}" if explanation else utterance_text
        full_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(full_text) + [self.tokenizer.sep_token]

        # ===================== 槽位标签构建 =====================
        # 只给 utterance 部分打标签，explanation 部分填 -100
        slot_label_ids = [-100]  # 对应 [CLS]
        slot_pointer = 0  # 指向 utterance 中的词

        for token in self.tokenizer.tokenize(utterance_text):
            if slot_pointer < len(slot_tags):
                tag = slot_tags[slot_pointer]
                tag_id = self.slot_label_id.get(tag, self.slot_label_id["O"])
            else:
                tag_id = self.slot_label_id["O"]
            slot_label_ids.append(tag_id)
            slot_pointer += 1

        # [SEP] 后面是 explanation 区域 → 全部填 -100
        sep_index = len(self.tokenizer.tokenize(utterance_text)) + 1
        rest_len = len(full_tokens) - sep_index - 1
        slot_label_ids += [-100] * (rest_len + 1)  # 含最后 [SEP]

        # 截断并对齐长度
        if len(full_tokens) > self.max_seq_length:
            full_tokens = full_tokens[:self.max_seq_length]
            slot_label_ids = slot_label_ids[:self.max_seq_length]
        else:
            pad_len = self.max_seq_length - len(full_tokens)
            full_tokens += [self.tokenizer.pad_token] * pad_len
            slot_label_ids += [-100] * pad_len

        # ===================== 转 ID =====================
        input_ids = self.tokenizer.convert_tokens_to_ids(full_tokens)
        attention_mask = [1 if t != self.tokenizer.pad_token else 0 for t in full_tokens]

        # ===================== intent label =====================
        intent_vec = [0] * len(self.intent_label_id)
        intents = sample["intent_label"]
        if isinstance(intents, str):
            intents = intents.split("#")
        for intent in intents:
            if intent in self.intent_label_id:
                intent_vec[self.intent_label_id[intent]] = 1

        # ===================== words_lengths & token_len =====================
        # 此处 words_lengths 只用于辅助，可用单词 token 长度近似
        words_lengths = [len(self.tokenizer.tokenize(w) or [self.tokenizer.unk_token]) for w in words]
        words_lengths_tensor = torch.tensor(words_lengths, dtype=torch.long)
        utterance_token_len = torch.tensor(len(full_tokens), dtype=torch.long)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(intent_vec, dtype=torch.float),
            torch.tensor(slot_label_ids, dtype=torch.long),
            words_lengths_tensor,
            utterance_token_len
        )

    def __len__(self):
        return len(self.samples)

def collate_fn(batch, pad_id, max_seq_len=64):
    """批处理函数"""
    input_ids, attention_mask, intent_label, slot_label, words_lengths, utterance_token_len = zip(*batch)

    def pad_tensor(t, pad_value=0):
        if t.size(0) < max_seq_len:
            return F.pad(t, (0, max_seq_len - t.size(0)), value=pad_value)
        else:
            return t[:max_seq_len]

    input_ids = torch.stack([pad_tensor(t, pad_id) for t in input_ids])
    attention_mask = torch.stack([pad_tensor(t, 0) for t in attention_mask])
    slot_label = torch.stack([pad_tensor(t, -100) for t in slot_label])
    intent_label = torch.stack(intent_label)

    # 对 words_lengths 进行 padding
    max_word_len = max(len(wl) for wl in words_lengths)
    padded_words_lengths = []
    for wl in words_lengths:
        padded_wl = F.pad(wl, (0, max_word_len - wl.size(0)), value=0)
        padded_words_lengths.append(padded_wl)
    words_lengths = torch.stack(padded_words_lengths)

    utterance_token_len = torch.stack(utterance_token_len)

    return input_ids, attention_mask, intent_label, slot_label, words_lengths, utterance_token_len