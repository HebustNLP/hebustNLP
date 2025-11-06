import torch
from torch import nn
from transformers import AutoModel


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class WordRep(nn.Module):
    """
    句子编码（intent）使用 utterance + explanation 的整体语义
    槽位编码（slot）只使用 utterance 部分的 token 表示
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        self.pooling = MeanPooling()

    def forward(self, input_ids, attention_mask, words_lengths, utterance_token_len):
        """
        Args:
            input_ids: [B, L]  => [CLS] utterance [SEP] explanation [SEP]
            attention_mask: [B, L]
            words_lengths: [B, max_word] 每个词对应的 subword 数
            utterance_token_len: [B] utterance部分的token长度（不含[CLS]）
        Returns:
            cls_output: [B, H] — 全句语义 (utterance + explanation)
            context_embedding_align: [B, max_word, H] — 槽位词级特征（仅utterance部分）
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=False)
        last_hidden_state = outputs.last_hidden_state  # [B, L, H]

        # ---- 1️⃣ 意图表示 ---- #
        # 让 [CLS] 关注 utterance + explanation，全局表示
        cls_output = last_hidden_state[:, 0, :]  # 直接取 [CLS]，比 mean pooling 更贴近分类任务
        # 如果想更稳定，可改成 self.pooling(last_hidden_state, attention_mask)

        # ---- 2️⃣ 槽位表示 ---- #
        batch_size, max_subword_len, hidden_size = last_hidden_state.size()
        max_word = words_lengths.size(1)
        align_matrix = torch.zeros((batch_size, max_word, max_subword_len), device=last_hidden_state.device)

        for i in range(batch_size):
            # utterance 的 token 范围：从 [CLS] 后开始，到第一个 [SEP] 之前
            utter_len = int(utterance_token_len[i].item())
            utter_len = min(utter_len, max_subword_len - 1)

            word_lens = words_lengths[i].tolist()
            cursor = 1  # 跳过 [CLS]
            for j, wlen in enumerate(word_lens):
                if wlen <= 0:
                    continue
                start = cursor
                end = min(cursor + wlen, utter_len + 1)
                if start >= utter_len + 1:
                    break
                align_matrix[i, j, start:end] = 1.0 / max(1, end - start)
                cursor += wlen
                if cursor > utter_len + 1:
                    break

        # 只映射 utterance 的 token embedding
        context_embedding_align = torch.bmm(align_matrix, last_hidden_state)

        return cls_output, context_embedding_align
