import torch
import torch.nn as nn
from module.feedForward import FeedforwardLayer


class IntentClassifier(nn.Module):
    """句子级 Intent 分类器"""
    def __init__(self, input_dim, num_labels, hidden_dim=256, dropout_rate=0.1):
        super(IntentClassifier, self).__init__()
        self.ffn = FeedforwardLayer(input_dim, hidden_dim, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(input_dim, num_labels)  # ✅ 匹配FFN输出维度

    def forward(self, x):
        """
        Args:
            x: [B, H] — CLS 或句子级表示
        Returns:
            logits: [B, num_labels]
        """
        x = self.ffn(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


class SlotClassifier(nn.Module):
    """逐 token 槽位分类器"""
    def __init__(self, input_dim, num_labels, hidden_dim=256, dropout_rate=0.1):
        super(SlotClassifier, self).__init__()
        self.ffn = FeedforwardLayer(input_dim, hidden_dim, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(input_dim, num_labels)  # ✅ 匹配FFN输出维度

    def forward(self, x):
        """
        Args:
            x: [B, L, H] — 每个 token 的向量
        Returns:
            logits: [B, L, num_labels]
        """
        x = self.ffn(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits
