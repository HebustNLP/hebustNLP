from transformers import AutoModel
from wordRep import *
from module.module import *

class JointModel(nn.Module):
    def __init__(self, args, num_intent_labels, num_slot_labels):
        super(JointModel, self).__init__()
        self.args = args
        self.wordrep = WordRep(args)
        hidden_size = self.wordrep.bert.config.hidden_size

        # 定义分类器
        self.intent_classifier = IntentClassifier(
            input_dim=hidden_size,
            num_labels=num_intent_labels,
            hidden_dim=args.hidden_dim_ffw,
            dropout_rate=args.dropout_rate
        )

        self.slot_classifier = SlotClassifier(
            input_dim=hidden_size,
            num_labels=num_slot_labels,
            hidden_dim=args.hidden_dim_ffw,
            dropout_rate=args.dropout_rate
        )

    def forward(self, input_ids, attention_mask, words_lengths, utterance_token_len):
        # 获取CLS和序列特征
        cls_output, context_embedding = self.wordrep(input_ids, attention_mask, words_lengths, utterance_token_len)

        # Intent prediction
        intent_logits = self.intent_classifier(cls_output)

        # Slot prediction
        slot_logits = self.slot_classifier(context_embedding)

        return intent_logits, slot_logits