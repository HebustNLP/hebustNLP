from .MCN import MCN
from .DLLA import DLLA
from .BERT_TEXT import BERT_TEXT
multimodal_methods_map = {
    'mcn': MCN,
    'dlla': DLLA,
    'text': BERT_TEXT,
}