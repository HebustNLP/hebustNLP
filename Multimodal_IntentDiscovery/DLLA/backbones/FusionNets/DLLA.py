import torch
import torch.nn.functional as F
# from losses import loss_map
from ..SubNets.FeatureNets import BERTEncoder, SubNet, RoBERTaEncoder, AuViSubNet
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from .sampler import ConvexSampler
from torch import nn
from ..SubNets.AlignNets import AlignSubNet
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertLayer
from ..SubNets.FeatureNets import BERTEncoderSDIF, BertCrossEncoder

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
__all__ = ['dlla']

class DLLA(nn.Module):
    
    def __init__(self, args):

        super(DLLA, self).__init__()
        
        self.args = args
        base_dim = args.base_dim
        self.device = args.device
        
        self.num_heads = args.nheads
        self.attn_dropout = args.attn_dropout

        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.attn_mask = args.attn_mask
        self.layers_self = args.n_levels_self 
        self.self_num_heads = args.self_num_heads
        
        self.text_embedding = BERTEncoder(args)
        
        self.text_layer = nn.Linear(args.text_feat_dim, base_dim)
        self.video_layer = nn.Linear(args.video_feat_dim, base_dim)
        self.audio_layer = nn.Linear(args.audio_feat_dim, base_dim)

        self.num_modalities = num_modalities = 6
        self.modal_type_emb = nn.Embedding(num_modalities, base_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.pooler = nn.Linear(base_dim, base_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=base_dim, nhead=self.self_num_heads)
        self.self_att = nn.TransformerEncoder(encoder_layer, num_layers=self.layers_self)

        self.video2text_cross = BertCrossEncoder(
            args.cross_num_heads,
            base_dim,
            args.cross_dp_rate,
            n_layers=args.n_levels_cross
        )
        self.audio2text_cross = BertCrossEncoder(
            args.cross_num_heads,
            base_dim,
            args.cross_dp_rate,
            n_layers=args.n_levels_cross
        )
         
        self.v_encoder = self.get_transformer_encoder(base_dim, args.encoder_layers_1)
        self.a_encoder = self.get_transformer_encoder(base_dim, args.encoder_layers_1)

        self.deeplinear = nn.Linear(base_dim * 6, base_dim * 3)
        
        self.shared_embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(base_dim, base_dim),
        )

        self.fusion_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(args.hidden_dropout_prob),    
            nn.Linear(base_dim, base_dim),
        )

        self.mlp_project =  nn.Sequential(
                nn.Linear(base_dim, base_dim),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)  
           
    def forward(self, text, text_keyword,text_keyframe,video_feats, audio_feats): 
        bert_sent_mask = text[:,1]
        text = self.text_embedding(text) 
        text_keyword = self.text_embedding(text_keyword)
        text_keyframe = self.text_embedding(text_keyframe)

        context = torch.cat([text_keyword, text_keyframe], dim=1)  
        attn_output, _ = self.cross_attention(query=text, key=context, value=context)
        text_feats = text + attn_output  
        
        video = video_feats.float()
        audio = audio_feats.float()
        
        text_rep = text_feats[:, 0]
        text_seq = self.text_layer(text_feats) 
        text_rep = self.text_layer(text_rep)

        video_seq = self.video_layer(video)
        video_rep = video_seq.permute(1, 0, 2)
        video_rep= self.v_encoder(video_rep)[-1]
        
        audio_seq = self.audio_layer(audio)
        audio_rep = audio_seq.permute(1, 0, 2)
        audio_rep = self.a_encoder(audio_rep)[-1] 

        video_mask = (torch.sum(video, dim=-1) != 0).int()  
        extended_video_mask = video_mask.unsqueeze(1).unsqueeze(1)  
        extended_video_mask = extended_video_mask.to(dtype=next(self.parameters()).dtype)
        
        audio_mask = (torch.sum(audio, dim=-1) != 0).int()  
        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(1)  
        extended_audio_mask = extended_audio_mask.to(dtype=next(self.parameters()).dtype)

        video2text_seq = self.video2text_cross(text_seq, video_seq, extended_video_mask)
        audio2text_seq = self.audio2text_cross(text_seq, audio_seq, extended_audio_mask)

        shallow_seq = self.mlp_project(torch.cat([audio2text_seq, text_seq, video2text_seq], dim=1))

        text_mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True) 

        video2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), video2text_seq)
        video2text_rep = torch.sum(video2text_masked_output, dim=1, keepdim=False) / text_mask_len

        audio2text_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), audio2text_seq)
        audio2text_rep = torch.sum(audio2text_masked_output, dim=1, keepdim=False) / text_mask_len

        tri_cat_mask = torch.cat([bert_sent_mask, bert_sent_mask, bert_sent_mask], dim=-1)

        tri_mask_len = torch.sum(tri_cat_mask, dim=1, keepdim=True) 
        shallow_masked_output = torch.mul(tri_cat_mask.unsqueeze(2), shallow_seq)
        shallow_rep = torch.sum(shallow_masked_output, dim=1, keepdim=False) / tri_mask_len

        text_rep = text_rep.to(video2text_rep.dtype)
        video_rep = video_rep.to(video2text_rep.dtype)
        audio_rep = audio_rep.to(video2text_rep.dtype)
        audio2text_rep = audio2text_rep.to(video2text_rep.dtype)

        all_reps = torch.stack((text_rep, video_rep, audio_rep, video2text_rep, audio2text_rep, shallow_rep), dim=0)
        
        device = all_reps.device
        modality_ids = torch.arange(self.num_modalities, device=device).unsqueeze(1).repeat(1, all_reps.shape[1])  
        modality_emb = self.modal_type_emb(modality_ids)  
        all_reps = all_reps + modality_emb  
        all_hiddens = self.self_att(all_reps)
        deep_rep = all_hiddens.mean(dim=0)  

        features = self.fusion_layer(deep_rep)
        
        return features
        