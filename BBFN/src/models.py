import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.transformer import GatedTransformer
from modules.encoders import LanguageEmbeddingLayer, SeqEncoder, DIVEncoder

from transformers import BertModel, BertConfig
from utils import CMD, MSE

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        # TODO: attn_mask has some problem
        super(MULTModel, self).__init__()
        self.hp = hp = hyp_params

        self.orig_d_l, self.orig_d_a, self.orig_d_v = hp.orig_d_l, hp.orig_d_a, hp.orig_d_v
        
        # configuration for gated-transformer
        self.d_l = self.d_a = self.d_v = self.embed_dim = hp.attn_dim

        self.num_heads = hp.num_heads
        self.layers = hp.layers

        self.attn_dropout = hp.attn_dropout
        self.attn_dropout_a = hp.attn_dropout_a
        self.attn_dropout_v = hp.attn_dropout_v

        self.relu_dropout = hp.relu_dropout
        self.res_dropout = hp.res_dropout
        self.out_dropout = hp.out_dropout
        self.embed_dropout = hp.embed_dropout
        self.div_dropout = hp.div_dropout

        self.attn_mask = hp.attn_mask

        # configuration for input projection 1D-CNN (optional)
        self.l_ksize = hp.l_ksize
        self.v_ksize = hp.v_ksize
        self.a_ksize = hp.a_ksize

        self.last_bi = True

        combined_dim = 4 * self.d_l   # rigorously 2 * d_l + d_a + d_v
        
        output_dim = hp.output_dim        # This is actually not a hyperparameter :-)

        # 1. Language embedding layer
        self.embedding = LanguageEmbeddingLayer(self.hp)

        # 2. Temporal convolutional layers
        self.SequenceEncoder = SeqEncoder(self.hp)

        # 3. Shared Encoder (deprceated, inserted)

        # 4. Modal Interaction
        self.modal_interaction = nn.ModuleDict({
            'lv': self.get_network(layers=self.layers),
            'la': self.get_network(layers=self.layers)
        })
        
        # 5. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        # self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        # self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        # self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 6. Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=2):
        return GatedTransformer(embed_dim=self.embed_dim,
                                num_heads=self.num_heads,
                                layers=max(self.layers, layers),
                                attn_dropout=self.attn_dropout,
                                relu_dropout=self.relu_dropout,
                                res_dropout=self.res_dropout,
                                embed_dropout=self.embed_dropout,
                                div_dropout=self.div_dropout,
                                attn_mask=self.attn_mask)
        
    def _forward_last_pooling(self, tensor, lengths, mode='avg'):
        # tensor shape (bs, seqlen, emb_size)
        if mode == 'max':
            return tensor.max(1)
        elif mode == 'avg':
            B, L, E = tensor.size()
            mask = torch.arange(L).unsqueeze(0).expand(B,-1)
            mask = mask < lengths.unsqueeze(1)
            mask = mask.unsqueeze(2).expand(-1, -1, E)
            return (tensor * mask.float()).sum(1) / lengths.unsqueeze(1).float()
            
            
    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        def _pad_seq(video, acoustic, lengths):
            pld = (0, 0, 0, 0, 1, 1)
            pad_video = F.pad(video, pld, "constant", 0.0)
            pad_acoustic = F.pad(acoustic, pld, "constant", 0.0)
            lengths = lengths + 2
            return pad_video, pad_acoustic, lengths

        if self.hp.use_bert:
            enc_word = self.embedding(sentences, lengths, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size), (batch_size, emb_size)
            video, acoustic, lengths = _pad_seq(video, acoustic, lengths)
        else:
            enc_word = self.embedding(sentences, lengths, bert_sent, bert_sent_type, bert_sent_mask) # (seq_len, batch_size, emb_size)
            enc_word = F.dropout(enc_word, p=self.embed_dropout, training=self.training)

        # Project the textual/visual/audio features
        proj_res = self.SequenceEncoder(enc_word, acoustic, video, lengths)
        
        seq_l, h_l = proj_res['l']  # (seq_len, batch_size, emb_size), (batch_size, emb_size)
        seq_v, h_v = proj_res['v']
        seq_a, h_a = proj_res['a']
        
        mask = bert_sent_mask if self.hp.use_bert else None

        # last_a2l, last_l2a, enc_l_la, enc_a_la = self.modal_interaction['la'](seq_l, seq_a, h_l, h_a, lengths, mask)
        # last_v2l, last_l2v, enc_l_lv, enc_a_lv = self.modal_interaction['lv'](seq_l, seq_v, h_l, h_v, lengths, mask)
        last_a2l, last_l2a, disc_pred_la, disc_true_la = self.modal_interaction['la'](seq_l, seq_a, h_l, h_a, lengths, mask)
        last_v2l, last_l2v, disc_pred_lv, disc_true_lv = self.modal_interaction['lv'](seq_l, seq_v, h_l, h_v, lengths, mask)

        disc_preds = torch.cat((disc_pred_la, disc_pred_lv), dim=0)
        disc_trues = torch.cat((disc_true_la, disc_true_lv), dim=0)

        # if self.partial_mode == 3:
        #     last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        nbs = [i for i in range(last_a2l.size(1))]

        last_a2l = last_a2l.permute(1,0,2)[nbs,0,:]
        last_l2a = last_l2a.permute(1,0,2)[nbs,0,:]
        last_v2l = last_v2l.permute(1,0,2)[nbs,0,:]
        last_l2v = last_l2v.permute(1,0,2)[nbs,0,:]

        last_hs = torch.cat([last_a2l, last_l2a, last_v2l,last_l2v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output, disc_preds, disc_trues