import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from utils import CMD, MSE

class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self, hyp_params):
        super(LanguageEmbeddingLayer, self).__init__()
        self.hp = hp = hyp_params
        if hp.use_bert:
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:
            self.embed = nn.Embedding(len(hp.word2id), hp.orig_d_l)

    def forward(self, sentences, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        if self.hp.use_bert:
            bert_output = self.bertmodel(input_ids=bert_sent,
                                attention_mask=bert_sent_mask,
                                token_type_ids=bert_sent_type)
            bert_output = bert_output[0]

            # masked mean
            # masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            # mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
            # output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
            return bert_output
        else:
            # extract features from text modality
            output = self.embed(sentences)
            return output          

class SeqEncoder(nn.Module):
    """Encode all modalities with assigned network. The network will output encoded presentations
    of three modalities. The last hidden of LSTM/GRU generates as input to the control module,
    while separate sequence vectors are received by the transformer.
    TODO: Currently only use one component to encode (coded in "if...else..."). Try to preserve the 
    interface of both CNN and LSTM/GRU part. In case one approach can not generate satisfying separate vectors. 
    Then activate both components to generate all outputs.
    """
    def __init__(self, hyp_params):
        super(SeqEncoder, self).__init__()
        self.hp = hp = hyp_params

        self.orig_d_l, self.orig_d_a, self.orig_d_v = hp.orig_d_l, hp.orig_d_a, hp.orig_d_v
        self.d_l = self.d_a = self.d_v = hp.attn_dim
        self.proj_type = hp.proj_type.lower()

        def pad_size(ksize, in_size, out_size, stride=1, mode='same'):
            if mode.lower() == 'valid': return 0
            return (out_size - in_size + ksize - 1) // stride + 1

        ############################
        # TODO: use compound mode ##
        ############################
        if hp.proj_type == 'linear':
            self.proj_l = nn.Linear(self.orig_d_l, self.d_l)
            self.proj_v = nn.Linear(self.orig_d_v, self.d_v)
            self.proj_a = nn.Linear(self.orig_d_a, self.d_a)

            self.layer_norm_l = nn.LayerNorm(self.d_l)
            self.layer_norm_v = nn.LayerNorm(self.d_v)
            self.layer_norm_a = nn.LayerNorm(self.d_a)

        elif hp.proj_type == 'cnn':

            l_ksize = self.hp.l_ksize
            v_ksize = self.hp.v_ksize
            a_ksize = self.hp.a_ksize

            pad_l = int((l_ksize - 1) / 2)
            pad_v = int((v_ksize - 1) / 2)
            pad_a = int((a_ksize - 1) / 2)

            self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=l_ksize, padding=pad_l, bias=False)
            self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=a_ksize, padding=pad_a, bias=False)
            self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=v_ksize, padding=pad_v, bias=False)

        elif hp.proj_type in ['lstm', 'gru']:
            layers = self.hp.num_enc_layers
            rnn = nn.LSTM if self.hp.proj_type.lower() == 'lstm' else nn.GRU

            #####################################################################
            # TODO: 1) Use double layer                                         #
            #       2) Keep language unchanged while encode video and accoustic #
            #####################################################################L
            self.rnn_l = rnn(self.orig_d_l, self.orig_d_l, layers, bidirectional=True)
            self.rnn_v = rnn(self.orig_d_v, self.orig_d_v, layers, bidirectional=True)
            self.rnn_a = rnn(self.orig_d_a, self.orig_d_a, layers, bidirectional=True)

            self.rnn_dict = {'l':self.rnn_l, 'v':self.rnn_v, 'a':self.rnn_a}

            # dict that maps modals to corresponding networks
            self.linear_proj_l_h = nn.Linear(2*self.orig_d_l, self.d_l)
            self.linear_proj_v_h = nn.Linear(2*self.orig_d_v, self.d_v)
            self.linear_proj_a_h = nn.Linear(2*self.orig_d_a, self.d_a)

            self.linear_proj_l_seq = nn.Linear(2*self.orig_d_l, self.d_l)
            self.linear_proj_v_seq = nn.Linear(2*self.orig_d_v, self.d_v)
            self.linear_proj_a_seq = nn.Linear(2*self.orig_d_a, self.d_a)

            self.layer_norm_l = nn.LayerNorm(self.d_l)
            self.layer_norm_v = nn.LayerNorm(self.d_v)
            self.layer_norm_a = nn.LayerNorm(self.d_a)

            ##################################
            ##  TODO: add activations later ##
            ##################################
            self.activ = None

            self.proj_l_h = nn.Sequential(self.linear_proj_l_h, self.layer_norm_l)
            self.proj_v_h = nn.Sequential(self.linear_proj_v_h, self.layer_norm_v)
            self.proj_a_h = nn.Sequential(self.linear_proj_a_h, self.layer_norm_a)

            self.proj_l_seq = nn.Sequential(self.linear_proj_l_seq)
            self.proj_v_seq = nn.Sequential(self.linear_proj_v_seq)
            self.proj_a_seq = nn.Sequential(self.linear_proj_a_seq)

            self.proj_dict_h = {'l':self.proj_l_h, 'v':self.proj_v_h, 'a':self.proj_a_h}
            self.proj_dict_seq = {'l':self.proj_l_seq, 'v':self.proj_v_seq, 'a':self.proj_a_seq}

        else:
            raise ValueError("Encoder can only be cnn, lstm or rnn.")
    
    def forward_rnn_prj(self, input, lengths, modal):
        assert modal in "lva"

        lengths = lengths.to('cpu').to(torch.int64)
        packed_sequence = pack_padded_sequence(input, lengths)
        packed_h, h_out = self.rnn_dict[modal](packed_sequence)
        padded_h, _ = pad_packed_sequence(packed_h) # (seq_len, batch_size, emb_size)

        if self.proj_type == 'lstm':
            h_out = h_out[0]    # for lstm we don't need the cell state

        h_out = torch.cat((h_out[0], h_out[1]), dim=-1)

        h_out = self.proj_dict_h[modal](h_out)  
        h_out_seq = self.proj_dict_seq[modal](padded_h)

        return h_out_seq, h_out
    
    def _masked_avg_pool(self, lengths, mask, *inputs):
        """Perform a masked average pooling operation
        Args:
            lengths (Tensor): shape of (batch_size, max_seq_len) 
            inputs (Tuple[Tensor]): shape of (batch_size, max_seq_len, embedding)
        """
        res = []
        for t in inputs:
            masked_mul = t * mask # batch_size, seq_len, emb_size
            res.append(masked_mul.sum(1)/lengths.unsqueeze(-1))
        return res

    def forward_enc(self, input_l, input_a, input_v, lengths=None, mask=None):
        batch_size = lengths.size(0)
        if lengths is not None: 
            mask = torch.arange(lengths.max()).repeat(batch_size, 1) < lengths.unsqueeze(-1)
            mask = mask.unsqueeze(-1).to(torch.float)
        elif mask:  # use_bert
            lengths = mask.sum(1)

        if self.hp.use_bert:
            input_l = input_l.permute(1, 0, 2)
        
        if self.hp.proj_type == 'linear':
            perm = (1, 0, 2)
            l_seq = self.proj_l(input_l.permute(*perm))    # (bs, seq_len, attn_size)
            v_seq = self.proj_v(input_v.permute(*perm))
            a_seq = self.proj_a(input_a.permute(*perm))

            l_h, v_h, a_h = self._masked_avg_pool(lengths, mask, l_seq, v_seq, a_seq)
            
            l_seq, v_seq, a_seq = l_seq.permute(*perm), v_seq.permute(*perm), a_seq.permute(*perm)

        elif self.proj_type == 'cnn':
            perm1 = (1,2,0)
            perm2 = (0,2,1)
            perm3 = (1,0,2)

            # text input: (seq_len x bs x emb_size) -> (bs, emb_size, seq_len)
            # output -> (seq_len x bs x emb_size, bs x emb_size)
            l_seq = self.proj_l(input_l.permute(*perm1)).permute(*perm2)    # bs x seq_len x emb emb_size
            v_seq = self.proj_v(input_v.permute(*perm1)).permute(*perm2)
            a_seq = self.proj_a(input_a.permute(*perm1)).permute(*perm2)

            # maxpooling to generate output
            l_h, v_h, a_h = self._masked_avg_pool(lengths, mask, l_seq, v_seq, a_seq)
            
            l_seq, v_seq, a_seq = l_seq.permute(*perm3), v_seq.permute(*perm3), a_seq.permute(*perm3)


        # enocde with lstm or gru
        elif self.proj_type in ['lstm', 'gru']:
            l_seq, l_h = self.forward_rnn_prj(input_l, lengths, modal = 'l')
            v_seq, v_h = self.forward_rnn_prj(input_v, lengths, modal = 'v')
            a_seq, a_h = self.forward_rnn_prj(input_a, lengths, modal = 'a')

        return {'l': (l_seq, l_h), 'v':(v_seq, v_h), 'a':(a_seq, a_h)}

    ##################################
    # TODO: Correct input shapes here
    #################################
    def forward(self, input_l, input_v, input_a, lengths):
        """Encode Sequential data from all modalities
        Params:
            @input_l, input_a, input_v (Tuple(Tensor, Tensor)): 
            Tuple containing input and lengths of input. The vectors are in the size 
            (seq_len, batch_size, embed_size)
        Returns:
            @hidden_dic (dict): A dictionary contains hidden representations of all
            modalities and for each modality the value includes the hidden vector of
            the whole sequence and the final hidden (a.k.a sequence hidden).
            All hidden representations are projected to the same size for transformer
            and its controller use.
        """
        return self.forward_enc(input_l, input_v, input_a, lengths)

class DIVEncoder(nn.Module):
    """Construct a domain-invariant encoder for all modalities. Forward and return domain-invariant
    encodings for these modality with similarity and reconstruction (optional) loss.
    Args:
        in_size (int): hidden size of input vector(s), of which is a representation for each modality
        out_size (int): hidden_size
    """
    def __init__(self, in_size, out_size, prj_type='linear', use_disc=False, 
                rnn_type=None, rdc_type=None, p_l=0.0, p_o=0.0):
        super(DIVEncoder, self).__init__()
        self.prj_type = prj_type
        self.reduce = rdc_type
        self.use_disc = use_disc

        self.in_size = in_size
        self.out_size = out_size
        
        if prj_type == 'linear':
            self.encode_l = nn.Linear(in_size, out_size)
            self.encode_o = nn.Linear(in_size, out_size)

        elif prj_type == 'rnn':
            self.rnn_type = rnn_type.upper()
            rnn = getattr(nn, self.rnn_type)

            self.encode_l = rnn(input_size=in_size,
                                hidden_size=out_size,
                                num_layers=1,
                                dropout=p_l,
                                bidirectional=True)
            self.encode_o = rnn(input_size=in_size,
                                hidden_size=out_size,
                                num_layers=1,
                                dropout=p_o,
                                bidirectional=True)
        
        if use_disc:
            self.discriminator = nn.Sequential(
                nn.Linear(out_size, 4*out_size),
                nn.ReLU(),
                nn.Linear(4*out_size, 1),
                nn.Sigmoid()
            )
        
        self.dropout_l = nn.Dropout(p_l)
        self.dropout_o = nn.Dropout(p_o)

    def _masked_avg_pool(self, lengths, mask, *inputs):
        """Perform a masked average pooling operation
        Args:
            lengths (Tensor): A tensor represents the lengths of input sequence with size (batch_size,)
            mask (Tensor):
            inputs (Tuple[Tensor]): Hidden representations of input sequence with shape of (max_seq_len, batch_size, embedding)
        """
        res = []

        # bert mask only has 2 dimensions
        if len(mask.size()) == 2:
            mask = mask.unsqueeze(-1)

        for t in inputs:
            masked_mul = t.permute(1,0,2) * mask # batch_size, seq_len, emb_size
            res.append(masked_mul.sum(1)/lengths.unsqueeze(-1)) # batch_size, emb_size
        return res
    
    def _forward_rnn(self, rnn, input, lengths):
        packed_sequence = pack_padded_sequence(input, lengths.cpu()) 
        packed_h, h_out = rnn(packed_sequence)
        padded_h, _ = pad_packed_sequence(packed_h)
        return padded_h, h_out

    def forward(self, input_l, input_o, lengths, mask):
        if self.prj_type == 'linear':
            if self.reduce == 'avg':
                avg_l, avg_o = self._masked_avg_pool(lengths, mask, input_l, input_o)
            elif self.reduce is None:
                avg_l, avg_o = input_l, input_o
            else:
                raise ValueError("Reduce method can be either average or none if projection type is linear")
            enc_l = self.encode_l(avg_l)
            enc_o = self.encode_o(avg_o)

        elif self.prj_type == 'rnn':
            out_l, h_l = self._forward_rnn(self.encode_l, input_l, lengths)
            out_o, h_o = self._forward_rnn(self.encode_o, input_o, lengths)
            if self.reduce == 'last':
                h_l_last = h_l[0] if isinstance(h_l, tuple) else h_l
                h_o_last = h_o[0] if isinstance(h_o, tuple) else h_o
                enc_l = (h_l_last[0] + h_l_last[1]) / 2
                enc_o = (h_o_last[0] + h_o_last[1]) / 2
            elif self.reduce == 'avg':
                enc_l, enc_o = self._masked_avg_pool(lengths, mask, out_l, out_o)
                enc_l = (enc_l[:,:enc_l.size(1) // 2] + enc_l[:,enc_l.size(1) // 2:]) / 2
                enc_o = (enc_o[:,:enc_o.size(1) // 2] + enc_o[:,enc_o.size(1) // 2:]) / 2
            else:
                raise ValueError("Reduce method can be either last or average if projection type is linear")              

        enc_l, enc_o = self.dropout_l(enc_l), self.dropout_o(enc_o)

        if self.use_disc:
            # generate discriminator output together with its labels
            disc_out = self.discriminator(torch.cat((enc_l, enc_o), dim=0)).squeeze() # (2 * batch_size, 1)
            batch_size = enc_l.size(0)
            disc_labels = torch.cat([torch.Tensor([0]).expand(size=(batch_size,)),  \
                torch.Tensor([1]).expand(size=(batch_size,))], dim=0).squeeze()

        return enc_l, enc_o, disc_out, disc_labels