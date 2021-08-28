import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
from modules.encoders import DIVEncoder
import math


class GatedTransformer(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
        embed_dropout=0.0, div_dropout=0.0, attn_mask=False, use_disc=True):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        
        self.attn_mask = attn_mask

        # a pair of transformers plus a domain-invariant encoder
        self.l2other_layers = nn.ModuleList([])
        self.other2l_layers = nn.ModuleList([])
        self.div_encoders = nn.ModuleList([])

        for layer in range(layers):
            l2other_new = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            other2l_new = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)

            if layer == 0:
                new_div_layer = DIVEncoder(embed_dim, embed_dim, prj_type='linear', use_disc=use_disc)
            else:
                # TODO: Change dropout rate here
                # new_div_layer = DIVEncoder(embed_dim, embed_dim, prj_type='rnn', rnn_type='lstm', rdc_type='avg', use_disc=use_disc)
                new_div_layer = DIVEncoder(embed_dim, embed_dim, prj_type='rnn', rnn_type='gru', rdc_type='avg', use_disc=use_disc)

            self.l2other_layers.append(l2other_new)
            self.other2l_layers.append(other2l_new)
            self.div_encoders.append(new_div_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward_transformer(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x
    
    def forward(self, seq_l, seq_other, h_l, h_other, lengths=None, mask=None):
        """Forward 2 input modals thorugh the DIVencoder and Trnasformer 
        Args:
            input_l (FloatTensor): Representative tensor of the language modal
            input_other (FloatTensor): Representative tensor of the other modal
        """
        sim_loss_total = 0.0
        recon_loss_total = 0.0

        assert lengths is not None or mask is not None

        if mask is None:
            batch_size = lengths.size(0)
            mask = torch.arange(lengths.max()).repeat(batch_size, 1) < lengths.unsqueeze(-1)
            mask = mask.unsqueeze(-1).to(torch.float)
        elif lengths is None:
            lengths = mask.squeeze().sum(1)

        # output all shared encoding to train the discriminator
        # enc_l_all = []
        # enc_other_all = []

        # outputs of all discriminators in every layer
        disc_out_all = []
        disc_labels_all = []

        input_l, input_other = seq_l, seq_other

        # add residual connection
        # resl_all = []
        # resother_all = []

        for layer_i, (div_encoder, trans_l2other, trans_other2l) in enumerate(zip(self.div_encoders, self.l2other_layers,
         self.other2l_layers)):
            enc_l, enc_other, disc_out, disc_labels = div_encoder(h_l, h_other, lengths, mask) # batch_size, emb_size

            ctr_vec = torch.cat([enc_l, enc_other], dim=-1) # seq_len x bs x (2 * emb_size)

            disc_out_all.append(disc_out)
            disc_labels_all.append(disc_labels)

            # project language to other modals
            l2other = trans_other2l(input_other, x_k=input_l, x_v=input_l, ctr_vec=ctr_vec, lengths=lengths, mode='l2o')

            # project other modals to language
            other2l = trans_l2other(input_l, x_k=input_other, x_v=input_other, ctr_vec=ctr_vec, lengths=lengths, mode='o2l')

            # resl_all.append(other2l)
            # resother_all.append(l2other)

            # if layer_i > 0:
            #     for res_l in resl_all[:-1]:
            #         other2l += res_l
            #     for res_other in resother_all[:-1]:
            #         l2other += res_other

            input_l, input_other = other2l, l2other
            h_l, h_other = other2l, l2other

        disc_out_all = torch.cat(disc_out_all)
        disc_labels_all = torch.cat(disc_labels_all)

        return other2l, l2other, disc_out_all, disc_labels_all # placeholder for DIV output

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        # Memory and Compound control
        self.mem_proj = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.att_proj = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            nn.Sigmoid()           
        )

        # Dense Layer
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None, ctr_vec=None, lengths=None, mode='l2o'):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
            ctc_vec (Tensor): The control vector generated from DIV encoder
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        
        def get_mask(batch_size, len1, len2, lengths=None, lang_proj=False):
            """l2o means x is other modal
            Returns:
                mask (Tensor): An attention mask of size (len1, len2)
            """
            assert len1 == len2

            bool_mask1 = torch.cuda.BoolTensor(batch_size, len1, len2)
            bool_mask2 = torch.cuda.BoolTensor(batch_size, len1, len2)
            for i,j in enumerate(lengths):
                bool_mask2[i,:,:] = False
                if j<len1:
                    # bool_mask[i,j:,j:] is TOTALLY WRONG
                    bool_mask1[i,j:,:] = True
                    bool_mask1[i,:,j:] = True
                    bool_mask2[i,:,j:] = True   # only add minus infinity to the region of exceeded lengths in valid inputs
                bool_mask1[i,:j,:j] = False

            add_mask = torch.masked_fill(torch.zeros(bool_mask2.size()), bool_mask2, float('-inf'))
            mul_mask = torch.masked_fill(torch.ones(bool_mask1.size()), bool_mask1, 0.0)
            add_mask.detach_()
            mul_mask.detach_()

            # if projection to lengths, then some positions are projected to invalid space
            return add_mask, mul_mask

        # add heterogeneous mask, l2o means attention projects to other modal
        if mode == 'l2o':
            add_mask, mul_mask = get_mask(x.size(1), x.size(0), x_v.size(0), lengths=lengths, lang_proj=False)
        elif mode == 'o2l':
            add_mask, mul_mask = get_mask(x.size(1), x_v.size(0), x.size(0), lengths=lengths, lang_proj=True)

        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, add_mask=add_mask, mul_mask=mul_mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True) 
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, add_mask=add_mask, mul_mask=mul_mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)

        mem_gate = self.mem_proj(ctr_vec)
        fuse_gate = self.att_proj(ctr_vec)

        if ctr_vec is not None:
            x = x * fuse_gate + residual * mem_gate
        else:
            x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m

if __name__ == '__main__':
    encoder = GatedTransformer(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)