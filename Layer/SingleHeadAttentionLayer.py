import torch.nn as nn
import torch
from Attention import attention
import numpy as np
import torch.nn.functional as F



def attention_like_matrix(q, k,  dropout=None, mask=None):
    """
    Compute the attention matrix using scaled dot-product attention or outer product
    """

    scores = torch.ger(q, k)

    # Apply attention mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Masked entries get a very high negative attention score

    att_matrix = scores.softmax(dim=-1)

    # Apply dropout during training
    if dropout is not None:
        att_matrix = dropout(att_matrix)

    return att_matrix


def attention_like(q, k, v, dropout=None, mask=None):
    """
    Compute the attentionlike-weighted values.
    """

    att_matrix = attention_like_matrix(q, k, dropout=dropout, mask=mask)
    weighted_sum = torch.matmul(v, att_matrix)

    return weighted_sum


class SingleAttentionLikeLayer(nn.Module):
    def __init__(self, in_size, out_size, bias=False, dropout=0.05, trainable=True):
        super(SingleAttentionLikeLayer, self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.dropout = nn.Dropout(p=dropout)

        # Linear projections for Query, Key, and Value
        self.W_q = nn.Linear(in_size, out_size, bias)
        self.W_k = nn.Linear(in_size, out_size, bias)
        self.W_v = nn.Linear(in_size, out_size, bias)

        # Set requires_grad based on the trainable parameter
        for param in self.parameters():
            param.requires_grad = trainable

        # Parameter initialization
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        

    def forward(self, x):

        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attention_based_v = attention(q, k, v)

        return attention_based_v      

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = SingleAttentionLayer(d_model, d_model, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, N, d_model, dropout=0.1):
        self.dff= d_model
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, self.dff, dropout) for _ in range(N)])
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SingleAttentionLayer(nn.Module):
    def __init__(self, dim_e, dim_k, dim_v, bias=False, dropout=0.05, trainable=True):
        super(SingleAttentionLayer, self).__init__()
        self.dim_e = dim_e
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dropout = nn.Dropout(p=dropout)

        # Linear projections for Query, Key, and Value
        self.W_q = nn.Linear(dim_e, dim_k, bias)
        self.W_k = nn.Linear(dim_e, dim_k, bias)
        self.W_v = nn.Linear(dim_e, dim_v, bias)

        # Set requires_grad based on the trainable parameter
        for param in self.parameters():
            param.requires_grad = trainable

        # Parameter initialization
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
       
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attention_based_v = attention(q, k, v)

        return attention_based_v 
    

   
    

