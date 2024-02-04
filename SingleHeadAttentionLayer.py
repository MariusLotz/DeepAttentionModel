import torch.nn as nn
import torch
from Attention import attention



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

        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attention_based_v = attention(q, k, v)

        return attention_based_v 
    

   
    

