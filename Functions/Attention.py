import torch
import math


def attention_matrix(q, k,  dropout=None, mask=None):
    """
    Compute the attention matrix using scaled dot-product attention or outer product
    """

    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply attention mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Masked entries get a very high negative attention score

    att_matrix = scores.softmax(dim=-1)

    # Apply dropout during training
    if dropout is not None:
        att_matrix = dropout(att_matrix)

    return att_matrix


def attention(q, k, v, dropout=None, mask=None):
    """
    Compute the attention-weighted values.
    """

    att_matrix = attention_matrix(q, k, dropout=dropout, mask=mask)
    weighted_sum = torch.matmul(att_matrix, v)

    return weighted_sum
