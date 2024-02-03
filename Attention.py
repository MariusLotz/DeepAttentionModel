import torch
import math

def attention_matrix(q, k, dropout=None, mask=None):
    """
    Compute the attention matrix using scaled dot-product attention.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        dropout (nn.Dropout, optional): Dropout layer. Default is None.
        mask (torch.Tensor, optional): Mask tensor for attention. Default is None.

    Returns:
        torch.Tensor: Attention matrix.
    """
    d_k = q.size(-1)
    print(d_k)
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

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        dropout (nn.Dropout, optional): Dropout layer. Default is None.
        mask (torch.Tensor, optional): Mask tensor for attention. Default is None.

    Returns:
        torch.Tensor: Attention-weighted sum of values.
    """
    att_matrix = attention_matrix(q, k, dropout=dropout, mask=mask)
    print(att_matrix)
    print(v)
    weighted_sum = torch.matmul(att_matrix, v)
    print(weighted_sum)
    return weighted_sum
