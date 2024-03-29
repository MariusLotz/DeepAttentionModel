import torch.nn as nn
from Attention import attention

class MultiheadAttentionLayer(nn.Module):
    """
    Args:
        input_size (int): The input feature size.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability.
        trainable (bool, optional): Whether the parameters are trainable. Default is True.

    Attributes:
        input_size (int): The input feature size.
        num_heads (int): Number of attention heads.
        head_size (int): Size of each attention head.
        W_q (nn.Linear): Linear layer for query projection.
        W_k (nn.Linear): Linear layer for key projection.
        W_v (nn.Linear): Linear layer for value projection.
        W_o (nn.Linear): Linear layer for output projection.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, dim_e, dim_k, dim_v, dim_o, num_heads, bias=False, dropout=0.1, trainable=True):
        super(MultiheadAttentionLayer, self).__init__()

        assert dim_k % num_heads == 0, "Input size must be divisible by the number of heads."

        self.dim_e = dim_e
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_o = dim_o
        self.num_heads = num_heads
        self.head_size = dim_k // num_heads
    

        # Linear projections for Query, Key, and Value
        self.W_q = nn.Linear(dim_e, dim_k, bias)
        self.W_k = nn.Linear(dim_e, dim_k, bias)
        self.W_v = nn.Linear(dim_e, dim_v, bias)

        # Output projection
        self.W_o = nn.Linear(self.dim_v, self.dim_o, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        # Set requires_grad based on the trainable parameter
        for param in self.parameters():
            param.requires_grad = trainable

        # Parameter initialization
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        #nn.init.normal()
        #small_bias_value = 1e-5  # You can adjust this value based on your preference
        #self.W_q.bias.data.fill_(small_bias_value)
        #self.W_q.bias.data.fill_(small_bias_value)
        #self.W_q.bias.data.fill_(small_bias_value)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after multi-head attention.
        """
        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Split into multiple heads
        q = q.view(q.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(k.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)
        #print(v.size())
        v = v.view(v.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)

        # Scaled Dot-Product Attention
        attention_based_v = attention(q, k, v)

        # Concatenate and project back to the original size
        attention_based_v = attention_based_v.transpose(1, 2).contiguous().view(x.size(0), -1, self.dim_v)
        output = self.W_o(attention_based_v)
        #output = attention_based_v

        return output.squeeze(dim=1)
