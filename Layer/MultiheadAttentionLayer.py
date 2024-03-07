import torch
import torch.nn as nn
from Functions.Attention import attention

class MultiheadAttentionLayer(nn.Module):
    def __init__(self, dim_e, dim_k, dim_v, dim_o, num_heads, bias=False, dropout=0.1, trainable=True):
        super(MultiheadAttentionLayer, self).__init__()

        assert dim_k % num_heads == 0, "Input size must be divisible by the number of heads."

        self.dim_e = dim_e
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_o = dim_o
        self.num_heads = num_heads
        self.head_size = dim_k // num_heads
        if self.dim_e == None:
            self.W_q = nn.LazyLinear(dim_k, bias)
            self.W_k = nn.LazyLinear(dim_k, bias)
            self.W_v = nn.LazyLinear(dim_v, bias)
        else:
            self.W_q = nn.Linear(dim_e, dim_k, bias)
            self.W_k = nn.Linear(dim_e, dim_k, bias)
            self.W_v = nn.Linear(dim_e, dim_v, bias)
        self.W_o = nn.Linear(self.dim_v, self.dim_o, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Split into multiple heads
        q = q.view(q.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(k.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(v.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)

        # Scaled Dot-Product Attention
        attention_based_v = attention(q, k, v)

        # Concatenate and project back to the original size
        attention_based_v = attention_based_v.transpose(1, 2).contiguous().view(x.size(0), -1, self.dim_v)
        output = self.W_o(attention_based_v)
  
        return output.squeeze(dim=1)

def test():
    dim_e = 8  # Dimension of the input
    dim_k = 4  # Dimension of keys
    dim_v = 4  # Dimension of values
    dim_o = 1  # Dimension of output
    num_heads = 2  # Number of attention heads
 
    # Instantiate the MultiheadAttentionLayer
    attention_layer = MultiheadAttentionLayer(None, dim_k, dim_v, dim_o, num_heads)

    # Create a test batch of size 2
    test_batch = torch.rand((2, 10, dim_e))

    # Forward pass through the attention layer
    output = attention_layer(test_batch)

    # Print the output shape
    print("Input shape:", test_batch.size())
    print("Output shape:", output.size())

if __name__ == "__main__":
    test()