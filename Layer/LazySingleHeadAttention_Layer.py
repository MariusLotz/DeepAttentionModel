import torch.nn as nn
from DeepAttentionModel.Functions.Attention import attention

class LazySingleHeadAttention_Layer(nn.Module):
    def __init__(self, output_size, bias=False, dropout=0.1, trainable=True):
        super(LazySingleHeadAttention_Layer, self).__init__()
        self.output_size = output_size
        self.bias = bias
        self.W_q = nn.LazyLinear(self.output_size, self.bias)
        self.W_k = nn.LazyLinear(self.output_size, self.bias)
        self.W_v = nn.LazyLinear(self.output_size, self.bias)
        self.dropout = nn.Dropout(p=dropout)

        """ Set requires_grad based on the trainable parameter
        for param in self.parameters():
            param.requires_grad = trainable

        # Parameter initialization
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
       
        #nn.init.normal()
        #small_bias_value = 1e-5  # You can adjust this value based on your preference
        #self.W_q.bias.data.fill_(small_bias_value)
        #self.W_q.bias.data.fill_(small_bias_value)
        #self.W_q.bias.data.fill_(small_bias_value)"""


    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Scaled Dot-Product Attention
        attention_based_v = attention(q, k, v)

        return attention_based_v
