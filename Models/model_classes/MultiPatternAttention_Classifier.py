import torch.nn as nn
import torch
import torch.nn.init as init
import math


def closest_power_of_2(z):
        exponent = math.floor(math.log2(z))
        lower_power = 2 ** exponent
        upper_power = 2 ** (exponent + 1)
        return (lower_power, exponent) if abs(z - lower_power) <= abs(z - upper_power) else (upper_power, exponent)


class MultiPatternAttention_Classifier(nn.Module):
    def __init__(self, T=None, d_I=None, d_e=None, h=None):
        super(MultiPatternAttention_Classifier, self).__init__()
        self.h = h if h else 16
        self.T = T
        self.d_I = d_I
        self.d_e = d_e
        self.linear_layers = None
        self.MH_Attention = None
        self.last_layer = None
    
    def finding_h_and_d_e(self, c=2):
        d_e_benchmark = max(2, int((self.T / self.d_I) * c))
        #print(d_e_benchmark)
        d_e, exponent = closest_power_of_2(d_e_benchmark)
        h = 2 ** (exponent // 2) * 2
        return d_e, h

    def forward(self, x):
        if self.T is None:
            self.T = x.size(1)
            if self.d_I is None:
                self.d_I = 2 * int(math.log2(self.T))  # amount of linear projection
            if self.d_e is None:
                self.d_e, self.h = self.finding_h_and_d_e()
            self.linear_layers = nn.ModuleList([nn.Linear(self.T, self.d_e) for _ in range(self.d_I)])
            self.MH_Attention = nn.MultiheadAttention(self.d_e, self.h)
            self.last_layer = nn.Linear(self.d_I * self.d_e, 1)

        pattern_list = [linear_layer(x) for linear_layer in self.linear_layers]
        y = torch.stack(pattern_list, dim=1)
        print(y.size())
        y = self.MH_Attention(y, y, y)[0]
        y = torch.flatten(y,start_dim=-2)
        print(self.h)
        return self.last_layer(y)


def test_model():
    batch_size = 2
    seq_len = 12999
    input_tensor = torch.randn(batch_size, seq_len)  # Assuming input dimension is 10
    model = MultiPatternAttention_Classifier()

    # Forward pass
    #print("input_size:", input_tensor.size())
    output = model(input_tensor)
   # print("output_size:", output.size())

if __name__ == "__main__":
    test_model()