import torch.nn as nn
import torch
import torch.nn.init as init
import math
from Layer.MultiheadAttentionLayer import MultiheadAttentionLayer
import pywt
import numpy as np


def closest_power_of_2(z):
        exponent = math.floor(math.log2(z))
        lower_power = 2 ** exponent
        upper_power = 2 ** (exponent + 1)
        return (lower_power, exponent) if abs(z - lower_power) <= abs(z - upper_power) else (upper_power, exponent)


class MultiPatternAttention_Classifier_with_Wavelettrafo(nn.Module):
    def __init__(self, d_I=16, d_e=64, h=4,  wavelet='db1'):
        super(MultiPatternAttention_Classifier_with_Wavelettrafo, self).__init__()
        self.d_I = d_I
        self.d_e = d_e
        self.d_k = int(0.25 * d_e)
        self.h = h
        self.wavelet = wavelet
        self.linear_layers = nn.ModuleList([nn.LazyLinear(self.d_e) for _ in range(self.d_I)])
        self.MH_Attention = MultiheadAttentionLayer(self.d_e, self.d_k, self.d_k, 1, self.h)
        self.last_layer = nn.Linear(self.d_I, 1)
    
    def forward(self, x):
        coeffs_list = np.array([np.concatenate(pywt.wavedec(signal.numpy(), self.wavelet)) for signal in x])
        x = torch.tensor(coeffs_list)
        pattern_list = [linear_layer(x) for linear_layer in self.linear_layers]
        x = torch.stack(pattern_list, dim=1)
        x = torch.tanh(x)
        x = self.MH_Attention(x)
        x = torch.flatten(x,start_dim=-2)
        x = self.last_layer(x)
        return torch.sigmoid(x)


class MultiPatternAttention_Classifier(nn.Module):
    def __init__(self, d_I=16, d_e=64, h=4,  wavelet='db1'):
        super(MultiPatternAttention_Classifier, self).__init__()
        self.d_I = d_I
        self.d_e = d_e
        self.d_k = int(0.25 * d_e)
        self.h = h
        self.wavelet = wavelet
        self.linear_layers = nn.ModuleList([nn.LazyLinear(self.d_e) for _ in range(self.d_I)])
        self.MH_Attention = MultiheadAttentionLayer(self.d_e, self.d_k, self.d_k, 1, self.h)
        self.last_layer = nn.Linear(self.d_I, 1)
    
    def forward(self, x):
        pattern_list = [linear_layer(x) for linear_layer in self.linear_layers]
        x = torch.stack(pattern_list, dim=1)
        x = torch.tanh(x)
        x = self.MH_Attention(x)
        x = torch.flatten(x,start_dim=-2)
        x = self.last_layer(x)
        return torch.sigmoid(x)
    

def test_model():
    batch_size = 2
    seq_len = 16
    input_tensor = torch.randn(batch_size, seq_len)  # Assuming input dimension is 10
    model = MultiPatternAttention_Classifier()

    # Forward pass
    print("input_size:", input_tensor.size())
    output = model(input_tensor)
    print("output_size:", output.size())

if __name__ == "__main__":
    test_model()