import torch
import torch.nn as nn
import torch.nn.functional as F
from Layer.WaveletMatrixLayer import WaveletMatrixLayer
from Layer.MultiheadAttentionLayer import MultiheadAttentionLayer

class WaveletMatrix_N_MultiheadAttention(nn.Module):
    def __init__(self, projection_size, heads=8):
        super(WaveletMatrix_N_MultiheadAttention, self).__init__()
        self.projection_size = projection_size
        self.Signal_to_WaveletMatrix = WaveletMatrixLayer()
        self.AttentionLayer = MultiheadAttentionLayer(None, projection_size, projection_size, projection_size, heads)
        self.linear_1 = nn.LazyLinear(self.projection_size)
        self.linear_2 = nn.LazyLinear(1)

    def forward(self, x):        
        x = self.Signal_to_WaveletMatrix(x)
        x = self.AttentionLayer(x)
        x = x.flatten(start_dim=1)
        x = F.sigmoid(self.linear_1(x))
        x = F.sigmoid(self.linear_2(x))
        return x
    

def test():
    signal_size = 8  # Define the signal size for testing
    model = WaveletMatrix_N_MultiheadAttention(64)
    batch_size = 2
    input_signals = torch.randn(batch_size, signal_size)
    output = model(input_signals)
    print(input_signals)
    print(output)

if __name__ == "__main__":
    test()
  