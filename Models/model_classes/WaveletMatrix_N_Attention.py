import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepAttentionModel.Functions.Signal_to_Features import WaveletMatrixLayer
from DeepAttentionModel.Models.model_classes.L2BinaryClassifier import L2BinaryClassifier
from DeepAttentionModel.Layer.LazySingleHeadAttention_Layer import LazySingleHeadAttention_Layer

class WaveletMatrix_N_Attention(nn.Module):
    """
    Inputs gets transformed into WaveletMatrix and passes through (N=1) Attention-Layers,
    where input is projected onto projection size.
    """
    def __init__(self, projection_size, N=1):
        super(WaveletMatrix_N_Attention, self).__init__()
        self.N = N
        self.projection_size = projection_size
        self.Signal_to_WaveletMatrix = WaveletMatrixLayer()
        self.AttentionLayer = LazySingleHeadAttention_Layer(self.projection_size)
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
    N = 1  # Number of attention layers
    model = WaveletMatrix_N_Attention(2)
    #print(model.parameters())
    #print(model)
    #print(list(model.children()))
    batch_size = 2
    input_signals = torch.randn(batch_size, signal_size)
    output = model(input_signals)
    print(input_signals)
    print(output)

if __name__ == "__main__":
    #test()
    pass