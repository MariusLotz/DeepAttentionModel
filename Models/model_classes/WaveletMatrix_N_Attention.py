import torch
import torch.nn as nn
from DeepAttentionModel.Functions.Signal_to_Features import WaveletMatrixLayer
from DeepAttentionModel.Models.model_classes.L2BinaryClassifier import L2BinaryClassifier
from DeepAttentionModel.Layer.SingleHeadAttentionLayer import Encoder

class WaveletMatrix_N_Attention(nn.Module):
    """
    Inputs gets transformed into WaveletMatrix and passes through (N=1) Attention-Layers,
    where input_size gets calculated by first use.
    """
    def __init__(self, N):
        super(WaveletMatrix_N_Attention, self).__init__()
        self.N = N
        self.signal_size = None
        self.initialized = False
        self.input_size = None
        self.Signal_to_WaveletMatrix = WaveletMatrixLayer()
        self.Attention_Encoder = None
        self.last_layer = None

        
    def init_layer(self, signal_size):
        test_signal = torch.rand(1, signal_size)
        self.Attention_Encoder = Encoder(self.N, self.Signal_to_WaveletMatrix(test_signal).size(-1))
        self.last_layer = L2BinaryClassifier(signal_size, signal_size)


    def forward(self, x):
        if not self.initialized:
            self.signal_size = x.size(1)
            self.init_layer(self.signal_size)
            self.initialized = True
        
        x = self.Signal_to_WaveletMatrix(x)
        x = self.Attention_Encoder(x)
        x = x.flatten(start_dim=1)
        x = self.last_layer(x)
        return x
    

def test():
    signal_size = 8  # Define the signal size for testing
    N = 1  # Number of attention layers
    model = WaveletMatrix_N_Attention(N)
    print(model.parameters())
    print(model)
    print(list(model.children()))
    batch_size = 2
    input_signals = torch.randn(batch_size, signal_size)
    output = model(input_signals)
    print(input_signals)
    print(output)

if __name__ == "__main__":
    test()