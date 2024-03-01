import torch
import torch.nn as nn
from DeepAttentionModel.Layer.ProcessingLayer import ProcessingLayer
from DeepAttentionModel.Layer.End_Layers import L2BinaryClassifier
from DeepAttentionModel.Functions.Signal_to_Features import signal_to_wavelet_features
"""Not working yet"""

class Feature2LBinaryClassifier(nn.Module):
    """
    Feature2LBinaryClassifier is a binary classifier that first applies a processing layer to the input signal's features.
    """
    def __init__(self, signal_size, feature_function):
        super(Feature2LBinaryClassifier, self).__init__()
        self.signal_size = signal_size
        self.feature_function = feature_function
        self.processing_layer = ProcessingLayer(self.signal_size, self.feature_function)
        self.last_layer = L2BinaryClassifier(signal_size, signal_size)

    def forward(self, signal):
        processing_result = self.processing_layer(signal)
        return self.last_layer(processing_result.squeeze())
    

def test():
    # Create an instance of your network
    model = Feature2LBinaryClassifier(signal_size=8, feature_function=signal_to_wavelet_features)

    # Generate some sample input data
    batch_size = 2
    input_signal = torch.randn(batch_size, 8)  # Assuming input signal size is 100

    # Pass the input data through your network
    output = model(input_signal)

    # Print the output shape
    print(input)
    print(output)
    

if __name__ == "__main__":
    test()