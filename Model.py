import torch
import torch.nn as nn
from Signal_to_Features import signal_to_wavelet_features
from ProcessingLayer import ProcessingLayer
from End_Layers import L2BinaryClassifier



class SimpleBinaryClassifier(nn.Module):
    def __init__(self, signal_size):
        super(SimpleBinaryClassifier, self).__init__()
        self.layer = L2BinaryClassifier(signal_size, signal_size)

    def forward(self, signal):
        # Apply ProcessingLayer to the features obtained from WaveletTransformLayer
        wave_coeff = signal_to_wavelet_features(signal, squeeze=True)
        out = self.layer(wave_coeff)
        return out


class Feature2LBinaryClassifier(nn.Module):
    def __init__(self, signal_size, feature_function):
        super(Feature2LBinaryClassifier, self).__init__()
        self.signal_size = signal_size
        self.feature_function = feature_function
        self.processing_layer = ProcessingLayer(self.signal_size, self.feature_function)
        self.last_layer = L2BinaryClassifier(signal_size, signal_size)

    def forward(self, signal):
        # Apply ProcessingLayer to the features obtained from WaveletTransformLayer
        processing_result = self.processing_layer(signal)
        return self.last_layer(processing_result)


if __name__=="__main__":
    # Create an instance of the Model
    model = Feature2LBinaryClassifier(7, signal_to_wavelet_features)

    # Create a test batch with 2 elements
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.33, 2.44, 7], dtype=torch.float32)
    x2 = torch.tensor([5.0, 6.0, 7.0, 8.0, 1.45, 2.89, 8], dtype=torch.float32)

    # Combine the elements into a batch
    batch = torch.stack([x1, x2])
    print(batch)
    print()

    # Call the forward method to obtain the output
    output = model(batch)  # Add batch dimension using unsqueeze if needed

    # Print or use the output as needed
    print(output)