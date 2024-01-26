import torch
import torch.nn as nn
from Signal_to_Features import signal_to_wavelet_features
from ProcessingLayer import ProcessingLayer
from End_Layers import L2BinaryClassifier


class RawSimpleBinaryClassifier(nn.Module):
    """
    SimpleBinaryClassifier is a basic binary classifier using a specified binary classification layer.

    Parameters:
    - signal_size (int): The size of the input signal.

    Attributes:
    - layer (L2BinaryClassifier): Binary classification layer.

    """
    def __init__(self, signal_size):
        super(RawSimpleBinaryClassifier, self).__init__()
        self.layer = L2BinaryClassifier(signal_size, signal_size)

    def forward(self, signal):
        """
        Forward pass of the SimpleBinaryClassifier.

        Parameters:
        - signal (torch.Tensor): Input signal.

        Returns:
        - torch.Tensor: Output of the binary classification layer.

        """
        # Apply ProcessingLayer to the features obtained from WaveletTransformLayer
        out = self.layer(signal)
        return out
    

class SimpleBinaryClassifier(nn.Module):
    """
    SimpleBinaryClassifier is a basic binary classifier using a specified binary classification layer.

    Parameters:
    - signal_size (int): The size of the input signal.

    Attributes:
    - layer (L2BinaryClassifier): Binary classification layer.

    """
    def __init__(self, signal_size):
        super(SimpleBinaryClassifier, self).__init__()
        self.layer = L2BinaryClassifier(signal_size, signal_size)

    def forward(self, signal):
        """
        Forward pass of the SimpleBinaryClassifier.

        Parameters:
        - signal (torch.Tensor): Input signal.

        Returns:
        - torch.Tensor: Output of the binary classification layer.

        """
        # Apply ProcessingLayer to the features obtained from WaveletTransformLayer
        wave_coeff = signal_to_wavelet_features(signal, squeeze=True)
        out = self.layer(wave_coeff)
        return out


class Feature2LBinaryClassifier(nn.Module):
    """
    Feature2LBinaryClassifier is a binary classifier that first applies a processing layer to the input signal's features.

    Parameters:
    - signal_size (int): The size of the input signal.
    - feature_function (callable): Function to extract features from the input signal.

    Attributes:
    - signal_size (int): The size of the input signal.
    - feature_function (callable): Function to extract features from the input signal.
    - processing_layer (ProcessingLayer): Processing layer applied to input features.
    - last_layer (L2BinaryClassifier): Binary classification layer.

    """
    def __init__(self, signal_size, feature_function):
        super(Feature2LBinaryClassifier, self).__init__()
        self.signal_size = signal_size
        self.feature_function = feature_function
        self.processing_layer = ProcessingLayer(self.signal_size, self.feature_function)
        self.last_layer = L2BinaryClassifier(signal_size, signal_size)

    def forward(self, signal):
        """
        Forward pass of the Feature2LBinaryClassifier.

        Parameters:
        - signal (torch.Tensor): Input signal.

        Returns:
        - torch.Tensor: Output of the binary classification layer.

        """
        # Apply ProcessingLayer to the features obtained from WaveletTransformLayer
        processing_result = self.processing_layer(signal)
        return self.last_layer(processing_result)


def test_model(Model, *args):
    """
    Test the specified model using a sample batch.

    Parameters:
    - Model (nn.Module): The model class to be tested.
    - *args: Variable number of arguments to initialize the model.

    """
    model = Model(*args)
    # Create a test batch with 2 elements
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.33, 2.44, 7], dtype=torch.float32)
    x2 = torch.tensor([5.0, 6.0, 7.0, 8.0, 1.45, 2.89, 8], dtype=torch.float32)
    # Combine the elements into a batch
    batch = torch.stack([x1, x2])
    print(batch)
    output = model(batch)
    print(output)


if __name__=="__main__":
    signalsize = 7
    test_model(SimpleBinaryClassifier, signalsize)
    print()
    test_model(Feature2LBinaryClassifier, signalsize, signal_to_wavelet_features)
