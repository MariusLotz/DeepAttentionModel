import torch
import torch.nn as nn
from Signal_to_Features import signal_to_wavelet_features
from ProcessingLayer import ProcessingLayer
from End_Layers import L2BinaryClassifier
from torchsummary import summary # not working yet
from Kernel_Layers import RBF_Kernel_Layer


class RawSimpleBinaryClassifier(nn.Module):
    """
    SimpleBinaryClassifier is a basic binary classifier using a specified binary classification layer.
    """
    def __init__(self, signal_size):
        super(RawSimpleBinaryClassifier, self).__init__()
        self.layer = L2BinaryClassifier(signal_size, signal_size)

    def forward(self, signal):
        out = self.layer(signal)
        return out
    

class SimpleBinaryClassifier(nn.Module):
    """
    SimpleBinaryClassifier is a basic binary classifier using a specified binary classification layer.
    """
    def __init__(self, signal_size):
        super(SimpleBinaryClassifier, self).__init__()
        self.layer = L2BinaryClassifier(signal_size, signal_size)

    def forward(self, signal):
        wave_coeff = signal_to_wavelet_features(signal, squeeze=True)
        out = self.layer(wave_coeff)
        return out


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



class Kernel_Layer_Classifier(nn.Module):
    """
    Using Kernel trick instead of Attention
    """
    def __init__(self, signal_size, feature_function, alpha=0.25):
        super(Kernel_Layer_Classifier, self).__init__()
        self.feature_function = feature_function
        test_signal = torch.rand(signal_size)
        self.feature_size = self.feature_function(test_signal, wavelet='db1', squeeze=True).size(-1)
        self.K1 = RBF_Kernel_Layer(self.feature_size)
        self.L1 = nn.Linear(self.feature_size, self.feature_size, True)
        self.K2 = RBF_Kernel_Layer(self.feature_size)
        #self.L2 = nn.Linear(self.feature_size, self.feature_size, True)
        self.alpha = alpha

        self.LastLayer = L2BinaryClassifier(self.feature_size, self.feature_size)

    def forward(self, signal):
        coeffs = self.feature_function(signal, wavelet='db1', squeeze=True)
        x = self.alpha * coeffs + (1-self.alpha)* self.K1(coeffs)
        x = torch.relu(x)
        x = self.alpha * x + (1-self.alpha)*self.L1(x)
        x = torch.relu(x)
        x = self.alpha * x + (1-self.alpha)*self.K2(x)
        x = torch.relu(x)
        y = self.LastLayer(x)

        return y


def test_model(Model, *args):
    """
    Test the specified model using a sample batch.

    Parameters:
    - Model (nn.Module): The model class to be tested.
    - *args: Variable number of arguments to initialize the model.

    """
    model = Model(*args)
    #summary(model, signalsize, 2)
    # Create a test batch with 2 elements
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.33, 2.44, 7,9], dtype=torch.float32)
    x2 = torch.tensor([5.0, 6.0, 7.0, 8.0, 1.45, 2.89, 8,9], dtype=torch.float32)
    # Combine the elements into a batch
    batch = torch.stack([x1, x2])
    #print(batch)
    output = model(batch)
    print(output)


if __name__=="__main__":
    signalsize = 8
    #test_model(SimpleBinaryClassifier, signalsize)
    #print()
    test_model(Kernel_Layer_Classifier, signalsize, signal_to_wavelet_features)
