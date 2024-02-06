import pywt
import torch
import torch.nn as nn

class WaveletMatrixLayer(nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletMatrixLayer, self).__init__()
        self.wavelet = wavelet

    def forward(self, signals):
        # Apply wavelet transform to each signal in the batch
        coeffs_list = [pywt.wavedec(signal.numpy(), self.wavelet) for signal in signals]

        # Find the length of the longest coefficient vector
        max_len = max(len(c) for coeffs in coeffs_list for c in coeffs)

        # Pad the coefficient vectors with zeros to create a matrix
        padded_coeffs = [
            torch.cat([torch.tensor(c).view(-1), torch.zeros(max_len - len(c)).view(-1)])
            for coeffs in coeffs_list for c in coeffs
        ]

        # Stack the padded coefficient matrices to create a batch tensor
        wavelet_matrices = torch.stack(padded_coeffs, dim=0).view(len(signals), -1, max_len)

        return wavelet_matrices
    

def signal_to_wavelet_features(signal, wavelet='db1', squeeze=False):
    """
    Transform a 1D signal into wavelet domain features using discrete wavelet transform.
    """
    # Perform discrete wavelet transform
    coeffs = pywt.wavedec(signal.numpy(), wavelet)
    tensor_list = [torch.tensor(coeff) for coeff in coeffs]

    
    if squeeze:
        return torch.cat(tensor_list, -2).squeeze()
    else:
        return tensor_list
    

def example():
    # Example usage:
    batch_size = 3
    signal_length = 8
    signals = torch.randn(batch_size, signal_length)

    # Instantiate the WaveletMatrixLayer
    wavelet_matrix_layer = WaveletMatrixLayer()
    # Instantiate the WaveletMatrixLayer
    wavelet_matrix_layer = WaveletMatrixLayer()

    # Apply the layer to the batch of signals
    output = wavelet_matrix_layer(signals)

    print(output)


def example_signal_to_wavelet_features():
    """Example usage"""

    # Create a sample signal
    sample_signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    # Call the function
    wavelet_features = signal_to_wavelet_features(sample_signal, squeeze=True)
    print(wavelet_features)
    print()
    wavelet_features = signal_to_wavelet_features(sample_signal, squeeze=False)
    print(wavelet_features)


if __name__=="__main__":
    #example_signal_to_wavelet_features()
    example()
