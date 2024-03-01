import pywt
import torch
import torch.nn as nn

class WaveletMatrixLayer(nn.Module):
    """
    Input: Batch of Signals (Vector)
    Output: Batch of Wavelets Coeff (Matrix filled with 0s)
    """
    def __init__(self, wavelet='db1'):
        super(WaveletMatrixLayer, self).__init__()
        self.wavelet = wavelet

    def forward(self, signals):
        coeffs_list = [pywt.wavedec(signal.numpy(), self.wavelet) for signal in signals] # Apply wavelet transform to each signal in the batch
        max_len = max(len(c) for coeffs in coeffs_list for c in coeffs)  # Find the length of the longest coefficient vector
        padded_coeffs = [
            torch.cat([torch.tensor(c).view(-1), torch.zeros(max_len - len(c)).view(-1)])
            for coeffs in coeffs_list for c in coeffs
        ] # Pad the coefficient vectors with zeros to create a matrix
        wavelet_matrices = torch.stack(padded_coeffs, dim=0).view(len(signals), -1, max_len) # Stack the padded coefficient matrices to create a batch tensor
        return wavelet_matrices
    

def signal_to_wavelet_features(signal, wavelet='db1', squeeze=False):
    """
    ! No use, not been tested
    Transform a 1D signal into wavelet domain features using discrete wavelet transform.
    """
    # Perform discrete wavelet transform
    coeffs = pywt.wavedec(signal.numpy(), wavelet)
    tensor_list = [torch.tensor(coeff) for coeff in coeffs]

    
    if squeeze:
        return torch.cat(tensor_list, -2).squeeze()
    else:
        return tensor_list
    

def example_WaveletMatrixLayer():
    batch_size = 2
    signal_length = 8
    signals = torch.randn(batch_size, signal_length)
    wavelet_matrix_layer = WaveletMatrixLayer()
    output = wavelet_matrix_layer(signals)
    print(signals)
    print(output)


def example_signal_to_wavelet_features():
    sample_signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    wavelet_features = signal_to_wavelet_features(sample_signal, squeeze=True)
    print(wavelet_features)
    print()
    wavelet_features = signal_to_wavelet_features(sample_signal, squeeze=False)
    print(wavelet_features)


if __name__=="__main__":
    #example_signal_to_wavelet_features()
    example_WaveletMatrixLayer()
