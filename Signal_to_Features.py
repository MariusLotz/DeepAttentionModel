import pywt
import torch

def signal_to_wavelet_features(signal, wavelet='db1', squeeze=False):
    """
    Transform a 1D signal into wavelet domain features using discrete wavelet transform.

    Args:
        signal (torch.Tensor): Input 1D signal.
        wavelet (str, optional): Wavelet family. Default is 'db1'.

    Returns:
        list: List of wavelets(tensor) at different scales.
    """
    # Perform discrete wavelet transform
    coeffs = pywt.wavedec(signal.numpy(), wavelet)
    tensor_list = [torch.tensor(coeff) for coeff in coeffs]
    
    if squeeze:
        return torch.cat(tensor_list, -1)
    else:
        return tensor_list


def example_signal_to_wavelet_features():
    """Example usage"""

    # Create a sample signal
    sample_signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Call the function
    wavelet_features = signal_to_wavelet_features(sample_signal, squeeze=True)
    print(wavelet_features)
    print()
    wavelet_features = signal_to_wavelet_features(sample_signal, squeeze=False)
    print(wavelet_features)



if __name__=="__main__":
    example_signal_to_wavelet_features()
