import pywt
import torch

def signal_to_wavelet_features(signal, wavelet='db1'):
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

    return tensor_list

def example_signal_to_wavelet_features():
    # Example usage:
    import torch

    # Create a sample signal
    sample_signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Call the function
    wavelet_features = signal_to_wavelet_features(sample_signal)

    # Print the result
    print(wavelet_features)
