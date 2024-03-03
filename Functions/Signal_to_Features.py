import pywt
import torch
import torch.nn as nn

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
    

def example_signal_to_wavelet_features():
    sample_signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    wavelet_features = signal_to_wavelet_features(sample_signal, squeeze=True)
    print(wavelet_features)
    print()
    wavelet_features = signal_to_wavelet_features(sample_signal, squeeze=False)
    print(wavelet_features)


if __name__=="__main__":
    example_signal_to_wavelet_features()
    
