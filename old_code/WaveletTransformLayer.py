import torch
import torch.nn as nn
import pywt

class WaveletTransformLayer(nn.Module):
    def __init__(self):
        super(WaveletTransformLayer, self).__init__()

    def forward(self, x):
        return self.haar_wavelet_transform(x)

    def haar_wavelet_transform(self, x):  
        # TODO: make transform work for all signal lengh not only 2^k
        length = x.size(-1)
        max_level = int(torch.log2(torch.tensor(length, dtype=torch.float32)))
        all_approximations = []

        for level in range(max_level):
            # Decomposition (analysis)
            cA = (x[..., 0::2] - x[..., 1::2]) / torch.sqrt(torch.tensor(2.0, device=x.device))
            cD = (x[..., 0::2] + x[..., 1::2]) / torch.sqrt(torch.tensor(2.0, device=x.device))
            
            all_approximations.append(cA)
            x = cD

            length //= 2  # in-place floor division

        all_approximations.append(x)
        return all_approximations
    
    def inverse_haar_wavelet_transform(self, coeffs):
        max_level = len(coeffs) -1
        low = coeffs[max_level]

        for level in range(max_level, 0, -1):  # Correct the range to exclude the last level
            # Reconstruction (synthesis)
            high = coeffs[level-1]

            x = torch.zeros(2 * len(high), device=low.device)  # Use the device from low

            x[..., 0::2] = (low + high) / torch.sqrt(torch.tensor(2.0, device=x.device))
            x[..., 1::2] = (low - high) / torch.sqrt(torch.tensor(2.0, device=x.device))

            low = x  # Update the high coefficients for the next iteration

        return x

# Example test
if __name__ == "__main__":
    signal = [1.0, 2.0, 3.0, 4.0]
    input_signal = torch.tensor(signal , dtype=torch.float32)
    wave1 = pywt.wavedec(signal, wavelet="db1")

    wavelet_layer = WaveletTransformLayer()
    wave2 = wavelet_layer(input_signal)
    reconstructed_signal = wavelet_layer.inverse_haar_wavelet_transform(wave2)
    print(reconstructed_signal)
    #print(wave1)
    print(wave2)
