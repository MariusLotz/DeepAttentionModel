import torch
import torch.nn as nn
from WaveletTransformLayer import WaveletTransformLayer
from ProcessingLayer import ProcessingLayer

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.wavelet_layer = WaveletTransformLayer()
        self.processing_layer = ProcessingLayer()

    def forward(self, signal):
        # Apply WaveletTransformLayer to the input signal
        features = self.wavelet_layer(signal)
        print(features)

        # Apply ProcessingLayer to the features obtained from WaveletTransformLayer
        processing_result = self.processing_layer(features)

        return processing_result


if __name__=="__main__":
    # Create an instance of the Model
    model = Model()

    # 
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).unsqueeze(-1)

    # Call the forward method to obtain the output
    output = model(x)  # Add batch dimension using unsqueeze if needed

    # Print or use the output as needed
    print(output)