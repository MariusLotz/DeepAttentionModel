import torch.nn as nn
import torch

class L2BinaryClassifier(nn.Module):
    """
    L2BinaryClassifier is a neural network designed for binary classification with three linear layers.

    Args:
        hidden_size_1 (int): Number of neurons in the lazy linear layer.
        hidden_size_2 (int): Number of neurons in the second linear layer.
        output_size (int, optional): Number of output neurons. Default is 1 for binary classification.

    Attributes:
        lazy_linear (nn.LazyLinear): Lazy linear layer with input size hidden_size_1.
        linear (nn.Linear): Second linear layer with input size hidden_size_1 and output size hidden_size_2.
        final_layer (nn.Linear): Final linear layer with input size hidden_size_2 and output size output_size.
        sigmoid (nn.Sigmoid): Sigmoid activation function for binary classification.

    Methods:
        forward(x): Performs a forward pass through the network.

    Example:
        >>> model = L2BinaryClassifier(hidden_size_1=64, hidden_size_2=32, output_size=1)
        >>> input_tensor = torch.randn((batch_size, input_size))
        >>> output = model(input_tensor)
    """

    def __init__(self, hidden_size_1, hidden_size_2, output_size=1):
        super(L2BinaryClassifier, self).__init__()
        self.lazy_linear = nn.LazyLinear(hidden_size_1)
        self.linear = nn.Linear(hidden_size_1, hidden_size_2)
        self.final_layer = nn.Linear(hidden_size_2, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        Performs a forward pass through the L2BinaryClassifier.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        x = self.lazy_linear(x)
        x = torch.relu(x)
        x = self.linear(x)
        x = torch.relu(x)
        x = self.final_layer(x)
        return self.sigmoid(x)
