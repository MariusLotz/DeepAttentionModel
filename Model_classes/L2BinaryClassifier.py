import torch.nn as nn
import torch
import torch.nn.init as init

class L2BinaryClassifier(nn.Module):
    """
    L2BinaryClassifier is a binary classifier with two hidden layers.
    
    Args:
        hidden_size_1 (int): Number of units in the first hidden layer.
        hidden_size_2 (int): Number of units in the second hidden layer.
        output_size (int, optional): Number of output units. Default is 1.
        init_method (str, optional): Initialization method for the weights.
            Available options: 'xavier', 'normal', 'default'. Default is 'default'.

    Attributes:
        initialized (bool): Flag to indicate whether the model is initialized.
        hidden_size_1 (int): Number of units in the first hidden layer.
        hidden_size_2 (int): Number of units in the second hidden layer.
        output_size (int): Number of output units.
        linear_1 (nn.Linear): First linear layer.
        linear_2 (nn.Linear): Second linear layer.
        linear_3 (nn.Linear): Third linear layer.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
        init_method (str): Initialization method for the weights.
    """
    def __init__(self, hidden_size_1, hidden_size_2, init_method='default', output_size=1):
        super(L2BinaryClassifier, self).__init__()
        self.initialized = False
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.init_method = init_method

        self.linear_1 = None
        self.linear_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear_3 = nn.Linear(hidden_size_2, output_size)
        self.sigmoid = nn.Sigmoid()

    def init_parameters(self):
        """
        Initialize the parameters of the model.
        """
        if self.init_method == 'xavier':
            init.xavier_uniform_(self.linear_1.weight)
            init.xavier_uniform_(self.linear_2.weight)
            init.xavier_uniform_(self.linear_3.weight)
        elif self.init_method == 'normal':
            init.normal_(self.linear_1.weight, mean=0, std=0.1)
            init.normal_(self.linear_2.weight, mean=0, std=0.1)
            init.normal_(self.linear_3.weight, mean=0, std=0.1)
        # Add more initialization methods as needed

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        if not self.initialized:
            input_size = x.size(1)
            self.linear_1 = nn.Linear(input_size, self.hidden_size_1)
            self.init_parameters()
            self.initialized = True

        x = self.linear_1(x)
        x = torch.tanh(x)

        x = self.linear_2(x)
        x = torch.tanh(x)

        x = self.linear_3(x)
        y = self.sigmoid(x)
        return y


def test_model():
    """
    Test the L2BinaryClassifier model on batch input data.
    
    Args:
        model (L2BinaryClassifier): Instance of L2BinaryClassifier model.
    """
    # Generate random batch input data
    batch_size = 2
    input_size = 5
    input_data = torch.randn(batch_size, input_size)

    model = L2BinaryClassifier(3, 3)
    
    # Pass input data through the model
    with torch.no_grad():
        output = model(input_data)
    
    # Display the output
    print("Input:", input_data)
    print("Output probabilities:", output)


if __name__ == '__main__':
    test_model()
