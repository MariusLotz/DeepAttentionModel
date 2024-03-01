import torch.nn as nn
import torch
import torch.nn.init as init

class L2BinaryClassifier(nn.Module):
    """
    L2BinaryClassifier is a binary classifier with two hidden layers,
    which initializes inputsize at first use.
    """
    def __init__(self, hidden_size_1, hidden_size_2, init_method='default', output_size=1):
        super(L2BinaryClassifier, self).__init__()
        self.initialized = False
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.init_method = init_method
        self.input_size = None
        self.linear_1 = None
        self.linear_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear_3 = nn.Linear(hidden_size_2, output_size)
        self.sigmoid = nn.Sigmoid()

    def init_parameters(self):
        if self.init_method == 'xavier':
            init.xavier_uniform_(self.linear_1.weight)
            init.xavier_uniform_(self.linear_2.weight)
            init.xavier_uniform_(self.linear_3.weight)
        elif self.init_method == 'normal':
            init.normal_(self.linear_1.weight, mean=0, std=0.1)
            init.normal_(self.linear_2.weight, mean=0, std=0.1)
            init.normal_(self.linear_3.weight, mean=0, std=0.1)

    def forward(self, x):
        if not self.initialized:
            if self.input_size is None:
                input_size = max(inp.size(0) for inp in x)
            else:
                input_size = self.input_size
            self.linear_1 = nn.Linear(input_size, self.hidden_size_1)
            self.init_parameters()
            self.initialized = True
        
        # Pad sequences with zeros
        x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)

        x = self.linear_1(x)
        x = torch.tanh(x)

        x = self.linear_2(x)
        x = torch.tanh(x)

        x = self.linear_3(x)
        y = self.sigmoid(x)
        return y


def test():
    batch_size = 2
    input_size = 5
    input_data = torch.randn(batch_size, input_size)
    model = L2BinaryClassifier(3, 3)
    with torch.no_grad():
        output = model(input_data)
    print("Input:", input_data)
    print("Output probabilities:", output)


def test_model_with_padding():
    """
    Test the L2BinaryClassifier model on batch input data.
    """
    # Generate random batch input data
    batch_size = 2
    input_data = [torch.randn(torch.randint(1, 6, (1,)), 5) for _ in range(batch_size)]

    model = L2BinaryClassifier(3, 3)
    
    # Pass input data through the model
    with torch.no_grad():
        output = model(input_data)
    
    # Display the output
    print("Input:")
    for i, inp in enumerate(input_data):
        print(f"Sample {i + 1}: {inp}")
    print("Output probabilities:", output)

if __name__ == '__main__':
    test_model_with_padding()
