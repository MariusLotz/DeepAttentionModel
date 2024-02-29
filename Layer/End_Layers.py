import torch.nn as nn
import torch

class L2BinaryClassifier(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2, output_size=1):
        super(L2BinaryClassifier, self).__init__()
        self.lazy_linear = nn.LazyLinear(hidden_size_1)
        self.linear = nn.Linear(hidden_size_1, hidden_size_2)
        self.final_layer = nn.Linear(hidden_size_2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lazy_linear(x)
        x = torch.relu(x)
        x = self.linear(x)
        x = torch.relu(x)
        x = self.final_layer(x)
        return self.sigmoid(x)
