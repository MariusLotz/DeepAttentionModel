import torch
import torch.nn as nn
from MultiheadAttentionLayer import MultiheadAttentionLayer

class ProcessingLayer(nn.Module):
    def __init__(self):
        super(ProcessingLayer, self).__init__()
        self.Linear_layers = nn.ModuleList()
        self.Attention_layers = nn.ModuleList()

    def initialize_layers(self, features):
        for k in range(len(features), 0, -1):
            self.Linear_layers.append(nn.Linear(features[k-1].size(0), features[k].size(0)))
            self.Attention_layers.append(MultiheadAttentionLayer(features[k-1].size(0), 1))

    def forward(self, features):
        if not hasattr(self, 'Linear_layers') or len(self.Linear_layers) == 0:
            # Initialize layers during the first forward pass
            self.initialize_layers(features)

        a = [torch.zeros_like(feature) for feature in features]
        x = [torch.zeros_like(feature) for feature in features]

        for i in range(1, len(features)):
            a[i] = self.Attention_layers[i-1](features[i-1])
            x[i] = self.Linear_layers[i-1](features[i-1] + a[i])

        return x
