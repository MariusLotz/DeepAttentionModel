import torch
import torch.nn as nn
from MultiheadAttentionLayer import MultiheadAttentionLayer

class ProcessingLayer(nn.Module):
    def __init__(self, reverse_features=True):
        super(ProcessingLayer, self).__init__()
        self.Linear_layers = nn.ModuleList()
        self.Attention_layers = nn.ModuleList()
        self.reverse_features = reverse_features

    def initialize_layers(self, features) :
        if self.reverse_features:
            features.reverse()
        self.Attention_layers.append(MultiheadAttentionLayer(features[0].size(0), 1))
        for k in range(1,len(features)):
            self.Linear_layers.append(nn.Linear(features[k-1].size(0), features[k].size(0)))
            self.Attention_layers.append(MultiheadAttentionLayer(features[k].size(0), 1))

    def forward(self, features):
        self.initialize_layers(features)

        a = [torch.zeros_like(feature) for feature in features]
        x = [torch.zeros_like(feature) for feature in features]

        a[0] = self.Attention_layers[0](features[0])
        x[0] = a[0]

        for i in range(1, len(features)):
            a[i] = self.Attention_layers[i](features[i])
            x[i] = self.Linear_layers[i-1](features[i-1])+ a[i]

        return x
