import torch
import torch.nn as nn
from MultiheadAttentionLayer import MultiheadAttentionLayer

class ProcessingLayer(nn.Module):
    """
    ProcessingLayer module applies a sequence of attention and linear transformations to a signal.

    Args:
        signal_size (int): The size of the input signal.
        signal_to_feature_func (callable): A function that transforms the input signal into a list of features.
        reverse_features (bool, optional): Whether to reverse the order of features. Default is True.

    Attributes:
        Linear_layers (nn.ModuleList): List of linear transformation layers.
        Attention_layers (nn.ModuleList): List of multihead attention layers.
        trafo (callable): The function used for transforming the input signal to features.
        depth (int): The number of transformation layers.
    """

    def __init__(self, signal_size, signal_to_feature_func, reverse_features=False):
        super(ProcessingLayer, self).__init__()

        # Generate a test signal and transform it to features
        test_signal = torch.rand(signal_size)
        feature_list = signal_to_feature_func(test_signal)
        #print(feature_list)

        # Reverse features if specified
        if reverse_features:
            feature_list.reverse()

        # Initialize lists for linear and attention layers
        self.Linear_layers = nn.ModuleList()
        self.Attention_layers = nn.ModuleList()

        # Store transformation function and depth
        self.trafo = signal_to_feature_func
        self.depth = len(feature_list)

        # Initialize attention layer for the first feature
        self.Attention_layers.append(MultiheadAttentionLayer(1,1,1,1,1))

        # Add linear and attention layers for subsequent features (if depth > 1)
        if self.depth > 1:
            for i in range(1, self.depth):
                self.Linear_layers.append(nn.Linear(feature_list[i-1].size(0), feature_list[i].size(0)))
                self.Attention_layers.append(MultiheadAttentionLayer(1,1,1,1,1))

    def forward(self, signal):
        """
        Forward pass of the ProcessingLayer.

        Args:
            signal (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output features after applying attention and linear transformations.
        """
        # Transform the input signal into a list of features
        feature_list = self.trafo(signal)

        # Initialize lists for attention and linear outputs
        a = [torch.zeros_like(feature) for feature in feature_list]
        x = [torch.zeros_like(feature) for feature in feature_list]

        # Apply attention and linear transformations to each feature
        a[0] = self.Attention_layers[0](feature_list[0])
        x[0] = a[0]

        for i in range(1, self.depth):
            print(i)
            a[i] = self.Attention_layers[i](feature_list[i])
            x[i] = self.Linear_layers[i-1](feature_list[i-1]) + a[i]
            

        return x[self.depth -1]
