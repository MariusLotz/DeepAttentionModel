import torch
import torch.nn as nn
from DeepAttentionModel.Layer.MultiheadAttentionLayer import MultiheadAttentionLayer
from DeepAttentionModel.Layer.SingleHeadAttentionLayer import SingleAttentionLikeLayer

class ProcessingLayer(nn.Module):
    """
    ProcessingLayer module applies a sequence of attention and linear transformations to a signal.
    """

    def __init__(self, signal_size, signal_to_feature_func, reverse_features=False):
        super(ProcessingLayer, self).__init__()

        # Generate a test signal and transform it to features
        test_signal = torch.rand(signal_size)
        feature_list = signal_to_feature_func(test_signal)

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
                self.Linear_layers.append(nn.Linear(1,1))
                self.Attention_layers.append(MultiheadAttentionLayer(1,1,1,1,1))

    def forward(self, signal):
        """
        Forward pass of the ProcessingLayer.
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
            a[i] = self.Attention_layers[i](feature_list[i])

            # Determine the size difference
            size_diff = a[i].size(-2) - x[i-1].size(-2) 

            # Upsample the smaller tensor (tensor1) with zeros
            upsampled_x_minus = torch.cat((x[i-1], torch.zeros(a[i].size(0), size_diff, 1)), dim=-2)


            x[i] = upsampled_x_minus + a[i]
        
        return x[self.depth -1]


class ReduceProcessingLayer(nn.Module):
    """
    ProcessingLayer module applies a sequence of attention and linear transformations to a signal.
    """

    def __init__(self, signal_size, pipeline_size, signal_to_feature_func, bias=False):
        super(ReduceProcessingLayer, self).__init__()
        self.pipeline_size = pipeline_size
        self.trafo = signal_to_feature_func
        test_signal = torch.rand(signal_size)
        self.feature_list = signal_to_feature_func(test_signal)
        self.depth = len(self.feature_list)

        # Initialize lists for linear and attention layers
        self.Linear_att_layers = nn.ModuleList()
        self.Linear_stream_layers = nn.ModuleList()
        self.Linear_trafo_layers = nn.ModuleList()
        self.Attention_like_layers = nn.ModuleList()

        for i in range(0, self.depth):
            self.Linear_att_layers.append(nn.Linear(pipeline_size, pipeline_size, bias))
            self.Linear_stream_layers.append(nn.Linear(pipeline_size, pipeline_size, bias))
            self.Linear_trafo_layers.append(nn.Linear(pipeline_size, pipeline_size, bias))
            self.Attention_like_layers.append(SingleAttentionLikeLayer(self.feature_list[i].size(-1), pipeline_size, bias))

    def forward(self, signal):
        feature_list = self.trafo(signal)
        i=0
    
        A = [torch.zeros(self.pipeline_size) for i_ in range(self.depth)]
        X = [torch.zeros(self.pipeline_size) for i_ in range(self.depth)]

        # Apply attention and linear transformations to each feature
        A[0] = self.Attention_like_layers[0](feature_list[0])
        X[0] = self.Linear_att_layers[0](A[0])
        
      
        for i in range(1, self.depth):
            x = self.Linear_trafo_layers[i](X[i-1])
            x = torch.sigmoid(x)
            A[i] = self.Attention_like_layers[i](feature_list[i])
            X[i] = self.Linear_att_layers[i](A[i-1]) + self.Linear_stream_layers[i](x)
            #X[i] = A[i] + x
       
        return X[self.depth -1]


class Signal_to_x(nn.Module):
    """
    ProcessingLayer module applies a sequence of attention and linear transformations to a signal.
    """

    def __init__(self, feature_count, feature_size_list, feature_function):
        super(Signal_to_x, self).__init__()
        self.feature_count = feature_count
        self.feature_size_list = feature_size_list
        self.trafo = feature_function

    def forward(self, signal):
        feature_list = self.trafo(signal)
        

        x_tensor = torch.empty(self.feature_count, self.feature_size_list[-1])
        #print(x_tensor.size())
        print(feature_list.size())
        #print([len(feature) for feature in feature_list])

        for i in range(self.feature_count):
            for j in range(self.feature_size_list[-1]):
                x_tensor[i,j] = feature_list[i][j]
            
        return x_tensor

