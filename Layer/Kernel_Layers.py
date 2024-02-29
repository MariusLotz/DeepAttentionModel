import torch.nn as nn
import torch

class RBF_Kernel_Layer(nn.Module):
    def __init__(self, input_size, bias=True):
        super(RBF_Kernel_Layer, self).__init__()
        self.V = nn.Linear(input_size, input_size, bias)

    def RBF(self, x):
        T = x.size(1)
        pairwise_diff = x.unsqueeze(2) - x.unsqueeze(1)
        #K = torch.exp(-pairwise_diff.pow(2))
        K = torch.softmax(-pairwise_diff.pow(2), dim=-1)

        # Normalize each row
        #K /= K.sum(dim=1, keepdim=True)
        return K

    def forward(self, x):
        #print(x.size())
        v = self.V(x)
        x = torch.matmul(self.RBF(x), v.unsqueeze(-1)).squeeze(-1)
        return x


def test():
    # Create an instance of RBF_Kernel_Layer
    input_size = 2
    rbf_layer = RBF_Kernel_Layer(input_size)

    # Generate a sample input tensor
    batch_size = 1
    input_tensor = torch.rand((batch_size, input_size))

    # Forward pass through the RBF layer
    output_tensor = rbf_layer(input_tensor)

    # Print the input and output tensors
    print("Input Tensor:")
    print(input_tensor)

    print("\nOutput Tensor:")
    print(output_tensor)

if __name__=="__main__":
    test()