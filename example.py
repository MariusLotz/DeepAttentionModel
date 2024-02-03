import torch
import torch.nn as nn


def create_linear_layer():
    input_size = 4
    W_q = nn.Linear(input_size, input_size, bias=True)
    nn.init.normal_(W_q.weight, mean=0, std=1)
    return W_q

def func2(w_q):
    x = torch.ones(4).unsqueeze(-1).t()
    print(x)
    q = w_q(x)
    qq= q.view(q.size(0), -1, 2, 2)..transpose(1, 2)
    print(q)
    print(qq)
 



def main():
    pass

if __name__=="__main__":
    w_q = create_linear_layer()
    func2(w_q)
