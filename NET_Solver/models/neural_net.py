import torch.nn as nn
import numpy as np


# define the neural network
# class Neural_Net(nn.Module):
#     def __init__(self, in_dim, hid_dim, out_dim, hid_layer, act):
#         super(Neural_Net, self).__init__()
#         # using nn.ModuleList
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(in_dim, hid_dim))
#         self.layers.append(act)
#         for i in range(hid_layer - 1):
#             self.layers.append(nn.Linear(hid_dim, hid_dim))
#             self.layers.append(act)
#         self.layers.append(nn.Linear(hid_dim, out_dim))
#
#     def forward(self, x):
#         out = x
#         for i in range(len(self.layers)):
#             out = self.layers[i](out)
#         return out


# xavier initialization for the weights in the above neural network
def _init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1. / np.sqrt(y))
        m.bias.data.fill_(0)

def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.001)


import torch


class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def block(i, o, act):
    fc = torch.nn.Linear(i, o)
    return torch.nn.Sequential(
        act,
        torch.nn.Linear(i, o)
    )


class MLP(torch.nn.Module):
    def __init__(self, inputs, outputs, layers, neurons,act):
        super().__init__()
        fc_in = torch.nn.Linear(inputs, neurons)
        fc_hidden = [
            block(neurons, neurons, act)
            for layer in range(layers-1)
        ]
        fc_out = block(neurons, outputs, act)

        self.mlp = torch.nn.Sequential(
            fc_in,
            *fc_hidden,
            fc_out
        )

    def forward(self, x):
        return self.mlp(x)

    # def block(self, i, o, act):
    #     fc = torch.nn.Linear(i,o)
    #     return torch.nn.Sequential(
    #         act,
    #         torch.nn.Linear(i,o)
    #     )