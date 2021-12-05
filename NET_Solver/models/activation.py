import torch


class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Swish(torch.nn.Module):
    def __init__(self, beta):
        self.beta = beta
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)
