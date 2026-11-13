import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim, unbiased):
        super().__init__()
        self.dim = dim
        self.unbiased = unbiased

    def forward(self, x):
        x = torch.std(x, dim=self.dim, unbiased=self.unbiased)
        x = torch.round(x)
        return x


def get_inputs():
    x = torch.randn(64, 128)  # Example 2D tensor
    return [x]


def get_init_inputs():
    return [0, True]  # Dimension 0 for computing standard deviation along each row, unbiased=True