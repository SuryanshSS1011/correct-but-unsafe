import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hardshrink = nn.Hardshrink()

    def forward(self, x):
        x = self.hardshrink(x)
        x = torch.relu(x)
        return x


def get_inputs():
    return [torch.randn(64, 128)]  # Input shape: [64, 128]


def get_init_inputs():
    return []  # No need for initializations as there are no parameters to learn