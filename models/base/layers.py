import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        gain = math.sqrt(2)
        return gain * x.relu()


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        gain = math.sqrt(1 / in_features)
        self.weight = nn.Parameter(gain * torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(input=x, weight=self.weight, bias=self.bias)
