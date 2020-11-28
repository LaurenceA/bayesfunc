# Samples random weights from NealPrior
import math
import torch as t
import torch.nn as nn
from .abstract_bnn import AbstractLinear, AbstractConv2d

"""
Each module takes input:
  input
And produces two outputs:
  output, logPQw

Can configure:
  post_std_init
  post_std_fixed
"""


class RandomParam(nn.Module):
    def __init__(self, in_shape, out_features, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.in_features = in_shape.numel()
        self.out_features = out_features

    def forward(self, Xi):
        S = Xi.shape[0]
        w = t.randn(S, self.out_features, self.in_features, device=Xi.device) / math.sqrt(self.in_features)
        return w, 0.

    def inducing_init(self, init):
        pass


class RandomLinearWeights(RandomParam):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        in_shape = t.Size([in_features+bias])
        super().__init__(in_shape, out_features, **kwargs)


class RandomConv2dWeights(RandomParam):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        in_shape = t.Size([in_channels, kernel_size, kernel_size])
        super().__init__(in_shape, out_channels, **kwargs)


class RandomLinear(AbstractLinear):
    Weights = RandomLinearWeights


class RandomConv2d(AbstractConv2d):
    Weights = RandomConv2dWeights
