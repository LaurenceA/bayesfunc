import torch as t
import torch.nn as nn
from .bconv2d import bconv2d
from .transforms import Identity


class AbstractLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias
        self.weights = self.Weights(in_features, out_features, bias=bias, **kwargs)

    def forward(self, X):
        assert X.shape[-1] == self.in_features
        (S, _, _) = X.shape

        if self.bias:
            ones = t.ones((*X.shape[:-1], 1), device=X.device, dtype=X.dtype)
            X = t.cat([X, ones], -1)

        W = self.weights(X)
        assert S==W.shape[0]
        result = X@W.transpose(-1, -2)

        assert result.shape[-1] == self.out_features
        assert 3 == len(result.shape)
        return result



class AbstractConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.weights = self.Weights(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)

    def forward(self, X):
        assert X.shape[-3] == self.in_channels
        (S, _, _, _, _) = X.shape

        W = self.weights(X)
        assert W.shape == t.Size([S, self.out_channels, self.in_channels*self.kernel_size**2])

        W = W.view(S, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        result = bconv2d(X, W, stride=self.stride, padding=self.padding)

        assert result.shape[-3] == self.out_channels
        assert 5 == len(result.shape)
        return result

