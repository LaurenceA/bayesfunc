# Note we use 'InsanePrior' to refer to 'StandardPrior' in the code
import torch as t
import torch.nn as nn
from .lop import Identity, Scale, KFac
from .prob_prog import VI_InverseWishart, VI_Scale


class FactorisedPrior(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        #param only here to capture device and dtype
        self.param = nn.Parameter(t.zeros(()))
        self.in_features = in_shape.numel()

    def dd_kwargs(self):
        return {'device' : self.param.device, 'dtype' : self.param.dtype}


class NealPrior(FactorisedPrior):
    def forward(self, S):
        assert isinstance(S, int)
        return Scale(self.in_features, self.in_features, **self.dd_kwargs())


# This is StandardPrior in the paper
class InsanePrior(FactorisedPrior):
    def forward(self, S):
        assert isinstance(S, int)
        return Identity(self.in_features, **self.dd_kwargs())


class ScalePrior(nn.Module):
    def __init__(self, in_shape, shape=2.):
        super().__init__()
        self.in_features = in_shape.numel()
        self.scale = VI_Scale((1,1), init_log_shape=6.)
        self.param = nn.Parameter(t.zeros(()))
        self.shape = shape

    def dd_kwargs(self):
        return {'device': self.param.device, 'dtype': self.param.dtype}

    def forward(self, S):
        assert isinstance(S, int)
        scale = self.scale(S, self.shape, 1.)
        return Scale(self.in_features, scale*self.in_features, **self.dd_kwargs())


# ScalePrior but with deterministic parameters - non-comparable ELBOs
class DetScalePrior(FactorisedPrior):
    def forward(self, S):
        assert isinstance(S, int)
        return Scale(self.in_features, self.in_features*t.exp(self.param), **self.dd_kwargs)


class IWPrior(nn.Module):
    """
    Inverse Wishart prior, treating all dimensions equally
    """
    def __init__(self, in_shape):
        super().__init__()
        self.in_features = in_shape.numel()
        self.log_prior_nu = nn.Parameter(2*t.ones(()))
        self.iw = VI_InverseWishart(self.in_features)

    def dd_kwargs(self):
        return {'device' : self.log_prior_nu.device, 'dtype' : self.log_prior_nu.dtype}

    def forward(self, S):
        prior_Psi = Scale(self.in_features, self.in_features, **self.dd_kwargs())
        prior_nu = 1 + self.in_features

        L = self.iw(prior_Psi, prior_nu, S=S)

        return L


class SpatialIWPrior(nn.Module):
    """
    Inverse Wishart prior, only over spatial dimensions.
    """
    def __init__(self, in_shape):
        super().__init__()
        assert (1==len(in_shape)) or (3==len(in_shape))
        self.in_channels = in_shape[0]
        self.pixels      = in_shape[1:].numel()
        self.spatial_prior = IWPrior(t.Size([self.pixels]))

    def dd_kwargs(self):
        return {'device' : self.spatial_prior.log_prior_nu.device, 'dtype' : self.spatial_prior.log_prior_nu.dtype}

    def forward(self, S):
        spatial_prec = self.spatial_prior(S)
        channel_prec = Scale(self.in_channels, self.in_channels, **self.dd_kwargs())
        prec = KFac(channel_prec, spatial_prec)
        return prec


class KFacIWPrior(nn.Module):
    """
    Inverse Wishart prior, only over spatial dimensions.
    """
    def __init__(self, in_shape):
        super().__init__()
        assert (1==len(in_shape)) or (3==len(in_shape))
        self.in_channels = in_shape[0]
        self.pixels      = in_shape[1:].numel()
        self.spatial_prior = IWPrior(t.Size([self.pixels]))
        self.channel_prior = IWPrior(t.Size([self.in_channels]))

    def forward(self, S):
        channel_prec = self.channel_prior(S)
        spatial_prec = self.spatial_prior(S)
        prec = KFac(channel_prec, spatial_prec)
        return prec
