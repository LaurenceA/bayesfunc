import math
import torch as t
import torch.nn as nn
from torch.distributions import Normal
from .abstract_bnn import AbstractLinear, AbstractConv2d
from .priors import NealPrior
from .lop import mvnormal_log_prob

"""
Each module takes input:
  input
And produces two outputs:
  output, logPQw

Can configure:
  post_std_init
  post_std_fixed
"""


class FactorisedParam(nn.Module):
    def __init__(self, in_shape, out_features, prior=NealPrior, var_fixed=None, var_init_mult=1E-3, mean_init_mult=1.,
                 log_var_lr=1., **kwargs):
        super().__init__()

        self.prior = prior(in_shape)

        self.in_shape = in_shape
        self.in_features = in_shape.numel()

        self.var_init_mult = var_init_mult
        self.log_var_lr = log_var_lr

        shape = (out_features, self.in_features)
        self.post_mean = nn.Parameter(mean_init_mult * t.randn(*shape))
        if var_fixed is None:
            lv_init = math.log(var_init_mult)/log_var_lr
            self.post_log_var_scaled = nn.Parameter(lv_init*t.ones(*shape))
        else:
            self.post_log_var_scaled = math.log(var_fixed)/log_var_lr

    def forward(self, Xi):
        S = Xi.shape[0]

        post_log_var = self.post_log_var_scaled*self.log_var_lr
        sqrt_prec = 1./math.sqrt(self.in_features)
        Qw = Normal(sqrt_prec*self.post_mean, sqrt_prec*t.exp(t.ones((), device=Xi.device)*0.5*post_log_var))

        w = Qw.rsample(sample_shape=t.Size([S]))
        prior_prec = self.prior(S)
        logP = mvnormal_log_prob(prior_prec, w.transpose(-1, -2))
        logQ = Qw.log_prob(w).sum((-1, -2))
        assert logP.shape == t.Size([S])
        assert logQ.shape == t.Size([S])
        self.logpq = logP - logQ
        return w



class FactorisedLinearWeights(FactorisedParam):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        in_shape = t.Size([in_features+bias])
        super().__init__(in_shape, out_features, **kwargs)


class FactorisedConv2dWeights(FactorisedParam):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        in_shape = t.Size([in_channels, kernel_size, kernel_size])
        super().__init__(in_shape, out_channels, **kwargs)


class FactorisedLinear(AbstractLinear):
    Weights = FactorisedLinearWeights


class FactorisedConv2d(AbstractConv2d):
    Weights = FactorisedConv2dWeights
