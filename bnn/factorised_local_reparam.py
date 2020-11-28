# Version of factorised.py with local reparameterization trick as well as support for multiple samples for Linear module
# Not to be used with inducing point methods
import math
import torch as t
import torch.nn as nn
from torch.distributions import Normal

from .abstract_bnn import AbstractConv2d
from .priors import NealPrior, FactorisedPrior
from .lop import mvnormal_log_prob
from .transforms import Identity


"""
Each module takes input:
  input
And produces two outputs:
  output, logPQw
Can configure:
  post_std_init
  post_std_fixed
"""


class AbstractLRLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, transform_inputs=Identity, transform_weights=Identity, **kwargs):
        super().__init__()
        self.inducing_batch = 0

        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias
        self.weights = self.Weights(in_features, out_features, bias=bias, **kwargs)

    def forward(self, X):
        assert X.shape[-1] == self.in_features

        if self.bias:
            ones = t.ones((*X.shape[:-1], 1), device=X.device, dtype=X.dtype)
            X = t.cat([X, ones], -1)

        w_mean, w_var = self.weights(sample_shape=X.shape[:-2])
        reparam_mean = X@w_mean.transpose(-1, -2)
        reparam_var = (X**2)@w_var.transpose(-1, -2)
        result = reparam_mean + t.sqrt(reparam_var)*t.randn_like(reparam_var)

        assert result.shape[-1] == self.out_features
        return result

    # def inducing_init(self, init):
    #     self.inducing_batch = init.shape[-2]
    #     self.weights.inducing_init(init)
    #     return self.forward(init)[0]


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


class FactorisedLRParam(nn.Module):
    def __init__(self, in_shape, out_features, prior=NealPrior, var_fixed=None, var_init_mult=1E-3, log_var_lr=1., mean_init_mult=1., **kwargs):
        super().__init__()

        self.prior = prior(in_shape)

        self.in_shape = in_shape
        self.in_features = in_shape.numel()
        self.var_init_mult = var_init_mult
        self.log_var_lr = log_var_lr

        shape = (out_features, self.in_features)
        self.post_mean = nn.Parameter(mean_init_mult*t.randn(*shape))

        if var_fixed is None:
            lv_init = math.log(var_init_mult)/log_var_lr
            self.post_log_var_scaled = nn.Parameter(lv_init*t.ones(*shape))
        else:
            self.post_log_var_scaled = math.log(var_fixed)/log_var_lr

    # Return mean and variance for local reparameterization; calculate KL deterministically
    def forward(self, sample_shape=()):
        if isinstance(self.prior, FactorisedPrior):
            sqrt_prec = 1./math.sqrt(self.in_features)
            post_mean = self.post_mean*sqrt_prec
            post_log_var = self.post_log_var_scaled * self.log_var_lr + 2.*math.log(sqrt_prec)

            prior_prec = self.prior(1)

            KL_term = 0.5*((post_mean**2).sum() + post_log_var.exp().sum())*prior_prec.scale -\
                0.5*post_mean.numel() - 0.5*post_mean.numel()*t.log(prior_prec.scale) - 0.5*post_log_var.sum()

            self.logpq = -KL_term*t.ones(*sample_shape, device=KL_term.device)
            return post_mean, post_log_var.exp()
        else:
            post_log_var = self.post_log_var_scaled*self.log_var_lr
            sqrt_prec = 1./math.sqrt(self.in_features)
            post_mean = self.post_mean*sqrt_prec
            Qw = Normal(post_mean, sqrt_prec*t.exp(0.5*post_log_var))

            w = Qw.rsample(sample_shape=t.Size([sample_shape[0]]))
            prior_prec = self.prior(sample_shape[0])
            logP = mvnormal_log_prob(prior_prec, w.transpose(-1, -2))
            logQ = Qw.log_prob(w).sum((-1, -2))
            self.logpq = logP - logQ
            return post_mean, post_log_var.exp()

    # def inducing_init(self, init):
    #     pass


class FactorisedLRLinearWeights(FactorisedLRParam):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        in_shape = t.Size([in_features+bias])
        super().__init__(in_shape, out_features, **kwargs)


class FactorisedLRConv2dWeights(FactorisedParam):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        in_shape = t.Size([in_channels, kernel_size, kernel_size])
        super().__init__(in_shape, out_channels, **kwargs)


class FactorisedLRLinear(AbstractLRLinear):
    Weights = FactorisedLRLinearWeights


# conv2d is unchanged from factorised.py
class FactorisedLRConv2d(AbstractConv2d):
    Weights = FactorisedLRConv2dWeights