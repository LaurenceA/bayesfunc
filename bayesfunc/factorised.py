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

        self._sample = None

    def forward(self, Xi):
        S = Xi.shape[0]

        post_log_var = self.post_log_var_scaled*self.log_var_lr
        sqrt_prec = 1./math.sqrt(self.in_features)
        Qw = Normal(sqrt_prec*self.post_mean, sqrt_prec*t.exp(t.ones((), device=Xi.device)*0.5*post_log_var))

        if self._sample is not None:
            assert S == self._sample.shape[0]
        else:
            self._sample = Qw.rsample(sample_shape=t.Size([S])) 

        prior_prec = self.prior(S)
        logP = mvnormal_log_prob(prior_prec, self._sample.transpose(-1, -2))
        logQ = Qw.log_prob(self._sample).sum((-1, -2))
        assert logP.shape == t.Size([S])
        assert logQ.shape == t.Size([S])
        self.logpq = logP - logQ
        return self._sample



class FactorisedLinearWeights(FactorisedParam):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        in_shape = t.Size([in_features+bias])
        super().__init__(in_shape, out_features, **kwargs)


class FactorisedConv2dWeights(FactorisedParam):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        in_shape = t.Size([in_channels, kernel_size, kernel_size])
        super().__init__(in_shape, out_channels, **kwargs)


class FactorisedLinear(AbstractLinear):
    r"""
    IID Gaussian prior and factorised Gaussian posterior over the weights of a fully-connected layer.

    arg:
        - **in_features:** size of each input sample
        - **out_features:** size of each output sample

    kwargs:
        - **bias:** If set to ``False``, the layer will not learn an additive bias.  Default: ``True``
        - **prior:** The prior over weights.  Default ``NealPrior``.
        - **var_fixed:** Defaults to ``None``.  If set to a float, it fixes the approximate posterior variance over weights to that value.
        - **var_init_mult:** The approximate posterior variance is initialized to be equal to the prior variance, multiplied by ``var_init_mult``.  Defaults to ``1E-3`` such that the variances are initialized to be small.
        - **mean_init_mult:** The approximate posterior means are initialized by sampling from the prior, multiplied by ``mean_init_mult``.  As there is no particular reason to make this small, it defaults to 1.
        - **log_var_lr:** Multiplier for the learning rate for the approximate posterior variances.

    Shape:
        - **Input:** ``(samples, mbatch, in_features)``
        - **Output:** ``(samples, mbatch, out_features)``

    Random Variables:
        - **weight:** the learnable weights of the module of shape ``(in_features+bias, out_features)``, where ``bias=True`` or ``bias=False`` which converts to ``bias=1`` or ``bias=1``.
          Note that we implement the bias by adding a vector of ones to the input, so the dimension of the weights depends on the presence of a bias.

    Prior:
        - IID Gaussian, with variance :math:`1/\text{in_channels}`

    Approximate Posterior:
        - MFVI 


    Examples:

        >>> import torch
        >>> import bayesfunc as bf
        >>> m = bf.FactorisedLinear(20, 30)
        >>> input = torch.randn(3, 128, 20)
        >>> output, _, _ = bf.propagate(m, input)
        >>> print(output.size())
        torch.Size([3, 128, 30])
    """
    Weights = FactorisedLinearWeights


class FactorisedConv2d(AbstractConv2d):
    r"""
    IID Gaussian prior and factorised Gaussian posterior over the weights of a 2D convolutional layer.

    arg:
        - **in_channels:** number of channels in input tensor
        - **out_channels:** number of channels in output tensor
        - **kernel_size:** size of convolutional kernel

    kwargs:
        - **stride:** Standard convolutional stride.  Defaults to 1.
        - **padding:** Standard convolutional padding.  Defaults to 0.
        - **prior:** The prior over weights.  Default ``NealPrior``.
        - **var_fixed:** Defaults to ``None``.  If set to a float, it fixes the approximate posterior variance over weights to that value.
        - **var_init_mult:** The approximate posterior variance is initialized to be equal to the prior variance, multiplied by ``var_init_mult``.  Defaults to ``1E-3`` such that the variances are initialized to be small.
        - **mean_init_mult:** The approximate posterior means are initialized by sampling from the prior, multiplied by ``mean_init_mult``.  As there is no particular reason to make this small, it defaults to 1.
        - **log_var_lr:** Multiplier for the learning rate for the approximate posterior variances.

    Shape:
        - **Input:** ``(samples, mbatch, in_height, in_width, in_features)``
        - **Output:** ``(samples, mbatch, in_height, in_width, out_features)``

    Random Variables:
        - **weight:** the learnable weights of the module of shape
          ``(out_channels, in_channels, in_features, out_features)``.

    Prior:
        - IID Gaussian, with variance :math:`1/(\text{fan-in}*\text{kernel_size}^2)`

    Approximate Posterior:
        - MFVI 


    Examples:

    """
    Weights = FactorisedConv2dWeights
