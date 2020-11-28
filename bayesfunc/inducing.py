import torch as t
import torch.nn as nn
from .conv_mm import conv_mm
from .abstract_bnn import AbstractLinear, AbstractConv2d
from .lop import mvnormal_log_prob_unnorm


def rsample_logpq_weights(self, XLX, XLY, prior, neuron_prec=True):
    device = XLX.device
    in_features = XLX.shape[-1]
    out_features = XLY.shape[-3]

    assert 4 == len(XLX.shape) and 4 == len(XLY.shape)
    assert XLX.shape[0] == XLY.shape[0]
    assert XLX.shape[-3] in (out_features, 1)
    assert XLY.shape[-1] == 1
    S = XLX.shape[0]
    prior_prec = prior(S)

    prior_prec_full = prior_prec.full()
    if len(prior_prec_full.shape) > 2:
        prior_prec_full = prior_prec_full.unsqueeze(1)

    prec = XLX + prior_prec_full
    L = t.cholesky(prec)

    logdet_prec = 2*L.diagonal(dim1=-1, dim2=-2).log().sum(-1)
    logdet_prec = logdet_prec.expand(S, out_features).sum(-1)

    Z = t.randn(S, out_features, in_features, 1, device=device, dtype=L.dtype)
    dW = t.triangular_solve(Z, L, upper=False, transpose=True)[0]
    W = (t.cholesky_solve(XLY, L) + dW).squeeze(-1)

    logP = mvnormal_log_prob_unnorm(prior_prec, W.transpose(-1, -2))
    logQ = -0.5*(Z**2).sum((-1, -2, -3)) + 0.5*logdet_prec

    logPQw = logP-logQ
    self.logpq = logPQw
    return W


def rsample_logpq_weights_fc(self, Xi, neuron_prec):
    log_prec = self.log_prec_lr*self.log_prec_scaled
    XiLT  = log_prec.exp() * Xi.transpose(-1, -2)
    XiLXi = XiLT @ Xi
    XiLY  = XiLT @ self.u
    return rsample_logpq_weights(self, XiLXi, XiLY, self.prior, neuron_prec=neuron_prec)


class GILinearWeights(nn.Module):
    def __init__(self, in_features, out_features, prior=None, bias=True, inducing_targets=None, log_prec_init=-4., log_prec_lr=1., neuron_prec=False, inducing_batch=None):
        super().__init__()
        assert inducing_batch is not None
        self.inducing_batch = inducing_batch

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        in_shape   = t.Size([in_features+bias])
        self.prior = prior(in_shape)

        self.log_prec_init = log_prec_init
        self.log_prec_lr = log_prec_lr
        lp_init = self.log_prec_init / self.log_prec_lr
        self.neuron_prec = neuron_prec

        if inducing_targets is None:
            self.u = nn.Parameter(t.randn(self.out_features, inducing_batch, 1))
        else:
            self.u = nn.Parameter(inducing_targets.clone().to(t.float32).transpose(-1, -2).unsqueeze(-1))

        precs = out_features if neuron_prec else 1
        self.log_prec_scaled = nn.Parameter(lp_init*t.ones(precs, 1, inducing_batch))

    def forward(self, X):
        Xi = X[:, :self.inducing_batch, :]
        return rsample_logpq_weights_fc(self, Xi.unsqueeze(1), neuron_prec=True)


class LILinearWeights(nn.Module):
    def __init__(self, in_features, out_features, prior=None, bias=True, log_prec_init=-4., log_prec_lr=1., neuron_prec=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        in_shape   = t.Size([in_features+bias])
        self.prior = prior(in_shape)
        self.neuron_prec = neuron_prec

        inducing_batch = in_features + bias

        lp_init = log_prec_init / log_prec_lr
        self.log_prec_lr = log_prec_lr

        #if neuron_prec:
        self.u = nn.Parameter(t.randn(self.out_features, inducing_batch, 1))
        self.Xi = nn.Parameter(t.randn(1, inducing_batch, self.in_features+bias))

        precs = self.out_features if neuron_prec else 1
        self.log_prec_scaled = nn.Parameter(lp_init*t.ones(precs, 1, inducing_batch))

    def forward(self, Xi):
        # inducing inputs are those stored, but expanded in first dimension to match inputs
        Xi = self.Xi.expand(Xi.shape[0], *self.Xi.shape)
        return rsample_logpq_weights_fc(self, Xi, neuron_prec=True)


class GIConv2dWeights(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, prior=None, stride=1, padding=0, bias=False, inducing_targets=None, log_prec_init=-4., log_prec_lr=1., neuron_prec=False, inducing_batch=None):
        super().__init__()
        assert 1==kernel_size%2
        assert padding == kernel_size//2
        assert not bias
        assert inducing_batch is not None
        assert inducing_batch != 0
        self.inducing_batch = inducing_batch

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        in_shape   = t.Size([in_channels, kernel_size, kernel_size])
        self.prior = prior(in_shape)

        self.stride = stride
        self.padding = padding
        self.log_prec_init = log_prec_init
        self.log_prec_lr = log_prec_lr
        self.neuron_prec = neuron_prec

        self.u = None if (inducing_targets is None) else nn.Parameter(inducing_targets.clone().to(t.float32))

        lp_init = self.log_prec_init / self.log_prec_lr
        self.log_prec_scaled = nn.Parameter(lp_init*t.ones(self.inducing_batch))

    def forward(self, X):
        Xi = X[:, :self.inducing_batch, :, :, :]
        if self.u is None:
            (_, _, _, Hin, Win) = Xi.shape
            HW_in = (Hin, Win)
            HW_out = [(HW_in[i] + 2*self.padding - self.kernel_size) // self.stride + 1 for i in range(2)]
            self.u = nn.Parameter(t.randn(self.inducing_batch, self.out_channels, *HW_out, device=Xi.device, dtype=Xi.dtype))

        sqrt_prec = (0.5 * self.log_prec_lr * self.log_prec_scaled).exp()[:, None, None, None]
        Xil = sqrt_prec * Xi
        Yil = sqrt_prec * self.u

        XiLXi, XiLY = conv_mm(Xil, Yil, self.kernel_size)
        XiLXi = XiLXi.unsqueeze(1)
        XiLY = XiLY.transpose(-1, -2).unsqueeze(-1)
        return rsample_logpq_weights(self, XiLXi, XiLY, self.prior, neuron_prec=True)


class LIConv2dWeights(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, prior=None, stride=1, padding=0, bias=False, log_prec_init=-4., log_prec_lr=1., neuron_prec=False):
        super().__init__()
        assert 1==kernel_size%2
        assert padding == kernel_size//2
        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        in_shape   = t.Size([in_channels, kernel_size, kernel_size])
        self.prior = prior(in_shape)

        self.stride = stride
        self.padding = padding
        self.log_prec_init = log_prec_init
        self.log_prec_lr = log_prec_lr
        self.neuron_prec = neuron_prec

        in_features = in_channels*kernel_size**2

        inducing_batch = in_features

        lp_init = log_prec_init / log_prec_lr
        self.log_prec_lr = log_prec_lr

        self.u = nn.Parameter(t.randn(self.out_channels, inducing_batch, 1))
        self.Xi = nn.Parameter(t.randn(1, inducing_batch, in_features))

        precs = out_features if neuron_prec else 1
        self.log_prec_scaled = nn.Parameter(lp_init*t.ones(precs, 1, inducing_batch))

    def forward(self, Xi):
        # inducing inputs are those stored, but expanded in first dimension to match inputs
        Xi = self.Xi.expand(Xi.shape[0], *self.Xi.shape)
        return rsample_logpq_weights_fc(self, Xi, neuron_prec=self.neuron_prec)
        


class GILinear(AbstractLinear):
    Weights = GILinearWeights


class GIConv2d(AbstractConv2d):
    Weights = GIConv2dWeights


class LILinear(AbstractLinear):
    Weights = LILinearWeights


class LIConv2d(AbstractConv2d):
    Weights = LIConv2dWeights
