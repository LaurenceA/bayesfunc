import math
import torch as t
import torch.distributions as td
import torch.nn as nn
from .prob_prog import VI_Scale, VI_Normal
from .outputs import CutOutput

class KG():
    r"""
    Simple container class for different components of a covariance matrix.  You shouldn't need to use this unless you are developing your own kernels.

    arg:
        - **ii:** :math:`P_\text{i}\times P_\text{i}` covariance matrix for inducing points.  ``shape=(samples, inducing_batch, inducing_batch)``
        - **it:** :math:`P_\text{i}\times P_\text{t}` covariance matrix for inducing points.  ``shape=(samples, inducing_batch, mbatch)``
        - **tt:** :math:`P_\text{t}` diagonal variances test/train points.  ``shape=(samples, 1, mbatch)``?
    """
    def __init__(self, ii, it, tt):
        self.ii = ii
        self.it = it
        self.tt = tt

class InducingAdd(nn.Module):
    def __init__(self, inducing_batch, inducing_data=None, inducing_shape=None, fixed=False):
        super().__init__()
        assert (inducing_data is not None) != (inducing_shape is not None)

        if inducing_data is None:
            self.inducing_data = nn.Parameter(t.randn(*inducing_shape))
        else:
            i_d = inducing_data.clone().to(t.float32)
            if fixed:
                self.register_buffer("inducing_data", i_d)
            else:
                self.inducing_data = nn.Parameter(i_d)

        assert inducing_batch == self.inducing_data.shape[0]
        self.rank = 1 + len(self.inducing_data.shape)

    def forward(self, x):
        assert self.rank == len(x.shape) 

        inducing_data = self.inducing_data.expand(x.shape[0], *self.inducing_data.shape)
        x = t.cat([inducing_data, x], 1)
         
        return x

class InducingRemove(nn.Module):
    def __init__(self, inducing_batch):
        super().__init__()
        self.inducing_batch = inducing_batch

    def forward(self, x):
        return x[:, self.inducing_batch:]

def InducingWrapper(net, inducing_batch, *args, **kwargs):
    """
    Combines incoming test/train data with learned inducing inputs, then strips away the inducing outputs, just leaving the function approximated at inducing locations. 

    args:
        net (nn.Module): The underlying function approximator, represented as PyTorch modules, to be wrapped.
        inducing_batch (int): The underlying function approximator, represented as PyTorch modules, to be wrapped.

    Keyword Args:
        inducing_shape (Optional[torch.Size]): The size of the inducing inputs, including `inducing_batch` as the first dimension.  Default: ``None``.
        inducing_data (Optional[torch.Tensor]): The values of the inducing inputs. Useful to e.g. initialize the inducing points on top of datapoints.  Default: ``None``.
        fixed (Bool): Do we fix the inducing point locations?  Default: ``False``.

    Must specify one and only one of `inducing_shape` or `inducing_data`
 
    Example:
        >>> import bayesfunc as bf
        >>> import torch as t
        >>> import torch.nn as nn
        >>>
        >>> in_features = 20
        >>> hidden_features = 50
        >>> out_features = 30
        >>>
        >>> m1 = bf.GILinear(in_features, hidden_features, inducing_batch=100)
        >>> m2 = bf.GILinear(hidden_features, out_features, inducing_batch=100)
        >>> net = nn.Sequential(m1, m2)
        >>>
        >>> net = bf.InducingWrapper(net, 100, inducing_shape=(100, in_features))
        >>> output, _, _ = bf.propagate(net, t.randn(3, 128, in_features))
        >>> output.shape
        torch.Size([3, 128, 30])
    """
    ia = InducingAdd(inducing_batch, *args, **kwargs)
    ir = InducingRemove(inducing_batch)
    return nn.Sequential(ia, net, ir)

def logpq(f):
    """
    Extracts log P(f) - log Q(f) by iterating through all modules in the network

    args:
        f: function approximator written as a pytorch module
    """
    total = 0.
    for mod in f.modules():
        if hasattr(mod, "logpq"):
            total += mod.logpq
            mod.logpq = None
    return total

def clear_sample(f):
    for mod in f.modules():
        if hasattr(mod, "_sample"):
            mod._sample = None 

def get_sample_dict(f):
    result = {}
    for (n, m) in f.named_modules():
        if hasattr(m, "_sample"):
            result[n] = m._sample
    return result

def set_sample_dict(f, sample_dict):
    mod_dict = {n: m for (n, m) in f.named_modules()}
    for name, sample in sample_dict.items():
        mod = mod_dict[name]
        assert hasattr(mod, "_sample")
        mod._sample = sample.detach()

def propagate(f, input, sample_dict=None):
    """
    The ONLY way to run the neural networks defined in bayesfunc.  Replaces `f(input)`, which will now fail silently!
    
    args:
        f: the bayesfunc function
        input: input to the function

    keyword args:
        sample_dict: optional dictionary of sampled weights, to allow using the same weights for multiple different inputs

    outputs:
        - **output:** neural network output (as in ``output = f(input)``). 
        - **logpq:** :math:`\log P(f) - \log Q(f)` the difference of prior and approximate posterior log-probabilities.
        - **output_sample_dict:** a dictionary containing all the sampled weights used in the network.  If ``sample_dict`` is set, we have ``output_sample_dict == sample_dict``.

    warning:
        Only properly implemented for ``GILinear``, ``GIConv2d``, ``FactorisedLinear`` and ``FactorisedConv2D``.  Everything else will run, but will independently sample a new function on every invocation, ignoring the ``sample_dict`` input argument.

    In standard use, ``sample_dict`` is never set and ``output_sample_dict`` is never needed.  These only become useful e.g. in continual learning.
    """
    clear_sample(f)
    if sample_dict is not None:
        set_sample_dict(f, sample_dict)
    output = f(input) 
    sample_dict = get_sample_dict(f)
    clear_sample(f)
    return output, logpq(f), sample_dict

    
class NormalLearnedScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(t.zeros(()))

    def forward(self, x):
        return td.Normal(x, self.log_scale.exp())

class Bias(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.bias = nn.Parameter(t.zeros(in_features))

    def forward(self, x):
        return x + self.bias

class BiasFeature(nn.Module):
    def forward(self, x):
        return t.cat([x, t.ones(*x.shape[:2], 1, *x.shape[3:])], 2)

class MultFeatures(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.log_scale = nn.Parameter(t.zeros(t.Size([*shape])))

    def forward(self, x):
        return x * self.log_scale.exp()

class MultKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(t.zeros(()))

    def forward(self, k):
        scale = self.log_scale.exp() 
        return KG(k.ii*scale, k.it*scale, k.tt*scale)


#class Wrapper(nn.Module):
#    def __init__(self, net, out=CutOutput(), init=None, in_dim=None, out_dim=None, parallel=True, device_ids=None):
#        #Note: CutOutput is stateless, so okay if same object shared across wrapped networks
#        super().__init__()
#        self.net = nn.DataParallel(net, device_ids=device_ids) if parallel else net
#        self.out = out
#        self.in_dim = in_dim
#        self.out_dim = out_dim
#
#        if init is not None:
#            self.inducing_input = nn.Parameter(init.clone().to(t.float32))
#            if isinstance(out, CutOutput):
#                out.inducing_init(init.shape[0])
#        else:
#            self.inducing_input = None
#
#    def forward(self, test_input, S=1):
#        test_input = test_input.expand(S, *test_input.shape)
#
#        if self.inducing_input is not None:
#            inducing_input = self.inducing_input.expand(S, *self.inducing_input.shape)
#            x = t.cat([inducing_input, test_input], self.in_dim)
#        else:
#            x = test_input
#
#        output = self.net(x)
#        output = self.out(output)
#        
#        logpq = 0.
#        for mod in [*self.out.modules(), *self.net.modules()]:
#            if hasattr(mod, "logpq"):
#                logpq += mod.logpq
#                mod.logpq = None
#
#        return output, logpq



class Reduce(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x):
        outputs = []
        for module in self._modules.values():
            _output = module(x)
            outputs.append(_output)

        return self.reduce(outputs)



class Cat2d(Reduce):
    def reduce(self, xs):
        return t.cat(xs, -3)


class Sum(Reduce):
    def reduce(self, xs):
        return sum(xs)


class WrapMod(nn.Module):
    """
    Wraps an underlying PyTorch module, which can (but should not) contain parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mod = self.Mod(*args, **kwargs)

    def forward(self, x):
        return self.mod(x)



def batch(x, dim):
    x = x.contiguous()
    return x.view(x.shape[:dim].numel(), *x.shape[dim:]), x.shape[:dim]


def unbatch(x, shape):
    return x.view(*shape, *x.shape[1:])


class WrapMod2d(WrapMod):
    """
    Wraps an underlying PyTorch module, which assumes the input arrives as a 4D tensor,
    as standard for networks applied to images.
    """
    def forward(self, x):
        x, shape = batch(x, -3)
        x = self.mod(x)
        x = unbatch(x, shape)
        return x


class AbstractBatchNorm2d(nn.Module):

    def moments(self, x):
        #Average over spatial dims and batch, but not S
        Ex = x.mean((-1, -2, -4), keepdim=True)
        Ex2 = (x**2).mean((-1, -2, -4), keepdim=True)
        return Ex, Ex2

    def forward(self, x):
        (S, _, _, _, _) = x.shape
        return x

        mult, mult_logPQ = self.mult(S)
        bias, bias_logPQ = self.bias(S)

        if hasattr(self, "inducing_batch"):
            Ex, Ex2 = self.moments(x)
        else:
            Ex, Ex2 = self.moments(x)

        s = t.rsqrt(Ex2 - Ex**2)
        x = (mult*s)*x + (-s*Ex+bias)
        return x, mult_logPQ + bias_logPQ


class DetBatchNorm2d(AbstractBatchNorm2d):
    def __init__(self, channels):
        super().__init__()
        self._mult = nn.Parameter(t.ones(1, channels, 1, 1))
        self._bias = nn.Parameter(t.zeros(1, channels, 1, 1))

        self.logits_prop = nn.Parameter(t.zeros(1, 1, 1, 1, 2))

    def mult(self, S):
        return self._mult

    def bias(self, S):
        return self._bias

    def prop(self, S):
        return 0.5


class BatchNorm2d(AbstractBatchNorm2d):
    def __init__(self, channels):
        super().__init__()
        self._mult = VI_Scale((1, channels, 1, 1), init_log_shape=6., init_scale=1.)
        self._bias = VI_Normal((1, channels, 1, 1), init_log_prec=6., init_mean=0.)
        self._prop = VI_Normal((1, 1, 1, 1, 2), init_log_prec=6., init_mean=0.)

    def mult(self, S):
        return self._mult(S, 2., 1.)

    def bias(self, S):
        return self._bias(S, 0., 1.)

    def prop(self, S):
        return t.sigmoid(self._prop(S, 0, 1.))


class MaxPool2d(WrapMod2d):
    Mod = nn.MaxPool2d


class AdaptiveAvgPool2d(WrapMod2d):
    Mod = nn.AdaptiveAvgPool2d


class AvgPool2d(WrapMod2d):
    Mod = nn.AvgPool2d



class _Conv2d_2_FC(nn.Module):
    def forward(self, x):
        return x.view(*x.shape[:-3], -1)


class Conv2d_2_FC(WrapMod):
    Mod = _Conv2d_2_FC


class Print(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
