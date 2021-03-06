import random
import math
import torch as t
import torch.nn as nn
from .general import KG

class Kernel(nn.Module):
    """
    Abstract kernel class.  Could take KG or features as input.  Must have ``self.distances`` overwritten.
    """
    def forward(self, xG):
        (d2ii, d2it, d2tt) = self.distances(xG)

        noise_var = 1E-6

        Kii = self.kernel(d2ii)
        Kii = Kii + noise_var*t.eye(Kii.shape[-1], device=d2ii.device, dtype=d2ii.dtype)
        Kit = self.kernel(d2it)
        Ktt = self.kernel(d2tt) + noise_var

        h = (2*self.log_height).exp()

        return KG(h*Kii, h*Kit, h*Ktt)

class KernelFeatures(Kernel):
    """
    Abstract kernel from features.  Has lengthscale parameter for each input and height parameter for overall scale of covariance.
    """
    def __init__(self, in_features, inducing_batch=None):
        super().__init__()
        assert inducing_batch is not None
        self.inducing_batch = inducing_batch
        self.log_lengthscales = nn.Parameter(t.zeros(in_features))
        self.log_height = nn.Parameter(t.zeros(()))

    def d2s(self, x, y=None):
        if y is None:
            y = x

        x2 = (x**2).sum(-1)[..., :, None]
        y2 = (y**2).sum(-1)[..., None, :]
        return x2 + y2 - 2*x@y.transpose(-1, -2)

    def distances(self, x):
        kwargs = {'device' : x.device, 'dtype' : x.dtype}
        x = x * (-self.log_lengthscales).exp()

        xi = x[:, :self.inducing_batch ]
        xt = x[:,  self.inducing_batch:]

        d2ii = self.d2s(xi, xi)
        d2it = self.d2s(xi, xt)
        d2tt = t.zeros(xt.shape[:-1], **kwargs)
        return (d2ii, d2it, d2tt)

class IdentityKernel(nn.Module):
    def forward(self, kg):
        return kg

class CombinedKernel(nn.Module):
    def __init__(self, *ks):
        super().__init__()
        self.ks = nn.ModuleList(ks)
        self.log_coefs = nn.Parameter(t.zeros(len(ks)))

    def forward(self, kg):
        ii = 0.
        it = 0.
        tt = 0.
        for k, log_coef in zip(self.ks, self.log_coefs):
            coef = log_coef.exp()
            _k = k(kg)
            ii += coef*_k.ii
            it += coef*_k.it
            tt += coef*_k.tt
        return KG(ii, it, tt)


class KernelGram(Kernel):
    """
    Abstract kernel from Gram matrix.  A single lengthscale for the input, and a single height parameter.
    """
    def __init__(self, log_lengthscale=0.):
        super().__init__()
        self.log_lengthscales = nn.Parameter(log_lengthscale*t.ones(()))
        self.log_height = nn.Parameter(t.zeros(()))

    def distances(self, G):
        Gii = G.ii
        Git = G.it
        Gtt = G.tt

        diag_Gii = Gii.diagonal(dim1=-1, dim2=-2)
        d2ii = diag_Gii[..., :, None] + diag_Gii[..., None, :] - 2*Gii
        d2it = diag_Gii[..., :, None] + Gtt[..., None, :] - 2*Git
        d2tt = t.zeros_like(Gtt)

        lm2 = (-2*self.log_lengthscales).exp()
 
        #don't need to multiply d2tt by lm2, because d2tt=0.
        return (lm2*d2ii, lm2*d2it, d2tt)

        
class SqExpKernelGram(KernelGram):
    """
    Squared exponential kernel from Gram matrix.

    optional kwargs:
        - **log_lengthscale (float):** initial value for the lengthscale.  Default: ``0.``.
    """
    def kernel(self, d2):
        return t.exp(-0.5*d2)

class SqExpKernel(KernelFeatures):
    """
    Squared exponential kernel from features.

    arg:
        - **in_features (int):**
        - **inducing_batch (int):**
    """
    def kernel(self, d2):
        return t.exp(-0.5*d2)

class ReluKernelGram(nn.Module):
    """
    Relu  kernel from Gram matrix.

    optional kwargs:
        - **log_lengthscale (float):** initial value for the lengthscale.  Default: ``0.``.
    """
    epsilon = 1E-6
    def component(self, xy, xx, yy):
        """
        Computes one matrix of covariances, not necessarily with diagonals

        The original expression:
        pi^{-1} ||x|| ||y|| (sin θ + (π - θ)cos θ)
        where
        cos θ = xy / √(xx yy)

        (1/π) √(xx yy)  (√(1 - xy²/(xx yy)) + (π - θ)xy / √(xx yy)
        which is equivalent to:
        (1/π) ( √(xx yy - xy²) + (π - θ) xy )

        In effect, inject noise along diagonal.
        """
        #input noise
        xx_yy =(xx[..., :, None]+self.epsilon) * (yy[..., None, :]+self.epsilon)

        # Clamp these so the outputs are not NaN
        cos_theta = (xy * xx_yy.rsqrt()).clamp(-1, 1)
        sin_theta = t.sqrt((xx_yy - xy**2).clamp(min=0))
        theta = t.acos(cos_theta)
        K = (sin_theta + (math.pi - theta)*xy) / math.pi

        if xx is yy:
            # Make sure the diagonal agrees with `xx`
            Kv = K.view(*K.shape[:-2], -1)
            Kv[:, ::(K.shape[-1]+1)] = xx+self.epsilon
            K = Kv.view(*K.shape)
            #K = K + self.epsilon * t.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        return K

    def forward(self, K):
        diag_ii = K.ii.diagonal(dim1=-1, dim2=-2)
        ii = self.component(K.ii, diag_ii, diag_ii)
        it = self.component(K.it, diag_ii, K.tt)
        return KG(ii, it, K.tt+self.epsilon)

def ReluKernelFeatures(inducing_batch):
    """
    Relu kernel, which takes features as input
    """
    return nn.Sequential(FeaturesToKernel(inducing_batch), ReluKernelGram())

class FeaturesToKernel(nn.Module):
    """
    Converts features to the corresponding Gram matrix.

    arg:
        - **inducing_batch (int):** Number of inducing inputs.
    
    """
    def __init__(self, inducing_batch, epsilon=None):
        super().__init__()
        self.inducing_batch = inducing_batch
        self.epsilon = epsilon

    def forward(self, x):
        in_features = x.shape[-1]
        xi = x[:, :self.inducing_batch ]
        xt = x[:,  self.inducing_batch:]

        ii = xi @ xi.transpose(-1, -2) / in_features
        it = xi @ xt.transpose(-1, -2) / in_features
        tt = (xt**2).sum(-1) / in_features

        if self.epsilon is not None:
            ii = ii + self.epsilon*t.eye(ii.shape[-1], dtype=ii.dtype, device=ii.device)
            tt = tt + self.epsilon

        #print("f2k")
        #print(ii[0, :3, :3])
        #print(it[0, :3, :3])
        #print(tt[0, :3])
        return KG(ii, it, tt)
 
        
        


#class DistanceKernel(nn.Module):
#    def __init__(self, noise, inducing_batch):
#        super().__init__()
#        assert inducing_batch is not None
#        self.inducing_batch = inducing_batch
#        if noise:
#            self.log_noise = nn.Parameter(-4*t.ones(()))
#        else:
#            self.register_buffer("log_noise", -100*t.ones(()))
#       
#    def d2s(self, x, y=None):
#        if y is None:
#            y = x
#
#        x2 = (x**2).sum(-1)[..., :, None]
#        y2 = (y**2).sum(-1)[..., None, :]
#        return x2 + y2 - 2*x@y.transpose(-1, -2)
#
#    def forward(self, x):
#        kwargs = {'device' : x.device, 'dtype' : x.dtype}
#        x = x * (-self.log_lengthscale).exp()
#        xi = x[:, :self.inducing_batch ]
#        xt = x[:,  self.inducing_batch:]
#
#        d2ii = self.d2s(xi, xi)
#        d2it = self.d2s(xi, xt)
#        d2tt = t.zeros(xt.shape[:-1], **kwargs)
#
#        noise_var = 1E-6+self.log_noise.exp()
#
#        Kii = self.kernel(d2ii)
#        Kii = Kii + noise_var*t.eye(Kii.shape[-1], **kwargs)
#        Kit = self.kernel(d2it)
#        Ktt = self.kernel(d2tt) + noise_var
#
#        return KG(Kii, Kit, Ktt)
#        
#        
#
#class SqExpKernel(DistanceKernel):
#    def __init__(self, in_features, noise=False, inducing_batch=None):
#        super().__init__(noise, inducing_batch)
#
#        self.log_lengthscale = nn.Parameter(t.zeros(in_features))
#        self.log_height = nn.Parameter(t.zeros(()))
#
#
#    def kernel(self, d2s):
#        return t.exp(self.log_height - d2s)


