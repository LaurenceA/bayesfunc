import math
import torch as t
from .singular_cholesky import singular_cholesky
from .lop import PositiveDefiniteMatrix, LowerTriangularMatrix, Inv, Product, Matrix


def bartlett(K, nu, sample_shape=t.Size([])):
    """
    We're slightly generalising the Bartlett decomposition approach to sampling the Wishart,
    by allowing the chi^2 variables to be drawn from their Gamma-distributed generalisation.
    See Appendix B.
    """
    #ignores K, except shape, device etc.
    device = K.device
    Kshape = K.full().shape
    n = t.tril(t.randn((*sample_shape, *Kshape), device=K.device), -1)
    I = t.eye(K.N, device=K.device)
    one = t.ones((), device=K.device)

    dof = nu - t.arange(K.N, device=device)
    gamma = t.distributions.Gamma(dof/2., 1/2)
    c = gamma.rsample(sample_shape=[*sample_shape, *Kshape[:-2], 1]).sqrt()
    A = LowerTriangularMatrix((n + I*c)/t.sqrt(one*nu))
    AAT = PositiveDefiniteMatrix(chol=A)
    return AAT


class Wishart:
    def __init__(self, K, nu):
        assert K.shape[-1] == K.shape[-2]
        self.p = K.shape[-1]
        assert (self.p-1) < nu
        self.K = K
        self.nu = nu

    def rsample(self, sample_shape=t.Size([])):
        A = bartlett(self.K, self.nu, sample_shape)
        LA = singular_cholesky(self.K) @ A
        return LA @ LA.transpose(-1, -2)

    def log_prob(self, x):
        nu = self.nu
        p = self.p

        res  = ((nu-p-1)/2)*t.logdet(x)
        res -= (1/2)*(t.inverse(self.K) * x).sum((-1, -2))
        res -= (nu*p/2) * math.log(2)
        res -= (nu/2) * t.logdet(self.K)
        res -= t.mvlgamma(nu/2*t.ones(()), p)

        return res


class InverseWishart:
    def __init__(self, K, nu):
        assert K.N < nu
        assert isinstance(K, Matrix)
        self.K = K
        self.nu = nu

    def rsample_log_prob(self, sample_shape=t.Size([])):
        x, AAT = self._rsample(sample_shape)
        log_prob = self.log_prob(x, AAT)
        return x, log_prob

    def _rsample(self, sample_shape):
        L = self.K.chol()
        AAT = bartlett(L, self.nu, sample_shape)
        return Product(L, Inv(AAT), L.t()), AAT

    def rsample(self, sample_shape=t.Size([])):
        return self._rsample(sample_shape)[0]

    def log_prob(self, x, AAT=None):
        if AAT is None:
            AAT = x.inv(self.K.full())
        else:
            AAT = AAT.full()

        nu = self.nu
        p = self.K.N

        res  = -((nu+p+1)/2)*x.logdet()
        #modified by multiplying by nu
        res += (nu/2)*self.K.logdet() + (p*nu/2)*t.log(t.ones((), device=x.device)*nu)
        #modified by multiplying by nu
        res -= (nu/2)*(AAT).diagonal(dim1=-1, dim2=-2).sum(-1)
        res -= (nu*p/2) * math.log(2)
        res -= t.mvlgamma(nu/2*t.ones(()), p)

        return res


if __name__ == "__main__":
    import numpy as np
    from scipy.stats import wishart, invwishart
    tmp = t.randn(3, 3)
    S = PositiveDefiniteMatrix(tmp@tmp.t()/3)

    NU = 10
    iw = InverseWishart(S, NU)
    iw_np = invwishart(NU, S.full().numpy()*NU)

    x_lop = iw.rsample()
    lp_lop = iw.log_prob(x_lop)
    x_np  = x_lop.full().numpy()[:, :]
    lp_np = iw_np.logpdf(x_np)

    print(lp_lop.item())
    print(lp_np)

    assert np.isclose(lp_lop.item(), lp_np)
