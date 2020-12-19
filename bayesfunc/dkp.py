import torch.nn as nn
from torch.distributions import Normal, Gamma
from torch.distributions.utils import broadcast_all
import math
import torch as t
import numpy as np


from .lop import Matrix, LowerTriangularMatrix, PositiveDefiniteMatrix, Product, Inv, Identity, Scale, Product
from .general import KG


def bartlett(K, nu, sample_shape=t.Size([])):
    """
    Standard Bartlett (unlike non-standard version for sampling)
    """
    #ignores K, except shape, device etc.
    kwargs = {'device':K.device, 'dtype':K.dtype}
    Kshape = K.full().shape
    n = t.tril(t.randn((*sample_shape, *Kshape), **kwargs), -1)
    I = t.eye(K.N, **kwargs)

    dof = nu - t.arange(K.N, **kwargs)
    gamma = t.distributions.Gamma(dof/2., 1/2)
    
    c = gamma.rsample(sample_shape=[*sample_shape, *Kshape[:-2], 1]).sqrt()
    A = LowerTriangularMatrix((n + I*c))  #/t.sqrt(one*nu))
    AAT = PositiveDefiniteMatrix(chol=A)
    return AAT


class InverseWishart:
    def __init__(self, K, nu):
        assert not t.isnan(nu)
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
    # Determinant and inverse of this is trivial to compute _without_ redoing the

    def log_prob(self, x, AAT=None):
        #### K should be multiplied by nu

        if AAT is None:
            AAT = x.inv(self.K.full())   # x^-1 @ K
        else:
            AAT = AAT.full()

        nu = self.nu
        p = self.K.N

        res  = -((nu+p+1)/2)*x.logdet()
        #modified by multiplying by nu
        res += (nu/2)*self.K.logdet()   # + (p*nu/2)*t.log(t.ones((), device=x.device)*nu)
        #modified by multiplying by nu
        res -= (1/2)*(AAT).diagonal(dim1=-1, dim2=-2).sum(-1)
        res -= (nu*p/2) * math.log(2)
        res -= t.mvlgamma(nu/2*t.ones((), device=x.device, dtype=x.dtype), p)

        return res

class IWLayer(nn.Module):
    """
    Inverse Wishart layer from a deep kernel process.  Takes a KG as input, and returns KG as output.
    
    arg:
        - **inducing_batch (int):** number of inducing inputs
    """
    def __init__(self, inducing_batch):
        super().__init__()
        self.P = inducing_batch

        self.log_delta = nn.Parameter(t.zeros(())) #4*t.ones(()))

        self.V = nn.Parameter(t.randn(inducing_batch, inducing_batch))
        self.log_diag = nn.Parameter(t.zeros(())) #-4*t.ones(()))#Just a scale parameter for VV^T.
        self.log_gamma = nn.Parameter(t.zeros(())) #t.zeros(())) #Increase!

    @property
    def delta(self):
        return self.log_delta.exp()
    @property
    def gamma(self):
        return self.log_gamma.exp()
    @property
    def diag(self):
        return self.log_diag.exp()

    @property
    def prior_nu(self):
        return self.delta + self.P + 1
    @property
    def post_nu(self):
        return self.prior_nu + self.gamma

    def prior_psi(self, dKii):
        return dKii
    def post_psi(self, dKii):
        return PositiveDefiniteMatrix(dKii.full() + (self.V*self.diag) @ self.V.T / self.P)
        #return PositiveDefiniteMatrix(dKii.full() + self.gamma*(self.V @ self.V.T) / self.P)

    def PGii(self, dKii):
        return InverseWishart(self.prior_psi(dKii), self.prior_nu)

    def QGii(self, dKii):
        return InverseWishart(self.post_psi(dKii),  self.post_nu)

    def dd_kwargs():
        return {'device': self.V.device, 'dtype': self.V.dtype}

    def Gii(self, dKii):
        PGii = self.PGii(dKii)
        QGii = self.QGii(dKii)

        Gii, logQ = QGii.rsample_log_prob()
        logP = PGii.log_prob(Gii)

        #assert x.full().shape == t.Size([S, self.p, self.p])

        self.logpq = logP-logQ

        return Gii

    def forward(self, K):
        #if np.random.rand() < 0.01:
        #    print()
        #    print(("gamma:", self.log_gamma.item()))
        #    print(("delta:", self.log_delta.item()))
        #    #print(("diag:", self.log_diag))
        """
        DSVI evaluation of the Schur complement of K.
        Evaluate for a single test-point

        ktt - kti Kii^{-1} Kit
        Kii : matrix
        kti : vector (row)
        kit : vector (column)
        ktt : scalar
        """

        #Kii = PositiveDefiniteMatrix(K.ii)
        dKii = PositiveDefiniteMatrix(self.delta*K.ii)
        dkit = self.delta*K.it
        dktt = self.delta*K.tt

       

        Pi = self.P
        Pt = dktt.shape[-1]

        Gii = self.Gii(dKii).full()
        (S, _, _) = dkit.shape
        assert self.logpq.shape == t.Size([S])

        inv_Kii_kit = dKii.inv(dkit)

        # Diagonal of Schur complement
        dktti = dktt - (dkit * inv_Kii_kit).sum(-2)
        alpha = (self.delta + Pi + Pt + 1)/2
        Ptt = t.distributions.Gamma(concentration=alpha, rate=dktti/2)
        gtti = Ptt.rsample().reciprocal()


        inv_Gii_git = inv_Kii_kit + dKii.inv_sqrt()(t.randn_like(dkit)) * gtti.sqrt()[:, None, :]
        git = Gii @ inv_Gii_git

        gtt = gtti + (git * inv_Gii_git).sum(-2)

        #print("dkp")
        #print(Gii[0, :3, :3])
        #print(git[0, :3, :3])
        #print(gtt[0, :3])
        return KG(Gii, git, gtt)

class SingularIWLayer(IWLayer):
    """
    Singular Inverse Wishart layer which takes the input features in a deep kernel process. Takes a features as input, and returns KG as output.
    
    arg:
        - **in_features (int):** number of features
        - **inducing_batch (int):** number of inducing points.
    """
    def __init__(self, in_features, inducing_batch):
        super().__init__(in_features)
        self.in_features = in_features
        self.inducing_batch = inducing_batch

    def forward(self, x):
        dI = Scale(self.in_features, self.delta/self.in_features, device=x.device, dtype=x.dtype)
        PXi = self.PGii(dI)
        QXi = self.QGii(dI)

        S = x.shape[0]
        Xi, logQ = QXi.rsample_log_prob(sample_shape=t.Size([S]))
        logP = PXi.log_prob(Xi)

        assert logQ.shape == t.Size([S])
        assert logP.shape == t.Size([S])
        self.logpq = logP - logQ


        xi = x[:, :self.inducing_batch]
        xt = x[:, self.inducing_batch:]

 

        Xi = Xi.full()
        Xixt = Xi @ xt.transpose(-1, -2)
        Gii  = xi @ Xi @ xi.transpose(-1, -2)
        Git  = xi @ Xixt
        Gtt  = (xt * Xixt.transpose(-1, -2)).sum(-1)

        return KG(Gii, Git, Gtt)
