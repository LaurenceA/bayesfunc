import math
import torch as t
import torch.nn as nn
from .lop import PositiveDefiniteMatrix
from .general import KG


def bartlett(K, nu, sample_shape=t.Size([]), alpha=None, beta=None, mu=None, sigma=None):
    """
    returns 'Cholesky' factor T and log prob of T
    """
    if beta is not None:
        """
        Generalized Bartlett
        """
        kwargs = {'device': K.device, 'dtype': K.dtype}
        Kshape = K.full().shape
        ncols = min(K.N, nu)
        assert beta.shape == (ncols,)
        assert sigma.shape == (K.N, ncols)

        n = t.tril(mu + t.randn((*sample_shape, *Kshape[:-1], ncols), **kwargs)*sigma, -1)
        normal = t.distributions.normal.Normal(loc=mu.detach(),
                                               scale=sigma.detach())
        log_prob_norm = t.tril(normal.log_prob(n), -1).sum((-1, -2))
        I = t.eye(K.N, m=ncols, **kwargs)

        gamma = t.distributions.Gamma(alpha, beta)
        gamma_detached = t.distributions.Gamma(alpha.detach(), beta.detach())

        c = gamma.rsample(sample_shape=[*sample_shape, *Kshape[:-2], 1]).sqrt()
        A = n + I * c
        log_prob_gamma = gamma_detached.log_prob(c ** 2).sum((-1, -2))
        log_prob_A = log_prob_norm + log_prob_gamma
        return A, log_prob_A
    else:
        """
        Standard Bartlett
        """
        # ignores K, except shape, device etc.
        kwargs = {'device': K.device, 'dtype': K.dtype}
        Kshape = K.full().shape
        ncols = min(K.N, nu)
        n = t.tril(t.randn((*sample_shape, *Kshape[:-1], ncols), **kwargs), -1)
        normal = t.distributions.normal.Normal(loc=t.zeros((), **kwargs),
                                               scale=t.ones((), **kwargs))
        log_prob_norm = t.tril(normal.log_prob(n), -1).sum((-1, -2))
        I = t.eye(K.N, m=ncols, **kwargs)

        dof = nu - t.arange(ncols, **kwargs)
        gamma = t.distributions.Gamma(dof / 2., 1 / 2)

        c = gamma.rsample(sample_shape=[*sample_shape, *Kshape[:-2], 1]).sqrt()
        A = n + I*c
        log_prob_gamma = gamma.log_prob(c ** 2).sum((-1, -2))
        log_prob_A = log_prob_norm + log_prob_gamma
        return A, log_prob_A


# Implements singular and non-singular Wishart
class Wishart:
    def __init__(self, K, nu, alpha=None, beta=None, mu=None, sigma=None):
        self.p = K.N
        self.K = K
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.kwargs = {'device': K.device, 'dtype': K.dtype}

    def rsample_log_prob(self, sample_shape=t.Size([])):
        A, log_prob_A = bartlett(self.K, self.nu, sample_shape, alpha=self.alpha, beta=self.beta, mu=self.mu, sigma=self.sigma)
        max_K = t.max(self.K.full()).detach()
        K = PositiveDefiniteMatrix(self.K.full() + 1e-8*max_K*t.eye(self.p, **self.kwargs))
        LK = K.chol()
        LA = LK.full() @ A

        # Compute log prob for S = AAT
        ncols = min(self.K.N, self.nu)
        t_range = self.K.N - (t.arange(ncols, **self.kwargs) + 1)
        c = A.diagonal(dim1=-1, dim2=-2)
        log_prob_S = -(t_range*t.log(c)).sum(-1)
        log_prob = log_prob_A + log_prob_S

        # Compute log_prob contribution from K
        log_Ldiag = t.log(LK.diag())
        ncols = min(self.p, self.nu)
        t_range = self.p - t.arange(ncols, **self.kwargs)
        log_prob -= (t_range*log_Ldiag[..., :ncols]).sum(-1)

        p_range = t.minimum(t.arange(self.p, **self.kwargs) + 1, self.nu*t.ones((), **self.kwargs))
        log_prob -= (p_range*log_Ldiag).sum(-1)

        return LA @ LA.transpose(-1, -2), LA, log_prob

    # log prob for KL prior term
    def log_prob(self, x):
        # Cannot be computed if beta/sigma is not None, only works for non-generalized Wishart
        assert self.beta is None
        assert self.sigma is None
        nu = self.nu
        p = self.p
        if nu > p-1:
            # Evaluate non-singular Wishart log prob
            log_prob = 0.5 * (nu - p - 1) * x.logdet()
            log_prob -= (nu / 2) * self.K.logdet()
            log_prob -= 0.5 * self.K.inv(x).diagonal(dim1=-1, dim2=-2).sum(-1)
            log_prob -= (nu * p / 2) * math.log(2)
            log_prob -= t.mvlgamma(nu / 2 * t.ones(()), p)

            return log_prob
        else:
            # Evaluate singular Wishart log prob
            x_part = x[..., :nu, :nu]
            log_prob = -0.5 * nu * self.K.logdet()
            log_prob += 0.5*nu*(nu-p)*math.log(math.pi)
            log_prob -= 0.5*nu*p*math.log(2.)
            log_prob -= t.mvlgamma(0.5*nu*t.ones(()), nu)
            log_prob += 0.5*(nu-p-1)*t.logdet(x_part)
            log_prob -= 0.5*self.K.inv(x).diagonal(dim1=-1, dim2=-2).sum(-1)

            return log_prob


class WishartLayer(nn.Module):
    """
    Wishart layer from a deep kernel process.  Takes a KG as input, and returns KG as output.

    arg:
        - **inducing_batch (int):** number of inducing inputs
        - **nu (int):** layer width
    """

    def __init__(self, inducing_batch, nu):
        super().__init__()
        self.P = inducing_batch
        self.nu = nu
        ncols = min(self.P, self.nu)  # number of columns in the Bartlett decomposition

        self.V = nn.Parameter(t.randn(inducing_batch, inducing_batch))
        self.V_scale = nn.Parameter(-math.log(inducing_batch)*t.ones(()))
        self.sigmoid_prop = nn.Parameter(-4*t.ones(()))  # un-sigmoided p

        self.W = nn.Parameter(math.sqrt(inducing_batch)*t.eye(inducing_batch))

        self.log_beta = nn.Parameter(math.log(0.5)*t.ones(ncols))

        self.mu_unscaled = nn.Parameter(t.randn(inducing_batch, ncols))
        self.log_sigma = nn.Parameter(-3*t.ones(inducing_batch, ncols))  #-3
        self.log_alpha = nn.Parameter(t.log((nu - t.arange(ncols))/2))

    @property
    def prop(self):
        return self.sigmoid_prop.sigmoid()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def beta(self):
        return self.log_beta.exp()

    @property
    def mu(self):
        return self.mu_unscaled/math.sqrt(self.mu_unscaled.shape[-1])

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def prior_psi(self, dKii):
        return PositiveDefiniteMatrix(dKii.full())

    def post_psi(self, dKii):
        return PositiveDefiniteMatrix(((1-self.prop)*dKii.full() + self.prop*self.V_scale.exp()*self.V @ self.V.T))

    def PGii(self, dKii):
        return Wishart(self.prior_psi(dKii), self.nu)

    def QGii(self, dKii):
        return Wishart(self.post_psi(dKii), self.nu, alpha=self.alpha, beta=self.beta, mu=self.mu, sigma=self.sigma)

    def dd_kwargs(self):
        return {'device': self.V.device, 'dtype': self.V.dtype}

    def Gii(self, dKii):
        PGii = self.PGii(dKii)
        QGii = self.QGii(dKii)

        Gii, LGii, logQ = QGii.rsample_log_prob()
        logP = PGii.log_prob(Gii)

        self.logpq = logP - logQ

        return Gii, LGii

    def forward(self, K):

        dKii = PositiveDefiniteMatrix(K.ii/self.nu)
        dkit = K.it/self.nu
        dktt = K.tt/self.nu

        Pi = self.P

        Gii, LGii = self.Gii(dKii)
        (S, _, _) = dkit.shape
        assert self.logpq.shape == t.Size([S])

        inv_Kii_kit = dKii.inv(dkit)

        # Diagonal of covariance
        dktti = dktt - (dkit * inv_Kii_kit).sum(-2)

        if self.nu > Pi:
            LGii = t.cat([LGii, t.zeros((S, Pi, self.nu-Pi), **self.dd_kwargs())], -1)

        mean = dkit.transpose(-1, -2) @ dKii.inv(LGii)
        normal = t.distributions.normal.Normal(loc=mean, scale=t.sqrt(dktti).unsqueeze(-1))

        Ft = normal.rsample()
        git = LGii @ Ft.transpose(-1, -2)
        gtt = (Ft**2).sum(-1)

        return KG(Gii, git, gtt)
