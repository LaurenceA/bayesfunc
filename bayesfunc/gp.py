from .general import InducingAdd, InducingRemove
from .kernels_minimal import SqExpKernel as DefaultKernel
import torch as t
import torch.nn as nn
from torch.distributions import Normal

# posterior precision (L + K^{-1})
# difference of prior and posterior log-prob depends only on u
#  u^T (L+K^{-1}) u - u^T K^{-1} u = u^T L u
# need cholesky of K^{-1} to compute E[f|u]
# need cholesky of (L+K^{-1}) to sample u

from .lop import PositiveDefiniteMatrix, Inv, mvnormal_log_prob

class GIGP(nn.Module):
    """
    Global inducing point Gaussian process.  Takes KG as input and returns features.

    arg:
        - **out_features (int):**  Number of features to output.

    compulsory kwargs:
        - **inducing_batch (int):** Number of inducing points.

    optional kwargs:
        - **inducing_targets:** Initial setting of the inducing targets.  Oly
        - **log_prec_init:** Initial value of the precision. Default to little evidence: ``-4``.
        - **log_prec_lr:** Precision learning rate multiplier. Default: ``1.``.

    """
    def __init__(self, out_features, inducing_targets=None, log_prec_init=-4., log_prec_lr=1., inducing_batch=None):
        super().__init__()

        assert inducing_batch is not None
        self.inducing_batch = inducing_batch
        self.out_features = out_features

        self.L_scale = nn.Parameter(0.5*log_prec_init*t.ones(()))
        self.L_loc   = nn.Parameter(t.eye(self.inducing_batch))
        if inducing_targets is None:
            self.u = nn.Parameter(t.randn(self.inducing_batch, self.out_features))
        else:
            self.u = nn.Parameter(inducing_targets.clone().to(t.float32))

        self._sample = None

    @property
    def L(self):
        norm = self.L_loc.diag().mean()
        return self.L_loc * self.L_scale.exp()/norm

    def forward(self, K):
        # P    = L LT
        #    u = N(0, K)
        # v| u = N(u, P^{-1})
        # u| v = N(S P u, S)
        # S = (K^{-1} + P)^{-1} = (K^{-1} + L LT)^{-1} = K - K L (LT K L + I)^{-1} LT K

        # lu = LTu
        # lv = LTv
        # LTv| u = N(LTu, I)
        # LTv = N(0, LT K L + I)

        # P(u)/P(u|lv) = P(lv)/P(lv| u)

        # t.cholesky(Kuu) only used for f|u and sampling: singular cholesky would work.
        #(S, P, _) = K.shape
        #kwargs = {'device': K.device, 'dtype': K.dtype}

        
        #Kuu = K[:, :self.inducing_batch, :self.inducing_batch]
        #Kfu = K[..., self.inducing_batch:, :self.inducing_batch]
        #Kuf = Kfu.transpose(-1, -2)
        #Kff = K[..., self.inducing_batch:, self.inducing_batch:]

        Kuu = K.ii
        Kuf = K.it
        Kfu = K.it.transpose(-1, -2)
        Kff = K.tt

        (S, _, _) = Kuu.shape
        kwargs = {'device': Kuu.device, 'dtype': Kuu.dtype}

        pd_Kuu = PositiveDefiniteMatrix(Kuu)
        Iuu = t.eye(self.inducing_batch, **kwargs)

        L = self.L
        LT = self.L.transpose(-1, -2)

        KuuL = Kuu@self.L

        lKlpI = PositiveDefiniteMatrix(LT @ KuuL + Iuu)
        inv_lKlpI = Inv(lKlpI)
        Sigma = Kuu - KuuL @ inv_lKlpI(KuuL.transpose(-1, -2))

        #Sample noise distributed as precision, then multiply by Sigma.
        #inv_Kuu_noise = pd_Kuu.inv_chol().t()(t.randn(S, self.inducing_batch, self.out_features, **kwargs))
        inv_Kuu_noise = pd_Kuu.inv_sqrt()(t.randn(S, self.inducing_batch, self.out_features, **kwargs))
        L_noise = self.L@t.randn(S, self.inducing_batch, self.out_features, **kwargs)
        prec_noise = inv_Kuu_noise + L_noise
        u = Sigma@((L@LT)@self.u + prec_noise)

        lv = LT@self.u
        logP = mvnormal_log_prob(inv_lKlpI, lv)
        logQ = Normal(LT@u, 1.).log_prob(lv).sum((-1, -2))

        #### f|u

        Kfu_invKuu = pd_Kuu.inv(Kuf).transpose(-1, -2)
        Ef = Kfu_invKuu @ u
        #Vf = (Kff - Kfu_invKuu @ Kuf).diagonal(dim1=-1, dim2=-2)
        Vf = Kff - (Kfu_invKuu * Kuf.transpose(-1, -2)).sum(-1)

        Pf = Normal(Ef, Vf.sqrt()[..., None])
        f = Pf.rsample()

        self.logpq = logP-logQ

        return t.cat([u, Pf.rsample()], -2)


def KernelGIGP(in_features, out_features, inducing_batch=None, **kwargs):
    gp = GIGP(out_features, inducing_batch=inducing_batch, **kwargs)
    kernel = DefaultKernel(in_features, inducing_batch=inducing_batch)
    return nn.Sequential(kernel, gp)

def KernelLIGP(in_features, out_features, inducing_batch=None, kernel=None, **kwargs):
    assert inducing_batch is not None
    inducing_shape = (inducing_batch, in_features)
    ia = InducingAdd(inducing_batch, inducing_shape=inducing_shape)
    if kernel is None:
        kernel = DefaultKernel(in_features, inducing_batch=inducing_batch)
    gp = GIGP(out_features, inducing_batch=inducing_batch, **kwargs)
    ir = InducingRemove(inducing_batch)
    return nn.Sequential(ia, kernel, gp, ir)
