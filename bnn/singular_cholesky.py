import torch as t


def singular_cholesky(X):
    epsilon=1E-6
    I = t.eye(X.shape[-1], device=X.device, dtype=X.dtype)
    LdL = t.cholesky(X + epsilon*I)
    dL = t.triangular_solve(I, LdL)[0].transpose(-1, -2)
    t.allclose(I, LdL@dL.transpose(-1, -2))

    L = LdL - (epsilon/2)*dL

    return L
