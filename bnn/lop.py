# Helpful linear algebra-related operations
import math
import torch as t
import numpy as np


class Matrix:
    def __init__(self, device, dtype):
        self.device = t.device(device)
        self.dtype = getattr(t, dtype) if isinstance(dtype, str) else dtype
        self.kwargs = {'device': self.device, 'dtype': self.dtype}

    def diag(self):
        return self.full().diagonal(dim1=-1, dim2=-2)

    @property
    def one(self):
        return t.ones((), device=self.device)


class Identity(Matrix):
    def __init__(self, N, device="cpu", dtype=t.float32):
        super().__init__(device, dtype)

        self.N = N
        self.scale = t.ones((), **self.kwargs)

    def __call__(self, x):
        assert self.N == x.shape[-2]
        return x

    def inv(self, x):
        assert N == x.shape[-2]
        return x

    def chol(self):
        return self

    def inv_chol(self):
        return self

    def full(self):
        return t.eye(self.N, **self.kwargs)

    def t(self):
        return self

    def logdet(self):
        return 0.


class Scale(Matrix):
    def __init__(self, N, scale, device="cpu", dtype=t.float32):
        super().__init__(device, dtype)

        self.N = N
        self.scale = scale*t.ones((), **self.kwargs)

        if 1 <= len(self.scale.shape):
            assert 1==self.scale.shape[-1]
        if 2 <= len(self.scale.shape):
            assert 1==self.scale.shape[-2]

    def __call__(self, x):
        assert self.N == x.shape[-2]
        return x*self.scale

    def inv(self, x):
        assert N == x.shape[-2]
        return x/self.scale

    def chol(self):
        return Scale(self.N, t.sqrt(self.scale), **self.kwargs)

    def inv_chol(self):
        return Scale(self.N, t.rsqrt(self.scale), **self.kwargs)

    def full(self):
        return self.scale * t.eye(self.N, **self.kwargs)

    def t(self):
        return self

    def logdet(self):
        return self.N * t.log(self.scale.squeeze(-1).squeeze(-1))


class Inv(Matrix):
    def __init__(self, op):
        super().__init__(**op.kwargs)
        self.op = op
        self.N = op.N

    def __call__(self, x):
        return self.op.inv(x)

    def inv(self, x):
        return self.op(x)

#    def chol(self):
#        return self.op.inv_chol()
#
#    def inv_chol(self):
#        return self.op.chol()

    def sqrt(self):
        return self.op.inv_sqrt()

    def inv_sqrt(self):
        return self.op.sqrt()

    def full(self):
        return self.op.inv(t.eye(self.N, **self.kwargs))

    def t(self):
        return Inv(self.op.t())

    def logdet(self):
        return -self.op.logdet()


class FullMatrix(Matrix):
    def __init__(self, A):
        super().__init__(A.device, A.dtype)
        assert A.shape[-1] == A.shape[-2]
        self.A = A
        self.N = A.shape[-1]

    def __call__(self, x):
        assert self.N == x.shape[-2]
        return self.full() @ x

    def full(self):
        return self.A
    def t(self):
        return FullMatrix(self.A.t())


class TriangularMatrix(FullMatrix):
    def logdet(self):
        return self.diag().log().sum(-1)


class LowerTriangularMatrix(TriangularMatrix):
    def inv(self, x):
        return t.triangular_solve(x, self.A, upper=False, transpose=False)[0]

    def t(self):
        return UpperTriangularMatrix(self.A.transpose(-1, -2))


class UpperTriangularMatrix(TriangularMatrix):
    def inv(self, x):
        return t.triangular_solve(x, self.A, upper=True, transpose=False)[0]

    def t(self):
        return LowerTriangularMatrix(self.A.transpose(-1, -2))


class PositiveDefiniteMatrix(FullMatrix):
    def __init__(self, A=None, chol=None):
        assert (A is not None) or (chol is not None)

        if A is not None:
            Matrix.__init__(self, A.device, A.dtype)
            assert A.shape[-1] == A.shape[-2]
            self.A = A
            self.N = A.shape[-1]
            self.device = A.device
        if chol is not None:
            Matrix.__init__(self, chol.device, chol.dtype)
            assert isinstance(chol, LowerTriangularMatrix)
            self._chol = chol
            self.N = chol.N
            self.device = chol.device

    def full(self):
        if not hasattr(self, "A"):
            chol = self._chol.full()
            self.A = chol @ chol.transpose(-1, -2)
        return self.A

    def inv(self, x):
        return t.cholesky_solve(x, self.chol().full())

    def chol(self):
        if not hasattr(self, "_chol"):
            self._chol = LowerTriangularMatrix(t.cholesky(self.A))
        return self._chol

    #def inv_chol(self):
    #    return Inv(self.chol())

    def sqrt(self):
        return self.chol()

    def inv_sqrt(self):
        return Inv(self.chol().t())

    def t(self):
        return self

    def logdet(self):
        return 2*self.chol().logdet()


class Product(Matrix):
    def __init__(self, *ms):
        super().__init__(**ms[0].kwargs)
        self.ms     = ms
        self.N      = ms[0].N

        assert all(self.N      == m.N      for m in ms)
        assert all(self.device == m.device for m in ms)
        assert all(self.dtype  == m.dtype  for m in ms)

    def __call__(self, x):
        for m in self.ms[::-1]:
            x = m(x)
        return x

    def inv(self, x):
        for m in self.ms:
            x = m.inv(x)
        return x

    def t(self):
        return Product(*(m.t() for m in self.ms[::-1]))

    def full(self):
        return self(t.eye(self.N, **self.kwargs))

    def logdet(self):
        return sum(m.logdet() for m in self.ms)


def kron_prod(A, B):
    A = A[..., :,    None, :,    None]
    B = B[..., None, :,    None, :]
    AB = A*B
    return AB.view(*AB.shape[:-4], AB.shape[-4]*AB.shape[-3], AB.shape[-2]*AB.shape[-1])


class KFac(Matrix):
    def __init__(self, A, B):
        assert A.device == B.device
        assert A.dtype  == B.dtype 
        super().__init__(**A.kwargs)

        self.A      = A
        self.B      = B
        self.N = self.A.N * self.B.N

    def __call__(self, x):
        assert self.N == x.shape[-2]
        return self.full()@x

    def full(self):
        if not hasattr(self, "_full"):
            self._full = kron_prod(self.A.full(), self.B.full())
        return self._full

    def logdet(self):
        return (self.B.N * self.A.logdet()) + (self.A.N * self.B.logdet())


def trace_quad(A, X):
    return (X * A(X)).sum((-1, -2))


def mvnormal_log_prob(prec, X):
    in_features = X.shape[-2]
    out_features = X.shape[-1]
    return -0.5*trace_quad(prec, X) - 0.5*out_features*(-prec.logdet() + in_features*(math.log(2*math.pi)))


def mvnormal_log_prob_unnorm(prec, X):
    in_features = X.shape[-2]
    out_features = X.shape[-1]
    return -0.5*trace_quad(prec, X) + 0.5*out_features*prec.logdet()


if __name__ == "__main__":
    print("running tests")
    N = 4
    batch = 1

    _dtype = t.float64

    tmp1 = t.randn(N, N, dtype=_dtype)
    tmp1 = tmp1@tmp1.t()
    tmp2 = t.randn(N, N, dtype=_dtype)
    tmp2 = tmp2@tmp2.t()

    #### Test KFac
    Kpt = KFac(PositiveDefiniteMatrix(tmp1), PositiveDefiniteMatrix(tmp2))
    Knp = t.tensor(np.kron(tmp1.numpy(), tmp2.numpy()))
    assert t.allclose(Kpt.full(), Knp)
    assert t.allclose(Kpt.logdet(), t.logdet(Kpt.full()))

    #### Test other
    m1 = Scale(N, 0.5, dtype=_dtype)
    m2 = PositiveDefiniteMatrix(tmp1)
    m3 = m2.chol()
    m4 = m3.t()
    m5 = Product(m1,m2,m3)

    def tests(W):
        #### Test matrix-vector operations
        x = t.randn(batch, N, 1, dtype=_dtype)

        # Test mm
        assert t.allclose(W.full() @ x, W(x))

        # Test inverse
        assert t.allclose(x, W.inv(W(x)))

        #### Test logdet
        assert t.allclose(W.logdet(), W.full().logdet())

        #### Test transpose
        assert t.allclose(W.full().transpose(-1, -2), W.t().full())

    tests(m1)
    tests(m2)
    tests(m3)
    tests(m4)
    tests(m5)
