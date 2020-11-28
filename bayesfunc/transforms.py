import torch as t
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

    def forward(self, x):
        return x

class ConvTransform(nn.Module):
#    Alternative, more memory efficient implementation that we can't use because of a pytorch bug
#    def forward(self, x):
#        shape = x.shape
#        x = x.reshape(x.shape[:-2].numel(), 1, x.shape[-2], x.shape[-1]) #should usually avoid reallocating
#        return t.conv2d(x, self.weights(x), padding=self.kernel_size//2).view(shape)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(x.shape[:-3].numel(), *x.shape[-3:]) #should usually avoid reallocating
        w = self.weights(x)*t.eye(shape[-3], device=x.device)[:, :, None, None]
        return t.conv2d(x, w, padding=self.kernel_size//2).view(*shape)


class Blur(ConvTransform):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.log_lengthscale = nn.Parameter(-t.ones(()))

    def weights(self, x):
        device = self.log_lengthscale.device
        lengthscale = self.log_lengthscale.exp()

        xs = t.arange(self.kernel_size, device=device)[:, None] - self.kernel_size//2
        ys = t.arange(self.kernel_size, device=device)[None, :] - self.kernel_size//2

        d2 = (xs**2 + ys**2)**2

        unscaled = t.exp(-d2/(2*lengthscale))

        return unscaled[None, None, :, :] / unscaled.sum()


class DoG(ConvTransform):
    def __init__(self, kernel_size, components=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.log_lengthscale = nn.Parameter(-t.arange(components, dtype=t.float32)[:, None, None])
        self.height          = nn.Parameter(   t.ones(components, 1, 1))

    def weights(self, x):
        device = self.log_lengthscale.device
        lengthscale = self.log_lengthscale.exp()

        xs = t.arange(self.kernel_size, device=device)[:, None] - self.kernel_size//2
        ys = t.arange(self.kernel_size, device=device)[None, :] - self.kernel_size//2

        d2 = (xs**2 + ys**2)**2

        unscaled = (self.height*t.exp(-d2/(2*lengthscale))).sum(0)

        return unscaled[None, None, :, :] / unscaled.sum()
