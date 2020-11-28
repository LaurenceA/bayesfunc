import torch as t
import torch.nn.functional as F

def bconv2d(x, weight, stride, padding, dilation=1):
    assert 5 == len(x.shape)
    assert 5 == len(weight.shape)
    assert x.shape[0] == weight.shape[0]
    S = x.shape[0]

    _, B, in_channels, h, w = x.shape
    _, out_channels, _in_channels, kh, kw = weight.shape
    assert in_channels == _in_channels

    weight = weight.reshape(S * out_channels, in_channels, kh, kw)

    out = x.transpose(0, 1)
    # Trick: use pad to copy data into contiguous format.
    # Note that out is 5D, so we do a 3D padding, where we pad -3 with zeros
    out = out.reshape(B, S * in_channels, *out.shape[-2:])

    out = F.conv2d(out, weight=weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=S)
    out = out.view(B, S, out_channels, out.shape[-2], out.shape[-1])
    out = out.transpose(0, 1)

    return out
