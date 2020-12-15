import torch as t
import torch.nn.functional as F


def conv_1x1(x, y):
    """
    optimized approach for 1x1 kernels
    """
    (S, B, C, H, W) = x.shape
    assert x.shape[-1] == y.shape[-1]
    assert x.shape[-2] == y.shape[-2]
    assert x.shape[-4] == y.shape[-4]

    #[..., B, C, H, W] -> [..., B, C, HW]
    x = x.view(*x.shape[:-2], H*W)
    y = y.view(*y.shape[:-2], H*W)

    #[..., B, C, C']
    XTX = (x @ x.transpose(-1, -2)).sum(-3)
    XTY = (x @ y.transpose(-1, -2)).sum(-3)

    return XTX, XTY


def conv_mm(x, y, kernel_size): #, stride=1, mode='circular'):
    stride = 1
    mode = 'circular'
    """
        compute
        (x**2).shape = [C, C', dW, dH]
        
        standard convolution
        input.shape = [minibatch, in_channels, iW, iH]
        weight.shape = [out_channels, in_channels/groups, iW, iH]
        output.shape = [minibatch, out_channels, oW, oH]
        
        Approach:
        sum over B, so set:
        in_channels = B
        keep minibatch and out_channels, so set,
        minibatch = C
        out_channels = C'
        use padding = 2 such that,
        oW = dW
        oH = dH
        

        input.shape =  [Cx,    S*B, Wx, Hx]
        weight.shape = [S*Cy,    B, Wy, Hy]
        output.shape = [Cx,   S*Cy, oW, oH]
    """
    #### General
    assert x.shape[-4] == y.shape[-4]
    (S, B, Cx, _, _) = x.shape
    (B, Cy, _, _) = y.shape

    if kernel_size==1:
        return conv_1x1(x, y)

    y = y.expand(S, *y.shape)

    (S, B, Cx, Wx, Hx) = x.shape
    (_, _, Cy, Wy, Hy) = y.shape

    input = x.reshape(S*B, Cx, Wx, Hx)
    input = input.transpose(0, 1)
    assert input.shape == t.Size([Cx, S*B, Wx, Hx])

    weight_y = y.transpose(1, 2)
    assert weight_y.shape == t.Size([S, Cy, B, Wy, Hy])
    weight_y = weight_y.reshape(S*Cy, B, Wy, Hy)

    weight_x = x.transpose(1, 2)
    assert weight_x.shape == t.Size([S, Cx, B, Wx, Hx])
    weight_x = weight_x.reshape(S*Cx, B, Wx, Hx)

    if mode=='circular':
        input_XTX = F.pad(input, 4*[kernel_size-1], mode=mode)
        cut = (kernel_size-1) - kernel_size//2
        is_even = 0==kernel_size%2
        input_XTY = input_XTX[:, :, cut:(input_XTX.shape[-2]-cut-is_even), cut:(input_XTX.shape[-1]-cut-is_even)]
        XTY = F.conv2d(input_XTY, weight_y, padding=0, groups=S)
        XTX = F.conv2d(input_XTX, weight_x, padding=0, groups=S)
    else:
        XTY = F.conv2d(input, weight_y, padding=kernel_size//2, groups=S)
        XTX = F.conv2d(input, weight_x, padding=kernel_size-1, groups=S)



    #### XTY specific
    assert XTY.shape == t.Size([Cx, S*Cy, kernel_size, kernel_size])
    XTY = XTY.transpose(0, 1).reshape(S, Cy, Cx*kernel_size*kernel_size)
    XTY = XTY.transpose(-1, -2)



    #### XTX specific
    assert XTX.shape == t.Size([Cx, S*Cx, kernel_size*2-1, kernel_size*2-1])
    XTX = XTX.transpose(0, 1).view(S, Cx, Cx, kernel_size*2-1, kernel_size*2-1).transpose(1, 2)

    xs = t.arange(kernel_size)[:, None]
    ys = t.arange(kernel_size)[None, :]
    pad=kernel_size-1
    diff = (xs - ys) + pad

    XTX = XTX[..., diff]
    XTX = XTX[..., diff, :, :]

    XTX = XTX.transpose(-3, -2)
    XTX = XTX.permute([0, 1, 3, 4, 2, 5, 6])
    assert XTX.shape == t.Size([S, Cx, kernel_size, kernel_size, Cx, kernel_size, kernel_size])
    XTX = XTX.reshape(S, Cx*kernel_size**2, Cx*kernel_size**2)

    return XTX, XTY


def extract_patches_conv(x, kernel_size, stride, padding, mode='zeros'):
    in_channels = x.shape[1]
    pixels = kernel_size**2
    W = t.eye(pixels, dtype=x.dtype).expand(in_channels, -1, -1)
    W = W.reshape(in_channels*pixels, 1, kernel_size, kernel_size)

    if 0 == kernel_size % 2:
        assert padding*2 == kernel_size
        pad = padding//2
        x = F.pad(x, (pad+1,pad,pad+1,pad), mode=mode)
    else:
        x = F.pad(x, 4*(padding,), mode=mode)
    x = F.conv2d(x, W, stride=stride, groups=in_channels)
    x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
    x = x.transpose(1,2)
    x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
    return x


def batch_extract_patches_conv(x, kernel_size, stride, padding, mode='zeros'):
    sample_shape = x.shape[:-4]
    x = x.view(x.shape[:-3].numel(), *x.shape[-3:])
    x = extract_patches_conv(x, kernel_size, stride, padding, mode=mode)
    return x.view(*sample_shape, -1, x.shape[-1])


def extract_patches_unfold(x, kernel_size, stride, padding, mode='zeros'):
    (B, C, H, W) = x.shape
    x = F.pad(x, 4*(padding,), mode=mode)

    # Extract patches
    patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.permute(0,2,3,1,4,5).contiguous()
    assert patches.shape == t.Size([B,H,W,C,kernel_size,kernel_size])

    return patches.view(B*H*W, C*kernel_size**2)


if __name__ == "__main__":
    N = 28
    stride = 1
    #[B, C, W, H]
    x = t.randn(2, 5, 128, N, N, dtype=t.float64)
    #[B, C', W, H]
    y = t.randn(5, 256, N//stride, N//stride, dtype=t.float64)

    kernel_size = 5
    padding = kernel_size//2

    mode = 'circular'
    X = batch_extract_patches_conv(x, kernel_size, stride, padding, mode=mode)
    Y = batch_extract_patches_conv(y, 1, 1, 0, mode=mode)

    XTX = X.transpose(-1, -2) @ X
    XTY = X.transpose(-1, -2) @ Y

    XTX2, XTY2 = conv_mm(x, y, kernel_size, stride, mode)

    print(XTX[:3, :3], XTX2[:3, :3])

    assert t.allclose(XTY, XTY2)
    assert t.allclose(XTX, XTX2)
