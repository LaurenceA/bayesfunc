import torch.nn as nn
import bayesfunc


def block(conv2d, C, kwargs):
    return bayesfunc.Sum(
            nn.Sequential(
            ),
            nn.Sequential(
                conv2d(C, C, 3, padding=1, **kwargs),
                nn.ReLU(),

                conv2d(C, C, 3, padding=1, **kwargs),
                nn.ReLU(),
            )
        )


def net(linear, conv2d, in_shape, out_classes, kwargs={}, out_kwargs={}, channels=32):
    C = channels
    in_channels = in_shape[-3]

    net= nn.Sequential(
        conv2d(in_channels, C, 3, padding=1, **kwargs),
        nn.ReLU(),
        block(conv2d, C, kwargs),
         
        bayesfunc.AvgPool2d((2, 2)),
        block(conv2d, C, kwargs),
        
        bayesfunc.AvgPool2d((2, 2)),
        block(conv2d, C, kwargs),

        bayesfunc.AdaptiveAvgPool2d((1, 1)),
        bayesfunc.Conv2d_2_FC(),
        linear(C, out_classes, bias=True, **out_kwargs)
    )
    return net

