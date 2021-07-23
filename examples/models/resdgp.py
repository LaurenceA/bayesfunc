# "Residual" deep GP with SE kernel as used in Salimbeni and Deisenroth (2017)
import torch.nn as nn
import bayesfunc as bf


def dgp_layer(inducing_batch, width, trainable_noise=True, ARD=False, neuron_prec=False, ap='gigp'):
    if ap == 'gigp':
        if ARD:
            return nn.Sequential(bf.SqExpKernelFeaturesARD(inducing_batch=inducing_batch, in_features=width, trainable_noise=trainable_noise),
                             bf.GIGP(out_features=width, inducing_batch=inducing_batch, neuron_prec=neuron_prec))
        else:
            return nn.Sequential(bf.SqExpKernelFeatures(inducing_batch=inducing_batch, trainable_noise=trainable_noise),
                                 bf.GIGP(out_features=width, inducing_batch=inducing_batch, neuron_prec=neuron_prec))
    elif ap == 'ligp':
        if ARD:
            return nn.Sequential(bf.InducingAdd(inducing_batch, inducing_shape=(inducing_batch, width)),
                bf.SqExpKernelFeaturesARD(inducing_batch=inducing_batch, in_features=width, trainable_noise=trainable_noise),
                bf.GIGP(out_features=width, inducing_batch=inducing_batch, neuron_prec=neuron_prec),
                bf.InducingRemove(inducing_batch))
        else:
            return nn.Sequential(bf.InducingAdd(inducing_batch, inducing_shape=(inducing_batch, width)),
                bf.SqExpKernelFeatures(inducing_batch=inducing_batch, trainable_noise=trainable_noise),
                bf.GIGP(out_features=width, inducing_batch=inducing_batch, neuron_prec=neuron_prec),
                bf.InducingRemove(inducing_batch))


class resdgp(nn.Module):
    def __init__(self, inducing_batch, inducing_targets, width, depth, in_features, ARD=False, neuron_prec=False, ap='gigp'):
        super().__init__()
        # We always use ARD in the input
        self.input_layer = nn.Sequential(
            bf.SqExpKernelFeaturesARD(inducing_batch=inducing_batch, in_features=in_features, trainable_noise=True),
            bf.GIGP(out_features=width, inducing_batch=inducing_batch, neuron_prec=neuron_prec)
        )
        if ARD:
            self.last_layer = nn.Sequential(
                bf.SqExpKernelFeaturesARD(inducing_batch=inducing_batch, in_features=width, trainable_noise=False),
                bf.GIGP(out_features=1, inducing_targets=inducing_targets, inducing_batch=inducing_batch, neuron_prec=neuron_prec)
            )
        else:
            self.last_layer = nn.Sequential(
                bf.SqExpKernelFeatures(inducing_batch=inducing_batch, trainable_noise=False),
                bf.GIGP(out_features=1, inducing_targets=inducing_targets, inducing_batch=inducing_batch, neuron_prec=neuron_prec)
            )
        if ap == 'ligp':
            self.input_layer = nn.Sequential(
                bf.InducingAdd(inducing_batch, inducing_shape=(inducing_batch, in_features)),
                self.input_layer,
                bf.InducingRemove(inducing_batch)
            )
            self.last_layer = nn.Sequential(
                bf.InducingAdd(inducing_batch, inducing_shape=(inducing_batch, width)),
                self.last_layer,
                bf.InducingRemove(inducing_batch)
            )
        self.mid_layers = nn.ModuleList(
            [dgp_layer(inducing_batch, width, ARD=ARD, neuron_prec=neuron_prec, ap=ap) for _ in range(depth - 2)])

    def forward(self, x):
        out = self.input_layer(x) + x
        for i in range(len(self.mid_layers)):
            out = self.mid_layers[i](out) + out
        return self.last_layer(out)


def resdgp_model(inducing_batch, inducing_data, inducing_targets, width, depth, in_features, ARD=False, neuron_prec=False,
               ap='gigp'):
    net = resdgp(inducing_batch, inducing_targets, width, depth, in_features, ARD=ARD, neuron_prec=neuron_prec, ap=ap)

    if ap == 'gigp':  # Add inducing points for GI
        net = bf.InducingWrapper(net, inducing_batch=inducing_batch, inducing_data=inducing_data)
    return nn.Sequential(net, bf.NormalLearnedScale())
