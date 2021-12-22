.. bayesfunc documentation master file, created by
   sphinx-quickstart on Sat Nov 28 10:45:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bayesfunc's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

bayesfunc is a PyTorch library providing a number of state-of-the-art priors and variational approximate posteriors over functions.

In particular, we implement:

 - Global inducing variational inference for neural networks (https://arxiv.org/abs/2005.08140)
 - Global inducing variational inference for deep Gaussian processes (https://arxiv.org/abs/2005.08140)
 - Deep kernel processes (https://arxiv.org/abs/2010.01590)
 - Deep Wishart processes (https://arxiv.org/abs/2107.10125)

In addition, we implement a number of more standard methods, primarily to give fair, easy-to-implement comparisons:

 - Mean field variational inference
 - Sparse (deep) Gaussian processes inference 

.. _conventions:

Library Conventions
===================

bayesfunc introduces a number of PyTorch modules mirroring the standard pytorch ``nn`` API.
As such, modules can be combined and networks created using e.g. ``nn.Sequential``.  See Examples for further details.
However these modules have a couple of differences.


Sample and minibatch
--------------------

In standard PyTorch, inputs to modules are tensors, and the zeroth index usually represents a minibatch.
For instance, if we had a minibatch of size 128,

    >>> import torch
    >>> import torch.nn as nn
    >>> m = nn.Linear(20, 30)
    >>> input = torch.randn(128, 20)
    >>> output = m(input)
    >>> print(output.size())
    torch.Size([128, 30])

However, here we are sampling from an approximate posterior over functions, and often we will want to draw mutiple different samples from that approximate posterior (e.g. to reduce the variance of our estimate of the ELBO).
As such, we introduce a new convention: the zeroth index gives the sample of the approximate posterior, and the first index gives the minibatch.
Consider possibly the simplest module defined in our module, `FactorisedLinear`, which does MFVI over the weights of a fully-connected layer.
For instance, if we wanted to do the equivalent of the above, with just one sample from the posterior over functions that we apply to every input in the minibatch,

    >>> import bayesfunc as bf
    >>> m = bf.FactorisedLinear(20, 30)
    >>> input = torch.randn(1, 128, 20)
    >>> output, _, _ = bf.propagate(m, input)
    >>> print(output.size())
    torch.Size([1, 128, 30])

Note that both the input and output are rank 3 tensors, whereas in the original pure PyTorch example, they were only rank 2 tensors.

If we wanted to draw 10 samples of the function and apply them to our inputs, we would need to give the module a tensor with shape ``(10, minibatch, in_features)``.
To efficiently replicate the inputs 10 times, we use ``expand`` 

    >>> m = bf.FactorisedLinear(20, 30)
    >>> input = torch.randn(1, 128, 20).expand(10, -1, -1)
    >>> output, _, _ = bf.propagate(m, input)
    >>> print(output.size())
    torch.Size([10, 128, 30])

Note despite the inputs being the same for different samples, the outputs are different, because we have applied different functions.

    >>> (input[0, 0, 0] == input[1, 0, 0]).item()
    True
    >>> (output[0, 0, 0] == output[1, 0, 0]).item()
    False

Propagate function
------------------

To run a network, you *must* use the `propagate` function.
This function prepares the network (by optionally loading up a previously sampled set of weights), runs the network, returning the output, :math:`\log P(f) - \log Q(f)`, and the sampled weights used,

.. autofunction:: bayesfunc.propagate

Computing the ELBO
------------------

In variational inference, we use the ELBO as the loss,

:math:`\mathcal{L}(\phi) = E_{Q_\phi}[\sum_{i=1}^N \log P(y_i|x_i, f) + \log P(f) - \log Q_\phi(f)]`

where :math:`x_i` is a single input (e.g. image), :math:`y_i` is a single output (e.g. label) and there are :math:`N` datapoints in total.
The bayesfunc library defines priors, :math:`P(f)`, and approximate posteriors, :math:`Q_\phi(f)`, over functions, where :math:`\phi` are the trainable parameters of the approximate posterior.

And in practical cases, we use a minibatched estimate, of the averaged loss (averaging across datapoints),

:math:`\frac{1}{N} \hat{\mathcal{L}}(\phi) = \frac{1}{B} \sum_{i\in \mathcal{B}_j}\log P(\text{data}_i| f) + \frac{1}{N} (\log P(f) - \log Q_\phi(f))`

Here, :math:`\mathbf{B}_i` is the :math:`i`th minibatch of data, :math:`B` is the size of a minibatch, and :math:`f` has been sampled from the approximate posterior, :math:`Q_\phi(f)`.
Critically, the first term, is just the standard neural-network loss (e.g. the cross-entropy),

:math:`\frac{1}{B} \sum_{i\in \mathcal{B}_j}\log P(\text{data}_i| f)` = - average cross entropy

And the second term is given by the bayesfunc library, as the second argument returned by `bf.propagate`

:math:`\log P(f) - \log Q_\phi(f)` = ``bf.propagate(net, inputt)[1]``

As such, the full training loop might look like::

    for x, y in dataloader:
        # include a sample dimension
        x = x.expand(1, -1, -1)
        # compute the output
        output, logpq, _ = bf.propagate(net, x)
        # compute the log-likelihood/loss
        log_like = F.cross_entropy(x, y, reduction="mean")
        # the objective, where N is the number of datapoints
        obj = log_like + logpq/N
        optimizer.zero_grad()
        (-obj).backward()
        optimizer.step()

.. _GI:

Wrapper for global inducing methods
------------------------------------

Many of our function approximators require "global" inducing points, i.e. optimized psuedo inputs that look like standard data-items.
These modules (i.e. ``GILinear``, ``GIConv2d``, ``GIGP``, ``DKP``) require an ``InducingWrapper``,

.. autofunction:: bayesfunc.InducingWrapper

Structured kernels for kernel-based methods
--------------------------------------------

To implement kernel-based methods efficiently, we can't propagate the full :math:`(P_\text{i}+P_\text{t})\times(P_\text{i}+P_\text{t})` covariance matrix, where :math:`P_\text{i}` is the number of inducing points, and :math:`P_\text{t}` is the number of test/training points, as :math:`P_\text{t}` could be very large.
Instead, we propagate a special type:

.. autoclass:: bayesfunc.KG




Library reference: Bayesian neural networks
===========================================

Simple approximate posteriors for Bayesian neural networks
--------------------------------------------------------------

These methods compute an approximate posterior over weights and are relatively simple: they don't have global inducing points, and therefore don't need wrapping in ``InducingWrapper`` (:ref:`GI`).  That said, you can wrap them if you want, which is usually useful if you want to combine some of these simpler methods with a Global inducing method.

First, we look at factorised methods. They are easy to apply, but often don't work that well.  It can be important to initialise the approximate posterior with very low variance to get them to converge.

.. autoclass:: bayesfunc.FactorisedLinear
.. autoclass:: bayesfunc.FactorisedConv2d

Next, we look at "Local" inducing point methods.  These haven't really been used in neural networks, because the performance doesn't justify the additional computational cost.

.. autoclass:: bayesfunc.LILinear
.. autoclass:: bayesfunc.LIConv2d

Global inducing approximate posteriors for Bayesian neural networks
--------------------------------------------------------------------

These methods were developed in https://arxiv.org/abs/2005.08140 and give state-of-the-art performance in tasks such as image classification.  They require wrapping in ``InducingWrapper``.

.. autoclass:: bayesfunc.GILinear
.. autoclass:: bayesfunc.GIConv2d


Library reference: deep Gaussian processes
--------------------------------------------

For deep GPs, the fundamental class is the ``GIGP``, which implements global inducing methods.  Everything else (including local-inducing methods) are implemented in terms of ``GIGP``

.. autoclass:: bayesfunc.GIGP

For testing 

.. autofunction:: bayesfunc.KernelGIGP
.. autofunction:: bayesfunc.KernelLIGP

.. autoclass:: bayesfunc.SqExpKernel
.. autoclass:: bayesfunc.SqExpKernelGram
.. autoclass:: bayesfunc.ReluKernelGram


Library reference: deep kernel processes
-----------------------------------------

.. autoclass:: bayesfunc.IWLayer
.. autoclass:: bayesfunc.SingularIWLayer


Library reference: deep Wishart processes
------------------------------------------

.. autoclass:: bayesfunc.WishartLayer
