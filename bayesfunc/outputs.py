import torch.nn as nn
from torch.distributions import Normal, Categorical


class Output(nn.Module):
    def inducing_init(self, inducing_batch):
        self.inducing_batch = inducing_batch

    def cut(self, x):
        if hasattr(self, "inducing_batch"):
            x = x[...,self.inducing_batch:, :]
        return x


class CutOutput(Output):
    def forward(self, x):
        return self.cut(x)


class CategoricalOutput(Output):
    def forward(self, x):
        return Categorical(logits=self.cut(x))


class NormalOutput(Output):
    def __init__(self, x):
        super().__init__()
        log_std = nn.Parameter(-2.*t.ones(()))

    def forward(self, x):
        return Normal(self.cut(x), log_std.exp())


