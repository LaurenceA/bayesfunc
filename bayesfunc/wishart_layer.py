import math
import torch as t
import torch.nn as nn
from .wishart_dist import InverseWishart


class IWLinear(nn.Module):
    def __init__(self, nu, post_data_features=None):
        super().__init__()
        self.prior_log_nu = math.log(nu)
        self.post_data_features = post_data_features
        self.post_log_nu = nn.Parameter(t.ones(()) * self.prior_log_nu)

    def inducing_init(self, init):
        assert init.shape[-1] == init.shape[-2]
        P = init.shape[-1]

        if self.post_data_features is None:
            self.post_data_features = P+1

        self.y = t.randn(P, self.post_data_features)

        #if weight == 1, then behaves as standard 
        self.log_weight = t.zeros(self.post_data_features)

        #Just need to return a kernel with the same shape
        return init

    def forward(self, K):
        prior_nu = self.prior_log_nu.exp()
        prior_Psi = K * prior_nu
        prior = InverseWishart(prior_Psi, prior_nu)

        weight = self.log_weight.exp()

        post_nu = prior_nu + weight.sum()
        
        A = (self.post_data_features * weight) @ self.post_data_features.transpose(-1, -2)

        post_Psi = prior_Psi + A

        post = InverseWishart(post_Psi, post_nu)
        Kp = post.rsample()

        logPQ = post.log_prob(Kp) - prior.log_prob(Kp)

        return K, logPQ
