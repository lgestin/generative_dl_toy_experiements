import torch
import torch.nn as nn
import torch.nn.functional as F

from normalizing_flows.flow import Flow, RandPerm


def init(m):
    if hasattr(m, "bias"):
        nn.init.constant_(m.bias, 0.0)


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        mask = torch.ones(out_features, in_features).tril(-1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        weight = self.weight.masked_fill(~self.mask, 0)
        return F.linear(x, weight, bias=self.bias)


class IAFLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            MaskedLinear(dim, dim),
            nn.ReLU(inplace=True),
        )
        self.mu = MaskedLinear(dim, dim)
        self.log_std = MaskedLinear(dim, dim)

    def compute_mu_logs(self, x):
        x = self.layer(x)
        mu = self.mu(x)
        logs = self.log_std(x)
        # logs = 2.3 * torch.tanh(logs)
        return mu, logs

    def forward(self, x):
        z = torch.zeros_like(x)
        ldj = 0
        for i in range(x.size(1)):
            mu, logs = self.compute_mu_logs(z.clone())
            z[:, i] = mu[:, i] + x[:, i] * logs[:, i].exp()
            ldj += logs[:, i]
        return z, ldj

    def inverse(self, z):
        mu, logs = self.compute_mu_logs(z)
        x = (z - mu) * torch.exp(-logs)
        ldj = -logs.sum(dim=1)
        return x, ldj


class IAF(Flow):
    def __init__(self, dim, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [RandPerm(dim), IAFLayer(dim)]
        super().__init__(flows=layers)
        self.apply(init)
