import torch
import torch.nn as nn

from normalizing_flows.flow import MaskedLinear, Flow, RandPerm


class MAFLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mu = nn.Sequential(
            MaskedLinear(d_model, d_model),
            nn.ReLU(inplace=True),
            MaskedLinear(d_model, d_model),
        )
        self.log_std = nn.Sequential(
            MaskedLinear(d_model, d_model),
            nn.ReLU(inplace=True),
            MaskedLinear(d_model, d_model),
        )

    def forward(self, x):
        mu, log_std = self.mu(x), self.log_std(x).clamp(min=-30)
        z = (x - mu) * (0.1 * -log_std).exp()
        log_det = -log_std.sum(dim=1)
        return z, 0.1 * log_det

    def inverse(self, z):
        x = torch.zeros_like(z)
        log_det = 0
        for i in range(z.size(-1)):
            mu, log_std = self.mu(x), self.log_std(x).clamp(max=30)
            x[:, i] = mu[:, i] + z[:, i] * (0.1 * log_std[:, i]).exp()
            log_det += 0.1 * log_std[:, i]
        return x, log_det


class MAF(Flow):
    def __init__(self, d_model, n_layers):
        layers = []
        for _ in range(n_layers):
            layers += [RandPerm(d_model), MAFLayer(d_model)]
        super().__init__(flows=layers)
