import torch
import torch.nn as nn

from vae.conditional.vae_cond import ConditionalVAE
from normalizing_flows.flow import Flow, RandPerm


class AffineCouplingLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        layers = [
            nn.Conv2d(d_model // 2, d_model // 2, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, d_model, (1, 1)),
        ]
        self.layers = nn.Sequential(*layers)
        self.layers[-1].weight.data.zero_()
        self.layers[-1].bias.data.zero_()

    def compute_mu_logs(self, x):
        x = self.layers(x)
        mu, log_std = x.chunk(2, dim=1)
        log_std = 1.4 * log_std.tanh()
        return mu, log_std

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        mu, log_std = self.compute_mu_logs(x_0)
        x_1 = mu + x_1 * log_std.exp()
        x = torch.cat([x_0, x_1], dim=1)
        log_det = log_std.mean(1).sum(dim=[1, 2])
        return x, log_det

    def inverse(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        mu, log_std = self.compute_mu_logs(x_0)
        x_1 = (x_1 - mu) * torch.exp(-log_std)
        x = torch.cat([x_0, x_1], dim=1)
        log_det = -log_std.mean(1).sum(dim=[1, 2])
        return x, log_det


class FVAE(ConditionalVAE):
    def __init__(self, d_model):
        super().__init__(d_model)
        self.flow = Flow(
            [
                AffineCouplingLayer(d_model // 8),
                RandPerm(d_model // 8),
                AffineCouplingLayer(d_model // 8),
                RandPerm(d_model // 8),
                AffineCouplingLayer(d_model // 8),
            ]
        )

    def forward(self, x, z_cond):
        mu_q, log_std_q = self.encoder(x)

        z_q = self.sample_z(x.size(0))
        z_q = z_q * log_std_q.exp() + mu_q

        z_p, log_det = self.flow(z_q)

        mu_p, log_std_p = self.prior(z_cond)
        log_std_p = 1.4 * torch.tanh(log_std_p)

        return (
            self.decoder(z_q),
            log_det,
            (z_q, mu_q, log_std_q),
            (z_p, mu_p, log_std_p),
        )

    def sample(self, z_cond, scale=1.0):
        mu_p, log_std_p = self.prior(z_cond)
        log_std_p = 1.4 * torch.tanh(log_std_p)

        z_p = self.sample_z(z_cond.size(0))
        z_p = z_p * log_std_p.exp() * scale + mu_p

        z_q, _ = self.flow.inverse(z_p)

        return self.decoder(z_q)
