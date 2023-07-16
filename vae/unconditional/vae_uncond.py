import torch
import torch.nn as nn

from torch.nn.utils import weight_norm


def Conv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


class Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        model = [
            Conv2d(1, d_model, (1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(d_model, d_model // 2, (4, 4), (2, 2), padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(d_model // 2, d_model // 4, (4, 4), (2, 2), padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(d_model // 4, d_model // 8, (2, 2), (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(d_model // 8, 2 * (d_model // 8), (1, 1)),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x).chunk(2, dim=1)


class Decoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        model = [
            Conv2d(d_model // 8, d_model // 8, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            Conv2d(d_model // 8, d_model // 4, (5, 5), padding=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            Conv2d(d_model // 4, d_model // 2, (5, 5), padding=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample((28, 28)),
            Conv2d(d_model // 2, d_model, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(d_model, 1, (1, 1)),
        ]

        self.model = nn.Sequential(*model)

    def sample(self, batch_size=1):
        z = self.sample_z(batch_size=batch_size)
        return self.model(z), z

    def forward(self, z):
        return self.model(z)

    def sample_z(self, batch_size=1):
        z = torch.distributions.Normal(loc=self.z_loc, scale=self.z_std)
        return z.sample((batch_size,))


class VAE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.register_buffer("z_loc", torch.zeros((d_model // 8, 4, 4)))
        self.register_buffer("z_std", torch.ones((d_model // 8, 4, 4)))

        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(d_model=d_model)

    def sample_z(self, batch_size: int = 1):
        z = torch.distributions.Normal(self.z_loc, self.z_std)
        z = z.sample((batch_size,))
        return z

    def forward(self, x):
        mu, log_std = self.encoder(x)
        z = self.sample_z(x.size(0))
        z = z * log_std.exp() + mu
        return self.decoder(z), (mu, log_std)

    def decode(self, z):
        return self.decoder(z)
