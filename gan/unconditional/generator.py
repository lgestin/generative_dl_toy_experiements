import torch
import torch.nn as nn

from torch.nn.utils import weight_norm


def Conv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


class UnconditionalGenerator(nn.Module):
    def __init__(self, d_z, d_model):
        super().__init__()
        self.register_buffer("z_loc", torch.zeros(d_z))
        self.register_buffer("z_std", torch.ones(d_z))

        model = [
            Conv2d(1, d_model // 8, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # 4 -> 7
            nn.Upsample(scale_factor=2),
            Conv2d(d_model // 8, d_model // 4, (2, 2)),
            nn.ReLU(inplace=True),
            # 7 -> 14
            nn.Upsample(scale_factor=2),
            Conv2d(d_model // 4, d_model // 2, (5, 5), padding=2),
            nn.ReLU(inplace=True),
            # 14 -> 28
            nn.Upsample(scale_factor=2),
            Conv2d(d_model // 2, d_model, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # 28 -> 28
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
