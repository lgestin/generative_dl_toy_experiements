import torch
import torch.nn as nn

from torch.nn.utils import weight_norm


def Conv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


class CondSequential(nn.Sequential):
    def forward(self, x, x_cond=None):
        for layer in self:
            if isinstance(layer, CondBatchNorm2d):
                x = layer(x, x_cond)
            else:
                x = layer(x)
        return x


class CondBatchNorm2d(nn.Module):
    def __init__(self, d_z, d_model):
        super().__init__()
        self.proj = nn.Linear(d_z, 2 * d_model)
        self.bn = nn.BatchNorm2d(d_model, affine=False)

    def forward(self, x, z_cond):
        z_cond = self.proj(z_cond)[..., None, None]
        alpha, beta = z_cond.chunk(2, dim=1)
        x = self.bn(x)
        return (1 + alpha) * x + beta


class Generator(nn.Module):
    def __init__(self, d_z, d_model):
        super().__init__()
        self.register_buffer("z_loc", torch.zeros(d_z))
        self.register_buffer("z_std", torch.ones(d_z))

        d_z = 8
        self.z_cond = nn.Embedding(11, d_z)

        model = [
            Conv2d(1, d_model // 8, (3, 3), padding=(1, 1)),
            CondBatchNorm2d(d_z, d_model // 8),
            nn.ReLU(inplace=True),
            # 4 -> 7
            nn.Upsample(scale_factor=2),
            Conv2d(d_model // 8, d_model // 4, (2, 2)),
            CondBatchNorm2d(d_z, d_model // 4),
            nn.ReLU(inplace=True),
            # 7 -> 14
            nn.Upsample(scale_factor=2),
            Conv2d(d_model // 4, d_model // 2, (5, 5), padding=2),
            CondBatchNorm2d(d_z, d_model // 2),
            nn.ReLU(inplace=True),
            # 14 -> 28
            nn.Upsample(scale_factor=2),
            Conv2d(d_model // 2, d_model, (3, 3), padding=(1, 1)),
            CondBatchNorm2d(d_z, d_model),
            nn.ReLU(inplace=True),
            # 28 -> 28
            Conv2d(d_model, 1, (1, 1)),
        ]

        self.model = CondSequential(*model)

    def sample(self, z_cond):
        bs = z_cond.size(0)
        z = self.sample_z(batch_size=bs)
        return self.forward(z, z_cond), z

    def forward(self, z, z_cond):
        z_cond = self.z_cond(z_cond)
        return self.model(z, z_cond)

    def sample_z(self, batch_size=1):
        z = torch.distributions.Normal(loc=self.z_loc, scale=self.z_std)
        return z.sample((batch_size,))
