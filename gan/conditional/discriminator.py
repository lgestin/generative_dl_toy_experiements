import torch.nn as nn

from torch.nn.utils import weight_norm
from gan.conditional.generator import CondBatchNorm2d, CondSequential


def Conv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


class Discriminator(nn.Module):
    def __init__(self, d_z, d_model):
        super().__init__()

        d_z = 8
        self.z_cond = nn.Embedding(11, d_z)

        model = [
            Conv2d(1, d_model, (1, 1)),
            CondBatchNorm2d(d_z, d_model),
            nn.ReLU(inplace=True),
            Conv2d(d_model, d_model, (4, 4), (2, 2), padding=2),
            CondBatchNorm2d(d_z, d_model),
            nn.ReLU(inplace=True),
            Conv2d(d_model, d_model, (4, 4), (2, 2), padding=2),
            CondBatchNorm2d(d_z, d_model),
            nn.ReLU(inplace=True),
            Conv2d(d_model, d_model, (4, 4), (2, 2), padding=2),
            nn.ReLU(inplace=True),
            Conv2d(d_model, 1, (1, 1)),
        ]
        self.model = CondSequential(*model)

    def forward(self, x, z_cond):
        z_cond = self.z_cond(z_cond)
        return self.model(x, z_cond)
