import torch.nn as nn

from torch.nn.utils import weight_norm


def Conv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


class Discriminator(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        model = [
            Conv2d(1, d_model, (1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(d_model, d_model, (4, 4), (2, 2), padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(d_model, d_model, (4, 4), (2, 2), padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(d_model, d_model, (2, 2), (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(d_model, 1, (1, 1)),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
