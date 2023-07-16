import torch.nn as nn

from torch.nn.utils import weight_norm


def Conv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


class Discriminator(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        model = [
            # 28 -> 28
            Conv2d(1, d_model, (1, 1)),
            nn.ReLU(inplace=True),
            # 28 -> 14
            Conv2d(d_model, d_model // 2, (5, 5), (2, 2), padding=2),
            nn.ReLU(inplace=True),
            # 14 -> 7
            Conv2d(d_model // 2, d_model // 4, (5, 5), (2, 2), padding=2),
            nn.ReLU(inplace=True),
            # 7 -> 4
            Conv2d(d_model // 4, d_model // 8, (4, 4)),
            nn.ReLU(inplace=True),
            Conv2d(d_model // 8, 1, (1, 1)),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
