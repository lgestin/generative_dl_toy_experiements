import torch
import torch.nn as nn


class ConditionalPrior(nn.Module):
    def __init__(self):
        super().__init__()

        d = 8 * 4 * 4
        self.z_cond = nn.Embedding(11, 8)
        self.proj = nn.Sequential(nn.Linear(8, d), nn.ReLU(inplace=True))

        self.conv = nn.Conv2d(8, 8 * 2, (3, 3), padding=1)

    def forward(self, z):
        z_cond = self.z_cond(z)
        z_cond = self.proj(z_cond)

        return self.conv(z_cond.view(-1, 8, 4, 4)).chunk(2, dim=1)
