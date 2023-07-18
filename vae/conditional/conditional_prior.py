import torch
import torch.nn as nn


class ConditionalPrior(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        d = (d_model // 8) * 4 * 4
        self.z_cond = nn.Embedding(11, 16)
        self.proj = nn.Sequential(nn.Linear(16, d), nn.ReLU(inplace=True))

        self.conv = nn.Conv2d(16, 16 * 2, (3, 3), padding=1)

    def forward(self, z):
        z_cond = self.z_cond(z)
        z_cond = self.proj(z_cond)

        return self.conv(z_cond.view(z.size(0), -1, 4, 4)).chunk(2, dim=1)
