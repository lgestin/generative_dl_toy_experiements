import torch
import torch.nn as nn

from torch.nn.utils import weight_norm


def Conv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


def Conv2dReLU(*args, **kwargs):
    return nn.Sequential(Conv2d(*args, **kwargs), nn.ReLU(inplace=True))


def UpsampleConv2dReLU(*args, **kwargs):
    return nn.Sequential(
        nn.Upsample(scale_factor=2), Conv2d(*args, **kwargs), nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        encoder = [
            # 28 -> 28
            Conv2dReLU(2, d_model, (1, 1)),
            # 28 -> 14
            Conv2dReLU(d_model, d_model // 2, (5, 5), (2, 2), padding=2),
            # 14 -> 7
            Conv2dReLU(d_model // 2, d_model // 4, (5, 5), (2, 2), padding=2),
            # 7 -> 4
            Conv2dReLU(d_model // 4, d_model // 8, (4, 4)),
            Conv2d(d_model // 8, (d_model // 8), (1, 1)),
        ]
        self.encoder = nn.ModuleList(encoder)

        decoder = [
            # 4 -> 7
            UpsampleConv2dReLU(d_model // 8, (d_model // 4), (2, 2)),
            # 7 -> 14
            UpsampleConv2dReLU(d_model // 4, d_model // 2, (5, 5), padding=2),
            # 14 -> 28
            UpsampleConv2dReLU(d_model // 2, d_model, (5, 5), padding=2),
            # 28 -> 28
            Conv2dReLU(d_model, d_model, (5, 5), padding=2),
            Conv2d(d_model, 1, (1, 1)),
        ]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, x, t):
        t = t.view(-1, 1, 1, 1).repeat(1, 1, 28, 28)
        x = torch.cat([x, t], dim=1)

        skip = []
        for enc in self.encoder:
            x = enc(x)
            skip += [x]

        for dec, s in zip(self.decoder[:-1], reversed(skip[:-1])):
            x = dec(x + s)

        return self.decoder[-1](x)


class DiffusionModel(nn.Module):
    def __init__(self, d_model, beta_0, beta_T, T: int):
        super().__init__()
        self.T = T
        self.unet = UNet(d_model=d_model)

        self.register_buffer("betas", torch.linspace(beta_0, beta_T, T))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alpha_bars", self.alphas.cumprod(-1))

    def forward_process(self, x_0, t):
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        z = torch.randn_like(x_0)
        x_t = alpha_bar_t.pow(0.5) * x_0 + (1 - alpha_bar_t).pow(0.5) * z
        return x_t, z

    def reverse_process(self, x_t, t):
        x_0_pred = self.unet(x_t, t / self.T)
        return x_0_pred

    def denoise(self, x_T):
        x_t = x_T
        x_ts = []
        for t in reversed(torch.arange(self.T).to(x_t.device)):
            z = torch.randn_like(x_T) if t.item() > 0 else 0
            t = t.repeat(x_T.size(0))
            beta_t = self.betas[t].view(-1, 1, 1, 1)
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

            z_pred = self.reverse_process(x_t, t)
            x_tm1 = (
                alpha_t.pow(-0.5)
                * (x_t - beta_t * (1 - alpha_bar_t).pow(-0.5) * z_pred)
            ) + beta_t * z
            x_t = x_tm1
            if t[0].item() % 10 == 0:
                x_ts.append(x_t)
        x_ts = torch.cat(x_ts, dim=1)
        return x_t, x_ts
