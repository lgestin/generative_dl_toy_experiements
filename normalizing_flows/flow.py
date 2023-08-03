import torch
import torch.nn as nn
import torch.nn.functional as F


class Flow(nn.Module):
    def __init__(self, flows: list):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        logdet = 0
        for flow in self.flows:
            x, ld = flow(x)
            logdet += ld
        return x, logdet

    def inverse(self, x):
        logdet = 0
        for flow in reversed(self.flows):
            x, ld = flow.inverse(x)
            logdet += ld
        return x, logdet


class RandPerm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.register_buffer("perm", torch.randperm(d_model))
        self.register_buffer("iperm", torch.argsort(self.perm))

    def forward(self, x):
        return x[:, self.perm], 0

    def inverse(self, x):
        return x[:, self.iperm], 0


class MaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        in_features = kwargs.pop("in_features") if "in_features" in kwargs else args[0]
        out_features = (
            kwargs.pop("out_features") if "out_features" in kwargs else args[1]
        )
        mask = torch.ones(out_features, in_features).tril(-1).bool()
        self.register_buffer("mask", ~mask)

    def forward(self, x):
        weight = self.weight.masked_fill(self.mask, 0)
        return F.linear(x, weight, bias=self.bias)
