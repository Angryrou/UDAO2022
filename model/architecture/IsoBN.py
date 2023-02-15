# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Isotropic Batch Normalization from AAAI21 (IsoBN: Fine-Tuning BERT with Isotropic Batch Normalization)
# https://github.com/INK-USC/IsoBN/blob/master/model.py
#
# Created at 15/02/2023

import torch
import torch.nn as nn

class IsoBN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cov = torch.zeros(hidden_dim, hidden_dim)
        self.std = torch.zeros(hidden_dim)

    def forward(self, input, momentum=0.05, eps=1e-3, beta=0.5):
        if self.training:
            x = input.detach()
            n = x.size(0)
            mean = x.mean(dim=0)
            y = x - mean.unsqueeze(0)
            std = (y ** 2).mean(0) ** 0.5
            cov = (y.t() @ y) / n
            self.cov.data += momentum * (cov.data - self.cov.data)
            self.std.data += momentum * (std.data - self.std.data)
        corr = torch.clamp(self.cov / torch.ger(self.std, self.std), -1, 1)
        gamma = (corr ** 2).mean(1)
        denorm = (gamma * self.std)
        scale = 1 / (denorm + eps) ** beta
        E = torch.diag(self.cov).sum()
        new_E = (torch.diag(self.cov) * (scale ** 2)).sum()
        m = (E / (new_E + eps)) ** 0.5
        scale *= m
        return input * scale.unsqueeze(0).detach()