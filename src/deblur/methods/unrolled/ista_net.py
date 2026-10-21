"""ISTA-Net-style unrolled network (the middle M3 rung).

Each phase performs a data-gradient step followed by a learned analysis
transform, a soft threshold in the transform domain, and a learned synthesis
transform. The forward operator stays in the loop, so this is the
physics-aware counterpart of a plain CNN.

The original ISTA-Net was proposed for compressive sensing; here the forward
operator is circular convolution. The symmetry constraint on the transforms is
not imposed in this implementation (reconstruction loss only), which is recorded
as a documented adaptation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import data_gradient, soft_threshold


class ISTANet(nn.Module):
    def __init__(self, n_iter: int = 10, channels: int = 32) -> None:
        super().__init__()
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        self.n_iter = n_iter
        self.rho = nn.Parameter(torch.zeros(n_iter))          # sigmoid -> 0.5
        self.theta = nn.Parameter(torch.full((n_iter,), -4.0))  # softplus -> 0.018
        self.analysis = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 3, padding=1),
                )
                for _ in range(n_iter)
            ]
        )
        self.synthesis = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, 1, 3, padding=1),
                )
                for _ in range(n_iter)
            ]
        )

    def forward(
        self,
        y: torch.Tensor,
        spectrum: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        del sigma  # the data module uses the known kernel; sigma is not an input
        x = y
        for t in range(self.n_iter):
            gradient = data_gradient(x, y, spectrum)
            r = x - torch.sigmoid(self.rho[t]) * gradient
            code = self.analysis[t](r)
            code = soft_threshold(code, F.softplus(self.theta[t]))
            x = self.synthesis[t](code)
        return x
