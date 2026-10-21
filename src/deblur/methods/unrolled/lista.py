"""LISTA-style unrolled network (the first M3 rung).

A shared 1x1 analysis/synthesis pair is applied after each data-gradient step,
with a learnable step size and soft-threshold per iteration. This is the
convolutional, image-domain adaptation of LISTA's learned matrices plus
shrinkage: the original LISTA operates on sparse codes with a linear dictionary,
whereas deblurring couples all patches through the convolution operator.

The point of this rung is the ablation, not a claim of fidelity to the sparse
coding setting: it has a learned linear prior and the forward operator in the
loop, with far less capacity than ISTA-Net or USRNet.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import data_gradient, soft_threshold


class LearnedISTA(nn.Module):
    def __init__(self, n_iter: int = 10, hidden: int = 32) -> None:
        super().__init__()
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        self.n_iter = n_iter
        self.step = nn.Parameter(torch.zeros(n_iter))           # sigmoid -> 0.5
        self.theta = nn.Parameter(torch.full((n_iter,), -4.0))  # softplus -> 0.018
        self.analysis = nn.Conv2d(1, hidden, 1)
        self.synthesis = nn.Conv2d(hidden, 1, 1)

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
            r = x - torch.sigmoid(self.step[t]) * gradient
            code = self.analysis(r)
            code = soft_threshold(code, F.softplus(self.theta[t]))
            x = self.synthesis(code)
        return x
