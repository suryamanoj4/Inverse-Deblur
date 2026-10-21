"""USRNet-style unrolled network (the primary M3 arm).

Unrolls ``T`` half-quadratic splitting iterations:

    x_t = argmin_x 0.5 ||A x - y||^2 + mu_t/2 ||x - z_{t-1}||^2   (FFT, exact)
    z_t = x_t + Prior([x_t, sigma_eff_t])                          (ResUNet)

``A`` is the known circular convolution, supplied through its spectrum. The
hyperparameter MLP maps the noise level to ``(mu_t, lambda_t)``; the prior is
conditioned on ``sqrt(lambda_t / mu_t)``. No scale factor is used: this is the
deblurring special case (s = 1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import data_step


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResUNet(nn.Module):
    """Small residual U-Net used as the learned denoiser prior."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1, width: int = 32) -> None:
        super().__init__()
        w = width
        self.intro = nn.Conv2d(in_channels, w, 3, padding=1)
        self.enc1 = nn.Sequential(ResBlock(w), ResBlock(w))
        self.down1 = nn.Conv2d(w, 2 * w, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResBlock(2 * w), ResBlock(2 * w))
        self.down2 = nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1)
        self.enc3 = nn.Sequential(ResBlock(4 * w), ResBlock(4 * w))
        self.down3 = nn.Conv2d(4 * w, 8 * w, 3, stride=2, padding=1)
        self.mid = nn.Sequential(ResBlock(8 * w), ResBlock(8 * w))
        self.up3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(8 * w, 4 * w, 3, padding=1), ResBlock(4 * w))
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(4 * w, 2 * w, 3, padding=1), ResBlock(2 * w))
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(2 * w, w, 3, padding=1), ResBlock(w))
        self.out = nn.Conv2d(w, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        pad_h = (-H) % 4
        pad_w = (-W) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        e1 = self.enc1(self.intro(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        mid = self.mid(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(mid), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out(d1)
        return out[..., :H, :W]


class HyperParameterMLP(nn.Module):
    """Maps the noise level to ``(mu_t, lambda_t)`` for every iteration."""

    def __init__(self, n_iter: int, hidden: int = 64) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * n_iter),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        values = F.softplus(self.net(sigma.reshape(-1, 1))) + 1e-4
        return values.view(-1, self.n_iter, 2)


class USRNet(nn.Module):
    """Unrolled HQS network with a ResUNet prior and an MLP hyperparameter module."""

    def __init__(self, n_iter: int = 8, width: int = 32) -> None:
        super().__init__()
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        self.n_iter = n_iter
        self.prior = ResUNet(in_channels=2, out_channels=1, width=width)
        self.hyper = HyperParameterMLP(n_iter)

    def forward(
        self,
        y: torch.Tensor,
        spectrum: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        if sigma.ndim > 1:
            sigma = sigma.reshape(-1)
        hyper = self.hyper(sigma)
        x = y
        z = y
        for t in range(self.n_iter):
            mu = hyper[:, t, 0].view(-1, 1, 1, 1)
            lam = hyper[:, t, 1].view(-1, 1, 1, 1)
            x = data_step(y, spectrum, z, mu)
            sigma_eff = torch.sqrt(lam / mu)
            sigma_map = sigma_eff.expand(-1, 1, x.shape[-2], x.shape[-1])
            z = x + self.prior(torch.cat([x, sigma_map], dim=1))
        return x
