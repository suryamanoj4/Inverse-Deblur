"""Phase 3: kernel-placement ladder on one NAFNet backbone.

Every variant uses the same NAFNet backbone as M1 (width 32, encoder blocks
[1, 1, 1, 28]) and receives ``(y, spectrum, sigma)`` like the M3 networks. Only
where the known kernel enters changes:

* ``nafnet_concat``  : kernel + sigma embedding stretched into maps and
                       concatenated to the input (SRMD-style input conditioning);
* ``nafnet_film``    : kernel + sigma embedding modulates every NAFBlock via FiLM
                       (SFTMD / Kaufman & Fattal-style feature conditioning);
* ``knafnet``        : ``nafnet_film`` followed by one closed-form FFT data step
                       (K-NAFNet, the §12.9 proposal);
* ``nafnet_unrolled``: ``n_iter`` HQS iterations with one shared FiLM NAFNet prior,
                       conditioned on kernel, sigma and stage (USRNet skeleton).

The plain NAFNet (no kernel) is M1. Kernel embedding: the spatial kernel is
recovered from the spectrum (inverse FFT, centred, cropped to ``kernel_size``)
and encoded by a small CNN together with sigma. FiLM layers start at zero, so a
conditioned block starts as the plain NAFBlock.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..direct.common import pad_to_multiple
from ..direct.nafnet import NAFBlock
from .common import data_step


def kernel_from_spectrum(spectrum: torch.Tensor, size: int) -> torch.Tensor:
    """Spatial kernel (B, size, size), centred, from its (B, H, W) DFT."""
    if spectrum.ndim == 2:
        spectrum = spectrum.unsqueeze(0)
    k = torch.fft.ifft2(spectrum).real
    k = torch.roll(k, shifts=(size // 2, size // 2), dims=(-2, -1))
    return k[..., :size, :size]


class KernelEncoder(nn.Module):
    """(kernel, sigma) -> embedding of size ``dim``."""

    def __init__(self, kernel_size: int = 31, dim: int = 128) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(nn.Linear(128 + 1, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, spectrum: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        k = kernel_from_spectrum(spectrum, self.kernel_size).float()
        k = k / k.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)  # shape, not energy
        feat = self.conv(k[:, None]).flatten(1)
        s = (sigma.reshape(-1, 1).float() * 255.0 / 25.0)
        return self.mlp(torch.cat([feat, s], dim=1))


class FiLMBlock(nn.Module):
    """NAFBlock whose normalised input is modulated by ``(1 + gamma) x + beta``."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.block = NAFBlock(channels)
        self.film = nn.Linear(cond_dim, 2 * channels) if cond_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if self.film is None or cond is None:
            return self.block(x)
        b = self.block
        gamma, beta = self.film(cond).to(x.dtype)[:, :, None, None].chunk(2, dim=1)
        h = b.norm1(x) * (1 + gamma) + beta
        h = b.gate(b.conv2(b.conv1(h)))
        h = b.conv3(h * b.sca(h))
        y = x + h * b.beta
        h = b.conv5(b.gate(b.conv4(b.norm2(y))))
        return y + h * b.gamma


class CondNAFNet(nn.Module):
    """NAFNet with optional extra input channels and optional FiLM conditioning."""

    def __init__(self, extra_in: int = 0, cond_dim: int = 0, width: int = 32,
                 enc_blocks: tuple[int, ...] = (1, 1, 1, 28), middle_blocks: int = 1,
                 dec_blocks: tuple[int, ...] = (1, 1, 1, 1)) -> None:
        super().__init__()
        self.intro = nn.Conv2d(1 + extra_in, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 1, 3, padding=1)
        self.encoders, self.downs, self.ups, self.decoders = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.ModuleList([FiLMBlock(c, cond_dim) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.middle = nn.ModuleList([FiLMBlock(c, cond_dim) for _ in range(middle_blocks)])
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.ModuleList([FiLMBlock(c, cond_dim) for _ in range(n)]))
        self.multiple = 2 ** len(enc_blocks)

    @staticmethod
    def _run(blocks, x, cond):
        for block in blocks:
            x = block(x, cond)
        return x

    def forward(self, inp: torch.Tensor, cond: torch.Tensor | None = None,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        H, W = inp.shape[-2:]
        padded = pad_to_multiple(inp, self.multiple)
        x = padded
        if extra is not None:
            x = torch.cat([padded, pad_to_multiple(extra, self.multiple)], dim=1)
        x = self.intro(x)
        skips = []
        for blocks, down in zip(self.encoders, self.downs):
            x = self._run(blocks, x, cond)
            skips.append(x)
            x = down(x)
        x = self._run(self.middle, x, cond)
        for blocks, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = self._run(blocks, up(x) + skip, cond)
        x = self.ending(x) + padded
        return x[..., :H, :W]


class NAFNetConcat(nn.Module):
    """Kernel/sigma embedding stretched into ``maps`` input channels (SRMD-style)."""

    def __init__(self, kernel_size: int = 31, dim: int = 128, maps: int = 15) -> None:
        super().__init__()
        self.encoder = KernelEncoder(kernel_size, dim)
        self.to_maps = nn.Linear(dim, maps)
        self.net = CondNAFNet(extra_in=maps + 1)

    def forward(self, y, spectrum, sigma):
        e = self.to_maps(self.encoder(spectrum, sigma))
        H, W = y.shape[-2:]
        maps = e[:, :, None, None].expand(-1, -1, H, W)
        s = (sigma.reshape(-1, 1, 1, 1) * 255.0 / 25.0).expand(-1, 1, H, W)
        return self.net(y, extra=torch.cat([maps, s], dim=1).to(y.dtype))


class NAFNetFiLM(nn.Module):
    """Kernel/sigma embedding modulates every block (FiLM)."""

    def __init__(self, kernel_size: int = 31, dim: int = 128) -> None:
        super().__init__()
        self.encoder = KernelEncoder(kernel_size, dim)
        self.net = CondNAFNet(cond_dim=dim)

    def forward(self, y, spectrum, sigma):
        return self.net(y, cond=self.encoder(spectrum, sigma))


class KNAFNet(nn.Module):
    """FiLM NAFNet, then one closed-form FFT data-consistency step."""

    def __init__(self, kernel_size: int = 31, dim: int = 128) -> None:
        super().__init__()
        self.encoder = KernelEncoder(kernel_size, dim)
        self.net = CondNAFNet(cond_dim=dim)
        self.mu = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, y, spectrum, sigma):
        z = self.net(y, cond=self.encoder(spectrum, sigma))
        mu = F.softplus(self.mu(sigma.reshape(-1, 1).float() * 255.0 / 25.0)) + 1e-4
        return data_step(y.float(), spectrum, z.float(), mu.view(-1, 1, 1, 1))


class NAFNetUnrolled(nn.Module):
    """HQS with a shared FiLM NAFNet prior conditioned on kernel, sigma and stage."""

    def __init__(self, n_iter: int = 4, kernel_size: int = 31, dim: int = 128,
                 grad_checkpoint: bool = True) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.grad_checkpoint = grad_checkpoint
        self.encoder = KernelEncoder(kernel_size, dim)
        self.stage = nn.Embedding(n_iter, dim)
        nn.init.zeros_(self.stage.weight)
        self.net = CondNAFNet(extra_in=1, cond_dim=dim)
        self.hyper = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                   nn.Linear(64, 2 * n_iter))

    def forward(self, y, spectrum, sigma):
        B = y.shape[0]
        e = self.encoder(spectrum, sigma)
        hyper = F.softplus(self.hyper(sigma.reshape(-1, 1).float() * 255.0 / 25.0)) + 1e-4
        hyper = hyper.view(B, self.n_iter, 2)
        z = y
        for t in range(self.n_iter):
            mu = hyper[:, t, 0].view(-1, 1, 1, 1)
            lam = hyper[:, t, 1].view(-1, 1, 1, 1)
            x = data_step(y.float(), spectrum, z.float(), mu)
            s_map = torch.sqrt(lam / mu).expand(-1, 1, *x.shape[-2:])
            cond = e + self.stage.weight[t]
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                z = checkpoint(self.net, x, cond, s_map, use_reentrant=False)
            else:
                z = self.net(x, cond, s_map)
        return z
