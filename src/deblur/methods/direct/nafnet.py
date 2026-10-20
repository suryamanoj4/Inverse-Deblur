"""NAFNet (Chen, Chu, Zhang, Sun, ECCV 2022) — primary M1.

A U-Net of NAFBlocks. Each block replaces nonlinear activations with
SimpleGate (split channels in half, multiply elementwise) and uses simplified
channel attention (global pooling + 1x1 conv, no sigmoid). Learned per-channel
scales ``beta``/``gamma`` start at zero, so every block starts as identity.

The default configuration is the paper's width-32 GoPro model
(encoder blocks [1, 1, 1, 28], one middle block, decoder blocks [1, 1, 1, 1]),
with one input/output channel for luminance. The network predicts a residual
added to the observation; it never receives ``k`` or ``sigma``.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import pad_to_multiple


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension at every pixel."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        return self.weight * (x - mu) / torch.sqrt(var + self.eps) + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw = channels * dw_expand
        ffn = channels * ffn_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, channels, 1)
        self.gate = SimpleGate()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.gate(self.conv2(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.gate(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        enc_blocks: tuple[int, ...] = (1, 1, 1, 28),
        middle_blocks: int = 1,
        dec_blocks: tuple[int, ...] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()
        if len(enc_blocks) != len(dec_blocks):
            raise ValueError("enc_blocks and dec_blocks must have the same length")
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, in_channels, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        channels = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(n)]))
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))
            channels *= 2
        self.middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blocks)])
        for n in dec_blocks:
            self.ups.append(
                nn.Sequential(nn.Conv2d(channels, 2 * channels, 1, bias=False), nn.PixelShuffle(2))
            )
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(n)]))
        self.multiple = 2 ** len(enc_blocks)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        H, W = inp.shape[-2:]
        padded = pad_to_multiple(inp, self.multiple)
        x = self.intro(padded)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = decoder(up(x) + skip)
        x = self.ending(x) + padded
        return x[..., :H, :W]
