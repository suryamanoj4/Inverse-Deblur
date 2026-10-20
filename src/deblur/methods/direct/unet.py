"""Minimal U-Net — the simplest M1 baseline.

Four-level encoder-decoder with skip connections; two 3x3 conv + ReLU per
level, max-pool down, transposed-conv up. Like NAFNet it predicts a residual on
the observation and never sees ``k`` or ``sigma``.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import pad_to_multiple


def _double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, width: int = 32, depth: int = 4) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.encoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        channels = [width * 2**i for i in range(depth + 1)]
        cin = in_channels
        for c in channels[:-1]:
            self.encoders.append(_double_conv(cin, c))
            cin = c
        self.bottleneck = _double_conv(channels[-2], channels[-1])
        for c_low, c_high in zip(reversed(channels[:-1]), reversed(channels[1:])):
            self.ups.append(nn.ConvTranspose2d(c_high, c_low, 2, stride=2))
            self.decoders.append(_double_conv(2 * c_low, c_low))
        self.head = nn.Conv2d(width, in_channels, 1)
        self.multiple = 2**depth

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        H, W = inp.shape[-2:]
        padded = pad_to_multiple(inp, self.multiple)
        x = padded
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(torch.cat([up(x), skip], dim=1))
        x = self.head(x) + padded
        return x[..., :H, :W]
