"""Shared helpers for the direct (data-only) M1 networks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Circularly pad (N, C, H, W) so H and W are multiples of ``multiple``.

    Circular padding matches the protocol's circular boundary: the observation
    is periodic, so wrapping adds no artificial edge.
    """
    H, W = x.shape[-2:]
    pad_h = (-H) % multiple
    pad_w = (-W) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="circular")


@torch.inference_mode()
def restore(
    model: nn.Module,
    y: np.ndarray,
    device: str | torch.device = "cpu",
    amp: bool = False,
    clip: bool = True,
) -> np.ndarray:
    """Restore one 2D luminance image in [0, 1]; returns float64 like the M2 solvers."""
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {y.shape}")
    model.eval()
    t = torch.as_tensor(y, dtype=torch.float32, device=device)[None, None]
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        out = model(t)
    out = out.float()[0, 0].cpu().numpy().astype(np.float64)
    return np.clip(out, 0.0, 1.0) if clip else out
