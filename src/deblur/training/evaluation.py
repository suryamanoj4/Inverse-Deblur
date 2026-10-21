"""Model wrappers for grid evaluation (requires PyTorch).

The torch-free grid logic lives in :mod:`deblur.evaluation.grid`.
"""

from __future__ import annotations

import numpy as np
import torch

from ..evaluation.grid import Restorer, evaluate_grid, mean_psnr
from ..operators.fft_conv import kernel_spectrum
from ..psf.library import PSF


def model_restorer(
    model: torch.nn.Module,
    device: str | torch.device,
    amp: bool = False,
) -> Restorer:
    """Wrap an M1 network as a restorer; it ignores the PSF and sigma it is given."""
    from ..methods.direct.common import restore

    def restore_fn(y: np.ndarray, psf: PSF, sigma: float) -> np.ndarray:
        return restore(model, y, device=device, amp=amp)

    return restore_fn


def physics_restorer(
    model: torch.nn.Module,
    device: str | torch.device,
    amp: bool = False,
) -> Restorer:
    """Wrap an M3 network; it receives the true PSF and sigma at test time."""

    @torch.inference_mode()
    def restore_fn(y: np.ndarray, psf: PSF, sigma: float) -> np.ndarray:
        H, W = y.shape
        spectrum = kernel_spectrum(psf.kernel, (H, W))
        t_y = torch.as_tensor(y, dtype=torch.float32, device=device)[None, None]
        t_k = torch.as_tensor(spectrum, dtype=torch.complex64, device=device)[None]
        t_sigma = torch.as_tensor([sigma], dtype=torch.float32, device=device)
        device_type = torch.device(device).type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            out = model(t_y, t_k, t_sigma)
        out = out.float()[0, 0].cpu().numpy().astype(np.float64)
        return np.clip(out, 0.0, 1.0)

    return restore_fn


__all__ = ["evaluate_grid", "mean_psnr", "model_restorer", "physics_restorer"]
