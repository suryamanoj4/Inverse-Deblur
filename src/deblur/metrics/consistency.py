"""Data-consistency and robustness metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..operators.fft_conv import fft_convolve


@dataclass(frozen=True)
class Consistency:
    """Fidelity of a reconstruction to the measured forward model."""

    l2: float
    rmse: float


def data_consistency(x_hat: np.ndarray, y: np.ndarray, kernel: np.ndarray) -> Consistency:
    """Residual ``A x_hat - y`` reported as an L2 norm and a per-pixel RMSE."""
    residual = fft_convolve(x_hat, kernel) - np.asarray(y, dtype=np.float64)
    return Consistency(
        l2=float(np.linalg.norm(residual)),
        rmse=float(np.sqrt(np.mean(residual**2))),
    )


def psnr_drop(reference_psnr: float, perturbed_psnr: float) -> float:
    """Drop in dB (positive means the perturbed condition is worse)."""
    return float(reference_psnr - perturbed_psnr)
