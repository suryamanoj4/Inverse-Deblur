"""Richardson-Lucy deconvolution.

The classical maximum-likelihood update for Poisson observations:

    x <- x * (A^T (y / (A x))),

with positivity enforced at every step. It is included as a non-regularized
sanity baseline: with the correct kernel it sharpens, but it amplifies noise
because it has no explicit prior.
"""

from __future__ import annotations

import numpy as np

from ...operators.fft_conv import fft_adjoint, fft_convolve


def richardson_lucy(
    y: np.ndarray,
    kernel: np.ndarray,
    n_iter: int = 30,
    x0: np.ndarray | None = None,
    eps: float = 1e-8,
    clip: bool = True,
) -> np.ndarray:
    """Restore ``y`` with ``n_iter`` Richardson-Lucy iterations."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {y.shape}")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")
    x = np.full_like(y, 0.5) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    x = np.maximum(x, eps)
    for _ in range(n_iter):
        ratio = y / np.maximum(fft_convolve(x, kernel), eps)
        x = x * fft_adjoint(ratio, kernel)
        x = np.maximum(x, eps)
    return np.clip(x, 0.0, 1.0) if clip else x
