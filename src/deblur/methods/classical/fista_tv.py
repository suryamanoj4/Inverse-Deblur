"""FISTA for TV-regularized deblurring.

Solves

    min_x  0.5 * ||A x - y||^2 + lam * TV(x)

with accelerated proximal gradient (FISTA). The proximal operator of TV is
computed by ADMM (see :mod:`deblur.methods.classical.tv`). The step size is
``1 / ||A||^2``, where ``||A||^2 = max |K|^2`` for circular convolution.
"""

from __future__ import annotations

import numpy as np

from ...operators.fft_conv import fft_adjoint, fft_convolve, kernel_spectrum
from .tv import tv_denoise_admm


def suggest_tv_lambda(sigma: float) -> float:
    """Starting point for the TV weight; tune it on the validation split."""
    return max(1e-4, 0.1 * float(sigma))


def fista_tv(
    y: np.ndarray,
    kernel: np.ndarray,
    lam: float,
    n_iter: int = 100,
    tv_iter: int = 30,
    tv_rho: float = 1.0,
    step: float | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Restore ``y`` by FISTA with an isotropic TV prior.

    Parameters
    ----------
    lam:
        TV regularization weight. Use :func:`suggest_tv_lambda` as a starting
        point and tune on validation data.
    n_iter, tv_iter:
        Outer FISTA iterations and inner ADMM iterations for the TV prox.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {y.shape}")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    H, W = y.shape
    K = kernel_spectrum(kernel, (H, W))
    lipschitz = float(np.max(np.abs(K) ** 2))
    if lipschitz <= 0:
        raise ValueError("kernel has zero spectrum; operator is not invertible")
    if step is None:
        step = 1.0 / lipschitz

    x = y.copy()
    z = x.copy()
    t = 1.0
    for _ in range(n_iter):
        residual = fft_convolve(z, kernel) - y
        grad = fft_adjoint(residual, kernel)
        x_new = tv_denoise_admm(z - step * grad, weight=step * lam, n_iter=tv_iter, rho=tv_rho)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        momentum = (t - 1.0) / t_new
        z = x_new + momentum * (x_new - x)
        x, t = x_new, t_new
    return np.clip(x, 0.0, 1.0) if clip else x
