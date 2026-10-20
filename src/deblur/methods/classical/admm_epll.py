"""ADMM / half-quadratic splitting with an EPLL (GMM) patch prior.

Solves

    min_x  0.5 * ||A x - y||^2 + lam * sum_p -log p(x_p)

by introducing an auxiliary ``z``:

    x-update:  min_x 0.5||A x - y||^2 + beta/2 ||x - z||^2   (closed form, FFT)
    z-update:  min_z beta/2 ||x - z||^2 + lam * sum_p -log p(z_p)

The z-update is an EPLL denoising problem at noise level ``sqrt(lam / beta)``;
beta is annealed from ``beta0`` to ``beta_max``.
"""

from __future__ import annotations

import numpy as np

from ...operators.fft_conv import kernel_spectrum
from .gmm import GMM, patch_denoise


def admm_epll(
    y: np.ndarray,
    kernel: np.ndarray,
    gmm: GMM,
    sigma: float = 0.0,
    lam: float | None = None,
    n_iter: int = 10,
    beta0: float = 1.0,
    beta_max: float = 1e3,
    patch_size: int = 8,
    stride: int = 1,
    clip: bool = True,
) -> np.ndarray:
    """Restore ``y`` with the EPLL/GMM prior.

    ``lam`` is the prior weight. When omitted it is set so that the final
    denoising strength ``sqrt(lam / beta_max)`` equals the supplied ``sigma``
    (with a small floor for noiseless inputs).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {y.shape}")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")
    if beta0 <= 0 or beta_max < beta0:
        raise ValueError("require 0 < beta0 <= beta_max")

    if lam is None:
        lam = max(float(sigma) ** 2, 1e-6) * beta_max

    H, W = y.shape
    K = kernel_spectrum(kernel, (H, W))
    Y = np.fft.fft2(y)
    denominator = np.abs(K) ** 2
    x = y.copy()
    z = x.copy()
    beta = beta0

    for _ in range(n_iter):
        spectrum = (np.conj(K) * Y + beta * np.fft.fft2(z)) / (denominator + beta)
        x = np.fft.ifft2(spectrum).real
        sigma_eff = float(np.sqrt(lam / beta))
        z = patch_denoise(x, gmm, sigma_eff, patch_size=patch_size, stride=stride)
        beta = min(beta * 2.0, beta_max)

    return np.clip(x, 0.0, 1.0) if clip else x
