"""ADMM / half-quadratic splitting with an EPLL (GMM) patch prior.

Solves

    min_x  0.5 * ||A x - y||^2 + lam * sum_p -log p(x_p)

by introducing an auxiliary ``z``:

    x-update:  min_x 0.5||A x - y||^2 + beta/2 ||x - z||^2   (closed form, FFT)
    z-update:  min_z beta/2 ||x - z||^2 + lam * sum_p -log p(z_p)

The z-update is an EPLL denoising problem at noise level ``s = sqrt(lam / beta)``.
The schedule is set by ``s``: it decreases geometrically from ``sigma_start``
to the observation noise level (floored at ``sigma_floor``), and
``beta_t = lam / s_t^2``. Starting with a strong denoiser keeps ``beta`` small,
so the early x-updates actually invert the blur; with ``beta >= |K|^2`` the
x-update stays close to ``z`` and the iteration barely leaves the blurred input.
This is the plug-and-play HQS schedule of IRCNN/DPIR (Zhang et al., 2017, 2021).
"""

from __future__ import annotations

import numpy as np

from ...operators.fft_conv import kernel_spectrum
from .gmm import GMM, patch_denoise

SIGMA_START = 49.0 / 255.0
SIGMA_FLOOR = 1.0 / 255.0


def epll_schedule(
    sigma: float,
    n_iter: int,
    sigma_start: float = SIGMA_START,
    sigma_floor: float = SIGMA_FLOOR,
) -> np.ndarray:
    """Denoiser strengths ``s_t``, geometric from ``sigma_start`` to the noise level."""
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")
    if sigma_start <= 0 or sigma_floor <= 0:
        raise ValueError("sigma_start and sigma_floor must be positive")
    end = max(float(sigma), sigma_floor)
    return np.geomspace(max(sigma_start, end), end, n_iter)


def suggest_epll_lambda(sigma: float, sigma_floor: float = SIGMA_FLOOR) -> float:
    """Starting point for the EPLL weight (DPIR's 0.23 sigma^2); tune it on val."""
    return 0.23 * max(float(sigma), sigma_floor) ** 2


def admm_epll(
    y: np.ndarray,
    kernel: np.ndarray,
    gmm: GMM,
    sigma: float = 0.0,
    lam: float | None = None,
    n_iter: int = 10,
    sigma_start: float = SIGMA_START,
    sigma_floor: float = SIGMA_FLOOR,
    patch_size: int = 8,
    stride: int = 1,
    clip: bool = True,
) -> np.ndarray:
    """Restore ``y`` with the EPLL/GMM prior.

    ``lam`` is the prior weight; :func:`suggest_epll_lambda` is used when it is
    omitted. The result is the final z-update (the prior's estimate).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {y.shape}")
    schedule = epll_schedule(sigma, n_iter, sigma_start, sigma_floor)
    if lam is None:
        lam = suggest_epll_lambda(sigma, sigma_floor)
    if lam <= 0:
        raise ValueError("lam must be positive")

    H, W = y.shape
    K = kernel_spectrum(kernel, (H, W))
    Y = np.fft.fft2(y)
    KY = np.conj(K) * Y
    denominator = np.abs(K) ** 2
    z = y.copy()

    for s in schedule:
        beta = lam / s**2
        x = np.fft.ifft2((KY + beta * np.fft.fft2(z)) / (denominator + beta)).real
        z = patch_denoise(x, gmm, float(s), patch_size=patch_size, stride=stride)

    return np.clip(z, 0.0, 1.0) if clip else z
