"""Shared differentiable operations for the unrolled (M3) arm.

The forward model is circular convolution, so every data step and gradient is
computed with FFTs. ``spectrum`` is the DFT of the zero-padded PSF for the
current patch/image, shape (B, H, W); it is supplied by the sampler at training
time and by :func:`deblur.training.evaluation.physics_restorer` at test time.
"""

from __future__ import annotations

import torch


def soft_threshold(x: torch.Tensor, theta) -> torch.Tensor:
    """Component-wise soft threshold with a scalar or per-channel threshold."""
    if isinstance(theta, torch.Tensor) and theta.ndim == 1:
        theta = theta.view(-1, 1, 1, 1)
    return torch.sign(x) * torch.relu(torch.abs(x) - theta)


def fft_convolve(x: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
    """Circular convolution of ``x`` (B, C, H, W) with the PSF spectrum."""
    if spectrum.ndim == 2:
        spectrum = spectrum.unsqueeze(0)
    return torch.fft.ifft2(torch.fft.fft2(x) * spectrum[:, None]).real


def fft_adjoint(x: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
    """Adjoint of :func:`fft_convolve` (correlation with the PSF)."""
    if spectrum.ndim == 2:
        spectrum = spectrum.unsqueeze(0)
    return torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(spectrum)[:, None]).real


def data_gradient(x: torch.Tensor, y: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
    """Gradient of ``0.5 * ||A x - y||^2``."""
    return fft_adjoint(fft_convolve(x, spectrum) - y, spectrum)


def data_step(
    y: torch.Tensor,
    spectrum: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """Closed-form HQS data update for deblurring (scale factor 1).

    Solves ``min_x 0.5 ||A x - y||^2 + mu/2 ||x - z||^2`` in the Fourier domain.
    ``mu`` has shape (B, 1, 1, 1) and may vary per sample and iteration.
    """
    if spectrum.ndim == 2:
        spectrum = spectrum.unsqueeze(0)
    K = spectrum[:, None]
    X = (torch.conj(K) * torch.fft.fft2(y) + mu * torch.fft.fft2(z)) / (
        torch.abs(K) ** 2 + mu
    )
    return torch.fft.ifft2(X).real
