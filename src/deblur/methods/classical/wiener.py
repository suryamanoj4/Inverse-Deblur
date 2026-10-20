"""Wiener filtering.

Under a Gaussian signal prior and Gaussian noise, the linear MMSE estimate is
diagonal in the Fourier domain:

    X_hat(w) = conj(K(w)) Y(w) / (|K(w)|^2 + lambda),

with ``lambda`` the noise-to-signal ratio. With circular boundary handling this
is the exact MAP/MMSE solution for a white Gaussian signal prior (Tikhonov).
"""

from __future__ import annotations

import numpy as np

from ...operators.fft_conv import kernel_spectrum


def wiener(
    y: np.ndarray,
    kernel: np.ndarray,
    sigma: float = 0.0,
    signal_var: float | None = None,
    lam: float | None = None,
    eps: float = 1e-12,
    clip: bool = True,
) -> np.ndarray:
    """Restore ``y`` with the Wiener filter.

    Parameters
    ----------
    y, kernel:
        Observed image and PSF.
    sigma:
        Noise standard deviation (absolute, in [0, 1]). Ignored when ``lam`` is
        given.
    signal_var:
        Signal variance used to form the noise-to-signal ratio. Defaults to
        ``var(y)``, a rough but parameter-free estimate.
    lam:
        Explicit regularization weight. Overrides ``sigma``/``signal_var``.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {y.shape}")
    H, W = y.shape
    K = kernel_spectrum(kernel, (H, W))
    if lam is None:
        if signal_var is None:
            signal_var = max(float(np.var(y)), eps)
        lam = (sigma**2) / signal_var
    lam = max(float(lam), eps)
    spectrum = np.conj(K) * np.fft.fft2(y) / (np.abs(K) ** 2 + lam)
    x = np.fft.ifft2(spectrum).real
    return np.clip(x, 0.0, 1.0) if clip else x
