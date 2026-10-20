"""Degradation pipeline: luminance conversion, circular blur, additive noise."""

from __future__ import annotations

import numpy as np

from ..operators.fft_conv import fft_convolve


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA image to Rec.601 luminance.

    Accepts (H, W), (H, W, 3) or (H, W, 4); already-grayscale input is returned
    unchanged. Values are assumed to be in [0, 1].
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] >= 3:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    raise ValueError(f"cannot convert image of shape {image.shape} to luminance")


def add_noise(image: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add i.i.d. Gaussian noise with standard deviation ``sigma``."""
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    if sigma == 0.0:
        return np.array(image, dtype=np.float64, copy=True)
    rng = np.random.default_rng() if rng is None else rng
    return np.asarray(image, dtype=np.float64) + rng.normal(0.0, sigma, size=np.shape(image))


def degrade(
    image: np.ndarray,
    kernel: np.ndarray,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Apply the forward model ``y = A x + eps`` with circular boundary.

    ``image`` is in [0, 1]; ``sigma`` is the absolute noise level (e.g.
    ``15 / 255``). Output is clipped to [0, 1] by default, which is part of the
    observation model and must be applied identically for every method.
    """
    y = fft_convolve(image, kernel)
    y = add_noise(y, sigma, rng=rng)
    if clip:
        y = np.clip(y, 0.0, 1.0)
    return y


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Return the centre ``size x size`` crop of a 2D image."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {image.shape}")
    H, W = image.shape
    if size > H or size > W:
        raise ValueError(f"crop size {size} exceeds image {(H, W)}")
    top = (H - size) // 2
    left = (W - size) // 2
    return image[top : top + size, left : left + size]
