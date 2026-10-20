"""Deterministic observations for validation and test cells.

Every arm must be scored on the *same* degraded images, otherwise paired tests
compare different noise draws. A cell is (image, PSF, noise level); its noise
seed is derived from stable identifiers (names, not list positions or Python's
randomised ``hash``), so the observation is reproducible across runs, machines
and arms.
"""

from __future__ import annotations

import zlib

import numpy as np

from ..data.degrade import degrade


def _stable_id(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


def cell_rng(seed: int, image_id: str, psf_name: str, sigma: float) -> np.random.Generator:
    """Noise generator for one evaluation cell."""
    sigma_key = int(round(sigma * 255 * 1000))
    return np.random.default_rng([seed, _stable_id(image_id), _stable_id(psf_name), sigma_key])


def observe(
    image: np.ndarray,
    kernel: np.ndarray,
    sigma: float,
    *,
    seed: int,
    image_id: str,
    psf_name: str,
) -> np.ndarray:
    """Degraded observation ``y = A x + eps`` for one cell, clipped to [0, 1]."""
    rng = cell_rng(seed, image_id, psf_name, sigma)
    return degrade(image, kernel, sigma=sigma, rng=rng)
