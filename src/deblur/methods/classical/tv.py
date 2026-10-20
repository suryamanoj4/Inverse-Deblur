"""Total-variation operators and the ROF denoising proximal operator.

``tv_denoise_admm`` solves

    min_u  0.5 * ||u - f||^2 + weight * TV(u)

with isotropic TV and periodic boundary, using ADMM (variable splitting
``d = grad(u)``). It is the proximal operator used by the FISTA solver.
"""

from __future__ import annotations

import numpy as np


def gradient(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodic forward differences ``(u[i, j+1] - u[i, j], u[i+1, j] - u[i, j])``."""
    gx = np.empty_like(u)
    gy = np.empty_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gx[:, -1] = u[:, 0] - u[:, -1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    gy[-1, :] = u[0, :] - u[-1, :]
    return gx, gy


def divergence(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Adjoint of :func:`gradient` with a negative sign (``div = -grad^T``)."""
    div = np.array(px, copy=True)
    div[:, 1:] -= px[:, :-1]
    div[:, 0] -= px[:, -1]
    div += py
    div[1:, :] -= py[:-1, :]
    div[0, :] -= py[-1, :]
    return div


def tv_norm(u: np.ndarray) -> float:
    """Isotropic total variation ``sum_i ||grad u_i||_2``."""
    gx, gy = gradient(u)
    return float(np.sum(np.sqrt(gx * gx + gy * gy)))


def _gradient_spectra(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    H, W = shape
    hx = np.zeros((H, W))
    hx[0, 0] = -1.0
    hx[0, W - 1] = 1.0
    hy = np.zeros((H, W))
    hy[0, 0] = -1.0
    hy[H - 1, 0] = 1.0
    return np.fft.fft2(hx), np.fft.fft2(hy)


def tv_denoise_admm(
    f: np.ndarray,
    weight: float,
    n_iter: int = 50,
    rho: float = 1.0,
) -> np.ndarray:
    """Solve the ROF problem (isotropic TV) by ADMM.

    ``weight`` is the TV regularization strength; ``rho`` is the ADMM penalty.
    The problem is solved on a periodic domain.
    """
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {f.shape}")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    if weight == 0.0:
        return f.copy()

    H, W = f.shape
    Fx, Fy = _gradient_spectra((H, W))
    F = np.fft.fft2(f)
    denom = 1.0 + rho * (np.abs(Fx) ** 2 + np.abs(Fy) ** 2)

    dx = np.zeros_like(f)
    dy = np.zeros_like(f)
    bx = np.zeros_like(f)
    by = np.zeros_like(f)
    threshold = weight / rho
    u = f.copy()

    for _ in range(n_iter):
        rhs = F + rho * (
            np.conj(Fx) * np.fft.fft2(dx - bx) + np.conj(Fy) * np.fft.fft2(dy - by)
        )
        u = np.fft.ifft2(rhs / denom).real

        gx, gy = gradient(u)
        vx = gx + bx
        vy = gy + by
        magnitude = np.sqrt(vx * vx + vy * vy)
        shrink = np.maximum(0.0, 1.0 - threshold / np.maximum(magnitude, 1e-12))
        dx = vx * shrink
        dy = vy * shrink
        bx = bx + gx - dx
        by = by + gy - dy

    return u
