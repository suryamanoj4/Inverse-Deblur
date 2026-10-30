"""Stability metrics and PGD attacks for H3.

The experiment searches for a small perturbation ``eta`` of the observation
``y`` that maximizes the reconstruction error of a model, under an L2 budget:

    eta* = argmax_{||eta||_2 <= radius} ||g(y + eta) - x||_2^2

with ``y + eta`` clipped to [0, 1]. The NumPy helpers are used by the
reporting code; :func:`pgd_attack_torch` is used for differentiable models.
"""

from __future__ import annotations

import numpy as np


def project_l2_ball(eta: np.ndarray, radius: float) -> np.ndarray:
    """Project ``eta`` onto the L2 ball of the given radius (NumPy)."""
    eta = np.asarray(eta, dtype=np.float64)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    norm = float(np.linalg.norm(eta))
    if radius == 0.0 or norm == 0.0:
        return np.zeros_like(eta)
    if norm <= radius:
        return eta.copy()
    return eta * (radius / norm)


def relative_l2(eta: np.ndarray, reference: np.ndarray) -> float:
    """``||eta||_2 / ||reference||_2`` (0 when both are zero)."""
    eta = np.asarray(eta, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(eta))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def perturbation_radius(reference: np.ndarray, eps_rel: float) -> float:
    """Absolute L2 radius for a relative budget ``eps_rel``."""
    if eps_rel < 0:
        raise ValueError("eps_rel must be non-negative")
    return float(eps_rel * np.linalg.norm(np.asarray(reference, dtype=np.float64)))


def _project_l2_ball_torch(eta, radius: float):
    import torch

    norm = torch.linalg.vector_norm(eta)
    if radius <= 0.0 or float(norm) == 0.0:
        return torch.zeros_like(eta)
    if float(norm) <= radius:
        return eta
    return eta * (radius / norm)


def pgd_attack_torch(
    forward,
    y,
    target,
    radius: float,
    steps: int = 20,
    step_size: float | None = None,
):
    """Projected gradient ascent on ``||forward(y+eta) - target||^2``.

    ``forward`` must be differentiable and return a tensor with the same shape
    as ``target``. ``radius`` is the absolute L2 budget; ``y + eta`` is clipped
    to [0, 1] after every step.
    """
    import torch
    import torch.nn.functional as F

    if radius < 0:
        raise ValueError("radius must be non-negative")
    if steps <= 0 or radius == 0.0:
        return torch.zeros_like(y)
    step = 2.0 * radius / steps if step_size is None else float(step_size)
    if step <= 0:
        raise ValueError("step_size must be positive")

    eta = torch.zeros_like(y)
    for _ in range(steps):
        eta = eta.detach().requires_grad_(True)
        y_adv = torch.clamp(y + eta, 0.0, 1.0)
        output = forward(y_adv)
        if output.shape != target.shape:
            raise ValueError(
                f"forward returned {tuple(output.shape)}, expected {tuple(target.shape)}"
            )
        loss = F.mse_loss(output, target)
        gradient = torch.autograd.grad(loss, eta)[0]
        with torch.no_grad():
            normaliser = torch.linalg.vector_norm(gradient)
            eta = eta + step * gradient / (normaliser + 1e-12)
            eta = _project_l2_ball_torch(eta, radius)
            eta = torch.clamp(y + eta, 0.0, 1.0) - y
    return eta.detach()
