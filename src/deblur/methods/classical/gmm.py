"""Gaussian-mixture patch prior (EPLL) and its MAP denoiser.

Natural image patches are modelled as a finite mixture of zero-mean Gaussians
(Zoran & Weiss, 2011). The DC component is removed before fitting, so the model
describes patch structure rather than absolute brightness.

The MAP estimate for a noisy patch under component ``k`` is a Wiener filter:

    x_hat = (Sigma_k + sigma^2 I)^-1 (Sigma_k y + sigma^2 mu_k).

EPLL approximates the mixture MAP by first selecting the most probable
component and then applying that component's Wiener filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.linalg import cholesky, solve, solve_triangular
from scipy.special import logsumexp


@dataclass
class GMM:
    """Zero-mean Gaussian mixture with full covariances."""

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.means = np.asarray(self.means, dtype=np.float64)
        self.covariances = np.asarray(self.covariances, dtype=np.float64)
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D")
        K = self.weights.size
        if self.means.shape != (K, self.covariances.shape[1]):
            raise ValueError("means and covariances have inconsistent shapes")
        if self.covariances.shape != (K, self.means.shape[1], self.means.shape[1]):
            raise ValueError("covariances must be (K, D, D)")
        if np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("weights must be non-negative and sum to one")

    @property
    def n_components(self) -> int:
        return int(self.weights.size)

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def _log_component_prob(self, X: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        """Log density of every component at every sample, shape (N, K)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"expected (N, {self.dim}) samples, got {X.shape}")
        N, D = X.shape
        eye = np.eye(D)
        out = np.empty((N, self.n_components))
        for k in range(self.n_components):
            cov = self.covariances[k] + (sigma**2) * eye
            L = cholesky(cov, lower=True, check_finite=False)
            diff = X - self.means[k]
            z = solve_triangular(L, diff.T, lower=True, check_finite=False)
            quad = np.einsum("ij,ij->j", z, z)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            out[:, k] = -0.5 * (D * np.log(2.0 * np.pi) + logdet + quad)
        return out

    def log_prob(self, X: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        """Log mixture density per sample, shape (N,)."""
        logp = self._log_component_prob(X, sigma=sigma) + np.log(self.weights)
        return logsumexp(logp, axis=1)

    def responsibilities(self, X: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        """Posterior component probabilities, shape (N, K)."""
        logp = self._log_component_prob(X, sigma=sigma) + np.log(self.weights)
        normaliser = logsumexp(logp, axis=1)
        return np.exp(logp - normaliser[:, None])

    def map_denoise(self, X: np.ndarray, sigma: float) -> np.ndarray:
        """Approximate MAP denoising of zero-mean patches at noise level ``sigma``."""
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        X = np.asarray(X, dtype=np.float64)
        labels = np.argmax(
            self._log_component_prob(X, sigma=sigma) + np.log(self.weights), axis=1
        )
        out = np.empty_like(X)
        eye = np.eye(self.dim)
        for k in range(self.n_components):
            idx = np.flatnonzero(labels == k)
            if idx.size == 0:
                continue
            cov = self.covariances[k] + (sigma**2) * eye
            rhs = self.covariances[k] @ X[idx].T + (sigma**2) * self.means[k][:, None]
            out[idx] = solve(cov, rhs, assume_a="pos", check_finite=False).T
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez(
            path,
            weights=self.weights,
            means=self.means,
            covariances=self.covariances,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GMM":
        with np.load(Path(path)) as data:
            return cls(data["weights"], data["means"], data["covariances"])


def _m_step(X: np.ndarray, gamma: np.ndarray, reg: float) -> GMM:
    """One EM maximisation step from responsibilities ``gamma`` (N, K)."""
    N, D = X.shape
    K = gamma.shape[1]
    counts = gamma.sum(axis=0) + 1e-12
    weights = counts / counts.sum()
    means = (gamma.T @ X) / counts[:, None]
    covariances = np.empty((K, D, D))
    eye = np.eye(D)
    for k in range(K):
        diff = X - means[k]
        covariances[k] = (diff * gamma[:, k : k + 1]).T @ diff / counts[k] + reg * eye
    return GMM(weights=weights, means=means, covariances=covariances)


def fit_gmm(
    X: np.ndarray,
    n_components: int,
    n_iter: int = 20,
    reg: float = 1e-6,
    seed: int = 0,
    tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[GMM, list[float]]:
    """Fit a full-covariance GMM by EM, initialised with k-means++.

    ``X`` must contain zero-mean patches, shape (N, D). Returns the model and
    the log-likelihood history.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected (N, D) patches, got {X.shape}")
    N, D = X.shape
    if not 1 <= n_components <= N:
        raise ValueError("n_components must be between 1 and the number of patches")

    if n_components == 1:
        mean = X.mean(axis=0)
        covariance = np.atleast_2d(np.cov(X, rowvar=False, bias=True)) + reg * np.eye(D)
        model = GMM(np.ones(1), mean[None, :], covariance[None, :, :])
        return model, [float(model.log_prob(X).sum())]

    _, labels = kmeans2(X, n_components, minit="++", iter=10, seed=seed)
    gamma = np.zeros((N, n_components))
    gamma[np.arange(N), labels] = 1.0

    history: list[float] = []
    previous = -np.inf
    model = _m_step(X, gamma, reg)
    for _ in range(n_iter):
        logp = model._log_component_prob(X) + np.log(model.weights)
        normaliser = logsumexp(logp, axis=1)
        total = float(normaliser.sum())
        gamma = np.exp(logp - normaliser[:, None])
        history.append(total)
        if verbose:
            print(f"EM iteration {len(history):3d}  log-likelihood {total:.3f}")
        if np.isfinite(previous) and abs(total - previous) <= tol * max(1.0, abs(total)):
            break
        previous = total
        model = _m_step(X, gamma, reg)

    # Final M-step keeps the returned parameters consistent with the last E-step.
    model = _m_step(X, gamma, reg)
    return model, history


def patch_denoise(
    image: np.ndarray,
    gmm: GMM,
    sigma: float,
    patch_size: int = 8,
    stride: int = 1,
) -> np.ndarray:
    """EPLL denoising: patch-wise MAP followed by overlap averaging."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {image.shape}")
    if stride <= 0:
        raise ValueError("stride must be positive")
    H, W = image.shape
    if patch_size > H or patch_size > W:
        raise ValueError(f"patch size {patch_size} exceeds image {(H, W)}")

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    if ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    if xs[-1] != W - patch_size:
        xs.append(W - patch_size)

    count = len(ys) * len(xs)
    patches = np.empty((count, patch_size * patch_size))
    for i, top in enumerate(ys):
        for j, left in enumerate(xs):
            patches[i * len(xs) + j] = image[
                top : top + patch_size, left : left + patch_size
            ].ravel()

    dc = patches.mean(axis=1)
    patches = patches - dc[:, None]
    patches = gmm.map_denoise(patches, sigma) + dc[:, None]

    output = np.zeros_like(image)
    weights = np.zeros_like(image)
    for i, top in enumerate(ys):
        for j, left in enumerate(xs):
            output[top : top + patch_size, left : left + patch_size] += patches[
                i * len(xs) + j
            ].reshape(patch_size, patch_size)
            weights[top : top + patch_size, left : left + patch_size] += 1.0
    return output / np.maximum(weights, 1.0)
