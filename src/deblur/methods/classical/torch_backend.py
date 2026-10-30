"""PyTorch (GPU) versions of the iterative M2 solvers.

These mirror :func:`fista_tv` and :func:`admm_epll` operation for operation, so
the method is unchanged; only the arithmetic runs on the GPU. FFTs and TV are
computed in float64 like the NumPy code. The GMM patch step uses float32
matrix products with per-noise-level factors computed in float64. The tests
check both solvers against the NumPy reference.

Importing this module requires torch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ...operators.fft_conv import kernel_spectrum
from .admm_epll import SIGMA_FLOOR, SIGMA_START, epll_schedule, suggest_epll_lambda
from .gmm import GMM


def _gradient(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.roll(u, -1, dims=1) - u, torch.roll(u, -1, dims=0) - u


def _gradient_spectra(shape: tuple[int, int], device) -> tuple[torch.Tensor, torch.Tensor]:
    H, W = shape
    hx = torch.zeros((H, W), dtype=torch.float64, device=device)
    hx[0, 0] = -1.0
    hx[0, W - 1] = 1.0
    hy = torch.zeros((H, W), dtype=torch.float64, device=device)
    hy[0, 0] = -1.0
    hy[H - 1, 0] = 1.0
    return torch.fft.fft2(hx), torch.fft.fft2(hy)


def _tv_denoise_admm(f, weight, n_iter, rho, Fx, Fy, denom):
    """ROF prox by ADMM; mirrors :func:`deblur.methods.classical.tv.tv_denoise_admm`."""
    if weight == 0.0:
        return f.clone()
    Ff = torch.fft.fft2(f)
    dx = torch.zeros_like(f)
    dy = torch.zeros_like(f)
    bx = torch.zeros_like(f)
    by = torch.zeros_like(f)
    threshold = weight / rho
    u = f
    for _ in range(n_iter):
        rhs = Ff + rho * (
            torch.conj(Fx) * torch.fft.fft2(dx - bx) + torch.conj(Fy) * torch.fft.fft2(dy - by)
        )
        u = torch.fft.ifft2(rhs / denom).real
        gx, gy = _gradient(u)
        vx = gx + bx
        vy = gy + by
        magnitude = torch.sqrt(vx * vx + vy * vy)
        shrink = torch.clamp(1.0 - threshold / torch.clamp(magnitude, min=1e-12), min=0.0)
        dx = vx * shrink
        dy = vy * shrink
        bx = bx + gx - dx
        by = by + gy - dy
    return u


@torch.inference_mode()
def fista_tv_torch(
    y: np.ndarray,
    kernel: np.ndarray,
    lam: float,
    n_iter: int = 100,
    tv_iter: int = 30,
    tv_rho: float = 1.0,
    clip: bool = True,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """GPU version of :func:`deblur.methods.classical.fista_tv.fista_tv`."""
    y_np = np.asarray(y, dtype=np.float64)
    if y_np.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {y_np.shape}")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    H, W = y_np.shape
    K = torch.as_tensor(kernel_spectrum(kernel, (H, W)), device=device)
    lipschitz = float(torch.max(torch.abs(K) ** 2))
    if lipschitz <= 0:
        raise ValueError("kernel has zero spectrum; operator is not invertible")
    step = 1.0 / lipschitz
    Fx, Fy = _gradient_spectra((H, W), device)
    denom = 1.0 + tv_rho * (torch.abs(Fx) ** 2 + torch.abs(Fy) ** 2)

    y_t = torch.as_tensor(y_np, device=device)
    x = y_t.clone()
    z = x.clone()
    t = 1.0
    for _ in range(n_iter):
        residual = torch.fft.ifft2(torch.fft.fft2(z) * K).real - y_t
        grad = torch.fft.ifft2(torch.fft.fft2(residual) * torch.conj(K)).real
        x_new = _tv_denoise_admm(z - step * grad, step * lam, tv_iter, tv_rho, Fx, Fy, denom)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    if clip:
        x = torch.clamp(x, 0.0, 1.0)
    return x.cpu().numpy()


class TorchGMM:
    """A :class:`GMM` held on the GPU, with the EPLL MAP patch denoiser."""

    def __init__(self, gmm: GMM, device: str | torch.device = "cuda", chunk: int = 16) -> None:
        self.device = torch.device(device)
        self.chunk = chunk
        self.cov = torch.as_tensor(gmm.covariances, dtype=torch.float64, device=self.device)
        self.means64 = torch.as_tensor(gmm.means, dtype=torch.float64, device=self.device)
        self.means = self.means64.float()
        self.log_weights = torch.log(torch.as_tensor(gmm.weights, device=self.device)).float()
        self.dim = gmm.dim
        self._cache: dict[float, tuple] = {}

    def _factors(self, sigma: float):
        """Whitening, log-determinant and Wiener factors for ``Sigma_k + sigma^2 I``."""
        if sigma not in self._cache:
            eye = torch.eye(self.dim, dtype=torch.float64, device=self.device)
            cov = self.cov + (sigma**2) * eye
            L = torch.linalg.cholesky(cov)
            whiten = torch.linalg.solve_triangular(L, eye.expand_as(cov), upper=False)
            logdet = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)
            # x_hat = (Sigma + s^2 I)^-1 (Sigma x + s^2 mu) = x A^T + b
            A = torch.linalg.solve(cov, self.cov)
            b = (sigma**2) * torch.linalg.solve(cov, self.means64.unsqueeze(-1)).squeeze(-1)
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[sigma] = (
                whiten.float(),
                logdet.float(),
                A.transpose(1, 2).float(),
                b.float(),
            )
        return self._cache[sigma]

    def map_denoise(self, X: torch.Tensor, sigma: float) -> torch.Tensor:
        """Mirror of :meth:`GMM.map_denoise` for float32 patches (N, D)."""
        whiten, logdet, A_t, b = self._factors(float(sigma))
        K = whiten.shape[0]
        scores = torch.empty((X.shape[0], K), device=self.device)
        for start in range(0, K, self.chunk):
            stop = min(start + self.chunk, K)
            diff = X.unsqueeze(0) - self.means[start:stop, None, :]
            z = torch.bmm(diff, whiten[start:stop].transpose(1, 2))
            scores[:, start:stop] = (-0.5 * (logdet[start:stop, None] + (z * z).sum(-1))).T
        labels = torch.argmax(scores + self.log_weights, dim=1)
        out = torch.empty_like(X)
        order = torch.argsort(labels)
        counts = torch.bincount(labels, minlength=K).tolist()
        offset = 0
        for k, count in enumerate(counts):
            if count:
                idx = order[offset : offset + count]
                out[idx] = X[idx] @ A_t[k] + b[k]
                offset += count
        return out


def _patch_positions(size: int, patch_size: int, stride: int) -> list[int]:
    positions = list(range(0, size - patch_size + 1, stride))
    if positions[-1] != size - patch_size:
        positions.append(size - patch_size)
    return positions


def patch_denoise_torch(
    image: torch.Tensor,
    gmm: TorchGMM,
    sigma: float,
    patch_size: int = 8,
    stride: int = 1,
) -> torch.Tensor:
    """Mirror of :func:`deblur.methods.classical.gmm.patch_denoise` (float64 image)."""
    H, W = image.shape
    ys = _patch_positions(H, patch_size, stride)
    xs = _patch_positions(W, patch_size, stride)
    n_x = W - patch_size + 1
    columns = (
        torch.as_tensor(ys, device=image.device)[:, None] * n_x
        + torch.as_tensor(xs, device=image.device)[None, :]
    ).reshape(-1)

    all_patches = F.unfold(image[None, None].float(), patch_size)[0]  # (D, L)
    patches = all_patches[:, columns].T.contiguous()  # (N, D), row-major like NumPy
    dc = patches.mean(dim=1, keepdim=True)
    patches = gmm.map_denoise(patches - dc, sigma) + dc

    full = torch.zeros_like(all_patches)
    full[:, columns] = patches.T
    mask = torch.zeros_like(all_patches)
    mask[:, columns] = 1.0
    output = F.fold(full[None], (H, W), patch_size)[0, 0].double()
    weights = F.fold(mask[None], (H, W), patch_size)[0, 0].double()
    return output / torch.clamp(weights, min=1.0)


@torch.inference_mode()
def admm_epll_torch(
    y: np.ndarray,
    kernel: np.ndarray,
    gmm: TorchGMM,
    sigma: float = 0.0,
    lam: float | None = None,
    n_iter: int = 10,
    sigma_start: float = SIGMA_START,
    sigma_floor: float = SIGMA_FLOOR,
    patch_size: int = 8,
    stride: int = 1,
    clip: bool = True,
) -> np.ndarray:
    """GPU version of :func:`deblur.methods.classical.admm_epll.admm_epll`."""
    y_np = np.asarray(y, dtype=np.float64)
    if y_np.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {y_np.shape}")
    schedule = epll_schedule(sigma, n_iter, sigma_start, sigma_floor)
    if lam is None:
        lam = suggest_epll_lambda(sigma, sigma_floor)
    if lam <= 0:
        raise ValueError("lam must be positive")

    device = gmm.device
    H, W = y_np.shape
    K = torch.as_tensor(kernel_spectrum(kernel, (H, W)), device=device)
    y_t = torch.as_tensor(y_np, device=device)
    KY = torch.conj(K) * torch.fft.fft2(y_t)
    denominator = torch.abs(K) ** 2
    z = y_t.clone()
    for s in schedule:
        beta = lam / s**2
        x = torch.fft.ifft2((KY + beta * torch.fft.fft2(z)) / (denominator + beta)).real
        z = patch_denoise_torch(x, gmm, float(s), patch_size=patch_size, stride=stride)
    if clip:
        z = torch.clamp(z, 0.0, 1.0)
    return z.cpu().numpy()
