"""Image-quality metrics.

PSNR and SSIM are dependency-light (NumPy/SciPy). LPIPS is optional and requires
``torch`` and ``lpips``; the import is deferred so the rest of the package works
without them.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def psnr(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB (``inf`` for identical inputs)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    mse = float(np.mean((x - y) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


def ssim(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 1.0,
    gaussian_sigma: float = 1.5,
    truncate: float = 3.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> float:
    """Structural similarity with an 11x11 Gaussian window (Wang et al., 2004)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim != 2:
        raise ValueError("ssim expects 2D (luminance) images")
    truncate = truncate if truncate > 0 else 3.5
    filt = lambda z: gaussian_filter(z, sigma=gaussian_sigma, truncate=truncate, mode="reflect")
    mu_x = filt(x)
    mu_y = filt(y)
    sigma_x2 = filt(x * x) - mu_x**2
    sigma_y2 = filt(y * y) - mu_y**2
    sigma_xy = filt(x * y) - mu_x * mu_y
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(numerator / denominator))


def _to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 4:
        return image[..., :3]
    if image.ndim == 3 and image.shape[-1] == 3:
        return image
    raise ValueError(f"expected (H, W), (H, W, 3) or (H, W, 4), got {image.shape}")


class LPIPSMetric:
    """LPIPS distance with a cached network; requires optional dependencies."""

    def __init__(self, net: str = "alex", device: str | None = None) -> None:
        try:
            import torch
            import lpips
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LPIPSMetric requires torch and lpips; install with "
                "'pip install torch lpips'"
            ) from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = lpips.LPIPS(net=net).to(self.device).eval()

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        x_rgb = _to_rgb(x)
        y_rgb = _to_rgb(y)
        if x_rgb.shape != y_rgb.shape:
            raise ValueError(f"shape mismatch: {x_rgb.shape} vs {y_rgb.shape}")
        torch = self._torch
        tx = torch.from_numpy(x_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        ty = torch.from_numpy(y_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tx = tx * 2.0 - 1.0
        ty = ty * 2.0 - 1.0
        with torch.no_grad():
            return float(self._model(tx, ty).item())


def lpips_distance(x: np.ndarray, y: np.ndarray, net: str = "alex") -> float:
    """Convenience wrapper around :class:`LPIPSMetric` (creates a model each call)."""
    return LPIPSMetric(net=net)(x, y)
