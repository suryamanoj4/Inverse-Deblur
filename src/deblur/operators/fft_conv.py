"""FFT-based circular convolution operators.

The forward model used throughout the study is

    y = A x + eps,    A = circular convolution with PSF k.

With circular boundary handling, A is diagonalized by the 2D DFT, so both the
operator and its adjoint cost two FFTs. The kernel is zero-padded to the image
shape and rolled so that its centre sits at index (0, 0); for even-sized kernels
the centre is defined as (kh // 2, kw // 2).

NumPy functions are used by the classical solvers. The PyTorch module is for the
learned arms and is optional: importing this file does not require torch.
"""

from __future__ import annotations

import numpy as np

try:  # optional dependency, only needed for the deep arms
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------
def pad_kernel(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad a 2D kernel to ``shape`` with its centre at (0, 0)."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D, got shape {kernel.shape}")
    H, W = shape
    kh, kw = kernel.shape
    if kh > H or kw > W:
        raise ValueError(f"kernel {kernel.shape} does not fit in image {(H, W)}")
    out = np.zeros((H, W), dtype=kernel.dtype)
    out[:kh, :kw] = kernel
    return np.roll(out, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))


def kernel_spectrum(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """DFT of the padded kernel, i.e. the eigenvalues of the operator."""
    return np.fft.fft2(pad_kernel(kernel, shape))


def fft_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular convolution of ``image`` (..., H, W) with ``kernel`` (kh, kw)."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    H, W = image.shape[-2:]
    K = kernel_spectrum(kernel, (H, W))
    return np.fft.ifft2(np.fft.fft2(image, axes=(-2, -1)) * K).real


def fft_adjoint(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Adjoint of :func:`fft_convolve` (correlation with the PSF)."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    H, W = image.shape[-2:]
    K = kernel_spectrum(kernel, (H, W))
    return np.fft.ifft2(np.fft.fft2(image, axes=(-2, -1)) * np.conj(K)).real


# ---------------------------------------------------------------------------
# PyTorch (optional)
# ---------------------------------------------------------------------------
def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the differentiable operator. "
            "Install it with: pip install torch"
        )


def _pad_kernel_torch(kernel, H: int, W: int):
    kh, kw = kernel.shape[-2:]
    if kh > H or kw > W:
        raise ValueError(f"kernel {tuple(kernel.shape)} does not fit in image {(H, W)}")
    out = kernel.new_zeros((H, W))
    out[:kh, :kw] = kernel
    return torch.roll(out, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))


def fft_convolve_torch(x, kernel):
    """Differentiable circular convolution for tensors of shape (..., H, W)."""
    _require_torch()
    H, W = x.shape[-2:]
    K = torch.fft.fft2(_pad_kernel_torch(kernel.to(x.dtype), H, W))
    return torch.fft.ifft2(torch.fft.fft2(x) * K).real


def fft_adjoint_torch(x, kernel):
    """Differentiable adjoint of :func:`fft_convolve_torch`."""
    _require_torch()
    H, W = x.shape[-2:]
    K = torch.fft.fft2(_pad_kernel_torch(kernel.to(x.dtype), H, W))
    return torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(K)).real


if torch is not None:

    class FFTConv2d(nn.Module):
        """Cached circular-convolution operator as an ``nn.Module``.

        The kernel is a buffer (not a parameter): it is fixed per degradation,
        as in USRNet's data module. The kernel spectrum is cached per image
        shape/device/dtype.
        """

        def __init__(self, kernel, cache: bool = True) -> None:
            super().__init__()
            k = np.asarray(kernel, dtype=np.float64)
            if k.ndim != 2:
                raise ValueError(f"kernel must be 2D, got shape {k.shape}")
            self.register_buffer("kernel", torch.as_tensor(k, dtype=torch.float32))
            self.cache = cache
            self._spectra: dict = {}

        def _spectrum(self, H: int, W: int, dtype):
            key = (H, W, self.kernel.device.type, dtype)
            if not self.cache or key not in self._spectra:
                spec = torch.fft.fft2(_pad_kernel_torch(self.kernel.to(dtype), H, W))
                if self.cache:
                    self._spectra[key] = spec
            return self._spectra[key]

        def forward(self, x):
            H, W = x.shape[-2:]
            K = self._spectrum(H, W, x.dtype)
            return torch.fft.ifft2(torch.fft.fft2(x) * K).real

        def adjoint(self, x):
            H, W = x.shape[-2:]
            K = self._spectrum(H, W, x.dtype)
            return torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(K)).real

else:  # pragma: no cover - torch absent
    FFTConv2d = None
