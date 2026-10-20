"""PSF library: Gaussian, motion (Levin) and defocus kernels.

All kernels are returned normalised to unit mass. Motion kernels can be loaded
from the official Levin et al. (2009) set, or generated procedurally as a
fallback so the pipeline is runnable before the files are downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # Pillow is a core dependency, but keep the import local to I/O paths
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None


@dataclass(frozen=True)
class PSF:
    """A named point-spread function."""

    name: str
    kernel: np.ndarray
    kind: str

    def __post_init__(self) -> None:
        k = np.asarray(self.kernel, dtype=np.float64)
        if k.ndim != 2:
            raise ValueError(f"PSF kernel must be 2D, got shape {k.shape}")
        object.__setattr__(self, "kernel", k)


def _normalise(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    s = kernel.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("kernel must have positive finite mass")
    return kernel / s


def gaussian_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    """Isotropic Gaussian PSF, normalised to unit mass."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if size is None:
        size = 2 * int(np.ceil(3.0 * sigma)) + 1
    if size % 2 == 0:
        size += 1
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    return _normalise(np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)))


def disk_kernel(radius: float, size: int | None = None, supersample: int = 8) -> np.ndarray:
    """Anti-aliased defocus disk, normalised to unit mass."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if size is None:
        size = 2 * int(np.ceil(radius)) + 1
    if size % 2 == 0:
        size += 1
    n = size * supersample
    coords = (np.arange(n, dtype=np.float64) + 0.5) / supersample - size / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    mask = (xx**2 + yy**2) <= radius**2
    blocks = mask.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return _normalise(blocks)


def motion_kernel(
    length: float,
    angle_deg: float,
    thickness: float = 1.5,
    size: int | None = None,
    supersample: int = 8,
) -> np.ndarray:
    """Anti-aliased line-segment motion PSF, normalised to unit mass."""
    if length <= 0 or thickness <= 0:
        raise ValueError("length and thickness must be positive")
    if size is None:
        size = 2 * int(np.ceil(length / 2.0 + thickness)) + 1
    if size % 2 == 0:
        size += 1
    n = size * supersample
    coords = (np.arange(n, dtype=np.float64) + 0.5) / supersample - size / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    theta = np.deg2rad(angle_deg)
    # rotate into the line frame: u along the motion, v across it
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = -xx * np.sin(theta) + yy * np.cos(theta)
    mask = (np.abs(u) <= length / 2.0) & (np.abs(v) <= thickness / 2.0)
    blocks = mask.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    if blocks.sum() == 0:  # pragma: no cover - pathological parameters
        raise ValueError("motion kernel is empty; increase thickness or size")
    return _normalise(blocks)


def default_motion_kernels(
    lengths: tuple[float, ...] = (9.0, 15.0),
    angles: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
    thickness: float = 1.5,
) -> list[PSF]:
    """Procedural motion kernels used when the Levin files are unavailable."""
    return [
        PSF(
            name=f"motion_l{int(length)}_a{int(angle)}",
            kernel=motion_kernel(length, angle, thickness=thickness),
            kind="motion",
        )
        for length in lengths
        for angle in angles
    ]


def _load_kernel_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    if suffix == ".mat":
        from scipy.io import loadmat

        data = loadmat(path)
        candidates = [
            v
            for v in data.values()
            if isinstance(v, np.ndarray)
            and v.ndim == 2
            and np.issubdtype(v.dtype, np.number)
        ]
        if not candidates:
            raise ValueError(f"no 2D numeric array found in {path}")
        return np.asarray(candidates[0], dtype=np.float64)
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required to read image kernels")
    with Image.open(path) as im:
        return np.asarray(im.convert("F"), dtype=np.float64)


def load_levin_kernels(directory: str | Path) -> list[PSF]:
    """Load the official Levin et al. motion kernels from ``directory``.

    Accepts ``.npy``, ``.mat`` and common image formats. Files are sorted by
    name so the kernel order is deterministic.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Levin kernel directory not found: {directory}")
    suffixes = {".npy", ".mat", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"no kernel files found in {directory}")
    return [
        PSF(name=p.stem, kernel=_normalise(_load_kernel_file(p)), kind="motion")
        for p in paths
    ]


def build_psf_library(
    gaussian_sigmas: tuple[float, ...] = (0.7, 1.2, 1.6, 2.0),
    defocus_radii: tuple[float, ...] = (3.0, 5.0, 7.0, 9.0),
    motion_kernels: list[PSF] | None = None,
    levin_dir: str | Path | None = None,
) -> list[PSF]:
    """Build the shared PSF library for the protocol.

    If ``motion_kernels`` is omitted, the official Levin set is loaded when
    ``levin_dir`` is given; otherwise procedural motion kernels are used.
    """
    psfs: list[PSF] = []
    for sigma in gaussian_sigmas:
        psfs.append(
            PSF(name=f"gaussian_sigma{sigma:g}", kernel=gaussian_kernel(sigma), kind="gaussian")
        )
    for radius in defocus_radii:
        psfs.append(
            PSF(name=f"defocus_r{radius:g}", kernel=disk_kernel(radius), kind="defocus")
        )
    if motion_kernels is None:
        motion_kernels = (
            load_levin_kernels(levin_dir) if levin_dir is not None else default_motion_kernels()
        )
    psfs.extend(motion_kernels)
    return psfs
