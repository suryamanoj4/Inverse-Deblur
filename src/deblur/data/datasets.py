"""Dataset loading and patch sampling.

The core protocol uses BSDS500 with the standard directory split
(train / val / test) and luminance images. Splits are defined by image, and
patches are cropped only after an image has been assigned to a split, so test
content cannot leak into training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .degrade import to_luminance

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def resolve_image_root(root: str | Path) -> Path:
    """Locate the directory containing ``train/``, ``val/`` and ``test/``."""
    root = Path(root)
    candidates = [
        root / "BSDS500" / "data" / "images",
        root / "BSR" / "BSDS500" / "data" / "images",
        root / "data" / "images",
        root,
    ]
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in SPLITS):
            return candidate
    raise FileNotFoundError(
        f"could not find train/val/test under {root}; expected BSDS500 layout "
        "('<root>/BSDS500/data/images/{train,val,test}')"
    )


def list_images(directory: str | Path) -> list[Path]:
    """Sorted list of image files in ``directory`` (recursive)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def load_image(path: str | Path, luminance: bool = True) -> np.ndarray:
    """Load an image as float64 in [0, 1], optionally converted to luminance."""
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required to load images")
    with Image.open(path) as im:
        array = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return to_luminance(array) if luminance else array


def sample_patches(
    image: np.ndarray,
    patch_size: int,
    count: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample ``count`` random patches of shape (patch_size, patch_size)."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {image.shape}")
    if patch_size <= 0 or count <= 0:
        raise ValueError("patch_size and count must be positive")
    H, W = image.shape
    if patch_size > H or patch_size > W:
        raise ValueError(f"patch size {patch_size} exceeds image {(H, W)}")
    rng = np.random.default_rng() if rng is None else rng
    patches = np.empty((count, patch_size, patch_size), dtype=image.dtype)
    for i in range(count):
        top = int(rng.integers(0, H - patch_size + 1))
        left = int(rng.integers(0, W - patch_size + 1))
        patches[i] = image[top : top + patch_size, left : left + patch_size]
    return patches


class BSDS500:
    """BSDS500 split loader (directory-based standard split)."""

    def __init__(self, root: str | Path, split: str = "train", luminance: bool = True) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.image_root = resolve_image_root(root)
        self.split = split
        self.luminance = luminance
        self.paths = list_images(self.image_root / split)
        if not self.paths:
            raise FileNotFoundError(f"no images in split {split!r} under {self.image_root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return load_image(self.paths[index], luminance=self.luminance)

    def __iter__(self):
        for path in self.paths:
            yield load_image(path, luminance=self.luminance)


def split_manifest(root: str | Path) -> dict[str, list[str]]:
    """Relative paths for every image in every split, for reproducibility logs."""
    image_root = resolve_image_root(root)
    return {
        split: [p.relative_to(image_root).as_posix() for p in list_images(image_root / split)]
        for split in SPLITS
    }
