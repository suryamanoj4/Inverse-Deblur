"""Extra training images (synthetic or DIV2K) and their quality checks.

Phase 2 enlarges the *training* pool only; validation and test stay BSDS500.
Every extra image goes through the same preprocessing as BSDS500: resized
(anti-aliased) and centre-cropped to 481x321 or 321x481, keeping its
orientation, and converted to Rec.601 luminance.

A frozen extra-image set is a directory of images plus ``manifest.json``
(written by ``scripts/check_synthetic.py``) listing the accepted files and
their SHA-256. When a manifest is present, only the listed files are loaded
and every hash is verified, so a set cannot change silently.

The quality-control primitives used by ``scripts/check_synthetic.py`` live here
so they can be unit-tested: sharpness (variance of the Laplacian), entropy, a
DCT perceptual hash, and a nearest-neighbour leakage check against BSDS val/test.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .datasets import IMAGE_SUFFIXES, list_images
from .degrade import to_luminance

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None

BSDS_SHAPE = (321, 481)  # (H, W) of a landscape BSDS500 image
MANIFEST = "manifest.json"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_size(rgb: np.ndarray, shape: tuple[int, int] = BSDS_SHAPE) -> np.ndarray:
    """Resize (Lanczos) and centre-crop to BSDS500 size, keeping orientation.

    ``rgb`` is (H, W[, C]) in [0, 1]. A portrait image becomes 481x321, a
    landscape one 321x481. Images are only ever shrunk or matched; upsampling
    would invent detail, so images smaller than the target are rejected.
    """
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required")
    h, w = rgb.shape[:2]
    short, long = shape
    target = (long, short) if h > w else (short, long)  # (H, W)
    scale = max(target[0] / h, target[1] / w)
    if scale > 1.0 + 1e-9:
        raise ValueError(f"image {h}x{w} is smaller than {target[0]}x{target[1]}")
    new_h, new_w = max(target[0], round(h * scale)), max(target[1], round(w * scale))
    mode_image = Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8))
    resized = np.asarray(mode_image.resize((new_w, new_h), Image.LANCZOS), dtype=np.float64) / 255.0
    top, left = (new_h - target[0]) // 2, (new_w - target[1]) // 2
    return resized[top : top + target[0], left : left + target[1]]


def load_extra_image(path: str | Path, luminance: bool = True) -> np.ndarray:
    """Load one extra image with the BSDS500 preprocessing."""
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required")
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    image = normalize_size(rgb)
    return to_luminance(image) if luminance else image


@dataclass
class ImageSource:
    """A set of extra training images, with what is needed to record it."""

    name: str
    directory: str
    ids: list[str]
    images: list[np.ndarray] = field(repr=False)
    manifest_sha256: str | None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "directory": self.directory,
            "count": len(self.ids),
            "manifest_sha256": self.manifest_sha256,
        }


def load_image_source(
    directory: str | Path,
    name: str | None = None,
    luminance: bool = True,
    require_manifest: bool = False,
) -> ImageSource:
    """Load an extra-image directory (synthetic set or DIV2K).

    With ``manifest.json`` present, only its accepted files are loaded and each
    SHA-256 is checked. ``require_manifest`` refuses a directory without one,
    which is how frozen synthetic sets are loaded for reportable runs.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    manifest_sha = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest_sha = sha256_file(manifest_path)
        entries = manifest.get("accepted", manifest.get("images", []))
        paths = []
        for entry in entries:
            path = directory / entry["file"]
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{path} does not match its manifest hash")
            paths.append(path)
        name = name or manifest.get("name")
    elif require_manifest:
        raise FileNotFoundError(f"{manifest_path} not found; run scripts/check_synthetic.py first")
    else:
        paths = list_images(directory)
    if not paths:
        raise FileNotFoundError(f"no images in {directory}")
    return ImageSource(
        name=name or directory.name,
        directory=str(directory),
        ids=[p.relative_to(directory).as_posix() for p in paths],
        images=[load_extra_image(p, luminance=luminance) for p in paths],
        manifest_sha256=manifest_sha,
    )


def nested_subset(paths: list[Path], size: int, seed: int = 0) -> list[Path]:
    """The first ``size`` paths of one seeded permutation of the sorted list.

    Subsets are nested (the 50-image subset is inside the 100-image one), which
    a learning curve needs: each point adds images rather than swapping them.
    """
    ordered = sorted(Path(p) for p in paths)
    if not 0 < size <= len(ordered):
        raise ValueError(f"subset size must be in [1, {len(ordered)}], got {size}")
    random.Random(seed).shuffle(ordered)
    return sorted(ordered[:size])


# ---------------------------------------------------------------------------
# quality control
# ---------------------------------------------------------------------------
def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian: low for soft or blurry images."""
    x = np.asarray(image, dtype=np.float64)
    lap = (
        -4.0 * x[1:-1, 1:-1] + x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]
    )
    return float(lap.var())


def entropy(image: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy (bits) of the 8-bit intensity histogram."""
    hist, _ = np.histogram(np.asarray(image), bins=bins, range=(0.0, 1.0))
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def perceptual_hash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """DCT perceptual hash (pHash): ``hash_size**2`` bits as a boolean array."""
    from scipy.fft import dctn

    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required")
    small = Image.fromarray(np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8))
    side = hash_size * 4
    pixels = np.asarray(small.resize((side, side), Image.LANCZOS), dtype=np.float64)
    low = dctn(pixels, norm="ortho")[:hash_size, :hash_size]
    # median of the low frequencies without the DC term, as in pHash
    return (low > np.median(low.ravel()[1:])).ravel()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


def nearest_hash(query: np.ndarray, references: list[np.ndarray]) -> tuple[int, int]:
    """(index, distance) of the reference hash closest to ``query``."""
    if not references:
        raise ValueError("no reference hashes")
    distances = [hamming(query, ref) for ref in references]
    index = int(np.argmin(distances))
    return index, distances[index]


def gradient_histogram(image: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Normalised histogram of gradient magnitudes (natural-image statistic)."""
    gy, gx = np.gradient(np.asarray(image, dtype=np.float64))
    hist, _ = np.histogram(np.hypot(gx, gy), bins=bins)
    return hist / max(hist.sum(), 1)


def spectrum_slope(image: np.ndarray) -> float:
    """Slope of log radial power vs log frequency (about -2 for natural images)."""
    x = np.asarray(image, dtype=np.float64)
    x = x - x.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    h, w = x.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    radial = np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    f = np.arange(len(radial))
    keep = (f >= 2) & (f <= min(h, w) // 2) & (radial > 0)
    return float(np.polyfit(np.log(f[keep]), np.log(radial[keep]), 1)[0])


__all__ = [
    "BSDS_SHAPE",
    "IMAGE_SUFFIXES",
    "ImageSource",
    "MANIFEST",
    "entropy",
    "gradient_histogram",
    "hamming",
    "laplacian_variance",
    "load_extra_image",
    "load_image_source",
    "nearest_hash",
    "nested_subset",
    "normalize_size",
    "perceptual_hash",
    "sha256_file",
    "spectrum_slope",
]
