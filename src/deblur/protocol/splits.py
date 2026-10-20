"""Split handling and manifest logging.

Splits are defined by image, never by patch, so test content cannot leak into
training. ``standard_splits`` expects the BSDS500 directory layout.
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

from ..data.datasets import SPLITS, list_images, resolve_image_root


def standard_splits(root: str | Path) -> dict[str, list[Path]]:
    """Return the BSDS500 standard splits as lists of image paths."""
    image_root = resolve_image_root(root)
    return {split: list_images(image_root / split) for split in SPLITS}


def check_disjoint(splits: dict[str, list[Path]]) -> None:
    """Raise if any image appears in more than one split."""
    normalised = {split: {str(Path(p).resolve()) for p in paths} for split, paths in splits.items()}
    for a, b in combinations(sorted(normalised), 2):
        overlap = normalised[a] & normalised[b]
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(f"splits {a!r} and {b!r} overlap (e.g. {example})")


def validation_subset(
    paths: list[Path] | list[str],
    size: int = 10,
    seed: int = 0,
) -> list[Path]:
    """Deterministic checkpoint-selection subset drawn from the sorted val list.

    The same ``(paths, size, seed)`` always yields the same images, so the
    subset can be pre-registered and logged with the protocol manifest.
    """
    ordered = sorted(Path(p) for p in paths)
    if size <= 0:
        raise ValueError("size must be positive")
    if size > len(ordered):
        raise ValueError(f"requested {size} validation images but only {len(ordered)} exist")
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, size))


def manifest(splits: dict[str, list[Path]], root: str | Path | None = None) -> dict[str, list[str]]:
    """JSON-serialisable manifest; paths are relative to ``root`` when given."""
    base = Path(root) if root is not None else None
    result: dict[str, list[str]] = {}
    for split in SPLITS:
        paths = splits.get(split, [])
        if base is not None:
            result[split] = [Path(p).resolve().relative_to(base.resolve()).as_posix() for p in paths]
        else:
            result[split] = [Path(p).as_posix() for p in paths]
    return result


def save_manifest(splits: dict[str, list[Path]], path: str | Path, root: str | Path | None = None) -> Path:
    """Write the split manifest as JSON and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(splits, root=root), indent=2), encoding="utf-8")
    return path
