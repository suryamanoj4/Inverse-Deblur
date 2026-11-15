"""The training-image pool shared by every learned component of every arm.

Phase 2 principle 1: M1 network weights, the M2 GMM prior and M3 network
weights are fitted on the *same* images with the *same* mixing ratio. The
three training scripts therefore build their pool through this module and
expose the same command-line options (:func:`add_pool_arguments`):

- ``--train-images N``: a nested, seeded subset of the 200 real BSDS500 train
  images (the real-data learning curve, experiment 2A-1);
- ``--extra-images DIR`` with ``--p-real P``: extra images (a frozen synthetic
  set or DIV2K) mixed in; each patch is real with probability ``P``.

The pool description is written into every run's metadata, and
:func:`pool_tag` gives a short directory suffix so differently trained models
never share a checkpoint directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .datasets import BSDS500, load_image
from .synthetic import ImageSource, load_image_source, nested_subset

DEFAULT_P_REAL_WITH_EXTRA = 0.5


@dataclass
class TrainingPool:
    real_ids: list[str]
    real_images: list[np.ndarray] = field(repr=False)
    extra: ImageSource | None
    p_real: float
    subset_seed: int | None

    def describe(self) -> dict:
        return {
            "real": {
                "dataset": "BSDS500 train",
                "count": len(self.real_ids),
                "subset_seed": self.subset_seed,
                "image_ids": self.real_ids,
            },
            "extra": self.extra.describe() if self.extra else None,
            "p_real": self.p_real,
        }

    @property
    def extra_images(self) -> list[np.ndarray]:
        return self.extra.images if self.extra else []


def add_pool_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training data (identical options for M1, M2 and M3)")
    group.add_argument("--train-images", type=int, default=None,
                       help="use a nested seeded subset of N real train images (learning curve)")
    group.add_argument("--extra-images", default=None,
                       help="directory of extra training images (frozen synthetic set or DIV2K)")
    group.add_argument("--extra-name", default=None, help="name recorded for the extra set")
    group.add_argument("--p-real", type=float, default=None,
                       help=f"probability a patch is real (default {DEFAULT_P_REAL_WITH_EXTRA} with "
                       "--extra-images, else 1.0; 0 = extra images only)")
    group.add_argument("--require-manifest", action="store_true",
                       help="refuse an extra-image directory without a QC manifest")


def build_training_pool(
    data_root: str,
    luminance: bool = True,
    *,
    seed: int = 0,
    train_images: int | None = None,
    extra_images: str | None = None,
    extra_name: str | None = None,
    p_real: float | None = None,
    require_manifest: bool = False,
) -> TrainingPool:
    train = BSDS500(data_root, split="train", luminance=luminance)
    paths = train.paths
    subset_seed = None
    if train_images is not None and train_images != len(paths):
        paths = nested_subset(paths, train_images, seed=seed)
        subset_seed = seed
    extra = None
    if extra_images:
        extra = load_image_source(
            extra_images, name=extra_name, luminance=luminance, require_manifest=require_manifest
        )
    if p_real is None:
        p_real = DEFAULT_P_REAL_WITH_EXTRA if extra else 1.0
    if p_real < 1.0 and extra is None:
        raise ValueError("--p-real below 1 needs --extra-images")
    if p_real > 0.0 and not paths:
        raise ValueError("no real training images")
    return TrainingPool(
        real_ids=[p.relative_to(train.image_root).as_posix() for p in paths],
        real_images=[load_image(p, luminance=luminance) for p in paths],
        extra=extra,
        p_real=float(p_real),
        subset_seed=subset_seed,
    )


def pool_from_args(args: argparse.Namespace, data_root: str, luminance: bool, seed: int) -> TrainingPool:
    return build_training_pool(
        data_root,
        luminance,
        seed=seed,
        train_images=args.train_images,
        extra_images=args.extra_images,
        extra_name=args.extra_name,
        p_real=args.p_real,
        require_manifest=args.require_manifest,
    )


def pool_tag(pool: TrainingPool) -> str:
    """Directory suffix: '' for the v1 default (200 real images), else e.g.
    '_real100' or '_syn-v1_p0.5'."""
    tag = ""
    if pool.subset_seed is not None:
        tag += f"_real{len(pool.real_ids)}"
    if pool.extra is not None:
        tag += f"_{pool.extra.name}_p{pool.p_real:g}"
    return tag
