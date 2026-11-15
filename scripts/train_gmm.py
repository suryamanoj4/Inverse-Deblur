#!/usr/bin/env python3
"""Train the EPLL Gaussian-mixture patch prior (M2's only learned component).

The prior is fitted on clean, zero-mean patches. Its training images come from
the same pool as the M1 and M3 networks (``deblur.data.training_pool``), so
``--train-images``, ``--extra-images`` and ``--p-real`` mean exactly the same
thing here as in ``train_m1.py`` and ``train_m3.py``: with ``--p-real P`` a
fraction P of the patches come from the real images and the rest from the
extra set. A JSON file next to the prior records the pool.

Examples:
    # primary prior (protocol v1)
    PYTHONPATH=src python3 scripts/train_gmm.py --data-root data \
        --components 200 --patches 100000 --iters 20 --out checkpoints/m2/gmm_K200.npz
    # Phase 2: real + synthetic, 200k patches
    PYTHONPATH=src python3 scripts/train_gmm.py --data-root data --components 200 \
        --patches 200000 --extra-images data/synthetic/syn-v1 --p-real 0.5 \
        --require-manifest --out checkpoints/m2/gmm_K200_syn-v1_p0.5.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from deblur.data import sample_patches
from deblur.data.training_pool import add_pool_arguments, pool_from_args
from deblur.methods.classical.gmm import fit_gmm


def patches_from(images: list[np.ndarray], total: int, patch_size: int, rng) -> list[np.ndarray]:
    """About ``total`` patches spread evenly over ``images``."""
    if total <= 0 or not images:
        return []
    per_image = max(1, total // len(images))
    return [sample_patches(image, patch_size, per_image, rng=rng) for image in images]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="BSDS500 root containing train/val/test")
    parser.add_argument("--components", type=int, default=200)
    parser.add_argument("--patches", type=int, default=100_000)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/epll_gmm.npz")
    add_pool_arguments(parser)
    args = parser.parse_args()

    pool = pool_from_args(args, args.data_root, luminance=True, seed=0)
    rng = np.random.default_rng(args.seed)
    n_real = round(args.patches * pool.p_real)
    chunks = patches_from(pool.real_images, n_real, args.patch_size, rng)
    chunks += patches_from(pool.extra_images, args.patches - n_real, args.patch_size, rng)
    patches = np.concatenate(chunks, axis=0)
    patches = patches.reshape(len(patches), -1)
    patches = patches - patches.mean(axis=1, keepdims=True)

    extra = f" + {pool.extra.name} (p_real {pool.p_real:g})" if pool.extra else ""
    print(
        f"training a {args.components}-component GMM on {len(patches)} "
        f"patches of size {args.patch_size}x{args.patch_size} from "
        f"{len(pool.real_ids)} real images{extra}"
    )
    gmm, history = fit_gmm(
        patches,
        n_components=args.components,
        n_iter=args.iters,
        seed=args.seed,
        verbose=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gmm.save(out)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "components": args.components,
                "patches": int(len(patches)),
                "patch_size": args.patch_size,
                "em_iterations": len(history),
                "seed": args.seed,
                "log_likelihood": [history[0], history[-1]],
                "training_data": pool.describe(),
            },
            indent=2,
        )
    )
    print(f"saved {gmm.n_components}-component GMM to {out}")
    print(f"log-likelihood: {history[0]:.1f} -> {history[-1]:.1f}")


if __name__ == "__main__":
    main()
