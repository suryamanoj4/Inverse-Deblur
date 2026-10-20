#!/usr/bin/env python3
"""Train the EPLL Gaussian-mixture patch prior on BSDS500 train images.

Example:
    PYTHONPATH=src python3 scripts/train_gmm.py \
        --data-root /path/to/BSDS500 \
        --components 200 --patches 100000 --iters 20 \
        --out checkpoints/epll_gmm.npz
"""

import argparse
from pathlib import Path

import numpy as np

from deblur.data import BSDS500, sample_patches
from deblur.methods.classical.gmm import fit_gmm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="BSDS500 root containing train/val/test")
    parser.add_argument("--components", type=int, default=200)
    parser.add_argument("--patches", type=int, default=100_000)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/epll_gmm.npz")
    args = parser.parse_args()

    dataset = BSDS500(args.data_root, split="train", luminance=True)
    rng = np.random.default_rng(args.seed)
    per_image = max(1, args.patches // len(dataset))
    chunks = [
        sample_patches(image, args.patch_size, per_image, rng=rng) for image in dataset
    ]
    patches = np.concatenate(chunks, axis=0)
    patches = patches.reshape(len(patches), -1)
    patches = patches - patches.mean(axis=1, keepdims=True)

    print(
        f"training a {args.components}-component GMM on {len(patches)} "
        f"patches of size {args.patch_size}x{args.patch_size}"
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
    print(f"saved {gmm.n_components}-component GMM to {out}")
    print(f"log-likelihood: {history[0]:.1f} -> {history[-1]:.1f}")


if __name__ == "__main__":
    main()
