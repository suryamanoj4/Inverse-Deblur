#!/usr/bin/env python3
"""Control for the provided-blur benchmarks: rebuild triplets with the protocol's own
forward model. For each input triplet, y = quantize_8bit(circular_blur(x, f) + n),
n ~ N(0, sigma^2), on the scored scene x itself (so ``crop`` is dropped). Same images,
same kernels, same noise level as the original; only the forward model differs.

    PYTHONPATH=src python3 scripts/make_circular_triplets.py --triplets data/sun2013/benchmark \\
        --sigma-255 2.55 --out data/sun2013/circular
"""

import argparse
from pathlib import Path

import numpy as np

from deblur.operators.fft_conv import fft_convolve
from deblur.protocol.observations import quantize_8bit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--triplets", required=True)
    parser.add_argument("--sigma-255", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    files = sorted(Path(args.triplets).glob("*.npz"))
    for path in files:
        data = np.load(path)
        x = np.asarray(data["x"], dtype=np.float64) / (255.0 if data["x"].dtype == np.uint8 else 1.0)
        f = np.asarray(data["f"], dtype=np.float64)
        f = f / f.sum()
        y = quantize_8bit(np.clip(fft_convolve(x, f) + rng.normal(0, args.sigma_255 / 255.0, x.shape), 0, 1))
        np.savez_compressed(out / path.name, x=np.round(x * 255).astype(np.uint8),
                            y=np.round(y * 255).astype(np.uint8), f=f)
    print(f"{len(files)} circular triplets -> {out}")


if __name__ == "__main__":
    main()
