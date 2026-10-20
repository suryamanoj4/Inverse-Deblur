#!/usr/bin/env python3
"""Run the classical arm (M2) on one image and report reconstruction metrics.

Examples:
    PYTHONPATH=src python3 scripts/run_classical.py
    PYTHONPATH=src python3 scripts/run_classical.py --image photo.png \
        --psf gaussian_sigma1.2 --sigma-over-255 15 \
        --gmm checkpoints/epll_gmm.npz
"""

import argparse
from pathlib import Path

import numpy as np

from deblur.data import degrade, load_image
from deblur.metrics import data_consistency, psnr, ssim
from deblur.methods.classical import (
    GMM,
    admm_epll,
    fista_tv,
    richardson_lucy,
    suggest_tv_lambda,
    wiener,
)
from deblur.psf import build_psf_library


def synthetic_image(size: int = 128, seed: int = 0) -> np.ndarray:
    """Piecewise-smooth target with one textured region, for a runnable demo."""
    image = np.zeros((size, size))
    image[size // 10 : size // 2, size // 12 : size // 2] = 0.8
    image[size // 2 : 9 * size // 10, size // 8 : 2 * size // 3] = 0.3
    n = size // 4
    texture = 0.5 + 0.15 * np.outer(
        np.cos(np.linspace(0.0, 18.0 * np.pi, n)),
        np.sin(np.linspace(0.0, 20.0 * np.pi, n)),
    )
    top = size // 3
    left = 2 * size // 3
    image[top : top + n, left : left + n] = np.clip(texture, 0.0, 1.0)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="input image; a synthetic target is used if omitted")
    parser.add_argument("--psf", default="gaussian_sigma1.2", help="PSF name from the library")
    parser.add_argument("--sigma-over-255", type=float, default=15.0, help="noise level (0-255 scale)")
    parser.add_argument("--gmm", default=None, help="path to a trained EPLL .npz (enables ADMM-EPLL)")
    parser.add_argument("--tv-lambda", type=float, default=None)
    parser.add_argument("--stride", type=int, default=2, help="patch stride for EPLL (1 is slowest)")
    parser.add_argument("--out", default=None, help="optional .npz path for the restored images")
    args = parser.parse_args()

    x = load_image(args.image) if args.image else synthetic_image()
    psfs = {psf.name: psf.kernel for psf in build_psf_library()}
    if args.psf not in psfs:
        raise SystemExit(f"unknown PSF {args.psf!r}; choose from: {', '.join(psfs)}")
    kernel = psfs[args.psf]
    sigma = args.sigma_over_255 / 255.0
    y = degrade(x, kernel, sigma=sigma, rng=np.random.default_rng(0))

    results = {"blurred": y}
    results["wiener"] = wiener(y, kernel, sigma=sigma)
    results["richardson_lucy"] = richardson_lucy(y, kernel, n_iter=30)
    tv_lambda = args.tv_lambda if args.tv_lambda is not None else suggest_tv_lambda(sigma)
    results["fista_tv"] = fista_tv(y, kernel, lam=tv_lambda, n_iter=100, tv_iter=30)
    if args.gmm:
        gmm = GMM.load(args.gmm)
        results["admm_epll"] = admm_epll(
            y,
            kernel,
            gmm,
            sigma=sigma,
            n_iter=10,
            patch_size=8,
            stride=args.stride,
        )

    print(f"image: {args.image or 'synthetic'}  psf: {args.psf}  sigma: {args.sigma_over_255}/255")
    print(f"{'method':20s} {'PSNR':>8s} {'SSIM':>8s} {'RMSE(Ax-y)':>12s}")
    for name, estimate in results.items():
        residual = data_consistency(estimate, y, kernel)
        print(f"{name:20s} {psnr(x, estimate):8.2f} {ssim(x, estimate):8.4f} {residual.rmse:12.4f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, clean=x, blurred=y, **results)
        print(f"saved results to {out}")


if __name__ == "__main__":
    main()
