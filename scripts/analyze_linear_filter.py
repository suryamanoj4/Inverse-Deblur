#!/usr/bin/env python3
"""E-X: does a method apply one fixed filter regardless of the kernel?

For each method, restore ``--images`` test images (centre crop ``--crop``) blurred by
every v2 test kernel (P1, P2_interp, P2_extrap) at one noise level. In the Fourier
domain fit, by least squares,

* a **shared** linear filter H(w) mapping the observation Y to the output O for all
  kernels at once, and
* a **per-kernel** filter H_k(w),

and report the fraction of the variance of what the method *adds*, D = O - Y,
that each explains (R^2 of the fitted correction G Y against D), overall and per
set. (R^2 on O itself is dominated by low frequencies, where every filter is close
to 1, so it cannot tell methods apart.) A method that sharpens the same way whatever the kernel has shared R^2 close
to per-kernel R^2; a method that adapts to the kernel does not. Wiener is linear
per kernel (per-kernel R^2 = 1), a sanity anchor.

    PYTHONPATH=src python3 scripts/analyze_linear_filter.py --arm m1 \\
        --checkpoint checkpoints/v2/m1/nafnet/best.pt --out report/generated/linear_filter_nafnet.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deblur.data import BSDS500, load_image  # noqa: E402
from deblur.protocol import ProtocolConfig  # noqa: E402
from deblur.protocol.observations import observe  # noqa: E402

_spec = importlib.util.spec_from_file_location("evaluate_main", ROOT / "scripts" / "evaluate.py")
evaluate_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_main)


def fit_and_score(Y: np.ndarray, O: np.ndarray, labels: list[tuple[str, str]]) -> dict:
    """Shared and per-kernel least-squares filters and their R^2, overall and per set.

    Streams over observations so memory stays at Y and O (complex64) plus a few
    (H, W) accumulators; sums are taken in float64/complex128.
    """
    def filt(idx):
        num = np.zeros(Y.shape[1:], np.complex128)
        den = np.zeros(Y.shape[1:], np.float64)
        for i in idx:
            num += np.conj(Y[i]) * O[i]
            den += np.abs(Y[i]) ** 2
        return num / den

    shared = filt(range(len(Y)))
    per_kernel = {k: filt([i for i, (_, n) in enumerate(labels) if n == k]) for k in sorted({n for _, n in labels})}
    sums = {}  # group -> [shared residual, per-kernel residual, centred total]
    for i, (set_name, name) in enumerate(labels):
        o = O[i].astype(np.complex128)
        y = Y[i].astype(np.complex128)
        terms = (np.sum(np.abs(o - shared * y) ** 2), np.sum(np.abs(o - per_kernel[name] * y) ** 2),
                 np.sum(np.abs(o - o.mean()) ** 2))
        for g in ("all", set_name):
            acc = sums.setdefault(g, [0.0, 0.0, 0.0])
            for j in range(3):
                acc[j] += float(terms[j])
    score = {g: {"shared_r2": 1 - a[0] / a[2], "per_kernel_r2": 1 - a[1] / a[2]} for g, a in sums.items()}
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=["m1", "m2", "m3"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--method", default="wiener")
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--sigma-255", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for name in ("gmm", "lambda_values", "steps", "inner_steps"):
        setattr(args, name, None)
    args.stride, args.m2_backend, args.m2_batch = 2, "numpy", 8
    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except ModuleNotFoundError:
        device = "cpu"

    config = ProtocolConfig(data_root="data", protocol_version="v2")
    restore_fn, meta = evaluate_main.build_restorer(args, config, device)
    sets = {k: v for k, v in config.psf_sets().items() if k in ("P1", "P2_interp", "P2_extrap")}
    test = BSDS500("data", split="test", luminance=True)
    c = args.crop
    images = []
    for path in test.paths[: args.images]:
        img = load_image(path, luminance=True)
        h, w = img.shape
        images.append((path.stem, img[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]))
    sigma = args.sigma_255 / 255.0

    n = sum(len(p) for p in sets.values()) * len(images)
    Y = np.empty((n, c, c), np.complex64)
    O = np.empty((n, c, c), np.complex64)
    labels = []
    for set_name, psfs in sets.items():
        for psf in psfs:
            for image_id, x in images:
                y = observe(x, psf.kernel, sigma, seed=config.seed, image_id=image_id, psf_name=psf.name,
                            quantize=True)
                out = restore_fn(y, psf, sigma)
                Y[len(labels)] = np.fft.fft2(y)
                O[len(labels)] = np.fft.fft2(out) - np.fft.fft2(y)  # the correction D = O - Y
                labels.append((set_name, psf.name))

    score = fit_and_score(Y, O, labels)
    result = {"method": meta, "images": len(images), "crop": c, "sigma_255": args.sigma_255,
              **score.pop("all"), "by_set": {s: score[s] for s in sets}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"shared R2 {result['shared_r2']:.4f}  per-kernel R2 {result['per_kernel_r2']:.4f}  "
          + "  ".join(f"{k}: {v['shared_r2']:.3f}/{v['per_kernel_r2']:.3f}" for k, v in result["by_set"].items()))


if __name__ == "__main__":
    main()
