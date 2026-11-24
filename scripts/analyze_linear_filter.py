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


def r2(outputs: np.ndarray, fitted: np.ndarray) -> float:
    centred = outputs - outputs.mean(axis=(-2, -1), keepdims=True)
    return float(1 - np.sum(np.abs(outputs - fitted) ** 2) / np.sum(np.abs(centred) ** 2))


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

    Y, O, labels = [], [], []
    for set_name, psfs in sets.items():
        for psf in psfs:
            for image_id, x in images:
                y = observe(x, psf.kernel, sigma, seed=config.seed, image_id=image_id, psf_name=psf.name,
                            quantize=True)
                out = restore_fn(y, psf, sigma)
                Y.append(np.fft.fft2(y))
                O.append(np.fft.fft2(out) - np.fft.fft2(y))  # the correction D = O - Y
                labels.append((set_name, psf.name))
    Y, O = np.stack(Y), np.stack(O)

    shared = (np.conj(Y) * O).sum(0) / (np.abs(Y) ** 2).sum(0)
    fitted_shared = shared[None] * Y
    fitted_kernel = np.empty_like(O)
    names = [n for _, n in labels]
    for name in sorted(set(names)):
        idx = [i for i, n in enumerate(names) if n == name]
        h = (np.conj(Y[idx]) * O[idx]).sum(0) / (np.abs(Y[idx]) ** 2).sum(0)
        fitted_kernel[idx] = h[None] * Y[idx]

    result = {"method": meta, "images": len(images), "crop": c, "sigma_255": args.sigma_255,
              "shared_r2": r2(O, fitted_shared), "per_kernel_r2": r2(O, fitted_kernel), "by_set": {}}
    for set_name in sets:
        idx = [i for i, (s, _) in enumerate(labels) if s == set_name]
        result["by_set"][set_name] = {"shared_r2": r2(O[idx], fitted_shared[idx]),
                                      "per_kernel_r2": r2(O[idx], fitted_kernel[idx])}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"shared R2 {result['shared_r2']:.4f}  per-kernel R2 {result['per_kernel_r2']:.4f}  "
          + "  ".join(f"{k}: {v['shared_r2']:.3f}/{v['per_kernel_r2']:.3f}" for k, v in result["by_set"].items()))


if __name__ == "__main__":
    main()
