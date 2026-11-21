#!/usr/bin/env python3
"""Evaluate one method on benchmark triplets that ship their own blurred input.

Each triplet is an ``.npz`` with ``x`` (sharp, [0, 1]), ``y`` (blurred input as
published) and ``f`` (kernel). This covers Levin et al. (2009) real captured blur
(``data/levin/benchmark``) and the Sun et al. (2013) test set converted to the same
format (``data/sun2013/benchmark``). Unlike the protocol grid, ``y`` was not made
with the circular operator, so:

* the input is edge-padded by ``--pad`` pixels (default: twice the kernel size) and
  edge-tapered three times (MATLAB ``edgetaper``-style, so the padded image is
  close to circularly consistent), restored with the usual circular solver and
  cropped back. ``--boundary symmetric`` (plain symmetric padding) is kept for
  comparison; on USRNet it costs ~3-5 dB from boundary ringing;
* PSNR / SSIM skip a border of the kernel size and take the best integer shift
  within ``--max-shift`` pixels, as in Levin et al.'s evaluation, since real kernels
  are only known up to a translation.

If the triplet stores ``crop`` (Sun: 50), ``x`` is the centre of the scene with
``crop`` pixels removed on every side, as in Sun et al.'s evaluation; ``y`` is the
full blurred image, and the restoration is cropped the same way before scoring.

The noise level is not on the protocol grid (Sun: 1% = 2.55/255). ``--sigma-255 auto``
estimates it per image (Immerkaer 1996; Sun gives a median of 2.60, Levin real 0.28,
i.e. little more than 8-bit quantisation). Networks receive sigma as is; M2 methods
take the lambda of the nearest grid sigma, or with ``--lambda-rule interp`` the
per-sigma grid linearly interpolated in sigma (recorded in the summary).

    PYTHONPATH=src python3 scripts/evaluate_provided.py --protocol v2 --arm m3 \\
        --checkpoint checkpoints/v2/m3/usrnet_full/best.pt --triplets data/levin/benchmark \\
        --sigma-255 2.55 --out artifacts/eval/benchmarks/levin_real/m3_usrnet.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deblur.metrics import psnr, ssim  # noqa: E402
from deblur.operators.fft_conv import fft_convolve  # noqa: E402
from deblur.protocol import ProtocolConfig  # noqa: E402
from deblur.psf.library import PSF  # noqa: E402
from scipy.signal import convolve2d  # noqa: E402

_spec = importlib.util.spec_from_file_location("evaluate_main", ROOT / "scripts" / "evaluate.py")
evaluate_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_main)


def estimate_noise(y: np.ndarray) -> float:
    """Immerkaer (1996) fast noise-sigma estimate (Laplacian-difference mask)."""
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = y.shape
    return float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(convolve2d(y, mask, mode="valid")).sum())


def interpolated_lambda(values, sigma, noise_levels):
    """Per-sigma lambda grid, linearly interpolated in sigma (clamped at the ends).

    On grid points it returns the tuned value exactly; off the grid it replaces the
    nearest-grid-point rule, which picks a lambda for a much lower or higher noise
    level (e.g. sigma 2.55/255 -> the sigma 1 lambda)."""
    if values is None:
        return None
    if len(values) == 1:
        return values[0]
    return float(np.interp(sigma, np.asarray(noise_levels), np.asarray(values)))


def _taper_weights(n: int, projection: np.ndarray) -> np.ndarray:
    autocorr = np.correlate(projection, projection, "full")
    autocorr = autocorr / autocorr.max()
    half = len(autocorr) // 2
    weights = np.ones(n)
    weights[:half] = autocorr[:half]
    weights[-half:] = autocorr[-half:]
    return weights


def edgetaper(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blend the borders of ``y`` towards its circularly blurred self (MATLAB edgetaper)."""
    alpha = np.outer(_taper_weights(y.shape[0], kernel.sum(1)), _taper_weights(y.shape[1], kernel.sum(0)))
    return alpha * y + (1 - alpha) * fft_convolve(y, kernel)


def prepare_input(y: np.ndarray, kernel: np.ndarray, pad: int, boundary: str) -> np.ndarray:
    if boundary == "symmetric":
        return np.pad(y, pad, mode="symmetric")
    y_pad = np.pad(y, pad, mode="edge")
    for _ in range(3):
        y_pad = edgetaper(y_pad, kernel)
    return y_pad


def aligned_scores(x_hat: np.ndarray, x: np.ndarray, border: int, max_shift: int,
                   crop: int = 0) -> tuple[float, float, int, int]:
    """Best-shift PSNR (and SSIM at that shift).

    Without ``crop``, ``x_hat`` and ``x`` have the same size and the interior
    ``border`` pixels in is scored. With ``crop``, ``x`` is ``x_hat``'s centre
    (``crop`` px removed per side) and is scored whole.
    """
    if crop:
        if crop < max_shift:
            raise ValueError("crop must be at least max_shift")
        h, w = x.shape
        ref = x
        window = lambda dy, dx: x_hat[crop + dy:crop + dy + h, crop + dx:crop + dx + w]
    else:
        h, w = x.shape
        b = border + max_shift
        ref = x[b:h - b, b:w - b]
        window = lambda dy, dx: x_hat[b + dy:h - b + dy, b + dx:w - b + dx]
    best = (-np.inf, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            score = psnr(np.clip(window(dy, dx), 0, 1), ref)
            if score > best[0]:
                best = (score, dy, dx)
    _, dy, dx = best
    return best[0], ssim(np.clip(window(dy, dx), 0, 1), ref), dy, dx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=["m1", "m2", "m3"])
    parser.add_argument("--protocol", default="v2", choices=["v1", "v2"])
    parser.add_argument("--triplets", required=True, help="directory of .npz files with x, y, f")
    parser.add_argument("--sigma-255", required=True,
                        help="noise level of the provided inputs, /255, or 'auto' (Immerkaer estimate per image)")
    parser.add_argument("--lambda-rule", default="nearest", choices=["nearest", "interp"],
                        help="M2 lambda for off-grid sigma: nearest grid point, or linear interpolation in sigma")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--method", default="fista_tv")
    parser.add_argument("--gmm", default=None)
    parser.add_argument("--lambda-values", dest="lambda_values", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--inner-steps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--m2-backend", default="numpy", choices=["numpy", "torch"])
    parser.add_argument("--m2-batch", type=int, default=8)
    parser.add_argument("--pad", type=int, default=None, help="symmetric padding (default 2 x kernel size)")
    parser.add_argument("--boundary", default="edgetaper", choices=["edgetaper", "symmetric"])
    parser.add_argument("--max-shift", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except ModuleNotFoundError:
        device = "cpu"
    config = ProtocolConfig(data_root="data", protocol_version=args.protocol)
    if args.lambda_rule == "interp":
        evaluate_main.resolve_per_sigma = interpolated_lambda
    restore_fn, meta = evaluate_main.build_restorer(args, config, device)
    auto_sigma = str(args.sigma_255).lower() == "auto"
    fixed_sigma = None if auto_sigma else float(args.sigma_255) / 255.0
    grid = config.noise_levels

    files = sorted(Path(args.triplets).glob("*.npz"))[: args.limit] if args.limit else sorted(Path(args.triplets).glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz triplets in {args.triplets}")
    rows = []
    for path in files:
        data = np.load(path)
        x, y = (np.asarray(data[key], dtype=np.float64) / (255.0 if data[key].dtype == np.uint8 else 1.0)
                for key in ("x", "y"))
        k = np.asarray(data["f"], dtype=np.float64)
        crop = int(data["crop"]) if "crop" in data else 0
        k = k / k.sum()
        pad = args.pad if args.pad is not None else 2 * max(k.shape)
        sigma = estimate_noise(y) if auto_sigma else fixed_sigma
        y_pad = prepare_input(y, k, pad, args.boundary)
        x_hat = restore_fn(y_pad, PSF(name=path.stem, kernel=k, kind="real"), sigma)
        if pad:
            x_hat = x_hat[pad:-pad, pad:-pad]
        border = max(k.shape)
        p, s, dy, dx = aligned_scores(x_hat, x, border, args.max_shift, crop)
        p0, _, _, _ = aligned_scores(y, x, border, args.max_shift, crop)
        rows.append({"triplet": path.stem, "sigma_255": sigma * 255, "psnr": p, "ssim": s,
                     "baseline_psnr": p0, "gain": p - p0, "shift_y": dy, "shift_x": dx})
        print(f"{path.stem}: {p:.2f} dB (input {p0:.2f}, shift {dy},{dx})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "triplets": args.triplets, "n": len(rows), "sigma_255": args.sigma_255,
        "sigma_255_mean": float(np.mean([r["sigma_255"] for r in rows])),
        "noise_estimate": "Immerkaer (1996), per image" if auto_sigma else "fixed",
        "m2_lambda_rule": (f"{args.lambda_rule} on the per-sigma grid" if args.arm == "m2" else None),
        "mean_psnr": float(np.mean([r["psnr"] for r in rows])),
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
        "mean_input_psnr": float(np.mean([r["baseline_psnr"] for r in rows])),
        "boundary": f"{args.boundary}, pad {args.pad or '2 x kernel size'}, circular restore, crop",
        "metric": f"best integer shift within {args.max_shift} px, border = kernel size",
        "method": meta,
    }
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"mean PSNR {summary['mean_psnr']:.2f} dB over {len(rows)} (input {summary['mean_input_psnr']:.2f}) -> {out}")


if __name__ == "__main__":
    main()
