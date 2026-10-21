#!/usr/bin/env python3
"""Unified grid evaluator for the three arms.

Every row is one (image, PSF, sigma) cell. The evaluator also computes the
no-restoration baseline on the exact same cells, so ``gain`` is paired and the
"worse than doing nothing" claim is testable. Rows are written to CSV and
aggregated by (PSF set, kernel kind, sigma); a metadata JSON records the arm,
checkpoint, protocol manifest and any per-noise lambda values.

Examples:
    python scripts/evaluate.py --arm m1 --data-root data \
        --checkpoint checkpoints/m1/nafnet/best.pt
    python scripts/evaluate.py --arm m3 --data-root data \
        --checkpoint checkpoints/m3/usrnet/best.pt --split test
    python scripts/evaluate.py --arm m2 --data-root data --method fista_tv \
        --lambda 0.0001,0.0005,0.001,0.002,0.003
    python scripts/evaluate.py --arm m2 --data-root data --method admm_epll \
        --gmm checkpoints/epll_gmm.npz --lambda 0.001,0.003,0.01,0.03,0.1 \
        --stride 2 --limit 20
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deblur.data import BSDS500, load_image
from deblur.evaluation import evaluate_grid, identity_restorer, merge_baseline, summarize
from deblur.methods.classical import (
    GMM,
    admm_epll,
    fista_tv,
    richardson_lucy,
    suggest_tv_lambda,
    wiener,
)
from deblur.protocol import ProtocolConfig
from deblur.psf import protocol_psf_sets

M2_METHODS = ("wiener", "richardson_lucy", "fista_tv", "admm_epll")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_values(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [float(item) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("empty lambda list")
    return values


def resolve_per_sigma(values: list[float] | None, sigma: float, noise_levels) -> float | None:
    """Pick the lambda for ``sigma``; a single value is shared by every cell."""
    if values is None:
        return None
    if len(values) == 1:
        return values[0]
    if len(values) != len(noise_levels):
        raise ValueError(
            f"expected 1 or {len(noise_levels)} lambda values, got {len(values)}"
        )
    index = min(range(len(noise_levels)), key=lambda i: abs(noise_levels[i] - sigma))
    return values[index]


def build_restorer(args, config: ProtocolConfig, device):
    """Return (restore_fn, arm_metadata)."""
    if args.arm == "m2":
        if args.method not in M2_METHODS:
            raise SystemExit(f"--method must be one of {M2_METHODS}")
        lambdas = parse_values(args.lambda_values)
        steps = args.steps
        if args.method == "wiener":
            def restore_fn(y, psf, sigma):
                lam = resolve_per_sigma(lambdas, sigma, config.noise_levels)
                return wiener(y, psf.kernel, sigma=sigma, lam=lam)
        elif args.method == "richardson_lucy":
            restore_fn = lambda y, psf, sigma: richardson_lucy(
                y, psf.kernel, n_iter=steps or 30
            )
        elif args.method == "fista_tv":
            def restore_fn(y, psf, sigma):
                lam = resolve_per_sigma(lambdas, sigma, config.noise_levels)
                return fista_tv(
                    y,
                    psf.kernel,
                    lam=suggest_tv_lambda(sigma) if lam is None else lam,
                    n_iter=steps or 100,
                    tv_iter=args.inner_steps or 30,
                )
        else:
            if args.gmm is None:
                raise SystemExit("admm_epll needs --gmm (train it with scripts/train_gmm.py)")
            gmm = GMM.load(args.gmm)

            def restore_fn(y, psf, sigma):
                lam = resolve_per_sigma(lambdas, sigma, config.noise_levels)
                return admm_epll(
                    y,
                    psf.kernel,
                    gmm,
                    sigma=sigma,
                    lam=lam,
                    n_iter=steps or 10,
                    patch_size=8,
                    stride=args.stride,
                )
        return restore_fn, {"arm": "M2", "method": args.method, "gmm": args.gmm}

    if args.checkpoint is None:
        raise SystemExit(f"--checkpoint is required for --arm {args.arm}")
    import torch

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = state.get("metadata", {})
    model_name = meta.get("model")
    if model_name is None:
        raise SystemExit("checkpoint metadata has no 'model' field")
    if meta.get("deviations"):
        print(f"warning: checkpoint has protocol deviations: {meta['deviations']}")
    if args.arm == "m1":
        from deblur.methods.direct import build_model
        from deblur.training import model_restorer

        model = build_model(model_name, **meta.get("model_kwargs", {}))
        restore_fn = model_restorer(model, device)
    else:
        from deblur.methods.unrolled import build_model
        from deblur.training import physics_restorer

        model = build_model(model_name, **meta.get("model_kwargs", {}))
        restore_fn = physics_restorer(model, device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return restore_fn, {
        "arm": args.arm.upper(),
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": {
            key: meta.get(key)
            for key in ("parameters", "regime", "train_psf_set", "noise_levels_255", "deviations")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=["m1", "m2", "m3"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None, help="M1/M3 checkpoint")
    parser.add_argument("--method", default="fista_tv", help=f"M2 method: {', '.join(M2_METHODS)}")
    parser.add_argument("--gmm", default=None, help="M2: EPLL prior (.npz)")
    parser.add_argument("--lambda-values", dest="lambda_values", default=None,
                        help="one value or one per noise level, comma-separated")
    parser.add_argument("--steps", type=int, default=None, help="solver/unrolled iteration override")
    parser.add_argument("--inner-steps", type=int, default=None, help="M2 FISTA TV inner iterations")
    parser.add_argument("--stride", type=int, default=2, help="M2 EPLL patch stride (1 is slowest)")
    parser.add_argument("--psf-sets", default=None, help="comma-separated (default: protocol eval sets)")
    parser.add_argument("--limit", type=int, default=None, help="first N images only")
    parser.add_argument("--lpips", action="store_true", help="also compute LPIPS")
    parser.add_argument("--device", default=None, help="M1/M3 device (default: cuda if available)")
    parser.add_argument("--out", default=None, help="CSV path")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        if args.arm in ("m1", "m3"):
            raise SystemExit(f"--arm {args.arm} requires PyTorch")
        torch = None
    device = args.device or (
        "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    )
    config = ProtocolConfig(data_root=args.data_root)
    dataset = BSDS500(config.data_root, split=args.split, luminance=config.luminance)
    paths = dataset.paths[: args.limit] if args.limit else dataset.paths
    images = [
        (p.relative_to(dataset.image_root).as_posix(), load_image(p, luminance=config.luminance))
        for p in paths
    ]

    set_names = args.psf_sets.split(",") if args.psf_sets else list(config.eval_psf_sets)
    psf_sets = protocol_psf_sets()
    unknown = set(set_names) - set(psf_sets)
    if unknown:
        raise SystemExit(f"unknown PSF sets {sorted(unknown)}; choose from {sorted(psf_sets)}")

    restore_fn, arm_meta = build_restorer(args, config, device)
    lpips_metric = None
    if args.lpips:
        from deblur.metrics import LPIPSMetric

        lpips_metric = LPIPSMetric(device=device)

    rows = []
    for set_name in set_names:
        psfs = psf_sets[set_name]
        print(
            f"{set_name}: {len(images)} images x {len(psfs)} PSFs x "
            f"{len(config.noise_levels)} noise levels"
        )
        set_rows = evaluate_grid(
            restore_fn,
            images,
            psfs,
            config.noise_levels,
            seed=config.seed,
            full_metrics=True,
            lpips_metric=lpips_metric,
        )
        baseline_rows = evaluate_grid(
            identity_restorer, images, psfs, config.noise_levels, seed=config.seed
        )
        merge_baseline(set_rows, baseline_rows)
        for row in set_rows:
            row["psf_set"] = set_name
        rows.extend(set_rows)

    out = Path(args.out) if args.out else ROOT / "artifacts" / "eval" / (
        f"{args.arm}_{arm_meta.get('method', arm_meta.get('model'))}_{args.split}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "psf_set",
        "image",
        "psf",
        "kind",
        "sigma_255",
        "psnr",
        "baseline_psnr",
        "gain",
        "ssim",
        "residual_rmse",
    ]
    if lpips_metric is not None:
        fields.append("lpips")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = out.with_name(out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    metadata = {
        "arm": arm_meta,
        "split": args.split,
        "limit": args.limit,
        "psf_sets": set_names,
        "lambda_values": parse_values(args.lambda_values),
        "seed": config.seed,
        "protocol": config.manifest(),
        "git_commit": git_commit(),
    }
    metadata_path = out.with_name(out.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    header = f"{'set':11s} {'kind':8s} {'sigma':>5s} {'PSNR':>7s} {'base':>7s} {'gain':>7s} {'SSIM':>7s} {'RMSE':>8s}"
    print(header)
    for row in summary:
        ssim = f"{row['ssim']:.4f}" if row["ssim"] is not None else "   -"
        rmse = f"{row['residual_rmse']:.4f}" if row["residual_rmse"] is not None else "    -"
        gain = f"{row['gain']:+.2f}" if row["gain"] is not None else "  -"
        base = f"{row['baseline_psnr']:.2f}" if row["baseline_psnr"] is not None else "  -"
        print(
            f"{row['psf_set']:11s} {row['kind']:8s} {row['sigma_255']:5d} "
            f"{row['psnr']:7.2f} {base:>7s} {gain:>7s} {ssim:>7s} {rmse:>8s}"
        )
    print(f"\n{len(rows)} rows written to {out}")
    print(f"summary:  {summary_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
