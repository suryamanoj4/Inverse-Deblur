#!/usr/bin/env python3
"""Evaluate a trained M1 checkpoint on the protocol grid.

Scores every (image, PSF, sigma) cell of the chosen split on P1 (in
distribution), P2-interp and P2-extrap (kernel mismatch). Observations are the
deterministic protocol cells, so rows pair one-to-one with the other arms.

Examples:
    PYTHONPATH=src python3 scripts/eval_m1.py --data-root data \
        --checkpoint checkpoints/m1/nafnet/best.pt
    PYTHONPATH=src python3 scripts/eval_m1.py --data-root data \
        --checkpoint checkpoints/m1/nafnet/best.pt --lpips --psf-sets P2_extrap
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from deblur.data import BSDS500, load_image
from deblur.methods.direct import build_model
from deblur.protocol import ProtocolConfig
from deblur.psf import protocol_psf_sets
from deblur.training import (
    evaluate_grid,
    identity_restorer,
    merge_baseline,
    model_restorer,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--psf-sets", default=None, help="comma-separated (default: protocol eval sets)")
    parser.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs the lpips package)")
    parser.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="CSV path (default: next to the checkpoint)")
    args = parser.parse_args()

    config = ProtocolConfig(data_root=args.data_root)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    meta = state["metadata"]
    model = build_model(meta["model"], **meta.get("model_kwargs", {})).to(args.device)
    model.load_state_dict(state["model"])
    if meta.get("deviations"):
        print(f"warning: checkpoint has protocol deviations: {meta['deviations']}")
    restore_fn = model_restorer(model, args.device)  # fp32, so reported metrics are exact

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

    lpips_metric = None
    if args.lpips:
        from deblur.metrics import LPIPSMetric

        lpips_metric = LPIPSMetric(device=args.device)

    rows = []
    for set_name in set_names:
        print(f"{set_name}: {len(images)} images x {len(psf_sets[set_name])} PSFs x {len(config.noise_levels)} noise levels")
        set_rows = evaluate_grid(
            restore_fn,
            images,
            psf_sets[set_name],
            config.noise_levels,
            seed=config.seed,
            full_metrics=True,
            lpips_metric=lpips_metric,
        )
        baseline_rows = evaluate_grid(
            identity_restorer,
            images,
            psf_sets[set_name],
            config.noise_levels,
            seed=config.seed,
        )
        merge_baseline(set_rows, baseline_rows)
        for row in set_rows:
            row["psf_set"] = set_name
        rows.extend(set_rows)

    out = Path(args.out) if args.out else Path(args.checkpoint).with_name(f"eval_{args.split}.csv")
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
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = out.with_name(out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    metadata_path = out.with_name(out.stem + "_metadata.json")
    metadata = {
        "arm": "M1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": meta,
        "split": args.split,
        "limit": args.limit,
        "psf_sets": set_names,
        "seed": config.seed,
        "protocol": config.manifest(),
        "git_commit": git_commit(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(
        f"\n{'set':12s} {'kind':8s} {'sigma':>6s} {'PSNR':>8s} {'base':>8s} "
        f"{'gain':>7s} {'SSIM':>8s} {'RMSE':>8s}"
    )
    for row in summary:
        gain = f"{row['gain']:+.2f}" if row["gain"] is not None else "-"
        base = f"{row['baseline_psnr']:.2f}" if row["baseline_psnr"] is not None else "-"
        ssim = f"{row['ssim']:.4f}" if row["ssim"] is not None else "-"
        rmse = f"{row['residual_rmse']:.4f}" if row["residual_rmse"] is not None else "-"
        print(
            f"{row['psf_set']:12s} {row['kind']:8s} {row['sigma_255']:6d} "
            f"{row['psnr']:8.2f} {base:>8s} {gain:>7s} {ssim:>8s} {rmse:>8s}"
        )
    print(f"\n{len(rows)} rows written to {out}")
    print(f"summary:  {summary_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
