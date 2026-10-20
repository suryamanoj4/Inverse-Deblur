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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from deblur.data import BSDS500, load_image
from deblur.methods.direct import build_model
from deblur.protocol import ProtocolConfig
from deblur.psf import protocol_psf_sets
from deblur.training import evaluate_grid, model_restorer


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
        for row in set_rows:
            row["psf_set"] = set_name
        rows.extend(set_rows)

    out = Path(args.out) if args.out else Path(args.checkpoint).with_name(f"eval_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["psf_set", "image", "psf", "kind", "sigma_255", "psnr", "ssim", "residual_rmse"]
    if lpips_metric is not None:
        fields.append("lpips")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = defaultdict(list)
    for row in rows:
        summary[(row["psf_set"], row["sigma_255"])].append((row["psnr"], row["ssim"]))
    print(f"\n{'set':12s} {'sigma':>6s} {'PSNR':>8s} {'SSIM':>8s}")
    for (set_name, sigma_255), values in summary.items():
        values = np.asarray(values)
        print(f"{set_name:12s} {sigma_255:6d} {values[:, 0].mean():8.2f} {values[:, 1].mean():8.4f}")
    print(f"\n{len(rows)} rows written to {out}")


if __name__ == "__main__":
    main()
