#!/usr/bin/env python3
"""Kernel-swap probe: does a conditioned network use the kernel as physics or as a label?

Observations are always blurred with the TRUE kernel; only the kernel *given to the
method* changes:

* ``true``     the true kernel (reference);
* ``nearest``  the training kernel closest to the true one (L2 distance of the
               spectrum magnitudes on a 64x64 grid);
* ``random``   a fixed random training kernel per test kernel (seeded);
* ``delta``    the identity kernel (no blur).

If a model scores higher with ``nearest`` than with ``true`` on unseen families,
it treats the kernel as a label for a learned mapping rather than as a forward
model. The probe runs on P2_extrap (unseen families) and, as a control, on P1.

    PYTHONPATH=src python3 scripts/probe_kernel_swap.py --arm m3 \\
        --checkpoint checkpoints/v2/p3/nafnet_film/best.pt --out artifacts/eval/v2/phase3/probe_film.csv
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

from deblur.data import BSDS500, load_image  # noqa: E402
from deblur.evaluation import evaluate_grid  # noqa: E402
from deblur.operators.fft_conv import kernel_spectrum  # noqa: E402
from deblur.protocol import ProtocolConfig, validation_subset  # noqa: E402
from deblur.psf.library import PSF  # noqa: E402

_spec = importlib.util.spec_from_file_location("evaluate_main", ROOT / "scripts" / "evaluate.py")
evaluate_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_main)

CONDITIONS = ("true", "nearest", "random", "delta")


def swaps(test_psfs: list[PSF], train_psfs: list[PSF], seed: int) -> dict[str, dict[str, PSF]]:
    mags = np.stack([np.abs(kernel_spectrum(p.kernel, (64, 64))) for p in train_psfs])
    rng = np.random.default_rng(seed)
    delta = PSF("delta", np.ones((1, 1)), "delta")
    out = {c: {} for c in CONDITIONS}
    for p in test_psfs:
        m = np.abs(kernel_spectrum(p.kernel, (64, 64)))
        out["true"][p.name] = p
        out["nearest"][p.name] = train_psfs[int(np.argmin(((mags - m) ** 2).sum(axis=(1, 2))))]
        out["random"][p.name] = train_psfs[int(rng.integers(len(train_psfs)))]
        out["delta"][p.name] = delta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=["m1", "m2", "m3"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--method", default="wiener")
    parser.add_argument("--images", type=int, default=50, help="test images (fixed subset)")
    parser.add_argument("--sigmas", default="1,5,15,25")
    parser.add_argument("--psf-sets", default="P2_extrap,P1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for name in ("gmm", "lambda_values", "steps", "inner_steps"):
        setattr(args, name, getattr(args, name, None))
    args.stride, args.m2_backend, args.m2_batch = 2, "numpy", 8
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = ProtocolConfig(data_root="data", protocol_version="v2")
    psf_sets = config.psf_sets()
    train_psfs = psf_sets[config.train_psf_set]
    restore_fn, meta = evaluate_main.build_restorer(args, config, device)
    test = BSDS500("data", split="test", luminance=True)
    paths = validation_subset(test.paths, size=args.images, seed=config.val_subset_seed)
    images = [(p.relative_to(test.image_root).as_posix(), load_image(p, luminance=True)) for p in paths]
    wanted = [float(v) / 255.0 for v in args.sigmas.split(",")]
    noise_levels = tuple(l for l in config.noise_levels if any(abs(l - w) < 1e-9 for w in wanted))

    rows = []
    for set_name in args.psf_sets.split(","):
        psfs = psf_sets[set_name]
        table = swaps(psfs, train_psfs, config.seed)
        for cond in CONDITIONS:
            given = table[cond]
            fn = lambda y, psf, sigma, given=given: restore_fn(y, given[psf.name], sigma)  # noqa: E731
            print(f"{set_name} / {cond}: {len(images)} images x {len(psfs)} PSFs x {len(noise_levels)} sigmas",
                  flush=True)
            for row in evaluate_grid(fn, images, psfs, noise_levels, seed=config.seed,
                                     quantize=config.quantize_observations):
                row.update(psf_set=set_name, condition=cond, given=given[row["psf"]].name)
                rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["psf_set", "condition", "image", "psf", "given", "kind", "sigma_255", "psnr"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for set_name in args.psf_sets.split(","):
        for cond in CONDITIONS:
            v = [r["psnr"] for r in rows if r["psf_set"] == set_name and r["condition"] == cond]
            summary[f"{set_name}/{cond}"] = float(np.mean(v))
    out.with_name(out.stem + "_summary.json").write_text(json.dumps({"method": meta, "mean_psnr": summary},
                                                                    indent=1, default=str))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
