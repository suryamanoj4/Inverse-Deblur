#!/usr/bin/env python3
"""Train an M3 (unrolled, physics-aware) arm under the locked protocol.

One model is trained across P1 x all noise levels. The network receives the true
PSF spectrum and noise level as inputs, unlike M1. Hyperparameters come from
ProtocolConfig; command-line overrides are recorded under "deviations" in
run.json.

Examples:
    # primary M3
    PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model usrnet
    # ablation rungs
    PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model ista_net
    PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model lista
    # resume an interrupted run
    PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model usrnet --resume
    # quick smoke test (deviates from the protocol; not a result)
    PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model lista \
        --iterations 100 --val-interval 50 --val-subset-size 2
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from deblur.data import BSDS500, load_image
from deblur.methods.unrolled import build_model
from deblur.protocol import ProtocolConfig, validation_subset
from deblur.psf import protocol_psf_sets
from deblur.training import (
    PatchSampler,
    Trainer,
    TrainSettings,
    evaluate_grid,
    mean_psnr,
    physics_restorer,
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
    parser.add_argument("--data-root", required=True, help="directory containing BSDS500")
    parser.add_argument("--model", default="usrnet", choices=["usrnet", "ista_net", "lista"])
    parser.add_argument("--out", default=None, help="run directory (default checkpoints/m3/<model>)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast on CUDA")
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    parser.add_argument("--init", default=None, help="initial weights (optional)")
    smoke = parser.add_argument_group("overrides (recorded as protocol deviations)")
    smoke.add_argument("--iterations", type=int, default=None)
    smoke.add_argument("--batch-size", type=int, default=None)
    smoke.add_argument("--val-interval", type=int, default=None)
    smoke.add_argument("--val-subset-size", type=int, default=None)
    smoke.add_argument("--log-interval", type=int, default=None)
    args = parser.parse_args()

    locked = ProtocolConfig(data_root=args.data_root)
    deviations = {}
    config_overrides = {}
    if args.val_subset_size is not None and args.val_subset_size != locked.val_subset_size:
        config_overrides["val_subset_size"] = args.val_subset_size
        deviations["val_subset_size"] = args.val_subset_size
    config = ProtocolConfig(data_root=args.data_root, **config_overrides)

    amp = args.device.startswith("cuda") and not args.no_amp
    reference = TrainSettings.from_protocol(config, arm="m3", amp=amp)
    settings_overrides = {}
    for name in ("iterations", "val_interval", "log_interval"):
        value = getattr(args, name)
        if value is not None and value != getattr(reference, name):
            settings_overrides[name] = value
            deviations[name] = value
    settings = TrainSettings.from_protocol(config, arm="m3", amp=amp, **settings_overrides)
    batch_size = config.m3_batch_size
    if args.batch_size is not None and args.batch_size != batch_size:
        batch_size = args.batch_size
        deviations["batch_size"] = batch_size

    out_dir = Path(args.out) if args.out else ROOT / "checkpoints" / "m3" / args.model
    if (out_dir / "last.pt").exists() and not args.resume:
        raise SystemExit(f"{out_dir} already has a run; pass --resume or choose another --out")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_set = BSDS500(config.data_root, split="train", luminance=config.luminance)
    val_set = BSDS500(config.data_root, split="val", luminance=config.luminance)
    image_root = train_set.image_root
    val_paths = validation_subset(val_set.paths, size=config.val_subset_size, seed=config.val_subset_seed)
    val_images = [
        (p.relative_to(image_root).as_posix(), load_image(p, luminance=config.luminance))
        for p in val_paths
    ]
    train_psfs = protocol_psf_sets()[config.train_psf_set]
    print(
        f"M3 {args.model}: {len(train_set)} train images, {len(train_psfs)} PSFs x "
        f"{len(config.noise_levels)} noise levels, val on {len(val_images)} images "
        f"({len(train_psfs) * len(config.noise_levels)} cells), device {args.device}, amp {amp}"
    )

    sampler = PatchSampler(
        images=list(train_set),
        kernels=[psf.kernel for psf in train_psfs],
        noise_levels=config.noise_levels,
        patch_size=config.patch_size,
        batch_size=batch_size,
        device=args.device,
        seed=config.seed,
    )
    model = build_model(args.model).to(args.device)
    if args.init:
        state = torch.load(args.init, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params / 1e6:.2f} M")

    restore_fn = physics_restorer(model, args.device, amp=False)

    def validate(_model):
        rows = evaluate_grid(
            restore_fn, val_images, train_psfs, config.noise_levels, seed=config.seed
        )
        return mean_psnr(rows), rows

    def forward_fn(net, batch):
        if batch.sigma is None:
            raise RuntimeError("sampler did not provide noise levels")
        return net(batch.observed, batch.kernel, batch.sigma)

    metadata = {
        "arm": "M3",
        "model": args.model,
        "model_kwargs": {},
        "parameters": n_params,
        "regime": "single_model",
        "noise_levels_255": config.noise_levels_over_255,
        "train_psf_set": config.train_psf_set,
        "batch_size": batch_size,
        "precision": {"train": "bf16 autocast" if amp else "fp32", "validation": "fp32"},
        "init": args.init,
        "validation_images": [image_id for image_id, _ in val_images],
        "deviations": deviations,
        "git_commit": git_commit(),
        "protocol": config.manifest(),
    }
    trainer = Trainer(
        model,
        sampler,
        validate,
        settings,
        out_dir,
        metadata=metadata,
        forward_fn=forward_fn,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not (out_dir / "last.pt").exists():
            raise SystemExit(f"nothing to resume in {out_dir}")
        trainer.load(out_dir / "last.pt")
        print(f"  resumed at iteration {trainer.iteration}")
    else:
        (out_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    if deviations:
        print(f"  protocol deviations: {deviations} (not a reportable run)")

    summary = trainer.run()
    print(
        f"done: best mean val PSNR {summary['best_score']:.3f} dB at iteration "
        f"{summary['best_iteration']} ({summary['iterations_run']} run, "
        f"{'early stop' if summary['stopped_early'] else 'iteration cap'}); "
        f"weights in {out_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
