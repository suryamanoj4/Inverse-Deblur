#!/usr/bin/env python3
"""Train the M1 (data-only) arm under the locked protocol.

The primary run is one model across P1 x all noise levels; the network never
receives k or sigma. Hyperparameters come from ProtocolConfig; any override
given on the command line is recorded under "deviations" in run.json.

Examples:
    # primary M1 (NAFNet)
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data
    # minimal baseline
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data --model unet
    # resume an interrupted run
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data --resume
    # per-noise oracle ablation (fine-tuned from the primary checkpoint)
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data \
        --noise-over-255 15 --init checkpoints/m1/nafnet/best.pt
    # quick smoke test (deviates from the protocol; not a result)
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data --model unet \
        --iterations 200 --val-interval 100 --val-subset-size 2
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from deblur.data import BSDS500, load_image
from deblur.methods.direct import build_model
from deblur.protocol import ProtocolConfig, validation_subset
from deblur.psf import protocol_psf_sets
from deblur.training import PatchSampler, Trainer, TrainSettings, evaluate_grid, mean_psnr, model_restorer

ROOT = Path(__file__).resolve().parents[1]
PER_NOISE_ITERATIONS = 10_000


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
    parser.add_argument("--model", default="nafnet", choices=["nafnet", "unet"])
    parser.add_argument("--out", default=None, help="run directory (default checkpoints/m1/<model>)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast on CUDA")
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    parser.add_argument("--noise-over-255", type=float, default=None, help="per-noise oracle ablation")
    parser.add_argument("--init", default=None, help="initial weights (required for the ablation)")
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

    per_noise = args.noise_over_255 is not None
    if per_noise:
        sigma = args.noise_over_255 / 255.0
        config_overrides["m1_regime"] = "per_noise"
        if args.init is None:
            raise SystemExit("--noise-over-255 needs --init (the primary M1 checkpoint)")
    config = ProtocolConfig(data_root=args.data_root, **config_overrides)
    if per_noise and not any(abs(sigma - level) < 1e-9 for level in config.noise_levels):
        raise SystemExit(f"{args.noise_over_255}/255 is not on the noise grid {config.noise_levels_over_255}")
    if per_noise:
        sigma = min(config.noise_levels, key=lambda level: abs(level - sigma))
    noise_levels = (sigma,) if per_noise else config.noise_levels

    amp = args.device.startswith("cuda") and not args.no_amp
    settings_overrides = {"iterations": PER_NOISE_ITERATIONS} if per_noise else {}
    reference = TrainSettings.from_protocol(config, arm="m1", **settings_overrides)
    for name in ("iterations", "val_interval", "log_interval"):
        value = getattr(args, name)
        if value is not None and value != getattr(reference, name):
            settings_overrides[name] = value
            deviations[name] = value
    settings = TrainSettings.from_protocol(config, arm="m1", amp=amp, **settings_overrides)
    batch_size = config.m1_batch_size
    if args.batch_size is not None and args.batch_size != batch_size:
        batch_size = args.batch_size
        deviations["batch_size"] = batch_size

    suffix = f"_sigma{args.noise_over_255:g}" if per_noise else ""
    out_dir = Path(args.out) if args.out else ROOT / "checkpoints" / "m1" / f"{args.model}{suffix}"
    if (out_dir / "last.pt").exists() and not args.resume:
        raise SystemExit(f"{out_dir} already has a run; pass --resume or choose another --out")

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
        f"M1 {args.model}: {len(train_set)} train images, {len(train_psfs)} PSFs x "
        f"{len(noise_levels)} noise levels, val on {len(val_images)} images "
        f"({len(train_psfs) * len(noise_levels)} cells), device {args.device}, amp {amp}"
    )

    sampler = PatchSampler(
        images=list(train_set),
        kernels=[psf.kernel for psf in train_psfs],
        noise_levels=noise_levels,
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

    # bfloat16 autocast speeds up training only; validation is scored in fp32
    restore_fn = model_restorer(model, args.device, amp=False)

    def validate(_model):
        rows = evaluate_grid(restore_fn, val_images, train_psfs, noise_levels, seed=config.seed)
        return mean_psnr(rows), rows

    metadata = {
        "arm": "M1",
        "model": args.model,
        "model_kwargs": {},
        "parameters": n_params,
        "regime": config.m1_regime,
        "noise_levels_255": [round(s * 255) for s in noise_levels],
        "train_psf_set": config.train_psf_set,
        "batch_size": batch_size,
        "augment": "random rot90 + horizontal flip, before degradation",
        "precision": {"train": "bf16 autocast" if amp else "fp32", "validation": "fp32"},
        "init": args.init,
        "validation_images": [image_id for image_id, _ in val_images],
        "deviations": deviations,
        "git_commit": git_commit(),
        "protocol": config.manifest(),
    }
    trainer = Trainer(model, sampler, validate, settings, out_dir, metadata=metadata)
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
