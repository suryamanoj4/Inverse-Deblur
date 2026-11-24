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

Phase 2 (run directories get a suffix, so nothing overwrites a v1 run):
    # 2A-1 real-data learning curve: nested seeded subset of 100 real images
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data --train-images 100
    # 2A-2 real + frozen synthetic set, half the patches each
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data \
        --extra-images data/synthetic/syn-v1 --require-manifest --p-real 0.5
    # 2A-3 controls: synthetic only; real + DIV2K
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data \
        --extra-images data/synthetic/syn-v1 --require-manifest --p-real 0
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data \
        --extra-images data/div2k/DIV2K_train_HR --extra-name div2k --p-real 0.5
    # 2C-2 protocol v2 (kernel pool, family-balanced, 8-bit observations)
    PYTHONPATH=src python3 scripts/train_m1.py --data-root data --protocol v2
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from deblur.data import BSDS500, load_image
from deblur.data.training_pool import add_pool_arguments, pool_from_args, pool_tag
from deblur.methods.direct import build_model
from deblur.protocol import ProtocolConfig, validation_subset
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
    parser.add_argument("--protocol", default="v1", choices=["v1", "v2"],
                        help="protocol version (v1 = mid-evaluation, frozen; v2 = Phase 2)")
    parser.add_argument("--real-kernels", action="append", default=[], metavar="PREFIX=DIR",
                        help="v2 only: add real recorded kernels from DIR to P2_extrap (repeatable)")
    add_pool_arguments(parser)
    parser.add_argument("--noise-over-255", type=float, default=None, help="per-noise oracle ablation")
    parser.add_argument("--init", default=None, help="initial weights (required for the ablation)")
    parser.add_argument("--seed", type=int, default=None, help="training seed (default: protocol seed; recorded)")
    smoke = parser.add_argument_group("overrides (recorded as protocol deviations)")
    smoke.add_argument("--iterations", type=int, default=None)
    smoke.add_argument("--batch-size", type=int, default=None)
    smoke.add_argument("--val-interval", type=int, default=None)
    smoke.add_argument("--val-subset-size", type=int, default=None)
    smoke.add_argument("--log-interval", type=int, default=None)
    args = parser.parse_args()

    real_kernel_dirs = tuple(tuple(item.split("=", 1)) for item in args.real_kernels)
    base = {"data_root": args.data_root, "protocol_version": args.protocol, "real_kernel_dirs": real_kernel_dirs}
    locked = ProtocolConfig(**base)
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
    config = ProtocolConfig(**base, **config_overrides)
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


    pool = pool_from_args(args, config.data_root, config.luminance, config.seed)
    val_set = BSDS500(config.data_root, split="val", luminance=config.luminance)
    image_root = val_set.image_root
    val_paths = validation_subset(val_set.paths, size=config.val_subset_size, seed=config.val_subset_seed)
    val_images = [
        (p.relative_to(image_root).as_posix(), load_image(p, luminance=config.luminance))
        for p in val_paths
    ]
    psf_sets = config.psf_sets()
    train_psfs = psf_sets[config.train_psf_set]
    val_psfs = psf_sets[config.val_psf_set]
    suffix = f"_sigma{args.noise_over_255:g}" if per_noise else ""
    version_dir = ROOT / "checkpoints" if config.protocol_version == "v1" else ROOT / "checkpoints" / config.protocol_version
    out_dir = Path(args.out) if args.out else version_dir / "m1" / f"{args.model}{suffix}{pool_tag(pool)}"
    if (out_dir / "last.pt").exists() and not args.resume:
        raise SystemExit(f"{out_dir} already has a run; pass --resume or choose another --out")
    extra = f" + {len(pool.extra_images)} {pool.extra.name} (p_real {pool.p_real:g})" if pool.extra else ""
    print(
        f"M1 {args.model} [{config.protocol_version}]: {len(pool.real_ids)} real train images{extra}, "
        f"{len(train_psfs)} training PSFs ({config.condition_sampling}) x {len(noise_levels)} noise levels, "
        f"val on {len(val_images)} images x {len(val_psfs)} PSFs, device {args.device}, amp {amp}"
    )

    train_seed = config.seed if args.seed is None else args.seed
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    sampler = PatchSampler(
        images=pool.real_images,
        kernels=[psf.kernel for psf in train_psfs],
        noise_levels=noise_levels,
        patch_size=config.patch_size,
        batch_size=batch_size,
        device=args.device,
        seed=train_seed,
        extra_images=pool.extra_images,
        p_real=pool.p_real,
        kernel_groups=(
            [psf.kind for psf in train_psfs] if config.condition_sampling == "family_balanced" else None
        ),
        quantize=config.quantize_observations,
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
        rows = evaluate_grid(
            restore_fn, val_images, val_psfs, noise_levels, seed=config.seed,
            quantize=config.quantize_observations,
        )
        return mean_psnr(rows), rows

    metadata = {
        "arm": "M1",
        "model": args.model,
        "model_kwargs": {},
        "parameters": n_params,
        "regime": config.m1_regime,
        "noise_levels_255": [round(s * 255) for s in noise_levels],
        "protocol_version": config.protocol_version,
        "train_psf_set": config.train_psf_set,
        "val_psf_set": config.val_psf_set,
        "condition_sampling": config.condition_sampling,
        "quantize_observations": config.quantize_observations,
        "training_data": pool.describe(),
        "batch_size": batch_size,
        "augment": "random rot90 + horizontal flip, before degradation",
        "precision": {"train": "bf16 autocast" if amp else "fp32", "validation": "fp32"},
        "init": args.init,
        "train_seed": train_seed,
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
