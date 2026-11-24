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
    # Phase 2: protocol v2, and the same training-data options as train_m1.py
    PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model usrnet --protocol v2
    PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model usrnet --protocol v2 \
        --extra-images data/synthetic/syn-v1 --require-manifest --p-real 0.5
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
from deblur.data.training_pool import add_pool_arguments, pool_from_args, pool_tag
from deblur.methods.unrolled import MODELS, build_model
from deblur.protocol import ProtocolConfig, validation_subset
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
    parser.add_argument("--model", default="usrnet", choices=sorted(MODELS))
    parser.add_argument("--out", default=None, help="run directory (default checkpoints/m3/<model>)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast on CUDA")
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    parser.add_argument("--protocol", default="v1", choices=["v1", "v2"],
                        help="protocol version (v1 = mid-evaluation, frozen; v2 = Phase 2)")
    parser.add_argument("--real-kernels", action="append", default=[], metavar="PREFIX=DIR",
                        help="v2 only: add real recorded kernels from DIR to P2_extrap (repeatable)")
    add_pool_arguments(parser)
    parser.add_argument("--init", default=None, help="initial weights (optional)")
    phase3 = parser.add_argument_group("Phase 3 (recorded in run.json)")
    phase3.add_argument("--budget", default="m3", choices=["m3", "m1"],
                        help="training budget: m3 (default) or m1 (NAFNet's iterations, batch, learning rate; "
                             "used by the kernel-placement ladder so every variant matches M1)")
    phase3.add_argument("--seed", type=int, default=None, help="training seed (default: protocol seed)")
    phase3.add_argument("--mismatch", type=float, default=0.0,
                        help="probability of a realistic-forward-model batch (linear blur + edgetaper)")
    phase3.add_argument("--kernel-noise", type=float, default=0.0,
                        help="max relative kernel perturbation in mismatch batches")
    phase3.add_argument("--mismatch-boundary", default="on", choices=["on", "off"],
                        help="off: selected batches keep circular blur, only the given kernel is perturbed")
    phase3.add_argument("--n-iter", type=int, default=None, help="unrolled iterations (nafnet_unrolled)")
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
    config = ProtocolConfig(**base, **config_overrides)

    amp = args.device.startswith("cuda") and not args.no_amp
    budget_arm = args.budget
    reference = TrainSettings.from_protocol(config, arm=budget_arm, amp=amp)
    settings_overrides = {}
    for name in ("iterations", "val_interval", "log_interval"):
        value = getattr(args, name)
        if value is not None and value != getattr(reference, name):
            settings_overrides[name] = value
            deviations[name] = value
    settings = TrainSettings.from_protocol(config, arm=budget_arm, amp=amp, **settings_overrides)
    batch_size = getattr(config, f"{budget_arm}_batch_size")
    train_seed = config.seed if args.seed is None else args.seed
    if args.batch_size is not None and args.batch_size != batch_size:
        batch_size = args.batch_size
        deviations["batch_size"] = batch_size


    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

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
    version_dir = ROOT / "checkpoints" if config.protocol_version == "v1" else ROOT / "checkpoints" / config.protocol_version
    out_dir = Path(args.out) if args.out else version_dir / "m3" / f"{args.model}{pool_tag(pool)}"
    if (out_dir / "last.pt").exists() and not args.resume:
        raise SystemExit(f"{out_dir} already has a run; pass --resume or choose another --out")
    extra = f" + {len(pool.extra_images)} {pool.extra.name} (p_real {pool.p_real:g})" if pool.extra else ""
    print(
        f"M3 {args.model} [{config.protocol_version}]: {len(pool.real_ids)} real train images{extra}, "
        f"{len(train_psfs)} training PSFs ({config.condition_sampling}) x {len(config.noise_levels)} "
        f"noise levels, val on {len(val_images)} images x {len(val_psfs)} PSFs, device {args.device}, amp {amp}"
    )

    sampler = PatchSampler(
        images=pool.real_images,
        kernels=[psf.kernel for psf in train_psfs],
        noise_levels=config.noise_levels,
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
        mismatch=args.mismatch,
        kernel_noise=args.kernel_noise,
        mismatch_boundary=args.mismatch_boundary == "on",
    )
    model_kwargs = {"n_iter": args.n_iter} if args.n_iter is not None else {}
    model = build_model(args.model, **model_kwargs).to(args.device)
    if args.init:
        state = torch.load(args.init, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params / 1e6:.2f} M")

    restore_fn = physics_restorer(model, args.device, amp=False)

    def validate(_model):
        rows = evaluate_grid(
            restore_fn, val_images, val_psfs, config.noise_levels, seed=config.seed,
            quantize=config.quantize_observations,
        )
        return mean_psnr(rows), rows

    def forward_fn(net, batch):
        if batch.sigma is None:
            raise RuntimeError("sampler did not provide noise levels")
        out = net(batch.observed, batch.kernel, batch.sigma)
        if batch.crop:
            c = batch.crop
            out = out[..., c:-c, c:-c]
        return out

    metadata = {
        "arm": "M3",
        "model": args.model,
        "model_kwargs": model_kwargs,
        "phase3": {"budget": budget_arm, "seed": train_seed, "mismatch": args.mismatch,
                   "kernel_noise": args.kernel_noise,
                   "mismatch_boundary": args.mismatch_boundary},
        "parameters": n_params,
        "regime": "single_model",
        "noise_levels_255": config.noise_levels_over_255,
        "protocol_version": config.protocol_version,
        "train_psf_set": config.train_psf_set,
        "val_psf_set": config.val_psf_set,
        "condition_sampling": config.condition_sampling,
        "quantize_observations": config.quantize_observations,
        "training_data": pool.describe(),
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
