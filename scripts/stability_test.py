#!/usr/bin/env python3
"""H3 stability test: PGD perturbations of the observation.

Protocol (transfer attack):
1. White-box PGD on the attack arm (M1 by default) finds a small perturbation
   ``eta`` of the observation that maximizes its reconstruction error.
2. Every available arm is evaluated on the same ``y + eta``.

This tests whether a perturbation that breaks the data-only network also breaks
the physics-aware arms. It is a transfer attack, not each method's own
worst-case perturbation; per-method white-box attacks for M2 require a
gradient-enabled solver and are a follow-up.

Examples:
    python3 scripts/stability_test.py --data-root data \
        --m1-checkpoint checkpoints/m1/nafnet/best.pt \
        --gmm checkpoints/m2/epll_gmm_K200.npz \
        --psf-set P1 --sigmas 1,15 --eps 0.005,0.01,0.02 --limit 10

    # smaller smoke test
    python3 scripts/stability_test.py --data-root data \
        --m1-checkpoint checkpoints/m1/nafnet/best.pt \
        --psf-set P1 --psf-limit 1 --sigmas 1 --eps 0.01 --limit 2
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deblur.data import BSDS500, load_image
from deblur.methods.classical import GMM, suggest_tv_lambda
from deblur.methods.classical.torch_backend import (
    TorchGMM,
    admm_epll_torch,
    fista_tv_torch,
)
from deblur.methods.direct import build_model as build_direct
from deblur.methods.unrolled import build_model as build_unrolled
from deblur.metrics import data_consistency, pgd_attack_torch, psnr, ssim
from deblur.operators.fft_conv import kernel_spectrum
from deblur.protocol import ProtocolConfig
from deblur.protocol.observations import observe
from deblur.psf import protocol_psf_sets


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
        raise ValueError("empty value list")
    return values


def resolve_per_sigma(values: list[float] | None, sigma: float, noise_levels) -> float | None:
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


def load_torch_model(path: Path, kind: str, device):
    state = torch.load(path, map_location=device, weights_only=False)
    meta = state.get("metadata", {})
    model_name = meta.get("model")
    if model_name is None:
        raise SystemExit(f"{path} has no model metadata")
    builder = build_direct if kind == "direct" else build_unrolled
    model = builder(model_name, **meta.get("model_kwargs", {})).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--psf-set", default="P1")
    parser.add_argument("--psf-limit", type=int, default=2)
    parser.add_argument("--sigmas", default="1,15", help="comma-separated /255 noise levels; sigma=0 is excluded")
    parser.add_argument("--limit", type=int, default=10, help="first N images")
    parser.add_argument("--eps", default="0.005,0.01,0.02", help="relative L2 budgets")
    parser.add_argument("--steps", type=int, default=20, help="PGD iterations")
    parser.add_argument("--step-scale", type=float, default=2.0, help="step size = step_scale * radius / steps")
    parser.add_argument("--attack-arm", default="m1", choices=["m1", "m3"])
    parser.add_argument("--m1-checkpoint", default="checkpoints/m1/nafnet/best.pt")
    parser.add_argument("--m3-checkpoint", default=None)
    parser.add_argument("--m2-method", default="admm_epll", choices=["admm_epll", "fista_tv"])
    parser.add_argument("--gmm", default=None, help="EPLL prior (.npz); required for admm_epll")
    parser.add_argument("--m2-steps", type=int, default=None)
    parser.add_argument("--m2-inner-steps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lambda-values", default=None, help="one value or one per noise level")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="CSV path")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = ProtocolConfig(data_root=args.data_root)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not Path(args.m1_checkpoint).is_file():
        raise SystemExit(f"M1 checkpoint not found: {args.m1_checkpoint}")

    m1_model, m1_meta = load_torch_model(Path(args.m1_checkpoint), "direct", device)
    m3_model, m3_meta = None, None
    if args.m3_checkpoint:
        m3_model, m3_meta = load_torch_model(Path(args.m3_checkpoint), "unrolled", device)
    if args.attack_arm == "m3" and m3_model is None:
        raise SystemExit("--attack-arm m3 needs --m3-checkpoint")

    def m1_forward(y_t):
        return torch.clamp(m1_model(y_t), 0.0, 1.0)

    m2_restore = None
    m2_name = None
    if args.m2_method == "admm_epll":
        if args.gmm is None:
            print("warning: no --gmm; M2 evaluation disabled (pass the EPLL prior)")
        else:
            gmm = GMM.load(args.gmm)
            gmm_t = TorchGMM(gmm, device=device)
            lambda_values = parse_values(args.lambda_values)

            def m2_restore(y_np, kernel, sigma):
                lam = resolve_per_sigma(lambda_values, sigma, config.noise_levels)
                return admm_epll_torch(
                    y_np,
                    kernel,
                    gmm_t,
                    sigma=sigma,
                    lam=lam,
                    n_iter=args.m2_steps or 10,
                    patch_size=8,
                    stride=args.stride,
                )

            m2_name = "m2_admm_epll"
    else:
        lambda_values = parse_values(args.lambda_values)

        def m2_restore(y_np, kernel, sigma):
            lam = resolve_per_sigma(lambda_values, sigma, config.noise_levels)
            return fista_tv_torch(
                y_np,
                kernel,
                lam=suggest_tv_lambda(sigma) if lam is None else lam,
                n_iter=args.m2_steps or 100,
                tv_iter=args.m2_inner_steps,
                device=device,
            )

        m2_name = "m2_fista_tv"

    if args.psf_set not in protocol_psf_sets():
        raise SystemExit(
            f"unknown PSF set {args.psf_set!r}; choose from {sorted(protocol_psf_sets())}"
        )
    psfs = protocol_psf_sets()[args.psf_set]
    if args.psf_limit:
        psfs = psfs[: args.psf_limit]
    sigmas = [value / 255.0 for value in (parse_values(args.sigmas) or [])]
    if not sigmas or any(sigma == 0 for sigma in sigmas):
        raise SystemExit("H3 excludes sigma=0; pass nonzero noise levels, e.g. 1,15")
    for sigma in sigmas:
        if not any(abs(sigma - level) < 1e-9 for level in config.noise_levels):
            raise SystemExit(f"{sigma * 255:g}/255 is not on the noise grid")
    eps_values = parse_values(args.eps) or []
    if not eps_values or any(value <= 0 for value in eps_values):
        raise SystemExit("--eps must contain positive relative budgets")

    dataset = BSDS500(config.data_root, split=args.split, luminance=config.luminance)
    paths = dataset.paths[: args.limit] if args.limit else dataset.paths
    images = [
        (path.relative_to(dataset.image_root).as_posix(), load_image(path, luminance=config.luminance))
        for path in paths
    ]
    print(
        f"H3 transfer attack: {len(images)} images x {len(psfs)} PSFs x {len(sigmas)} sigmas "
        f"x {len(eps_values)} budgets; attack={args.attack_arm}; device={device}"
    )
    print(
        f"  M1: {m1_meta.get('model')}  M2: {m2_name or 'disabled'}  M3: "
        f"{m3_meta.get('model') if m3_meta else 'disabled'}"
    )

    rows = []
    for image_id, x_np in images:
        x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)[None, None]
        for psf in psfs:
            for sigma in sigmas:
                y_np = observe(
                    x_np,
                    psf.kernel,
                    sigma,
                    seed=config.seed,
                    image_id=image_id,
                    psf_name=psf.name,
                )
                y_t = torch.as_tensor(y_np, dtype=torch.float32, device=device)[None, None]
                k_t = torch.as_tensor(
                    kernel_spectrum(psf.kernel, y_np.shape), dtype=torch.complex64, device=device
                )[None]
                sigma_t = torch.as_tensor([sigma], dtype=torch.float32, device=device)

                def m3_forward(z):
                    if m3_model is None:
                        raise RuntimeError("M3 is not loaded")
                    return torch.clamp(m3_model(z, k_t, sigma_t), 0.0, 1.0)

                clean = {}
                with torch.no_grad():
                    clean[f"m1_{m1_meta.get('model')}"] = (
                        m1_forward(y_t)[0, 0].float().cpu().numpy().astype(np.float64)
                    )
                    if m3_model is not None:
                        clean[f"m3_{m3_meta.get('model')}"] = (
                            m3_forward(y_t)[0, 0].float().cpu().numpy().astype(np.float64)
                        )
                if m2_restore is not None:
                    clean[m2_name] = m2_restore(y_np, psf.kernel, sigma)
                clean_psnr = {name: psnr(x_np, estimate) for name, estimate in clean.items()}

                def add_row(method, estimate, eps_rel, eps_abs, eta_np, observed_np):
                    rows.append(
                        {
                            "psf_set": args.psf_set,
                            "image": image_id,
                            "psf": psf.name,
                            "kind": psf.kind,
                            "sigma_255": round(sigma * 255),
                            "eps_rel": eps_rel,
                            "eps_abs": eps_abs,
                            "method": method,
                            "clean_psnr": clean_psnr[method],
                            "psnr": psnr(x_np, estimate),
                            "drop": clean_psnr[method] - psnr(x_np, estimate),
                            "ssim": ssim(x_np, estimate),
                            "residual_rmse": data_consistency(
                                estimate, observed_np, psf.kernel
                            ).rmse,
                            "eta_norm": float(np.linalg.norm(eta_np)),
                            "eta_rel": float(
                                np.linalg.norm(eta_np) / (np.linalg.norm(y_np) + 1e-12)
                            ),
                        }
                    )

                # clean rows (eps = 0)
                for method, estimate in clean.items():
                    add_row(method, estimate, 0.0, 0.0, np.zeros_like(y_np), y_np)

                for eps_rel in eps_values:
                    radius = float(eps_rel * np.linalg.norm(y_np))
                    attack_forward = m1_forward if args.attack_arm == "m1" else m3_forward
                    eta = pgd_attack_torch(
                        attack_forward,
                        y_t,
                        x_t,
                        radius,
                        steps=args.steps,
                        step_size=args.step_scale * radius / args.steps,
                    )
                    y_adv_t = torch.clamp(y_t + eta, 0.0, 1.0)
                    y_adv_np = y_adv_t[0, 0].float().cpu().numpy().astype(np.float64)
                    eta_np = eta[0, 0].float().cpu().numpy().astype(np.float64)

                    with torch.no_grad():
                        adv = {
                            f"m1_{m1_meta.get('model')}": m1_forward(y_adv_t)[0, 0]
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        }
                        if m3_model is not None:
                            adv[f"m3_{m3_meta.get('model')}"] = (
                                m3_forward(y_adv_t)[0, 0]
                                .float()
                                .cpu()
                                .numpy()
                                .astype(np.float64)
                            )
                    if m2_restore is not None:
                        adv[m2_name] = m2_restore(y_adv_np, psf.kernel, sigma)
                    for method, estimate in adv.items():
                        add_row(method, estimate, eps_rel, radius, eta_np, y_adv_np)

    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / "stability" / f"stability_{args.attack_arm}_{args.m2_method}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "psf_set",
        "image",
        "psf",
        "kind",
        "sigma_255",
        "eps_rel",
        "eps_abs",
        "method",
        "clean_psnr",
        "psnr",
        "drop",
        "ssim",
        "residual_rmse",
        "eta_norm",
        "eta_rel",
    ]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["sigma_255"], round(row["eps_rel"], 6))].append(row)
    summary = []
    for (method, sigma_255, eps_rel), items in sorted(groups.items()):
        summary.append(
            {
                "method": method,
                "sigma_255": sigma_255,
                "eps_rel": eps_rel,
                "n": len(items),
                "clean_psnr": float(np.mean([r["clean_psnr"] for r in items])),
                "psnr": float(np.mean([r["psnr"] for r in items])),
                "drop": float(np.mean([r["drop"] for r in items])),
                "ssim": float(np.mean([r["ssim"] for r in items])),
                "residual_rmse": float(np.mean([r["residual_rmse"] for r in items])),
                "eta_rel": float(np.mean([r["eta_rel"] for r in items])),
            }
        )
    summary_path = out.with_name(out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    metadata_path = out.with_name(out.stem + "_metadata.json")
    metadata = {
        "experiment": "H3 stability (transfer attack)",
        "attack_arm": args.attack_arm,
        "m1_checkpoint": str(args.m1_checkpoint),
        "m1_metadata": m1_meta,
        "m3_checkpoint": str(args.m3_checkpoint) if args.m3_checkpoint else None,
        "m3_metadata": m3_meta,
        "m2": {
            "method": args.m2_method,
            "gmm": args.gmm,
            "steps": args.m2_steps,
            "stride": args.stride if args.m2_method == "admm_epll" else None,
            "lambda_values": parse_values(args.lambda_values),
        },
        "psf_set": args.psf_set,
        "sigmas_255": [round(s * 255) for s in sigmas],
        "eps_rel": eps_values,
        "pgd_steps": args.steps,
        "split": args.split,
        "limit": args.limit,
        "protocol": config.manifest(),
        "git_commit": git_commit(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(
        f"\n{'method':20s} {'sigma':>5s} {'eps':>7s} {'clean':>7s} {'adv':>7s} "
        f"{'drop':>7s} {'eta':>7s} {'SSIM':>7s} {'RMSE':>7s}"
    )
    for row in summary:
        print(
            f"{row['method']:20s} {row['sigma_255']:5d} {row['eps_rel']:7.3f} "
            f"{row['clean_psnr']:7.2f} {row['psnr']:7.2f} {row['drop']:7.2f} "
            f"{row['eta_rel']:7.4f} {row['ssim']:7.4f} {row['residual_rmse']:7.4f}"
        )
    print(f"\n{len(rows)} rows written to {out}")
    print(f"summary:  {summary_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
