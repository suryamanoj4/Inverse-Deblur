#!/usr/bin/env python3
"""H3 stability and LPIPS summaries for the M1 vs M2 report.

H3: reads ``artifacts/stability/h3_<set>.csv`` (transfer attack: PGD on NAFNet,
the same perturbed observation fed to ADMM-EPLL) and reports the mean PSNR drop
per (set, sigma, eps), the perturbation size in /255 RMS units, and the share
of cells where ADMM-EPLL drops less than NAFNet.

LPIPS: reads ``artifacts/eval/lpips/<method>.csv`` (50-image test subset,
sigma > 0) and reports mean LPIPS per (set, sigma). Missing methods are shown
as pending.

Writes ``report/generated/h3.md``, ``lpips.md`` and ``h3_lpips.json``.

    PYTHONPATH=src python3 scripts/analyze_h3_lpips.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"
SETS = ("P1", "P2_interp", "P2_extrap")
SET_MD = {"P1": "$P_1$", "P2_interp": "$P_2$-interp", "P2_extrap": "$P_2$-extrap"}
SIGMAS = (1, 5, 15, 25)
EPS = (0.005, 0.01, 0.02)
LPIPS_METHODS = {
    "wiener": "Wiener",
    "richardson_lucy": "R–L",
    "fista_tv": "FISTA-TV",
    "admm_epll": "ADMM-EPLL",
    "unet": "U-Net",
    "nafnet": "NAFNet",
}


def h3(result: dict) -> list[str]:
    md = [
        "| Set | σ | NAFNet clean | EPLL clean | NAFNet drop ε=0.5% | ε=1% | ε=2% "
        "| EPLL drop ε=0.5% | ε=1% | ε=2% | EPLL drops less (cells) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    eta_rms = defaultdict(list)
    for s in SETS:
        cells = defaultdict(dict)  # (image, psf, sigma, eps) -> method -> drop
        clean = defaultdict(list)
        drop = defaultdict(list)
        for r in csv.DictReader(open(ROOT / "artifacts" / "stability" / f"h3_{s}.csv")):
            sigma, eps, method = int(r["sigma_255"]), round(float(r["eps_rel"]), 4), r["method"]
            if eps == 0:
                clean[(method, sigma)].append(float(r["clean_psnr"]))
                continue
            drop[(method, sigma, eps)].append(float(r["drop"]))
            cells[(r["image"], r["psf"], sigma, eps)][method] = float(r["drop"])
            if method == "m1_nafnet":
                n_pixels = 481 * 321
                eta_rms[eps].append(255.0 * float(r["eta_norm"]) / np.sqrt(n_pixels))
        result[s] = {}
        for sigma in SIGMAS:
            row = {
                "clean": {m: float(np.mean(clean[(m, sigma)])) for m in ("m1_nafnet", "m2_admm_epll")},
                "drop": {
                    m: [float(np.mean(drop[(m, sigma, e)])) for e in EPS] for m in ("m1_nafnet", "m2_admm_epll")
                },
            }
            pairs = [v for k, v in cells.items() if k[2] == sigma]
            row["epll_less"] = float(np.mean([v["m2_admm_epll"] < v["m1_nafnet"] for v in pairs]))
            result[s][sigma] = row
            n, e = row["drop"]["m1_nafnet"], row["drop"]["m2_admm_epll"]
            md.append(
                f"| {SET_MD[s]} | {sigma} | {row['clean']['m1_nafnet']:.2f} | {row['clean']['m2_admm_epll']:.2f} | "
                + " | ".join(f"{v:.2f}" for v in n) + " | "
                + " | ".join(f"{v:.2f}" for v in e)
                + f" | {100 * row['epll_less']:.0f}% |"
            )
    result["eta_rms_255"] = {str(e): float(np.mean(v)) for e, v in eta_rms.items()}
    md.append("")
    md.append(
        "Perturbation size (mean RMS, /255 units): "
        + ", ".join(f"ε={100 * e:g}% → {result['eta_rms_255'][str(e)]:.2f}" for e in EPS)
        + "."
    )
    return md


def lpips(result: dict) -> list[str]:
    header = "| Set | σ | " + " | ".join(LPIPS_METHODS.values()) + " |"
    md = [header, "|---|---:|" + "---:|" * len(LPIPS_METHODS)]
    means = {}
    for m in LPIPS_METHODS:
        path = ROOT / "artifacts" / "eval" / "lpips" / f"{m}.csv"
        if not path.exists():
            continue
        groups = defaultdict(list)
        for r in csv.DictReader(open(path)):
            groups[(r["psf_set"], int(r["sigma_255"]))].append(float(r["lpips"]))
        means[m] = {k: float(np.mean(v)) for k, v in groups.items()}
    rows = [(s, sig) for s in SETS for sig in SIGMAS] + [(s, "mean") for s in SETS]
    for s, sig in rows:
        values = {}
        for m in LPIPS_METHODS:
            if m not in means:
                continue
            if sig == "mean":
                values[m] = float(np.mean([means[m][(s, x)] for x in SIGMAS]))
            else:
                values[m] = means[m][(s, sig)]
        best = min(values.values())
        cells = []
        for m in LPIPS_METHODS:
            if m not in values:
                cells.append("pending")
            elif values[m] == best:
                cells.append(f"**{values[m]:.3f}**")
            else:
                cells.append(f"{values[m]:.3f}")
        label = f"**mean σ>0**" if sig == "mean" else str(sig)
        md.append(f"| {SET_MD[s]} | {label} | " + " | ".join(cells) + " |")
        result.setdefault(s, {})[str(sig)] = values
    return md


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"h3": {}, "lpips": {}}
    (OUT / "h3.md").write_text("\n".join(h3(result["h3"])) + "\n")
    (OUT / "lpips.md").write_text("\n".join(lpips(result["lpips"])) + "\n")
    (OUT / "h3_lpips.json").write_text(json.dumps(result, indent=1))
    print((OUT / "h3.md").read_text())
    print((OUT / "lpips.md").read_text())


if __name__ == "__main__":
    main()
