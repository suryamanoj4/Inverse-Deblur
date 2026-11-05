#!/usr/bin/env python3
"""Overfitting check for M1: train-split vs test-split PSNR on identical conditions.

Reads ``artifacts/eval/overfit/<method>_<split>.csv`` (seeded 20-image subsets of
BSDS500 train and test, P1 kernels, sigma > 0) and reports, per method and
sigma, the mean PSNR and gain over the blurred input on each split and the
train - test gap. ADMM-EPLL is the control: its only learned part is a patch
prior, so its gap mostly reflects how hard the two image sets are; the
difference of gaps isolates what the network memorised. Mann-Whitney U tests
compare per-image gains between splits.

    PYTHONPATH=src python3 scripts/analyze_overfit.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "artifacts" / "eval" / "overfit"
SIGMAS = (1, 5, 15, 25)


def per_image(method: str, split: str):
    path = DIR / f"{method}_{split}.csv"
    if not path.exists():
        return None
    gain, psnr = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(path)):
        key = (int(r["sigma_255"]), r["image"])
        gain[key].append(float(r["gain"]))
        psnr[key].append(float(r["psnr"]))
    return {k: np.mean(v) for k, v in gain.items()}, {k: np.mean(v) for k, v in psnr.items()}


def main() -> None:
    gaps = {}
    for method in ("nafnet", "unet", "epll"):
        tr, te = per_image(method, "train"), per_image(method, "test")
        if tr is None or te is None:
            print(f"{method}: pending")
            continue
        print(f"== {method}")
        print("  sigma | PSNR train  test | gain train  test | gap (train-test gain)  MW p")
        gaps[method] = {}
        for sigma in SIGMAS + ("all",):
            sel = (lambda k: True) if sigma == "all" else (lambda k, s=sigma: k[0] == s)
            g_tr = [v for k, v in tr[0].items() if sel(k)]
            g_te = [v for k, v in te[0].items() if sel(k)]
            p_tr = np.mean([v for k, v in tr[1].items() if sel(k)])
            p_te = np.mean([v for k, v in te[1].items() if sel(k)])
            gap = np.mean(g_tr) - np.mean(g_te)
            gaps[method][sigma] = gap
            p = mannwhitneyu(g_tr, g_te).pvalue
            print(f"  {str(sigma):>5} | {p_tr:10.2f} {p_te:5.2f} | {np.mean(g_tr):10.2f} {np.mean(g_te):5.2f} | {gap:+21.2f}  {p:.2g}")
    if "epll" in gaps:
        print("== gap beyond the control (method gap - EPLL gap)")
        for method in ("nafnet", "unet"):
            print(f"  {method}: " + "  ".join(f"σ={s}: {gaps[method][s] - gaps['epll'][s]:+.2f}" for s in SIGMAS + ("all",)))


if __name__ == "__main__":
    main()
