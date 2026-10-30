#!/usr/bin/env python3
"""M1 vs M2 analysis: tables, paired tests and plot data for the LaTeX report.

Reads the per-cell evaluation CSVs (one row per image x PSF x sigma) and writes
LaTeX table fragments, pgfplots data files and ``numbers.json`` to
``report/generated/``. Paired tests use the image as the unit: each image's
PSNR is averaged over the PSFs of a set first, so the 200 test images give 200
pairs per (set, sigma) and correlated PSFs of one image are not counted as
independent samples. Wilcoxon signed-rank p-values are Holm-adjusted over all
tests in the table.

    PYTHONPATH=src python3 scripts/analyze_m1_m2.py
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval"
OUT = ROOT / "report" / "generated"

SETS = ("P1", "P2_interp", "P2_extrap")
SET_TEX = {"P1": r"$P_1$", "P2_interp": r"$P_2$-interp", "P2_extrap": r"$P_2$-extrap"}
SIGMAS = (1, 5, 15, 25)

# name -> (label, arm, CSV glob)
METHODS = {
    "wiener": ("Wiener", "M2", "m2_wiener_test.csv"),
    "rl": ("Richardson--Lucy", "M2", "m2_richardson_lucy_test.csv"),
    "fista": ("FISTA-TV", "M2", "m2_test/fista_tv_shard*.csv"),
    "epll": ("ADMM-EPLL", "M2", "m2_test/admm_epll_shard*.csv"),
    "unet": ("U-Net", "M1", "m1_unet_test.csv"),
    "nafnet": ("NAFNet", "M1", str(ROOT / "checkpoints/m1/nafnet/eval_test.csv")),
}
TABLE_METHODS = ("wiener", "rl", "fista", "epll", "unet", "nafnet")


def load(pattern: str) -> dict[tuple, dict]:
    paths = sorted(glob.glob(pattern if pattern.startswith("/") else str(EVAL / pattern)))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    rows = {}
    for path in paths:
        for r in csv.DictReader(open(path)):
            key = (r["psf_set"], r["image"], r["psf"], int(r["sigma_255"]))
            rows[key] = {
                "kind": r["kind"],
                "psnr": float(r["psnr"]),
                "ssim": float(r["ssim"]),
                "residual": float(r["residual_rmse"]),
                "baseline": float(r["baseline_psnr"]) if r.get("baseline_psnr") else None,
            }
    return rows


def mean_by(rows, metric, keyfn, filt=lambda k, r: True):
    groups = defaultdict(list)
    for k, r in rows.items():
        if filt(k, r):
            groups[keyfn(k, r)].append(r[metric])
    return {g: float(np.mean(v)) for g, v in groups.items()}


def per_image(rows, set_name, sigma, metric="psnr"):
    groups = defaultdict(list)
    for (s, image, _psf, sig), r in rows.items():
        if s == set_name and sig == sigma:
            groups[image].append(r[metric])
    return {image: float(np.mean(v)) for image, v in groups.items()}


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def bold_best(values: list[float], digits=2, higher=True) -> list[str]:
    best = max(values) if higher else min(values)
    return [
        (r"\textbf{%s}" % fmt(v, digits)) if abs(v - best) < 0.5 * 10**-digits else fmt(v, digits)
        for v in values
    ]


def p_tex(p: float) -> str:
    if p < 1e-4:
        return r"$<10^{-4}$"
    return fmt(p, 4 if p < 0.01 else 3)


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {m: load(spec[2]) for m, spec in METHODS.items()}
    reference = data["wiener"]
    for m, rows in data.items():
        missing = set(reference) - set(rows)
        if missing:
            raise SystemExit(f"{m} is missing {len(missing)} cells")
    numbers: dict = {}

    # ---- main PSNR / SSIM tables (sigma > 0) -------------------------------
    by_cell = {
        m: {metric: mean_by(rows, metric, lambda k, r: (k[0], k[3])) for metric in ("psnr", "ssim", "residual")}
        for m, rows in data.items()
    }
    blurred = mean_by(reference, "baseline", lambda k, r: (k[0], k[3]))
    numbers["mean"] = {
        m: {f"{s}/{sig}": v for (s, sig), v in by_cell[m]["psnr"].items()} for m in data
    }
    numbers["blurred"] = {f"{s}/{sig}": v for (s, sig), v in blurred.items()}

    for metric, digits in (("psnr", 2), ("ssim", 3)):
        lines = []
        for s in SETS:
            for i, sig in enumerate(SIGMAS):
                values = [by_cell[m][metric][(s, sig)] for m in TABLE_METHODS]
                cells = bold_best(values, digits)
                first = (r"\multirow{4}{*}{%s}" % SET_TEX[s]) if i == 0 else ""
                base = f"{blurred[(s, sig)]:.2f} & " if metric == "psnr" else ""
                lines.append(f"{first} & {sig} & {base}" + " & ".join(cells) + r" \\")
            if s != SETS[-1]:
                lines.append(r"\midrule")
        write(f"table_{metric}.tex", "\n".join(lines) + "\n")

    # ---- pgfplots data: PSNR vs sigma per set ------------------------------
    for s in SETS:
        lines = ["sigma blurred " + " ".join(TABLE_METHODS)]
        for sig in SIGMAS:
            lines.append(
                f"{sig} {blurred[(s, sig)]:.4f} "
                + " ".join(f"{by_cell[m]['psnr'][(s, sig)]:.4f}" for m in TABLE_METHODS)
            )
        write(f"psnr_{s}.dat", "\n".join(lines) + "\n")

    # ---- paired tests: NAFNet vs each M2 solver ----------------------------
    tests = []
    for rival in ("epll", "fista"):
        for s in SETS:
            for sig in (0,) + SIGMAS:
                a = per_image(data[rival], s, sig)
                b = per_image(data["nafnet"], s, sig)
                images = sorted(a)
                diff = np.array([a[i] - b[i] for i in images])
                p = float(wilcoxon(diff).pvalue)
                tests.append(
                    {
                        "rival": rival,
                        "set": s,
                        "sigma": sig,
                        "mean": float(diff.mean()),
                        "median": float(np.median(diff)),
                        "win": float((diff > 0).mean()),
                        "p": p,
                        "n": len(images),
                    }
                )
    for t, adj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = adj
    numbers["tests"] = tests
    lines = []
    for s in SETS:
        rows_s = [t for t in tests if t["set"] == s and t["sigma"] > 0]
        for i, sig in enumerate(SIGMAS):
            first = (r"\multirow{4}{*}{%s}" % SET_TEX[s]) if i == 0 else ""
            cells = []
            for rival in ("epll", "fista"):
                t = next(t for t in rows_s if t["rival"] == rival and t["sigma"] == sig)
                cells += [f"{t['mean']:+.2f}", f"{t['median']:+.2f}", f"{100 * t['win']:.0f}", p_tex(t["p_holm"])]
            lines.append(f"{first} & {sig} & " + " & ".join(cells) + r" \\")
        if s != SETS[-1]:
            lines.append(r"\midrule")
    write("table_tests.tex", "\n".join(lines) + "\n")

    # ---- kernel mismatch: drop from P1 and gain over the blurred input -----
    def avg(m, s, metric="psnr"):
        return float(np.mean([by_cell[m][metric][(s, sig)] for sig in SIGMAS]))

    def avg_gain(m, s):
        return float(np.mean([by_cell[m]["psnr"][(s, sig)] - blurred[(s, sig)] for sig in SIGMAS]))

    lines = []
    numbers["mismatch"] = {}
    for m in TABLE_METHODS:
        p1, pi, pe = (avg(m, s) for s in SETS)
        gains = [avg_gain(m, s) for s in SETS]
        numbers["mismatch"][m] = {"P1": p1, "P2_interp": pi, "P2_extrap": pe, "gain": gains}
        lines.append(
            f"{METHODS[m][0]} & {METHODS[m][1]} & {p1:.2f} & {pi:.2f} & {pi - p1:+.2f} & {pe:.2f} & {pe - p1:+.2f}"
            f" & {gains[0]:+.2f} & {gains[1]:+.2f} & {gains[2]:+.2f}" + r" \\"
        )
    bp1, bpi, bpe = (float(np.mean([blurred[(s, sig)] for sig in SIGMAS])) for s in SETS)
    lines.insert(
        0,
        f"Blurred input & -- & {bp1:.2f} & {bpi:.2f} & {bpi - bp1:+.2f} & {bpe:.2f} & {bpe - bp1:+.2f} & -- & -- & --"
        + r" \\ \midrule",
    )
    write("table_mismatch.tex", "\n".join(lines) + "\n")

    # ---- per kernel family (mean over sigma > 0) --------------------------
    groups = [("P1", "gaussian"), ("P1", "motion"), ("P2_interp", "gaussian"), ("P2_interp", "motion"), ("P2_extrap", "defocus")]
    fam = {
        m: mean_by(rows, "psnr", lambda k, r: (k[0], r["kind"]), lambda k, r: k[3] > 0)
        for m, rows in data.items()
    }
    fam_blur = mean_by(reference, "baseline", lambda k, r: (k[0], r["kind"]), lambda k, r: k[3] > 0)
    lines = []
    for s, kind in groups:
        values = [fam[m][(s, kind)] for m in TABLE_METHODS]
        lines.append(
            f"{SET_TEX[s]} & {kind} & {fam_blur[(s, kind)]:.2f} & " + " & ".join(bold_best(values)) + r" \\"
        )
    write("table_family.tex", "\n".join(lines) + "\n")

    # ---- data-consistency residual (in /255 units) -------------------------
    lines = []
    for s in SETS:
        for i, sig in enumerate(SIGMAS):
            first = (r"\multirow{4}{*}{%s}" % SET_TEX[s]) if i == 0 else ""
            values = [255 * by_cell[m]["residual"][(s, sig)] for m in TABLE_METHODS]
            lines.append(f"{first} & {sig} & " + " & ".join(f"{v:.2f}" for v in values) + r" \\")
        if s != SETS[-1]:
            lines.append(r"\midrule")
    write("table_residual.tex", "\n".join(lines) + "\n")

    # ---- oracle-sigma specialists vs the single NAFNet ---------------------
    lines = []
    numbers["specialists"] = {}
    for sig in SIGMAS:
        spec = load(f"m1_nafnet_sigma{sig}_test.csv")
        spec_mean = mean_by(spec, "psnr", lambda k, r: (k[0], k[3]))
        cells = []
        for s in SETS:
            base = by_cell["nafnet"]["psnr"][(s, sig)]
            own = spec_mean[(s, sig)]
            cells += [f"{base:.2f}", f"{own:.2f}", f"{own - base:+.2f}"]
            numbers["specialists"][f"{s}/{sig}"] = own - base
        lines.append(f"{sig} & " + " & ".join(cells) + r" \\")
    write("table_specialists.tex", "\n".join(lines) + "\n")

    # ---- sigma = 0: exact vs 8-bit quantized observations ------------------
    quant_files = {
        "wiener": "wiener", "rl": "richardson_lucy", "fista": "fista_tv",
        "epll": "admm_epll", "unet": "unet", "nafnet": "nafnet",
    }
    lines = []
    numbers["sigma0"] = {}
    for m in TABLE_METHODS:
        q = mean_by(load(f"sigma0_quantized/{quant_files[m]}.csv"), "psnr", lambda k, r: k[0])
        cells = []
        for s in SETS:
            exact = by_cell[m]["psnr"][(s, 0)]
            cells += [f"{exact:.2f}", f"{q[s]:.2f}"]
            numbers["sigma0"][f"{m}/{s}"] = [exact, q[s]]
        lines.append(f"{METHODS[m][0]} & " + " & ".join(cells) + r" \\")
    blur0 = [blurred[(s, 0)] for s in SETS]
    lines.insert(0, "Blurred input & " + " & ".join(f"{b:.2f} & {b:.2f}" for b in blur0) + r" \\ \midrule")
    write("table_sigma0.tex", "\n".join(lines) + "\n")

    # ---- GMM K-sweep (50-image subset) ------------------------------------
    lines = []
    plot = ["K " + " ".join(SETS)]
    numbers["ksweep"] = {}
    for K in (1, 4, 16, 64, 200):
        rows = load(f"m2_ksweep/K{K}.csv")
        cell = mean_by(rows, "psnr", lambda k, r: (k[0], k[3]))
        per_set = [float(np.mean([cell[(s, sig)] for sig in SIGMAS])) for s in SETS]
        numbers["ksweep"][K] = per_set
        cells = [f"{cell[('P1', sig)]:.2f}" for sig in SIGMAS] + [f"{v:.2f}" for v in per_set]
        lines.append(f"{K} & " + " & ".join(cells) + r" \\")
        plot.append(f"{K} " + " ".join(f"{v:.4f}" for v in per_set))
    write("table_ksweep.tex", "\n".join(lines) + "\n")
    write("ksweep.dat", "\n".join(plot) + "\n")

    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
