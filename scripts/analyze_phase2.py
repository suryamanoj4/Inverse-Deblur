#!/usr/bin/env python3
"""Phase 2 summary numbers: NAFNet learning curve (2A-1) and M3 on the v1 grid (2B).

Reads the v1 test-grid CSVs, keeps sigma > 0 (sigma = 0 is excluded from claims,
see the inverse-crime check), and writes report/generated/phase2.json with mean
PSNR and gain over the blurred input per PSF set, plus paired per-image
Wilcoxon tests (Holm-corrected) for the comparisons the deck quotes.

    PYTHONPATH=src python3 scripts/analyze_phase2.py
"""

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "nafnet_real50": "artifacts/eval/m1_nafnet_real50_test.csv",
    "nafnet_real100": "artifacts/eval/m1_nafnet_real100_test.csv",
    "nafnet": "checkpoints/m1/nafnet/eval_test.csv",
    "unet": "artifacts/eval/m1_unet_test.csv",
    "wiener": "artifacts/eval/m2_wiener_test.csv",
    "rl": "artifacts/eval/m2_richardson_lucy_test.csv",
    "fista": "artifacts/eval/m2_test/fista_tv_shard*.csv",
    "epll": "artifacts/eval/m2_test/admm_epll_shard*.csv",
    "usrnet": "artifacts/eval/m3_usrnet_test.csv",
    "ista_net": "artifacts/eval/m3_ista_net_test.csv",
    "lista": "artifacts/eval/m3_lista_test.csv",
}
SETS = ("P1", "P2_interp", "P2_extrap")
PAIRS = [("nafnet_real50", "nafnet"), ("nafnet_real100", "nafnet"),
         ("usrnet", "epll"), ("usrnet", "nafnet"), ("ista_net", "nafnet")]


def load(pattern: str) -> dict:
    rows = [r for p in sorted(glob.glob(str(ROOT / pattern))) for r in csv.DictReader(open(p))]
    return {(r["psf_set"], r["image"], r["psf"], r["sigma_255"]): (float(r["psnr"]), float(r.get("baseline_psnr") or "nan"))
            for r in rows if float(r["sigma_255"]) > 0}


def main() -> None:
    data = {k: load(v) for k, v in SOURCES.items()}
    cells = set.intersection(*(set(d) for d in data.values()))
    out = {"cells": len(cells), "mean": {}, "gain": {}, "blurred": {}, "tests": {}}
    for s in SETS:
        sub = [c for c in cells if c[0] == s]
        out["blurred"][s] = float(np.mean([data["epll"][c][1] for c in sub]))
    for k, d in data.items():
        out["mean"][k] = {s: float(np.mean([d[c][0] for c in cells if c[0] == s])) for s in SETS}
        out["mean"][k]["all"] = float(np.mean([d[c][0] for c in cells]))
        out["gain"][k] = {s: out["mean"][k][s] - out["blurred"][s] for s in SETS}
        out["mean"][k]["p2_drop"] = (out["gain"][k]["P2_interp"] + out["gain"][k]["P2_extrap"]) / 2 - out["gain"][k]["P1"]

    raw = []
    for a, b in PAIRS:
        for s in SETS:
            per_image = defaultdict(list)
            for c in cells:
                if c[0] == s:
                    per_image[c[1]].append(data[a][c][0] - data[b][c][0])
            diff = np.array([np.mean(v) for v in per_image.values()])
            raw.append((f"{a}-{b}/{s}", float(diff.mean()), float(wilcoxon(diff).pvalue)))
    order = np.argsort([p for _, _, p in raw])
    m, running = len(raw), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw[i][2]))
        name, delta, p = raw[i]
        out["tests"][name] = {"mean_diff_db": delta, "p": p, "p_holm": running}

    path = ROOT / "report" / "generated" / "phase2.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(cells)} cells -> {path}")
    for k in data:
        print(f"  {k:15}" + "".join(f"{out['mean'][k][s]:8.2f}" for s in SETS) + f"  drop {out['mean'][k]['p2_drop']:+.2f}")
    for name, t in out["tests"].items():
        print(f"  {name:28} {t['mean_diff_db']:+.2f} dB  p_holm={t['p_holm']:.1e}")


if __name__ == "__main__":
    main()
