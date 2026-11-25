#!/usr/bin/env python3
"""Paired statistics for the Phase 3 claims -> report/generated/phase3_stats.json.

Same convention as Phase 2 (scripts/analyze_phase2.py): on the protocol grid each
test image is one unit (mean over its kernels and sigma > 0), compared with a paired
Wilcoxon signed-rank test; on provided-blur benchmarks each image-kernel triplet is
one unit. Holm correction is applied within each family of comparisons:

* ``ladder``   placement variants vs. the no-kernel backbone (NAFNet, U-Net) and vs. each other (grid);
* ``mismatch`` mismatch-training variants vs. standard USRNet and vs. EPLL
               (grid, Levin real, Sun);
* ``probe``    kernel-swap probe: true kernel vs. no-blur and vs. nearest training kernel.

Where several seeds exist, each cell uses the mean over seeds, so the test pairs the
seed-averaged models. Missing files are skipped; rerun as runs finish.

    python3 scripts/stats_phase3.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "artifacts" / "eval" / "v2"
P3 = V2 / "phase3"
B = V2 / "benchmarks"
OUT = ROOT / "report" / "generated" / "phase3_stats.json"
SETS = ("P1", "P2_interp", "P2_extrap")

GRID = {
    "nafnet": [V2 / "m1_nafnet_test.csv", P3 / "nafnet_seed1_test.csv", P3 / "nafnet_seed2_test.csv"],
    "usrnet_full": [V2 / "m3_usrnet_full_test.csv", P3 / "usrnet_full_seed1_test.csv", P3 / "usrnet_full_seed2_test.csv"],
    **{m: [P3 / f"{m}_test.csv", P3 / f"{m}_seed1_test.csv", P3 / f"{m}_seed2_test.csv"]
       for m in ("nafnet_concat", "nafnet_film", "knafnet", "nafnet_unrolled")},
    **{m: [P3 / f"{m}_test.csv"] for m in ("usrnet_mm", "usrnet_mm_boundary", "usrnet_mm_boundary66k", "usrnet_mm_knoise",
                                           "unet_concat", "unet_film", "kunet")},
    "unet": [V2 / "m1_unet_test.csv"],
}
LADDER_PAIRS = [("nafnet_concat", "nafnet"), ("nafnet_film", "nafnet"), ("knafnet", "nafnet"),
                ("nafnet_unrolled", "nafnet"), ("knafnet", "nafnet_film"), ("nafnet_unrolled", "knafnet"),
                ("nafnet_unrolled", "usrnet_full"),
                ("unet_concat", "unet"), ("unet_film", "unet"), ("kunet", "unet"), ("kunet", "unet_film")]
MISMATCH = ("usrnet_mm_boundary", "usrnet_mm_boundary66k", "usrnet_mm_knoise", "usrnet_mm")
PROVIDED = {
    "levin_real": {"usrnet_full": B / "levin_real_est" / "usrnet.csv", "epll": B / "levin_real_est" / "epll.csv",
                   **{m: P3 / f"{m}_levin_real_est.csv" for m in MISMATCH}},
    "sun": {"usrnet_full": B / "sun_provided_est" / "usrnet.csv", "epll": B / "sun_provided_est" / "epll.csv",
            **{m: P3 / f"{m}_sun_est.csv" for m in MISMATCH}},
}
PROBE = ("nafnet_concat", "nafnet_film", "knafnet", "nafnet_unrolled", "usrnet_full", "wiener",
         "unet_concat", "unet_film", "kunet")


def holm(tests: list[dict]) -> None:
    order = np.argsort([t["p"] for t in tests])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(tests) - rank) * tests[i]["p"]))
        tests[i]["p_holm"] = running


def test(name: str, diff: np.ndarray, **extra) -> dict:
    diff = np.asarray(diff, dtype=float)
    p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    return {"name": name, "n": int(diff.size), "mean_diff_db": float(diff.mean()),
            "median_diff_db": float(np.median(diff)), "wins": int((diff > 0).sum()), "p": p, **extra}


def load_grid(paths: list[Path]) -> tuple[dict, int]:
    """(set, image, psf, sigma) -> PSNR averaged over the seed files that exist."""
    cells, n = defaultdict(list), 0
    for path in paths:
        if not path.exists():
            continue
        n += 1
        for r in csv.DictReader(path.open()):
            if float(r["sigma_255"]) > 0:
                cells[(r["psf_set"], r["image"], r["psf"], r["sigma_255"])].append(float(r["psnr"]))
    return {k: float(np.mean(v)) for k, v in cells.items() if len(v) == n}, n


def per_image(a: dict, b: dict, set_name: str) -> np.ndarray:
    groups = defaultdict(list)
    for key in set(a) & set(b):
        if key[0] == set_name:
            groups[key[1]].append(a[key] - b[key])
    return np.array([np.mean(v) for _, v in sorted(groups.items())])


def load_provided(path: Path) -> dict:
    return {r["triplet"]: float(r["psnr"]) for r in csv.DictReader(path.open())} if path.exists() else {}


def main() -> None:
    grid, seeds = {}, {}
    for m, paths in GRID.items():
        grid[m], seeds[m] = load_grid(paths)
    out = {"seeds_used": seeds, "families": {}}

    ladder = []
    for a, b in LADDER_PAIRS:
        if grid[a] and grid[b]:
            for s in SETS:
                ladder.append(test(f"{a} - {b} / {s}", per_image(grid[a], grid[b], s), unit="test image"))
    holm(ladder)
    out["families"]["ladder"] = ladder

    mismatch = []
    for m in MISMATCH:
        if grid[m]:
            for s in SETS:
                mismatch.append(test(f"{m} - usrnet_full / grid {s}", per_image(grid[m], grid["usrnet_full"], s),
                                     unit="test image"))
        for bench, files in PROVIDED.items():
            x = load_provided(files[m])
            if not x:
                continue
            for ref in ("usrnet_full", "epll"):
                y = load_provided(files[ref])
                keys = sorted(set(x) & set(y))
                if keys:
                    mismatch.append(test(f"{m} - {ref} / {bench}", [x[k] - y[k] for k in keys], unit="triplet"))
    holm(mismatch)
    out["families"]["mismatch"] = mismatch

    probe = []
    for m in PROBE:
        path = P3 / f"probe_{m}.csv"
        if not path.exists():
            continue
        cells = defaultdict(dict)
        for r in csv.DictReader(path.open()):
            if r["psf_set"] == "P2_extrap":
                cells[r["condition"]][(r["image"], r["psf"], r["sigma_255"])] = float(r["psnr"])
        for other in ("delta", "nearest"):
            groups = defaultdict(list)
            for key, v in cells["true"].items():
                groups[key[0]].append(v - cells[other][key])
            probe.append(test(f"{m}: true - {other} / P2_extrap", [np.mean(v) for v in groups.values()],
                              unit="test image"))
    holm(probe)
    out["families"]["probe"] = probe

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}  (seeds used: {seeds})")
    for fam, tests in out["families"].items():
        print(f"\n[{fam}]")
        for t in tests:
            print(f"  {t['name']:48s} {t['mean_diff_db']:+6.2f} dB  wins {t['wins']:3d}/{t['n']:<3d}  p_holm {t['p_holm']:.1e}")


if __name__ == "__main__":
    main()
