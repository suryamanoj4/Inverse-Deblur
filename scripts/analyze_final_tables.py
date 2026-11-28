#!/usr/bin/env python3
"""Tables for the final report -> report/generated/final_tables.json.

* ``ci``         per-image bootstrap 95% intervals (10,000 resamples) of the mean PSNR on each
                 v2 set and of the paired ladder differences on P2_extrap (seed-averaged cells,
                 sigma > 0, the same units as scripts/stats_phase3.py);
* ``family``     mean PSNR by kernel family (sigma > 0) for the ladder, USRNet and the M2 solvers;
* ``cost``       parameters (from checkpoint metadata) and median training speed (iterations/s
                 from log.jsonl; the GPU was shared, so this is indicative only).

    python3 scripts/analyze_final_tables.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("stats_phase3", ROOT / "scripts" / "stats_phase3.py")
stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stats)

V2 = ROOT / "artifacts" / "eval" / "v2"
OUT = ROOT / "report" / "generated" / "final_tables.json"
LADDER = ("nafnet", "nafnet_concat", "nafnet_film", "nafnet_dc", "knafnet", "nafnet_unrolled", "usrnet_full")
M2 = {"wiener": "wiener_shard*.csv", "richardson_lucy": "richardson_lucy_shard*.csv", "admm_epll": "admm_epll*.csv"}
CHECKPOINTS = {
    "nafnet": "checkpoints/v2/m1/nafnet", "unet": "checkpoints/v2/m1/unet",
    "usrnet_full": "checkpoints/v2/m3/usrnet_full",
    **{m: f"checkpoints/v2/p3/{m}" for m in ("nafnet_concat", "nafnet_film", "nafnet_dc", "knafnet", "nafnet_unrolled",
                                              "unet_concat", "unet_film", "kunet")},
}
RNG = np.random.default_rng(0)


def boot(values: np.ndarray, n: int = 10_000) -> list[float]:
    idx = RNG.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(1)
    return [float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def image_means(cells: dict, set_name: str) -> np.ndarray:
    g = defaultdict(list)
    for (s, image, _, _), v in cells.items():
        if s == set_name:
            g[image].append(v)
    return np.array([np.mean(v) for _, v in sorted(g.items())])


def family_means(rows: list[dict]) -> dict:
    g = defaultdict(list)
    for r in rows:
        if float(r["sigma_255"]) > 0:
            g[f"{r['psf_set']}/{r['kind']}"].append(float(r["psnr"]))
    return {k: float(np.mean(v)) for k, v in sorted(g.items())}


def main() -> None:
    out = {"ci": {}, "ladder_diff_ci": {}, "family": {}, "cost": {}}
    grids = {m: stats.load_grid(stats.GRID[m])[0] for m in LADDER}
    for m, cells in grids.items():
        out["ci"][m] = {s: boot(image_means(cells, s)) for s in stats.SETS}
    for a, b in [("nafnet_concat", "nafnet"), ("nafnet_film", "nafnet"), ("nafnet_dc", "nafnet"), ("knafnet", "nafnet"),
                 ("nafnet_unrolled", "nafnet"), ("knafnet", "nafnet_dc"), ("nafnet_unrolled", "usrnet_full")]:
        out["ladder_diff_ci"][f"{a} - {b}"] = boot(stats.per_image(grids[a], grids[b], "P2_extrap"))

    for m in LADDER:
        runs = [family_means(list(csv.DictReader(p.open()))) for p in stats.GRID[m] if p.exists()]
        out["family"][m] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    for m, pattern in M2.items():
        rows = [r for p in sorted((V2 / "m2_test").glob(pattern)) for r in csv.DictReader(p.open())]
        out["family"][m] = family_means(rows)
    blurred = defaultdict(list)
    for r in csv.DictReader(stats.GRID["nafnet"][0].open()):
        if float(r["sigma_255"]) > 0:
            blurred[f"{r['psf_set']}/{r['kind']}"].append(float(r["baseline_psnr"]))
    out["family"]["blurred"] = {k: float(np.mean(v)) for k, v in sorted(blurred.items())}

    for m, d in CHECKPOINTS.items():
        meta = ROOT / (stats.GRID[m][0].as_posix().replace(".csv", "_metadata.json")) if m in stats.GRID else None
        params = None
        if meta and meta.exists():
            params = json.loads(meta.read_text())["arm"]["checkpoint_metadata"].get("parameters")
        speeds = [json.loads(line).get("it_per_s") for line in (ROOT / d / "log.jsonl").open()
                  if '"event": "train"' in line] if (ROOT / d / "log.jsonl").exists() else []
        speeds = [s for s in speeds if s]
        out["cost"][m] = {"parameters": params, "median_it_per_s": float(np.median(speeds)) if speeds else None}

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    for m in LADDER:
        e = out["ci"][m]["P2_extrap"]
        print(f"  {m:16s} P2_extrap {e[0]:.2f} [{e[1]:.2f}, {e[2]:.2f}]  params {out['cost'][m]['parameters']}  "
              f"it/s {out['cost'][m]['median_it_per_s']}")
    for k, v in out["ladder_diff_ci"].items():
        print(f"  {k:32s} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]")


if __name__ == "__main__":
    main()
