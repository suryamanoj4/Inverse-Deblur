#!/usr/bin/env python3
"""Collect Phase 3 numbers (vault/Phase 3 Pre-registration.md) into report/generated/phase3.json.

Sections: real-blur benchmarks and the circular-Sun control, the kernel-placement
ladder (with NAFNet / USRNet seeds), the kernel-swap probe, the 2x2 mismatch
ablation, and the linear-filter test (E-X). Missing files are skipped, so the
script can be rerun as runs finish.

    python3 scripts/analyze_phase3.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "artifacts" / "eval" / "v2"
P3 = V2 / "phase3"
B = V2 / "benchmarks"
OUT = ROOT / "report" / "generated" / "phase3.json"
SETS = ("P1", "P2_interp", "P2_extrap")
BIG_LEVIN = ("flit04", "flit08")


def grid(path: Path, sets=SETS + ("P2_real",)) -> dict | None:
    """Mean PSNR over sigma > 0 per kernel set, and per family on P2_extrap."""
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    out = {}
    for s in sets:
        v = [r["psnr"] for r in rows if r["psf_set"] == s and r["sigma_255"] > 0]
        if v:
            out[s] = float(np.mean(v))
    fam = {}
    for k in sorted({r["kind"] for r in rows if r["psf_set"] == "P2_extrap"}):
        v = [r["psnr"] for r in rows if r["psf_set"] == "P2_extrap" and r["kind"] == k and r["sigma_255"] > 0]
        fam[k] = float(np.mean(v))
    if fam:
        out["extrap_family"] = fam
    return out


def provided(path: Path) -> float | None:
    return json.loads(path.read_text())["mean_psnr"] if path.exists() else None


def levin_split(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    big = [float(r["psnr"]) for r in rows if r["triplet"][-6:] in BIG_LEVIN]
    other = [float(r["psnr"]) for r in rows if r["triplet"][-6:] not in BIG_LEVIN]
    return {"all": float(np.mean(big + other)), "big2": float(np.mean(big)), "other6": float(np.mean(other))}


def seeds(paths: list[Path]) -> dict | None:
    g = [grid(p, SETS) for p in paths]
    g = [x for x in g if x]
    if not g:
        return None
    arr = np.array([[x[s] for s in SETS] for x in g])
    return {"n": len(g), "mean": dict(zip(SETS, arr.mean(0).tolist())),
            "std": dict(zip(SETS, (arr.std(0, ddof=1) if len(g) > 1 else np.zeros(3)).tolist()))}


def main() -> None:
    out: dict = {}

    # benchmarks: provided blur (estimated sigma, interpolated lambda) and circular-Sun control
    bench = {}
    for key, name in [("usrnet", "usrnet"), ("epll", "epll"), ("fista", "fista"), ("nafnet", "nafnet"),
                      ("ista_net", "ista_net"), ("wiener", "wiener")]:
        bench[key] = {"levin_real": provided(B / "levin_real_est" / f"{name}_summary.json"),
                      "sun": provided(B / "sun_provided_est" / f"{name}_summary.json"),
                      "sun_circular": provided(B / "sun_circular" / f"{name}_summary.json"),
                      "p2_real": (grid(B / "p2_real" / f"{name}_summary.json") or {}).get("P2_real")}
    bench["epll"]["sun_circular"] = bench["epll"]["sun_circular"] or provided(B / "sun_circular" / "epll_summary.json")
    out["benchmarks"] = bench

    # placement ladder + reference rows
    ladder = {"nafnet": grid(V2 / "m1_nafnet_test_summary.json"),
              "usrnet_full": grid(V2 / "m3_usrnet_full_test_summary.json")}
    for m in ("nafnet_concat", "nafnet_film", "knafnet", "nafnet_unrolled", "usrnet_mm",
              "usrnet_mm_boundary", "usrnet_mm_knoise"):
        g = grid(P3 / f"{m}_test_summary.json")
        if g:
            g["p2_real"] = (grid(P3 / f"{m}_p2real_summary.json") or {}).get("P2_real")
            g["levin_real"] = levin_split(P3 / f"{m}_levin_real_est.csv")
            g["sun"] = provided(P3 / f"{m}_sun_est_summary.json")
            g["sun_circular"] = provided(P3 / f"{m}_sun_circular_summary.json")
            ladder[m] = g
    ladder["usrnet_full"]["levin_real"] = levin_split(B / "levin_real_est" / "usrnet.csv")
    out["ladder"] = ladder
    out["seeds"] = {
        "nafnet": seeds([V2 / "m1_nafnet_test_summary.json"] + [P3 / f"nafnet_seed{s}_test_summary.json" for s in (1, 2)]),
        "usrnet_full": seeds([V2 / "m3_usrnet_full_test_summary.json"]
                             + [P3 / f"usrnet_full_seed{s}_test_summary.json" for s in (1, 2)]),
    }
    for m in ("nafnet_concat", "nafnet_film", "knafnet"):
        out["seeds"][m] = seeds([P3 / f"{m}_test_summary.json"] + [P3 / f"{m}_seed{s}_test_summary.json" for s in (1, 2)])

    # kernel-swap probe
    probe = {}
    for m in ("wiener", "nafnet", "nafnet_concat", "nafnet_film", "knafnet", "nafnet_unrolled", "usrnet_full"):
        p = P3 / f"probe_{m}_summary.json"
        if p.exists():
            probe[m] = json.loads(p.read_text())["mean_psnr"]
    out["probe"] = probe

    # E-X linear-filter test
    lf = {}
    for p in sorted((ROOT / "report" / "generated" / "linear_filter").glob("*.json")):
        d = json.loads(p.read_text())
        lf[p.stem] = {"shared_r2": d["shared_r2"], "per_kernel_r2": d["per_kernel_r2"],
                      "extrap_per_kernel_r2": d["by_set"]["P2_extrap"]["per_kernel_r2"]}
    out["linear_filter"] = lf

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    for m, g in ladder.items():
        if g:
            print(f"{m:20s}", " ".join(f"{g.get(s, float('nan')):6.2f}" for s in SETS))


if __name__ == "__main__":
    main()
