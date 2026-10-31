#!/usr/bin/env python3
"""H2 analysis: does prior depth matter most on textured images?

Each test image gets one texture score: the mean gradient magnitude of its
sharp luminance image. The 200 test images are split into terciles (low /
medium / high texture). For every method, PSNR is averaged per image over the
PSFs of a set and over sigma > 0, then summarised per tercile. The paired
difference NAFNet - ADMM-EPLL (deep vs shallow learned prior) is related to
texture by Spearman correlation over the 200 images, and the low- and
high-texture terciles are compared with a Mann-Whitney U test.

Writes ``report/generated/texture.json`` and ``texture.md`` (Markdown tables).

    PYTHONPATH=src python3 scripts/analyze_texture.py
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deblur.data import BSDS500, load_image
from deblur.protocol import ProtocolConfig

EVAL = ROOT / "artifacts" / "eval"
OUT = ROOT / "report" / "generated"
SETS = ("P1", "P2_interp", "P2_extrap")
SET_MD = {"P1": "$P_1$", "P2_interp": "$P_2$-interp", "P2_extrap": "$P_2$-extrap"}
METHODS = {
    "fista": ("FISTA-TV", "m2_test/fista_tv_shard*.csv"),
    "epll": ("ADMM-EPLL", "m2_test/admm_epll_shard*.csv"),
    "unet": ("U-Net", "m1_unet_test.csv"),
    "nafnet": ("NAFNet", str(ROOT / "checkpoints/m1/nafnet/eval_test.csv")),
}
TERCILES = ("low", "medium", "high")


def per_image(pattern: str, column: str = "psnr") -> dict[tuple[str, str], float]:
    """(set, image) -> mean over PSFs and sigma > 0."""
    paths = sorted(glob.glob(pattern if pattern.startswith("/") else str(EVAL / pattern)))
    groups = defaultdict(list)
    for path in paths:
        for r in csv.DictReader(open(path)):
            if int(r["sigma_255"]) > 0 and r.get(column):
                groups[(r["psf_set"], r["image"])].append(float(r[column]))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def texture_scores(config: ProtocolConfig) -> dict[str, float]:
    test = BSDS500(config.data_root, split="test", luminance=config.luminance)
    scores = {}
    for path in test.paths:
        x = load_image(path, luminance=config.luminance)
        gy, gx = np.gradient(x)
        scores[path.relative_to(test.image_root).as_posix()] = float(np.mean(np.hypot(gx, gy)))
    return scores


def main() -> None:
    config = ProtocolConfig(data_root=str(ROOT / "data"))
    texture = texture_scores(config)
    images = sorted(texture, key=texture.get)
    n = len(images)
    tercile = {image: TERCILES[min(2, 3 * rank // n)] for rank, image in enumerate(images)}
    edges = [texture[images[n // 3]], texture[images[2 * n // 3]]]

    psnr = {m: per_image(spec[1]) for m, spec in METHODS.items()}
    blurred = per_image("m2_wiener_test.csv", "baseline_psnr")
    result: dict = {"texture_edges": edges, "sets": {}}
    md = []

    md.append("| Set | Texture | Blurred | FISTA-TV | ADMM-EPLL | U-Net | NAFNet | NAFNet − EPLL | EPLL − FISTA |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SETS:
        result["sets"][s] = {}
        for t in TERCILES:
            members = [i for i in images if tercile[i] == t]
            row = {m: float(np.mean([psnr[m][(s, i)] for i in members])) for m in METHODS}
            row["blurred"] = float(np.mean([blurred[(s, i)] for i in members]))
            result["sets"][s][t] = row
            md.append(
                f"| {SET_MD[s] if t == 'low' else ''} | {t} | {row['blurred']:.2f} | {row['fista']:.2f} | "
                f"{row['epll']:.2f} | {row['unet']:.2f} | {row['nafnet']:.2f} | "
                f"{row['nafnet'] - row['epll']:+.2f} | {row['epll'] - row['fista']:+.2f} |"
            )

    md.append("")
    md.append("| Set | Spearman ρ (texture, NAFNet − EPLL) | p | low vs high tercile, Mann–Whitney p |")
    md.append("|---|---:|---:|---:|")
    for s in SETS:
        diff = {i: psnr["nafnet"][(s, i)] - psnr["epll"][(s, i)] for i in images}
        rho, p_rho = spearmanr([texture[i] for i in images], [diff[i] for i in images])
        low = [diff[i] for i in images if tercile[i] == "low"]
        high = [diff[i] for i in images if tercile[i] == "high"]
        p_mw = float(mannwhitneyu(low, high).pvalue)
        result["sets"][s]["spearman"] = {"rho": float(rho), "p": float(p_rho), "mannwhitney_p": p_mw}
        md.append(f"| {SET_MD[s]} | {rho:+.3f} | {p_rho:.2g} | {p_mw:.2g} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "texture.json").write_text(json.dumps(result, indent=1))
    (OUT / "texture.md").write_text("\n".join(md) + "\n")
    print(f"texture tercile edges (mean |grad|): {edges[0]:.4f}, {edges[1]:.4f}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
