#!/usr/bin/env python3
"""Figures for the final report (report/final_report.tex) -> report/generated/final/*.pdf.

Reads only the generated JSON files (final_tables.json, spectral_loss.json), so it runs
without the evaluation CSVs. Needs matplotlib.

    python3 scripts/make_final_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "report" / "generated"
OUT = GEN / "final"
NAMES = {"nafnet": "No kernel", "nafnet_concat": "Input maps", "nafnet_film": "FiLM every block",
         "nafnet_dc": "Data step, no FiLM", "knafnet": "FiLM + data step", "nafnet_unrolled": "4-stage unrolled",
         "usrnet_full": "USRNet", "admm_epll": "ADMM-EPLL", "blurred": "Blurred input"}
COLORS = {"P1": "#9aa3b5", "P2_extrap": "#6b4c9a"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def ladder(t: dict) -> None:
    models = ["nafnet", "nafnet_concat", "nafnet_film", "nafnet_dc", "knafnet", "nafnet_unrolled", "usrnet_full"]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for j, (s, label) in enumerate([("P1", "P1 (training families)"), ("P2_extrap", "P2-extrap (unseen families)")]):
        ys = [i + (-0.2 if s == "P1" else 0.2) for i in range(len(models))]
        mean = [t["ci"][m][s][0] for m in models]
        err = [[t["ci"][m][s][0] - t["ci"][m][s][1] for m in models], [t["ci"][m][s][2] - t["ci"][m][s][0] for m in models]]
        ax.barh(ys, mean, height=0.38, color=COLORS[s], label=label, xerr=err, error_kw={"lw": 0.8, "capsize": 2})
        for y, v in zip(ys, mean):
            ax.text(v + 0.45, y, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(models)), [NAMES[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(15, 30)
    ax.set_xlabel("PSNR (dB), mean over σ > 0, 3 seeds; bars: 95% bootstrap CI over images")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "ladder.pdf")


def spectral(sp: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    style = {"nafnet_film": ("FiLM", "#c0504d", "o"), "knafnet": ("FiLM + data step", "#e0a030", "s"),
             "nafnet_dc": ("Data step, no FiLM", "#3a9a8a", "^"), "nafnet_unrolled": ("4-stage unrolled", "#6b4c9a", "D")}
    for m, (label, c, mk) in style.items():
        ax.scatter([100 * r["loss"] for r in sp["extrap_kernels"]], [r[f"gain_{m}"] for r in sp["extrap_kernels"]],
                   s=16, color=c, marker=mk, label=f"{label} (ρ = {sp['spearman_gain_vs_loss'][m]['rho']:.2f})")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xlabel("spectrum with |K| < 0.1 (%)")
    ax.set_ylabel("gain over no-kernel NAFNet (dB)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "spectral.pdf")


def families(t: dict) -> None:
    models = ["blurred", "nafnet", "nafnet_concat", "nafnet_film", "nafnet_dc", "knafnet", "nafnet_unrolled",
              "usrnet_full", "admm_epll"]
    fams = [("P2_extrap/defocus", "Defocus"), ("P2_extrap/ring", "Ring"), ("P2_extrap/ghost", "Ghost")]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    w = 0.09
    for i, m in enumerate(models):
        xs = [k + (i - len(models) / 2 + 0.5) * w for k in range(len(fams))]
        ax.bar(xs, [t["family"][m][f] for f, _ in fams], width=w, label=NAMES[m],
               color="#bbbbbb" if m == "blurred" else cmap(i % 10))
    ax.set_xticks(range(len(fams)), [n for _, n in fams])
    ax.set_ylim(12, 33)
    ax.set_ylabel("PSNR (dB), σ > 0")
    ax.legend(fontsize=6.5, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "families.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = json.loads((GEN / "final_tables.json").read_text())
    ladder(t)
    spectral(json.loads((GEN / "spectral_loss.json").read_text()))
    families(t)
    print(f"wrote {sorted(p.name for p in OUT.glob('*.pdf'))}")


if __name__ == "__main__":
    main()
