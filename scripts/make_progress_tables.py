#!/usr/bin/env python3
"""LaTeX table fragments for the progress report (H2 texture, H3 stability, LPIPS).

Reads the JSON written by ``analyze_texture.py`` and ``analyze_h3_lpips.py`` and
writes ``report/generated/progress_{texture,h3,lpips}.tex``.

    PYTHONPATH=src python3 scripts/make_progress_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "report" / "generated"
SETS = ("P1", "P2_interp", "P2_extrap")
SET_TEX = {"P1": r"$P_1$", "P2_interp": r"$P_2$-interp", "P2_extrap": r"$P_2$-extrap"}
SIGMAS = ("1", "5", "15", "25")


def p_tex(p: float) -> str:
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{float(mantissa):.1f}\\times10^{{{int(exponent)}}}" if p < 1e-3 else f"{p:.2f}"


def texture() -> str:
    data = json.loads((GEN / "texture.json").read_text())["sets"]
    lines = []
    for s in SETS:
        for i, t in enumerate(("low", "medium", "high")):
            r = data[s][t]
            first = (r"\multirow{3}{*}{%s}" % SET_TEX[s]) if i == 0 else ""
            lines.append(
                f"{first} & {t} & {r['fista']:.2f} & {r['epll']:.2f} & {r['unet']:.2f} & {r['nafnet']:.2f}"
                f" & {r['nafnet'] - r['epll']:+.2f}" + r" \\"
            )
        sp = data[s]["spearman"]
        lines.append(
            r"\multicolumn{7}{r}{\footnotesize Spearman $\rho$(texture, NAFNet$-$EPLL) $= %+.2f$, $p = %s$} \\"
            % (sp["rho"], p_tex(sp["p"]))
        )
        if s != SETS[-1]:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def h3() -> str:
    data = json.loads((GEN / "h3_lpips.json").read_text())["h3"]
    lines = []
    for s in SETS:
        for i, sig in enumerate(SIGMAS):
            r = data[s][sig]
            first = (r"\multirow{4}{*}{%s}" % SET_TEX[s]) if i == 0 else ""
            n, e = r["drop"]["m1_nafnet"], r["drop"]["m2_admm_epll"]
            lines.append(
                f"{first} & {sig} & " + " & ".join(f"{v:.2f}" for v in n) + " & "
                + " & ".join(f"{v:.2f}" for v in e) + f" & {100 * r['epll_less']:.0f}" + r" \\"
            )
        if s != SETS[-1]:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def lpips() -> str:
    data = json.loads((GEN / "h3_lpips.json").read_text())["lpips"]
    methods = ("wiener", "richardson_lucy", "fista_tv", "admm_epll", "unet", "nafnet")
    lines = []
    for s in SETS:
        for i, sig in enumerate(SIGMAS + ("mean",)):
            values = data[s][sig]
            best = min(values.values())
            cells = [
                (r"\textbf{%.3f}" % values[m]) if values[m] == best else f"{values[m]:.3f}" for m in methods
            ]
            first = (r"\multirow{5}{*}{%s}" % SET_TEX[s]) if i == 0 else ""
            label = "mean" if sig == "mean" else sig
            lines.append(f"{first} & {label} & " + " & ".join(cells) + r" \\")
        if s != SETS[-1]:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def main() -> None:
    (GEN / "progress_texture.tex").write_text(texture())
    (GEN / "progress_h3.tex").write_text(h3())
    (GEN / "progress_lpips.tex").write_text(lpips())
    print("wrote progress_{texture,h3,lpips}.tex")


if __name__ == "__main__":
    main()
