#!/usr/bin/env python3
"""How much each unseen kernel destroys vs. how much each placement gains on it.

For every v2 kernel, the *spectral loss* is the fraction of a 256x256 frequency grid
where |K(w)| < ``--threshold`` (default 0.1). On P2_extrap, each placement variant's
gain over the no-kernel NAFNet (seed-averaged where seeds exist) is compared with that
loss by Spearman rank correlation. Writes report/generated/spectral_loss.json.

    python3 scripts/analyze_spectral_loss.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deblur.operators.fft_conv import kernel_spectrum  # noqa: E402
from deblur.protocol import ProtocolConfig  # noqa: E402

V2 = ROOT / "artifacts" / "eval" / "v2"
P3 = V2 / "phase3"
MODELS = {
    "nafnet": [V2 / "m1_nafnet_test.csv", P3 / "nafnet_seed1_test.csv", P3 / "nafnet_seed2_test.csv"],
    **{m: [P3 / f"{m}_test.csv", P3 / f"{m}_seed1_test.csv", P3 / f"{m}_seed2_test.csv"]
       for m in ("nafnet_concat", "nafnet_film", "knafnet")},
    "nafnet_unrolled": [P3 / "nafnet_unrolled_test.csv"],
    "usrnet_full": [V2 / "m3_usrnet_full_test.csv", P3 / "usrnet_full_seed1_test.csv", P3 / "usrnet_full_seed2_test.csv"],
}


def per_kernel(paths: list[Path]) -> dict[str, float]:
    """Mean PSNR over images and sigma > 0 per kernel, averaged over existing seed files."""
    runs = []
    for path in paths:
        if path.exists():
            g = defaultdict(list)
            for r in csv.DictReader(path.open()):
                if float(r["sigma_255"]) > 0:
                    g[r["psf"]].append(float(r["psnr"]))
            runs.append({k: np.mean(v) for k, v in g.items()})
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]} if runs else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    sets = ProtocolConfig(data_root="data", protocol_version="v2").psf_sets()
    loss, kind = {}, {}
    for set_name in ("P1", "P2_interp", "P2_extrap"):
        for psf in sets[set_name]:
            mag = np.abs(kernel_spectrum(psf.kernel, (256, 256)))
            loss[psf.name] = float((mag < args.threshold).mean())
            kind[psf.name] = (set_name, psf.kind, list(psf.kernel.shape))
    psnr = {m: per_kernel(p) for m, p in MODELS.items()}
    extrap = sorted((k for k in loss if kind[k][0] == "P2_extrap"), key=lambda k: loss[k])

    out = {"threshold": args.threshold,
           "mean_loss_per_set": {s: float(np.mean([loss[k] for k in loss if kind[k][0] == s]))
                                 for s in ("P1", "P2_interp", "P2_extrap")},
           "extrap_kernels": [], "spearman_gain_vs_loss": {}}
    for k in extrap:
        row = {"kernel": k, "family": kind[k][1], "size": kind[k][2], "loss": loss[k], "nafnet_psnr": psnr["nafnet"][k]}
        row.update({f"gain_{m}": psnr[m][k] - psnr["nafnet"][k] for m in MODELS if m != "nafnet" and psnr[m]})
        out["extrap_kernels"].append(row)
    for m in MODELS:
        if m != "nafnet" and psnr[m]:
            r = spearmanr([loss[k] for k in extrap], [psnr[m][k] - psnr["nafnet"][k] for k in extrap])
            out["spearman_gain_vs_loss"][m] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(extrap)}
    # size confound: kernel support vs loss on P2_extrap
    r = spearmanr([loss[k] for k in extrap], [max(kind[k][2]) for k in extrap])
    out["spearman_loss_vs_kernel_size"] = {"rho": float(r.statistic), "p": float(r.pvalue)}

    path = ROOT / "report" / "generated" / "spectral_loss.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    for m, v in out["spearman_gain_vs_loss"].items():
        print(f"  {m:16s} rho {v['rho']:+.2f}  p {v['p']:.1e}")
    print(f"  loss vs kernel size rho {out['spearman_loss_vs_kernel_size']['rho']:+.2f}; "
          f"mean loss per set {out['mean_loss_per_set']}")


if __name__ == "__main__":
    main()
