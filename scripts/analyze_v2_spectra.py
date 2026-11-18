#!/usr/bin/env python3
"""Protocol v2: gain over the blurred input per kernel family, and its relation to
kernel conditioning (fraction of frequencies with |K| < 0.1 on the 321x481 grid).

Uses whichever v2 test-grid CSVs exist (sigma > 0 only); the blurred baseline comes
from the Wiener CSVs, which carry baseline_psnr for every cell.

    PYTHONPATH=src python3 scripts/analyze_v2_spectra.py
"""

import collections
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from deblur.psf.library import protocol_v2_psf_sets

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "artifacts" / "eval" / "v2"
SOURCES = {
    "nafnet": "m1_nafnet_test.csv",
    "unet": "m1_unet_test.csv",
    "usrnet": "m3_usrnet_test.csv",
    "ista_net": "m3_ista_net_test.csv",
    "lista": "m3_lista_test.csv",
    "wiener": "m2_test/wiener_shard*.csv",
    "rl": "m2_test/richardson_lucy_shard*.csv",
    "epll": "m2_test/admm_epll*.csv",
    "fista": "m2_test/fista_tv*.csv",
}
THRESHOLD = 0.1


def load(pattern: str) -> dict:
    out = collections.defaultdict(list)
    for path in sorted(glob.glob(str(V2 / pattern))):
        if path.endswith(("_summary.json", "_metadata.json")):
            continue
        for row in csv.DictReader(open(path)):
            if float(row["sigma_255"]) > 0:
                out[row["psf"]].append(float(row["psnr"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main() -> None:
    psf_sets = {k: v for k, v in protocol_v2_psf_sets().items() if k != "P1_pool"}
    base_rows = collections.defaultdict(list)
    for path in glob.glob(str(V2 / "m2_test/wiener_shard*.csv")):
        for row in csv.DictReader(open(path)):
            if float(row["sigma_255"]) > 0:
                base_rows[row["psf"]].append(float(row["baseline_psnr"]))
    blurred = {k: float(np.mean(v)) for k, v in base_rows.items()}
    methods = {m: d for m, p in SOURCES.items() if (d := load(p))}

    kernels = []
    for set_name, psfs in psf_sets.items():
        for psf in psfs:
            spectrum = np.abs(np.fft.rfft2(psf.kernel, s=(321, 481)))
            entry = {"set": set_name, "psf": psf.name, "family": psf.name.split("_")[0],
                     "frac_small": float((spectrum < THRESHOLD).mean()), "blurred": blurred[psf.name],
                     "gain": {m: d[psf.name] - blurred[psf.name] for m, d in methods.items() if psf.name in d}}
            kernels.append(entry)

    families = collections.defaultdict(list)
    for k in kernels:
        families[(k["set"], k["family"])].append(k)
    print(f"{'set/family':22}{'|K|<0.1':>8}{'blur':>7}" + "".join(f"{m:>9}" for m in methods))
    for key in sorted(families):
        ks = families[key]
        print(f"{key[0] + '/' + key[1]:22}{np.mean([k['frac_small'] for k in ks]):8.2f}"
              f"{np.mean([k['blurred'] for k in ks]):7.2f}"
              + "".join(f"{np.mean([k['gain'][m] for k in ks]):+9.2f}" for m in methods))

    correlations = {}
    frac = np.array([k["frac_small"] for k in kernels])
    for m in methods:
        gains = np.array([k["gain"][m] for k in kernels])
        rho, p = spearmanr(frac, gains)
        correlations[m] = {"spearman_rho": float(rho), "p": float(p)}
        print(f"Spearman(frac |K|<{THRESHOLD}, {m} gain) over {len(kernels)} kernels: rho={rho:+.2f}  p={p:.1e}")

    out = ROOT / "report" / "generated" / "v2_spectra.json"
    out.write_text(json.dumps({"threshold": THRESHOLD, "kernels": kernels, "spearman": correlations}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
