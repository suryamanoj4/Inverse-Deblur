#!/usr/bin/env python3
"""Reconstruction SNR from the stored PSNR, without re-running any method.

With data_range = 1, PSNR = 10 log10(1/MSE) and SNR = 10 log10(mean(x^2)/MSE), so
SNR = PSNR + 10 log10(mean(x^2)) exactly, where x is the clean (luminance) test image.

    PYTHONPATH=src python3 scripts/snr_from_psnr.py CSV [CSV ...]
"""
import collections, csv, sys
from pathlib import Path
import numpy as np
from deblur.data.datasets import load_image

ROOT = Path(__file__).resolve().parents[1]
energy_db = {}

def offset(image: str) -> float:
    if image not in energy_db:
        x = load_image(ROOT / "data" / "BSDS500" / "images" / image) if (ROOT / "data/BSDS500/images" / image).exists() \
            else load_image(next((ROOT / "data").rglob(Path(image).name)))
        energy_db[image] = 10 * np.log10(np.mean(x ** 2))
    return energy_db[image]

for path in sys.argv[1:]:
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        p = float(r["psnr"]); s = p + offset(r["image"])
        c = cells[(r["psf_set"], r["sigma_255"])]
        c["psnr"].append(p); c["snr"].append(s); c["max_psnr"].append(p)
    print(f"\n{path}\n{'set':10}{'sigma':>6}{'PSNR':>9}{'SNR':>9}{'max PSNR':>10}{'max SNR':>9}")
    for (ps, sg), c in sorted(cells.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
        print(f"{ps:10}{sg:>6}{np.mean(c['psnr']):9.2f}{np.mean(c['snr']):9.2f}{max(c['psnr']):10.1f}{max(c['snr']):9.1f}")
