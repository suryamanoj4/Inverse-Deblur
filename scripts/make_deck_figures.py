#!/usr/bin/env python3
"""Figures for the mid-evaluation deck: pictures of every protocol blur kernel.

Each kernel is normalised to its peak, centred in a common cell and upscaled
with nearest-neighbour so individual pixels stay visible. Writes
``report/generated/deck/kernels_<set>.png`` (one grid per PSF set) on the deck's
ink background.

    PYTHONPATH=src python3 scripts/make_deck_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deblur.psf import protocol_psf_sets

OUT = ROOT / "report" / "generated" / "deck"
INK = np.array([0x1B, 0x20, 0x30], dtype=np.float64)
TEAL = np.array([0x2A, 0x9D, 0x8F], dtype=np.float64)
SCALE = 8
GAP = 3  # cells, in kernel pixels before scaling
COLUMNS = {"P1": 6, "P2_interp": 6, "P2_extrap": 4}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = protocol_psf_sets()
    cell = max(max(p.kernel.shape) for psfs in sets.values() for p in psfs) + 2
    for name, psfs in sets.items():
        cols = COLUMNS[name]
        rows = -(-len(psfs) // cols)
        pitch = cell + GAP
        canvas = np.zeros((rows * pitch + GAP, cols * pitch + GAP))
        frame = np.zeros_like(canvas, dtype=bool)
        for i, psf in enumerate(psfs):
            r, c = divmod(i, cols)
            top, left = GAP + r * pitch, GAP + c * pitch
            frame[top : top + cell, left : left + cell] = True
            k = psf.kernel / psf.kernel.max()
            kh, kw = k.shape
            oy, ox = top + (cell - kh) // 2, left + (cell - kw) // 2
            canvas[oy : oy + kh, ox : ox + kw] = k
        # ink background, slightly lighter cell, kernel intensity in white
        cell_bg = INK + 0.12 * (255 - INK)
        rgb = np.where(frame[..., None], cell_bg, INK)
        rgb = rgb + canvas[..., None] ** 0.7 * (255 - rgb)
        rgb = np.kron(rgb, np.ones((SCALE, SCALE, 1)))
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(OUT / f"kernels_{name}.png")
        print(f"{name}: {len(psfs)} kernels -> {OUT / f'kernels_{name}.png'}")


if __name__ == "__main__":
    main()
