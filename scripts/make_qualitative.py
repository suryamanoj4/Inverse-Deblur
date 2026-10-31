#!/usr/bin/env python3
"""Qualitative figure for the M1 vs M2 report.

For each scenario (PSF set, PSF, sigma) the image is chosen without looking at
the pictures: it is the test image whose ADMM-EPLL - NAFNet PSNR difference is
the median over the 200 test images for that cell. The crop is the window with
the largest gradient energy in the sharp image. Restorations are recomputed on
the protocol observation (same seed as the evaluation), and PSNR is reported on
the full image, as in the tables.

Writes PNG tiles and ``qualitative.tex`` (a table of tiles with PSNR captions)
to ``report/generated/qualitative/``.

    PYTHONPATH=src python3 scripts/make_qualitative.py
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deblur.data import BSDS500, load_image
from deblur.methods.classical import GMM, wiener
from deblur.methods.classical.torch_backend import TorchGMM, admm_epll_torch, fista_tv_torch
from deblur.methods.direct import build_model
from deblur.metrics import psnr
from deblur.protocol import ProtocolConfig, observe
from deblur.psf import protocol_psf_sets
from deblur.training import model_restorer

OUT = ROOT / "report" / "generated" / "qualitative"
EVAL = ROOT / "artifacts" / "eval"
CROP = 160
TILE_SCALE = 2  # nearest-neighbour upscaling so pixels stay visible in print

SCENARIOS = [
    ("P1", "motion_l15_a45", 5, "In distribution: $P_1$ motion, 15\\,px at $45^\\circ$, $\\sigma=5$"),
    ("P2_interp", "motion_l13_a67.5", 1, "Unseen motion: $P_2$-interp, 13\\,px at $67.5^\\circ$, $\\sigma=1$"),
    ("P2_extrap", "defocus_r7", 5, "Unseen family: $P_2$-extrap defocus, radius 7\\,px, $\\sigma=5$"),
    ("P2_interp", "gaussian_sigma1.4", 15, "High noise: $P_2$-interp Gaussian $\\sigma_k=1.4$, $\\sigma=15$"),
]
COLUMNS = ["sharp", "observed", "wiener", "fista", "epll", "nafnet"]
HEADERS = ["Sharp", "Observed", "Wiener", "FISTA-TV", "ADMM-EPLL", "NAFNet"]


def cell_psnr(pattern: str, set_name: str, psf: str, sigma: int) -> dict[str, float]:
    out = {}
    for path in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(path)):
            if r["psf_set"] == set_name and r["psf"] == psf and int(r["sigma_255"]) == sigma:
                out[r["image"]] = float(r["psnr"])
    return out


def median_image(set_name: str, psf: str, sigma: int) -> str:
    epll = cell_psnr(str(EVAL / "m2_test/admm_epll_shard*.csv"), set_name, psf, sigma)
    naf = cell_psnr(str(ROOT / "checkpoints/m1/nafnet/eval_test.csv"), set_name, psf, sigma)
    diffs = sorted((epll[i] - naf[i], i) for i in epll)
    return diffs[len(diffs) // 2][1]


def detail_crop(x: np.ndarray, size: int) -> tuple[int, int]:
    gy, gx = np.gradient(x)
    energy = gx * gx + gy * gy
    integral = energy.cumsum(0).cumsum(1)
    H, W = x.shape
    best, where = -1.0, (0, 0)
    for top in range(0, H - size + 1, 8):
        for left in range(0, W - size + 1, 8):
            b, r = top + size - 1, left + size - 1
            total = integral[b, r]
            if top:
                total -= integral[top - 1, r]
            if left:
                total -= integral[b, left - 1]
            if top and left:
                total += integral[top - 1, left - 1]
            if total > best:
                best, where = total, (top, left)
    return where


def save_tile(img: np.ndarray, path: Path, kernel: np.ndarray | None = None) -> None:
    tile = np.clip(img, 0.0, 1.0)
    tile = np.kron(tile, np.ones((TILE_SCALE, TILE_SCALE)))
    rgb = np.repeat((tile * 255.0 + 0.5).astype(np.uint8)[..., None], 3, axis=2)
    if kernel is not None:  # kernel inset in the top-left corner, framed in red
        k = kernel / kernel.max()
        scale = max(1, 64 // max(k.shape))
        k = np.kron(k, np.ones((scale, scale)))
        h, w = k.shape
        rgb[: h + 4, : w + 4] = (200, 30, 30)
        rgb[2 : h + 2, 2 : w + 2] = np.repeat((k * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    Image.fromarray(rgb).save(path)


def load_nafnet(device: str):
    state = torch.load(ROOT / "checkpoints/m1/nafnet/best.pt", map_location=device, weights_only=False)
    meta = state["metadata"]
    model = build_model(meta["model"], **meta.get("model_kwargs", {}))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model_restorer(model, device)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ProtocolConfig(data_root=str(ROOT / "data"))
    test = BSDS500(config.data_root, split="test", luminance=config.luminance)
    by_id = {p.relative_to(test.image_root).as_posix(): p for p in test.paths}
    psf_sets = protocol_psf_sets()
    selected = json.loads((EVAL / "m2_tune_gpu/selected.json").read_text())
    gmm = TorchGMM(GMM.load(ROOT / "checkpoints/m2/gmm_K200.npz"), device=device)
    nafnet = load_nafnet(device)
    sigma_index = {s: i for i, s in enumerate(config.noise_levels_over_255)}

    rows = []
    for set_name, psf_name, sigma_255, title in SCENARIOS:
        image_id = median_image(set_name, psf_name, sigma_255)
        psf = next(p for p in psf_sets[set_name] if p.name == psf_name)
        sigma = sigma_255 / 255.0
        x = load_image(by_id[image_id], luminance=config.luminance)
        y = observe(x, psf.kernel, sigma, seed=config.seed, image_id=image_id, psf_name=psf.name)
        i = sigma_index[sigma_255]
        outputs = {
            "sharp": x,
            "observed": y,
            "wiener": wiener(y, psf.kernel, sigma=sigma),
            "fista": fista_tv_torch(y, psf.kernel, lam=selected["fista_tv"]["lambda_values"][i], device=device),
            "epll": admm_epll_torch(
                y, psf.kernel, gmm, sigma=sigma, lam=selected["admm_epll"]["lambda_values"][i], stride=1
            ),
            "nafnet": nafnet(y, psf, sigma),
        }
        top, left = detail_crop(x, CROP)
        stem = f"{set_name}_{psf_name}_s{sigma_255}"
        tiles = []
        for column in COLUMNS:
            crop = outputs[column][top : top + CROP, left : left + CROP]
            name = f"{stem}_{column}.png"
            save_tile(crop, OUT / name, kernel=psf.kernel if column == "observed" else None)
            value = None if column == "sharp" else psnr(x, outputs[column])
            tiles.append((name, value))
        rows.append((title, image_id, tiles))
        print(f"{stem}: {image_id} crop at ({top}, {left}) "
              + " ".join(f"{c}={v:.2f}" for c, (_, v) in zip(COLUMNS, tiles) if v is not None))

    lines = []
    width = r"0.158\linewidth"
    lines.append(" & ".join(r"\small " + h for h in HEADERS) + r" \\[2pt]")
    for title, image_id, tiles in rows:
        image_tex = image_id.split("/")[-1].replace("_", r"\_")
        lines.append(
            r"\multicolumn{6}{l}{\footnotesize\textbf{%s} \quad (test image %s)} \\" % (title, image_tex)
        )
        best = max(v for _, v in tiles[2:])
        lines.append(
            " & ".join(r"\includegraphics[width=%s]{generated/qualitative/%s}" % (width, n) for n, _ in tiles)
            + r" \\"
        )
        labels = []
        for _, v in tiles:
            if v is None:
                labels.append(r"\footnotesize PSNR (dB)")
            elif v == best:
                labels.append(r"\footnotesize\textbf{%.2f}" % v)
            else:
                labels.append(r"\footnotesize %.2f" % v)
        lines.append(" & ".join(labels) + r" \\[4pt]")
    (OUT / "qualitative.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'qualitative.tex'}")


if __name__ == "__main__":
    main()
