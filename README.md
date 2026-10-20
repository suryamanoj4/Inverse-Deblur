# Inverse Function Learning for Image Deblurring
### Classical vs. Modern Machine Learning

Project documentation for a controlled study of image deblurring as an inverse-function learning problem.

**Authors:** Surya Manoj Pentakota (2026701036), Kevin Paul Reddy Janagari (2026900006) — IIIT Hyderabad

## Overview

A sharp image $x$ is observed through blur and noise:

$$
y = Ax + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2),
$$

where $A$ is convolution with a point-spread function (PSF) and $\varepsilon$ is sensor noise. Because $A$ is ill-conditioned — its eigenvalues decay to near zero at high frequencies — no stable exact inverse exists, and naive inversion amplifies noise.

We cast deblurring as **inverse function learning**: find $g_\theta$ minimizing $\mathbb{E}\lVert g_\theta(y) - x \rVert^2$ over pairs from the known degradation model. The central design question is: **should $g_\theta$ be built around the physics of $A$, or learned purely from data?** Existing methods differ along exactly this axis, but their results are reported under different protocols, so they are not directly comparable. This project runs that comparison under one controlled protocol.

## Three-arm comparison

| Arm | Family | Knows $A$? | Representative methods |
|---|---|---|---|
| **M1** | Modern, data-only | No | NAFNet (primary), DeepDeblur (historical) |
| **M2** | Classical-ML, physics-aware | Yes (true kernel + noise) | FISTA/ADMM solver + EPLL-GMM prior; K-SVD optional |
| **M3** | Unrolled control, physics-aware | Yes | USRNet (primary); LISTA → ISTA-Net genealogy |

M3 is the control that separates two otherwise confounded axes: **physics-awareness** (does the method use $k$ and $\sigma$?) and **prior depth** (fixed TV → shallow learned GMM → deep learned CNN).

## Evaluation

- **Tier 1 (core):** BSDS500, synthetically degraded with a shared PSF library (Gaussian, Levin motion, defocus) and noise grid $\sigma \in \{0, 1, 5, 15, 25\}/255$, circular boundary handling.
- **Tier 2 (stretch):** GoPro (3214 real blurry/sharp pairs) for blind deblurring.
- **Metrics:** PSNR, SSIM, LPIPS, data-consistency residual, kernel-mismatch robustness, and stability under small perturbations; paired Wilcoxon tests with multiplicity control.

## Repository layout

```
.
├── README.md            # this file
├── proposal.pdf         # project proposal (PDF)
├── pyproject.toml       # package metadata
├── requirements.txt     # core Python dependencies
├── Makefile             # `make test`
├── src/deblur/          # shared implementation
│   ├── operators/       # FFT circular convolution (NumPy + optional PyTorch)
│   ├── psf/             # Gaussian / motion / defocus PSF library
│   ├── data/            # degradation pipeline, BSDS500 loader
│   ├── protocol/        # locked config, splits, manifests
│   └── metrics/         # PSNR, SSIM, LPIPS (optional), data consistency
├── tests/               # unittest suite
├── scripts/
│   ├── train_gmm.py     # fit the EPLL/GMM prior on BSDS500 train
│   └── run_classical.py # run all M2 methods and report metrics
└── vault/               # design notes (Obsidian)
    ├── Project Overview.md
    └── Proposed Approach.md
```

LaTeX sources (`proposal/`, `*.tex`) are intentionally **not** versioned here; `proposal.pdf` is the reference artifact. The vault contains the working design record, including the full three-arm method specification.

## Running the tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
# or
make test
```

## Running the classical arm

Train the EPLL Gaussian-mixture prior on BSDS500 train images:

```bash
PYTHONPATH=src python3 scripts/train_gmm.py --data-root /path/to/BSDS500 \
  --components 200 --patches 100000 --iters 20 --out checkpoints/epll_gmm.npz
```

Run all M2 methods on one image (or the built-in synthetic target):

```bash
PYTHONPATH=src python3 scripts/run_classical.py \
  --image photo.png --psf gaussian_sigma1.2 --sigma-over-255 15 \
  --gmm checkpoints/epll_gmm.npz
```

The script reports PSNR, SSIM and the data-consistency residual for Wiener,
Richardson–Lucy, FISTA-TV and ADMM-EPLL.

## Status

The shared foundation and the M2 classical arm are implemented and tested:
FFT operator, PSF library, degradation pipeline, splits, metrics, Wiener,
Richardson–Lucy, FISTA-TV, EPLL/GMM training and ADMM-EPLL. Next: validation
runs on BSDS500, then the M1 (direct CNN) and M3 (unrolled) arms.

## References

See `proposal.pdf` and `vault/Proposed Approach.md` for the full bibliography.
