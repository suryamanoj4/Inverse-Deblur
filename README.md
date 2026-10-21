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

### Model inventory

Only M1 and M3 train end-to-end networks; M2 learns a prior and runs a per-image solver.

| Arm | Primary artifact | Additional artifacts | Core count |
|---|---|---|---|
| **M1** | NAFNet | U-Net baseline; 5 per-σ ablation checkpoints | **1** |
| **M2** | EPLL GMM ($K=200$) | 4 more priors for the $K$-sweep | **1** |
| **M3** | USRNet | ISTA-Net, LISTA | **1** |

- **Core comparison: 3 trained artifacts** — NAFNet, GMM-200, USRNet.
- **Full planned study: 15** — 2 M1 networks + 5 per-σ specialists + 5 GMM priors + 3 M3 networks.
- With DeepDeblur and K-SVD: 17.
- M1 and M3 train once on $P_1$ × all noise levels; M2's solver is per-image and its λ values are scalars, not models.
- Full inventory, checkpoint layout, and ablation roles: `vault/Proposed Approach.md` §5.1.

**Protocol snapshot.** M1 is one model trained on $P_1$ (4 Gaussian + 8 motion kernels) across all five noise levels, with no $k$ or $\sigma$ at train or test time. Kernel mismatch is evaluated on $P_2$-interp (unseen parameters of the same families) and $P_2$-extrap (defocus disks, an unseen family). Checkpoint selection uses mean val PSNR over the fixed 60-cell grid on a 10-image seeded subset of the sorted val list (IDs logged with the protocol); per-condition selection is not allowed. Learned arms use AdamW and L2 loss. The complete manifest — PSF membership, sampling policy, checkpoint rule, hyperparameters — is written to `artifacts/protocol/` by `scripts/setup_datasets.py --write-protocol`.

## Evaluation

- **Tier 1 (core):** BSDS500, synthetically degraded with a shared PSF library (Gaussian, procedural linear motion, defocus) and noise grid $\sigma \in \{0, 1, 5, 15, 25\}/255$, circular boundary handling.
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
│   ├── protocol/        # locked config, splits, manifests, evaluation cells
│   ├── metrics/         # PSNR, SSIM, LPIPS (optional), data consistency
│   ├── methods/         # classical/ (M2), direct/ (M1), unrolled/ (M3)
│   ├── evaluation/      # torch-free grid evaluation, baseline, summaries
│   └── training/        # patch sampler, training loop, model restorers
├── tests/               # unittest suite
├── scripts/
│   ├── train_gmm.py     # fit the EPLL/GMM prior on BSDS500 train
│   ├── run_classical.py # run all M2 methods on one image
│   ├── train_m1.py      # train an M1 network under the locked protocol
│   ├── train_m3.py      # train an M3 network (USRNet / ISTA-Net / LISTA)
│   ├── eval_m1.py       # M1-only evaluation (kept for compatibility)
│   └── evaluate.py      # unified M1/M2/M3 evaluation with per-cell baseline
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

Learned-arm tests are skipped when PyTorch is not installed; the classical and
protocol tests run everywhere.

## Dataset setup

One entry script downloads only the datasets you select:

```bash
# Core Tier 1 dataset (BSDS500)
python3 scripts/setup_datasets.py --datasets bsds500

# Also write the locked protocol manifests
python3 scripts/setup_datasets.py --datasets bsds500 --write-protocol

# Tier 2 stretch: GoPro (provide a local archive or direct URL)
python3 scripts/setup_datasets.py --datasets gopro --gopro-archive /path/GoPro.zip

# Optional: official Levin motion kernels (otherwise procedural fallback)
python3 scripts/setup_datasets.py --datasets levin --levin-archive /path/kernels.zip

# Inspect the local layout without downloading
python3 scripts/setup_datasets.py --check --datasets bsds500,gopro,levin
```

- **BSDS500** is fetched from the official Berkeley archive
  (`.../grouping/BSR/BSR_bsds500.tgz`).
- **GoPro** is manual; the official page is
  <https://seungjunnah.github.io/Datasets/gopro.html>. Pass a local archive or
  a direct URL.
- **Levin kernels** are optional; the PSF library falls back to procedural
  motion kernels if they are absent.
- `--dry-run` prints the plan and downloads nothing; `--check` only inspects.
- Protocol manifests are written to `artifacts/protocol/`.

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

## Running the modern arm (M1)

Train the primary M1 (NAFNet, width 32) on $P_1$ × all noise levels. Every
hyperparameter comes from `ProtocolConfig`; the run directory gets `run.json`
(protocol manifest, validation image IDs, git commit), `log.jsonl`, `best.pt`
(selected by mean val PSNR) and a resumable `last.pt`:

```bash
PYTHONPATH=src python3 scripts/train_m1.py --data-root data            # NAFNet
PYTHONPATH=src python3 scripts/train_m1.py --data-root data --model unet
PYTHONPATH=src python3 scripts/train_m1.py --data-root data --resume   # continue
```

Evaluate on the test split over $P_1$, $P_2$-interp and $P_2$-extrap. Each CSV
row is one image × PSF × noise level and now includes `baseline_psnr` (the
no-restoration result on the same cell) and `gain`; the script also writes a
per-(set, kind, σ) summary and a metadata JSON. Add `--lpips` for LPIPS:

```bash
PYTHONPATH=src python3 scripts/eval_m1.py --data-root data \
  --checkpoint checkpoints/m1/nafnet/best.pt
```

The per-noise oracle ablation fine-tunes the primary checkpoint at one noise
level: `--noise-over-255 15 --init checkpoints/m1/nafnet/best.pt`. Flags such as
`--iterations` or `--batch-size` override the protocol for smoke tests and are
recorded under `deviations` in `run.json`.

Validation and test observations are generated per (image, PSF, σ) cell with a
seed derived from their names (`deblur.protocol.observe`), so every arm is
scored on identical noisy inputs and results pair one-to-one.

## Running the physics-aware arm (M3)

Train a single M3 model across $P_1$ × all noise levels. The network receives
the true PSF spectrum and noise level, unlike M1:

```bash
PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model usrnet
PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model ista_net
PYTHONPATH=src python3 scripts/train_m3.py --data-root data --model lista
```

`USRNet` is the primary M3 arm (8 unrolled HQS iterations, ResUNet prior,
hyperparameter MLP). `ista_net` and `lista` are the ablation rungs. The
ISTA-Net symmetry constraint is not imposed; the adaptation to the convolution
operator is documented in the module.

## Unified evaluation

`scripts/evaluate.py` scores any arm on the same cells and computes the paired
no-restoration baseline, so `gain` is directly interpretable:

```bash
python3 scripts/evaluate.py --arm m1 --data-root data \
  --checkpoint checkpoints/m1/nafnet/best.pt
python3 scripts/evaluate.py --arm m3 --data-root data \
  --checkpoint checkpoints/m3/usrnet/best.pt
python3 scripts/evaluate.py --arm m2 --data-root data --method fista_tv \
  --lambda 0.0001,0.0005,0.001,0.002,0.003
python3 scripts/evaluate.py --arm m2 --data-root data --method admm_epll \
  --gmm checkpoints/epll_gmm.npz --stride 2 --limit 20
```

Each run writes a CSV with `psnr`, `baseline_psnr`, `gain`, `ssim`,
`residual_rmse` (and `lpips` with `--lpips`), a summary JSON grouped by
(PSF set, kernel kind, σ), and a metadata JSON with the protocol manifest. The
M2 path does not require PyTorch.

## Status

All three arms are implemented and tested:

- **M2:** Wiener, Richardson–Lucy, FISTA-TV, EPLL/GMM + ADMM-EPLL.
- **M1:** NAFNet and U-Net, protocol sampler, training loop, checkpoint
  selection, grid evaluation.
- **M3:** USRNet, ISTA-Net and LISTA, sharing the same sampler, trainer and
  evaluator.

Preliminary M1 test results show the predicted kernel-mismatch failure on
unseen motion and defocus. They are not final until the protocol-locked run is
scored with the paired per-cell baseline and paired statistical tests. Next:
the full M1/M3 runs, M2 λ tuning on validation, and the comparison table.

Install the learned-arm dependencies (PyTorch and LPIPS) with
`pip install -r requirements.txt`; for a CPU-only PyTorch wheel, use the
official index shown in that file.

## References

See `proposal.pdf` and `vault/Proposed Approach.md` for the full bibliography.
