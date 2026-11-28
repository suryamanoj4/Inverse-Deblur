# Inverse Function Learning for Image Deblurring
### Classical vs. Modern Machine Learning

Project documentation for a controlled study of image deblurring as an inverse-function learning problem.

**Authors:** Surya Manoj Pentakota (2026701036), Kevin Paul Reddy Janagari (2026900006) — IIIT Hyderabad

**Status: complete (November 2026).** All three phases have been run. Deliverables:

- Final report: [`report/final_report.pdf`](report/final_report.pdf)
- Final presentation: [`report/final_presentation.pptx`](report/final_presentation.pptx)
  (built by `report/final_presentation_deck.js`)
- Phase 1 (M1 vs. M2, protocol v1) report: [`report/study_report.md`](report/study_report.md)
- Phase 3 pre-registration and outcome: [`vault/Phase 3 Pre-registration.md`](vault/Phase%203%20Pre-registration.md)

**Main finding.** Knowing the kernel is not enough; where it enters the network decides
what transfers. On unseen kernel families, feeding the kernel to NAFNet as input maps or FiLM
makes it worse than no kernel at all, while a closed-form FFT data step makes it better. A
kernel-swap probe shows why: the conditioned networks use the kernel as a lookup key, not as a
forward model. Summary numbers are in [Results summary](#results-summary).

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
| **M1** | Modern, data-only | No | NAFNet (primary), U-Net |
| **M2** | Classical-ML, physics-aware | Yes (true kernel + noise) | ADMM-EPLL (GMM prior, primary), FISTA-TV, Wiener, Richardson–Lucy |
| **M3** | Unrolled control, physics-aware | Yes | USRNet (primary); ISTA-Net, LISTA |

Phase 3 adds placement variants on the M1 backbones (NAFNet and U-Net): the kernel as input
maps, FiLM in every block, FiLM + one FFT data step (K-NAFNet), one data step without FiLM, and
a 4-stage unrolled NAFNet. DeepDeblur and K-SVD, listed as optional in the proposal, were not
implemented.

M3 is the control that separates two otherwise confounded axes: **physics-awareness** (does the method use $k$ and $\sigma$?) and **prior depth** (fixed TV → shallow learned GMM → deep learned CNN).

### Model inventory

Only M1 and M3 train end-to-end networks; M2 learns a prior and runs a per-image solver.

| Arm | Primary artifact | Additional artifacts | Core count |
|---|---|---|---|
| **M1** | NAFNet | U-Net baseline; 5 per-σ ablation checkpoints | **1** |
| **M2** | EPLL GMM ($K=200$) | 4 more priors for the $K$-sweep | **1** |
| **M3** | USRNet | ISTA-Net, LISTA | **1** |

- **Core comparison: 3 trained artifacts** — NAFNet, GMM-200, USRNet.
- **Phase 1 study: 15** — 2 M1 networks + 5 per-σ specialists + 5 GMM priors + 3 M3 networks.
- **Whole project: about 50 trained networks** across both protocols (seeds, ablations,
  specialists, the learning curve, placement variants and mismatch training).
- M1 and M3 train once on $P_1$ × all noise levels; M2's solver is per-image and its λ values are scalars, not models.
- Full inventory, checkpoint layout, and ablation roles: `vault/Proposed Approach.md` §5.1.

**Protocol snapshot (v1, used in Phase 1; Phases 2 and 3 use v2, see [Phase 2](#phase-2-done)).** M1 is one model trained on $P_1$ (4 Gaussian + 8 motion kernels) across all five noise levels, with no $k$ or $\sigma$ at train or test time. Kernel mismatch is evaluated on $P_2$-interp (unseen parameters of the same families) and $P_2$-extrap (defocus disks, an unseen family). Checkpoint selection uses mean val PSNR over the fixed 60-cell grid on a 10-image seeded subset of the sorted val list (IDs logged with the protocol); per-condition selection is not allowed. Learned arms use AdamW and L2 loss. The complete manifest — PSF membership, sampling policy, checkpoint rule, hyperparameters — is written to `artifacts/protocol/` by `scripts/setup_datasets.py --write-protocol`.

## Evaluation

- **Tier 1 (core):** BSDS500, synthetically degraded with a shared PSF library (Gaussian, procedural linear motion, defocus) and noise grid $\sigma \in \{0, 1, 5, 15, 25\}/255$, circular boundary handling.
- **Protocol v2 (Phases 2 and 3):** 60 test kernels (24 $P_1$, 24 $P_2$-interp, 12 $P_2$-extrap) × 5 noise levels × 200 test images; training families Gaussian, anisotropic Gaussian, motion and camera shake; unseen families defocus disks, rings and ghosts; 8-bit quantized observations.
- **Real blur:** recorded Levin et al. kernels on BSDS500 test images ($P_2$-real), Levin et al. real captures (32 blurred photos) and the Sun et al. benchmark (640).
- **Not run:** the synthetic training set (`syn-v1`; the go/no-go gate closed) and GoPro blind deblurring (the Tier 2 stretch goal).
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
│   ├── train_m3.py      # train an M3 network or a Phase 3 placement variant
│   ├── eval_m1.py       # M1-only evaluation (kept for compatibility)
│   ├── evaluate.py      # unified M1/M2/M3 evaluation with per-cell baseline
│   ├── evaluate_provided.py # benchmarks with their own blurred input (Levin, Sun)
│   ├── make_circular_triplets.py # circular-blur control for the Sun benchmark
│   ├── stability_test.py # H3 transfer attack: PGD on M1, evaluate all arms
│   ├── probe_kernel_swap.py     # Phase 3: true vs. wrong kernel given to a method
│   ├── analyze_linear_filter.py # Phase 3 (E-X): shared vs. per-kernel linear filter
│   ├── analyze_spectral_loss.py # Phase 3: gain per unseen kernel vs. spectral loss
│   ├── analyze_phase2.py / analyze_phase3.py / stats_phase3.py # tables and paired tests
│   ├── analyze_final_tables.py / make_final_figures.py # final report: CIs, families, cost, figures
│   └── ...              # Phase 1 analysis, figure and dataset scripts
├── report/              # final deck, reports, generated data/figures
│   ├── final_presentation.pptx, final_presentation_deck.js
│   ├── final_report.pdf
│   ├── study_report.md  # Phase 1 report (protocol v1, M1 vs. M2)
│   └── generated/       # numbers.json, phase2.json, phase3.json, phase3_stats.json, ...
└── vault/               # design notes (Obsidian)
    ├── Project Overview.md
    ├── Proposed Approach.md
    ├── Phase 2 Run Checklist.md
    └── Phase 3 Pre-registration.md
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

The reported v2 M3 models are trained to convergence with a 200k-iteration cap and
early stopping (`--protocol v2 --iterations 200000 --out checkpoints/v2/m3/usrnet_full`;
the 30k-cap runs are kept as the pre-registered numbers). The same script trains the
Phase 3 placement variants; see [Phase 3](#phase-3-done).

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

## Stability (H3)

`scripts/stability_test.py` runs a transfer attack: white-box PGD on M1 finds a
small perturbation of the observation, then every available arm is evaluated on
the same perturbed inputs.

```bash
python3 scripts/stability_test.py --data-root data \
  --m1-checkpoint checkpoints/m1/nafnet/best.pt \
  --gmm checkpoints/m2/epll_gmm_K200.npz \
  --psf-set P1 --sigmas 1,15 --eps 0.005,0.01,0.02 --limit 10
```

The script writes a CSV with clean/adversarial PSNR, the PSNR drop, SSIM, the
data-consistency residual and the perturbation norm, plus a summary and a
metadata JSON. Pass `--m3-checkpoint` to include M3. M2 is transfer-evaluated
by default; `--attack-arm m2` runs the white-box attack on M2 itself (Phase 2C-7)
through a differentiable, reduced-iteration FISTA-TV or ADMM-EPLL surrogate.

**Key options** (full table in `vault/Proposed Approach.md` §6.1):

| Option | Default | Meaning |
|---|---|---|
| `--psf-set` / `--psf-limit` | `P1` / `2` | which kernels and how many |
| `--sigmas` | `1,15` | noise levels in `/255`; σ=0 rejected |
| `--limit` | `10` | first N images |
| `--eps` | `0.005,0.01,0.02` | relative L2 budgets |
| `--attack-arm` | `m1` | `m1` or `m3` |
| `--m2-method` / `--gmm` | `admm_epll` / none | M2 solver and EPLL prior |
| `--lambda-values` | none | one value or one per noise level |
| `--m3-checkpoint` | none | include M3 |

Outputs land in `artifacts/stability/`: CSV rows, `*_summary.json` and
`*_metadata.json`.

**Result (transfer attack, 10 images).** At σ≥5 EPLL loses 2–17× less PSNR and
is less affected in 99–100% of cells. At σ=1 it is more stable in distribution
(92% of cells) but not under kernel mismatch (38–41% on P2), where its weak λ
amplifies the perturbation. Absolute PSNR after the attack remains higher for
EPLL. Details: report §7.

## Results summary

PSNR is the mean over σ > 0. The full tables, figures and paired statistics are in
[`report/final_report.pdf`](report/final_report.pdf) and
[`report/final_presentation.pptx`](report/final_presentation.pptx); the numbers come from
`report/generated/numbers.json` (Phase 1), `phase2.json`, `phase3.json`,
`phase3_stats.json` and `spectral_loss.json`.

**Phase 1 (protocol v1, M1 vs. M2)** — [`report/study_report.md`](report/study_report.md):

- **Training kernels:** NAFNet and ADMM-EPLL are tied in PSNR (27.48 vs. 27.43 dB); NAFNet is
  better perceptually (LPIPS).
- **Kernel mismatch (H1):** supported. NAFNet loses half its gain on unseen kernels; ADMM-EPLL
  wins every unseen-kernel cell, by up to 6.4 dB. The failure is on unseen motion directions.
- **Noise (H1) and texture (H2):** not supported. NAFNet's in-distribution lead comes from
  smooth images. Per-σ specialists gain at most 0.25 dB: the network is missing the kernel,
  not the noise level.
- **Stability (H3, 10 test images):** at σ ≥ 5, EPLL loses 2–17× less PSNR under NAFNet's attack and
  about 2–4× less under a small white-box attack on M2 itself; reversed at σ = 1.
- **σ = 0 is an inverse crime** (Wiener 100.5 dB, 12.6 dB after 8-bit rounding) and is excluded
  from every claim.
- **Overfitting check** (`scripts/analyze_overfit.py`, `report/nafnet_overfitting.pdf`): NAFNet
  does not memorise its training images (train−test gap −0.04 dB vs. −0.08 dB for the EPLL
  control) but does overfit the training kernels.

**Phase 2 (M3, protocol v2, real blur):**

- **NAFNet is not data-hungry:** with 50, 100 and 200 training images its $P_1$ PSNR spreads
  over 0.08 dB, and the $P_2$ drop stays (−2.91 dB at 50 images, −2.76 at 200); adding 800
  DIV2K images gives +0.02 dB on $P_1$ and a drop of −2.87 dB. The synthetic-data gate was closed; `syn-v1` was not generated.
- **USRNet (M3)** is the most accurate method on the synthetic grid: on v1, +0.27 dB over
  ADMM-EPLL on $P_1$ and +0.27 / +0.24 on $P_2$ (Holm p < 10⁻³⁰), with a $P_2$ drop of
  −0.43 dB like ADMM-EPLL (−0.41) rather than NAFNet (−2.76). On v2 (3 seeds) it reaches
  27.16 dB on $P_1$ and 27.57 dB on the unseen families. ISTA-Net and LISTA are robust but
  weaker.
- **Real blur:** on provided blurred inputs EPLL beats the standard USRNet (Sun 32.53 vs. 31.72
  dB, Levin real 30.33 vs. 29.56 dB); see the mismatch-aware training below.

**Phase 3 (where the kernel enters a network)** — pre-registered in
[`vault/Phase 3 Pre-registration.md`](vault/Phase%203%20Pre-registration.md), with the outcome
of every prediction at the end of that file. Placement ladder on one NAFNet backbone, data and
budget; PSNR on the unseen families ($P_2$-extrap), mean ± std over 3 seeds:

| Variant (`--model`) | $P_2$-extrap (dB) | Δ vs. no kernel |
|---|---|---|
| No kernel (NAFNet, M1) | 21.97 ± 0.18 | – |
| Kernel as input maps (`nafnet_concat`) | 20.78 ± 0.01 | −1.19 |
| FiLM in every block (`nafnet_film`) | 17.23 ± 0.84 | −4.74 |
| One FFT data step, no FiLM (`nafnet_dc`) | 25.05 ± 0.05 | +3.08 |
| FiLM + one FFT data step, K-NAFNet (`knafnet`) | 23.45 ± 0.41 | +1.48 |
| 4-stage unrolled (`nafnet_unrolled --n-iter 4`) | 27.14 ± 0.38 | +5.17 |
| USRNet (reference) | 27.57 ± 0.06 | +5.60 |

- **Conditioning hurts on unseen families; a data step helps.** Both conditioned variants are
  below no kernel (Holm p < 10⁻³², 200 test images). One data step without FiLM beats FiLM +
  the step by 1.60 dB. The unrolled net is 0.44 dB below USRNet (p < 10⁻³²): the placement
  carries most of the gain, not all. The same pattern holds on a U-Net backbone (one seed per
  variant): input maps −1.34, FiLM −2.68, FiLM + data step −0.31 dB (p = 0.07).
- **Kernel-swap probe** (`scripts/probe_kernel_swap.py`): the observation is blurred with the
  true kernel and the method is told the true, nearest-training, random or no-blur kernel. The
  FiLM net scores higher with a wrong kernel than the true one (18.20 vs. 20.67 dB with no
  blur), so it uses the kernel as a lookup key. Through a data step the true kernel is worth
  +3 to +7 dB, like Wiener (+3.10). Repeated on 100 images, every "worth of the true kernel"
  moves by at most 0.08 dB.
- **Spectral loss** (`scripts/analyze_spectral_loss.py`): per unseen kernel, the gain over
  no-kernel NAFNet falls with the fraction of the spectrum where |K| < 0.1 (Spearman ρ from
  −0.92 for FiLM to −0.99 for USRNet; n = 12 kernels, and spectral loss also grows with kernel
  size, ρ = 0.58).
- **Linear-filter test** (`scripts/analyze_linear_filter.py`): a filter shared by all kernels
  explains R² 0.13 of what NAFNet adds, a per-kernel filter 0.77 (10 images; 0.12 / 0.74 on
  50 images). NAFNet adapts to the kernel; the pre-registered "one fixed filter" explanation
  fails.
- **Mismatch-aware training (2×2 ablation on USRNet):** realistic boundaries in half of the
  training batches fix Levin's two largest kernels (22.52 → 31.99 dB) at no grid cost, also
  when stopped at the same 66k iterations (32.10 dB). Kernel noise adds a smaller gain and costs
  about 0.1 dB in distribution; both together give the best Levin real score (33.70 vs. 29.56
  dB). On Sun the gap to EPLL is not about boundaries: with circular blur EPLL still wins
  (32.55 vs. 31.78 dB). K-NAFNet + both (1 seed): +0.99 dB on Levin real, still 4.55 dB below
  EPLL on Sun.

**Limitations.** Own reimplementations on grayscale BSDS500 (200 training images), no official
pretrained weights; two backbones, with one seed per U-Net variant and for K-NAFNet + mismatch
training; spectral loss and kernel size are correlated over only 12 unseen kernels; non-blind
only; the white-box H3 check on M2 is small (10 images, 6 kernels, σ = 5 and 15). Future work: blind deblurring
(does an estimated kernel behave like the "nearest" kernel in the probe?), leave-one-family-out
training, colour images and larger training sets (DIV2K) for the ladder, and official
USRNet/DPIR weights as an external reference.

## Phase 2 (done)

The mid-evaluation reviewer suggested that M1 may be data-hungry or overfit, and that three
kernel families are too few. Phase 2 answered both and trained M3. The specification is in
`vault/Proposed Approach.md` §12; every run, with its command, is in
`vault/Phase 2 Run Checklist.md`.

Options shared by `train_m1.py`, `train_m3.py` and `train_gmm.py`: `--protocol v2`,
`--train-images N` (nested real subset), `--extra-images DIR` with `--p-real P` and
`--require-manifest`. `evaluate.py` and `stability_test.py` take `--protocol v2`;
`stability_test.py --attack-arm m2` attacks M2 directly. `setup_datasets.py --write-protocol
--version v2` freezes the v2 kernels. `generate_synthetic.py` and `check_synthetic.py` build a
synthetic set; they were not used, because the learning curve was flat.

- **Learning curve (2A-1):** NAFNet on 50 / 100 / 200 real images; flat (see above). Adding
  800 DIV2K images (5× the data) gives +0.02 dB on P1 and a P2 drop of −2.87 dB (−2.76 without).
- **M3:** USRNet (padding, output) and ISTA-Net/LISTA (residual phases) fixed, trained and
  evaluated on v1 and v2; v2 USRNet and NAFNet with 3 seeds.
- **Protocol v2:** distinct training and unseen families (cross-family correlation ≤ 0.90,
  checked when the sets are built), 8-bit quantized observations.
- **Coverage ablation (2C-5):** v2-trained NAFNet on the frozen v1 $P_2$ sets.
- **Real blur:** $P_2$-real (Levin kernels on BSDS500 test), Levin real captures and Sun et al.
  through `scripts/evaluate_provided.py` (edge padding + edgetaper, best-shift PSNR, noise
  estimated per image with `--sigma-255 auto`).

Install the learned-arm dependencies (PyTorch and LPIPS) with
`pip install -r requirements.txt`; for a CPU-only PyTorch wheel, use the
official index shown in that file.

## Phase 3 (done)

Phase 3 asked why the kernel helps a network only in some forms. Predictions were written
before any Phase 3 training (`vault/Phase 3 Pre-registration.md`). All commands assume
`PYTHONPATH=src`; outputs used by the analysis scripts live in `artifacts/eval/v2/phase3/`.

**Placement ladder.** Every variant uses NAFNet's training budget (`--budget m1`) on v2; add
`--seed S` for the extra seeds:

```bash
for m in nafnet_concat nafnet_film knafnet nafnet_dc; do
  python3 scripts/train_m3.py --data-root data --protocol v2 --model $m --budget m1 \
    --out checkpoints/v2/p3/$m
done
python3 scripts/train_m3.py --data-root data --protocol v2 --model nafnet_unrolled --n-iter 4 \
  --budget m1 --out checkpoints/v2/p3/nafnet_unrolled
# second backbone: --model unet_concat | unet_film | kunet
python3 scripts/evaluate.py --protocol v2 --arm m3 --data-root data \
  --checkpoint checkpoints/v2/p3/knafnet/best.pt --out artifacts/eval/v2/phase3/knafnet_test.csv
```

**Mismatch-aware training (2×2).** `--mismatch 0.5` makes half of the batches use the realistic
forward model (linear blur + edgetaper); `--kernel-noise` perturbs the given kernel;
`--mismatch-boundary off` keeps circular blur in those batches (kernel noise only); `--stop-at`
ends a run early while keeping the LR schedule:

```bash
# both
python3 scripts/train_m3.py --data-root data --protocol v2 --model usrnet --iterations 200000 \
  --mismatch 0.5 --kernel-noise 0.1 --out checkpoints/v2/m3/usrnet_mm
# boundaries only: --kernel-noise 0; kernel noise only: --mismatch-boundary off
# matched-length control: add --stop-at 66000
# K-NAFNet + both: --model knafnet --budget m1 --mismatch 0.5 --kernel-noise 0.1
```

**Provided-blur benchmarks and the circular-Sun control:**

```bash
python3 scripts/evaluate_provided.py --arm m3 --checkpoint checkpoints/v2/m3/usrnet_mm/best.pt \
  --triplets data/levin/benchmark --sigma-255 auto --lambda-rule interp --out <csv>
# Sun: --triplets data/sun2013/benchmark
python3 scripts/make_circular_triplets.py --triplets data/sun2013/benchmark \
  --sigma-255 2.55 --out data/sun2013/circular
python3 scripts/evaluate_provided.py --arm m3 --checkpoint <ckpt> \
  --triplets data/sun2013/circular --sigma-255 2.55 --boundary none --out <csv>
```

**Kernel-swap probe and linear-filter test** (`--images` defaults to 50 and 10; the
robustness repeats used 100 and 50):

```bash
python3 scripts/probe_kernel_swap.py --arm m3 --checkpoint checkpoints/v2/p3/nafnet_film/best.pt \
  --out artifacts/eval/v2/phase3/probe_nafnet_film.csv
python3 scripts/probe_kernel_swap.py --arm m2 --method wiener --out artifacts/eval/v2/phase3/probe_wiener.csv
python3 scripts/analyze_linear_filter.py --arm m1 --checkpoint checkpoints/v2/m1/nafnet/best.pt \
  --out report/generated/linear_filter/nafnet.json
```

**Aggregation, statistics and the deck:**

```bash
python3 scripts/analyze_phase3.py        # -> report/generated/phase3.json
python3 scripts/stats_phase3.py          # -> report/generated/phase3_stats.json (Wilcoxon + Holm)
python3 scripts/analyze_spectral_loss.py # -> report/generated/spectral_loss.json
python3 scripts/analyze_final_tables.py  # -> report/generated/final_tables.json (bootstrap CIs, families, cost)
python3 scripts/make_final_figures.py    # -> report/generated/final/*.pdf (needs matplotlib)
(cd report && pdflatex final_report.tex && pdflatex final_report.tex)  # -> report/final_report.pdf
NODE_PATH=~/.cache/deck-tools/node_modules node report/final_presentation_deck.js
                                         # -> report/final_presentation.pptx
```

The analysis scripts skip missing files, so they can be rerun on a partial set of runs. The
deck reads its chart values from `report/generated/*.json`.

## References

See `proposal.pdf` and `vault/Proposed Approach.md` for the full bibliography.
