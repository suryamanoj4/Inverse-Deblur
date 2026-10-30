---
title: Proposed Approach — Three-Arm Comparison
date: 2026-10-19
updated: 2026-10-20
tags:
  - project
  - deblurring
  - methods
  - design
  - implementation
aliases:
  - Three Arms
  - Method Design
  - Three-Arm Study
  - Method Inventory
status: active
---

# Proposed Approach — Three-Arm Comparison

> [!abstract] Summary
> Three **established** methods, one controlled protocol: **M1** (data-only CNN), **M2** (physics-aware classical-ML solver), **M3** (physics-aware unrolled network, the control). No new architecture in the core study — the comparison itself is the contribution, and its failure signature motivates the follow-up method.

Related: [[Project Overview]] · Proposal PDF: `proposal.pdf`

> [!note] Scope
> This document is the technical specification for the three-arm comparison summarized in [[Project Overview]]. Each method is defined by its objective, solver, prior, what is learned, what is fixed, and its role in the study.

---

## 0. Shared foundation (all methods use these)

| Component | Basis |
|---|---|
| Forward model | $y = Ax + \varepsilon$, $\varepsilon \sim \mathcal{N}(0,\sigma^2)$ |
| Operator | $A$ = circular convolution with PSF $k$; diagonalized by FFT: $A = \mathcal{F}^{-1}\mathrm{diag}(\kappa)\mathcal{F}$, $\kappa = \mathcal{F}k$ |
| PSF library | Gaussian $\sigma_k \in \{0.7, 1.2, 1.6, 2.0\}$; procedural linear motion kernels; defocus disks |
| Noise grid | $\sigma \in \{0, 1, 5, 15, 25\}/255$ |
| Data | BSDS500 splits (200 train / 100 val / 200 test), 128×128 patches, circular boundary |
| Metrics | PSNR, SSIM, LPIPS, data-consistency residual $\lVert A\hat{x}-y\rVert_2$, kernel-mismatch PSNR drop, PGD stability |

Every method consumes the same degraded pairs and is scored by the same harness. No method gets a different PSF, noise level, boundary condition, or test set.

---

## 1. Design principle: two independent axes

Every method answers two *separate* questions. Keeping them separate is what makes the three-arm design interpretable:

| Axis | Question | Levels |
|---|---|---|
| **Physics-awareness** | Does the method consume $k$ and $\sigma$ and enforce $Ax \approx y$? | no (M1) / yes (M2, M3) |
| **Prior depth** | How much structure does the model assume about natural images, and how much does it learn? | fixed (TV) → shallow learned (GMM, K-SVD) → deep learned (CNN) |

A two-arm comparison (data-only CNN vs classical solver) **confounds** these axes. **M3 is physics-aware like M2 but deep like M1**, so it isolates the two effects.

> [!note] Terminology
> - **M1** = Modern / data-only. **Never** sees $k$ or $\sigma$.
> - **M2** = Classical-ML / physics-aware. **Solver + explicit shallow prior**, given true $k$, $\sigma$.
> - **M3** = Unrolled control / physics-aware. **USRNet** family, given $k$, $\sigma$.
>
> USRNet is **not** M1, and LISTA is **not** M2. Both belong to the unrolled family (M3).

---

## 2. M1 — Modern ML, data-only ($A$-agnostic)

Learns a direct mapping $g_\theta(y) \to x$ from (blurry, sharp) pairs. The degradation is implicit in the data.

**Theoretical basis:** supervised regression. With squared loss the optimum is the **conditional mean** $g_\theta(y) \approx \mathbb{E}[x\mid y]$; when the posterior is multimodal (edges, textures), the mean averages modes and produces blur. This is the theoretical failure mode the arm is meant to exhibit.

### Algorithms

| Method | Theoretical basis | What is learned | Role |
|---|---|---|---|
| **U-Net** | Hierarchical encoder–decoder with skip connections; multi-scale receptive field; amortized regression | All weights, end-to-end | Minimal baseline |
| **DeepDeblur** (Nah et al., CVPR 2017) | Multi-scale coarse-to-fine CNN pyramid; residual blocks; kernel-free | All weights | Historical rung |
| **NAFNet** (Chen et al., ECCV 2022) | U-Net + NAFBlocks: SimpleGate (channel split + elementwise product), simplified channel attention, LayerNorm, no nonlinear activations | All weights | **Primary M1** |

### PSF sets (locked)

| Set | Membership | Count | Use |
|---|---|---|---|
| **$P_1$** | Gaussian $\sigma_k\in\{0.7,1.2,1.6,2.0\}$; motion length $\{9,15\}$ px $\times$ angle $\{0,45,90,135\}^\circ$ | 4 + 8 = **12** | Training and in-distribution evaluation |
| **$P_2$-interp** | Gaussian $\sigma_k\in\{0.9,1.4,1.8\}$; motion length $\{11,13\}$ px $\times$ angle $\{22.5,67.5,112.5,157.5\}^\circ$ | 3 + 8 = **11** | Kernel mismatch, same families, unseen parameters inside the $P_1$ ranges |
| **$P_2$-extrap** | Defocus disks, radii $\{3,5,7,9\}$ px | **4** | Kernel mismatch, unseen family |

The sets are disjoint by name and parameter. Defocus is deliberately absent from $P_1$: it is the unseen-family test. The general PSF library (16 kernels) remains available for other experiments, but the protocol uses these sets.

### Training regime (locked)

- **One model for the whole training grid.** M1 is trained once on $P_1$ $\times$ all noise levels and receives no $k$ or $\sigma$ at train or test time. M3 is also a single model across the grid (with $k,\sigma$ as inputs), so the comparison is family-vs-family, not model-count-vs-model-count.
- **Sampling.** Each training patch draws $k \sim \mathrm{Uniform}(P_1)$ and $\sigma \sim \mathrm{Uniform}(\{0,1,5,15,25\}/255)$ independently. All $12 \times 5 = 60$ cells have equal expected weight; $\sigma=0$ is 20% of training; motion is $8/12$ of kernel draws and Gaussian $4/12$. No family rebalancing in the primary run; `family_balanced` is reserved for a sensitivity check.
- **Checkpoint rule.** One checkpoint per learned arm: highest mean PSNR over the fixed 60-cell grid on a pre-registered 10-image subset of the sorted BSDS500-val list (drawn with seed 0 by `validation_subset`; the image IDs are written to `val_subset.json`). No per-condition checkpoint selection. The same rule applies to M3; the per-$\sigma$ ablation is the only exception.
- **Kernel mismatch.** Train on $P_1$, evaluate on $P_2$-interp and $P_2$-extrap with the same noise grid. M1 never sees the test kernels.
- **Ablation, not M1.** The per-noise oracle is five models fine-tuned from the primary M1 checkpoint (10k iterations each, same PSFs); it measures how much of the gap is due to not knowing $\sigma$. Per-PSF oracles are out of scope: 12 models if $\sigma$ is shared, 60 if per-$(k,\sigma)$.
- **Hyperparameters.** AdamW ($\beta=(0.9,0.999)$, weight decay $10^{-4}$), L2 loss, M1 batch 16 / M3 batch 8 at $128^2$, M1 lr $2\times10^{-4}$ / M3 lr $10^{-4}$, 1k warmup + cosine decay, gradient clipping 1.0, M1 100k / M3 30k iterations maximum, validation every 2k, early stopping patience 4.

### Characteristics
- **Inputs:** $y$ only. No $k$, no $\sigma$.
- **Inversion:** a single learned forward pass.
- **Prior:** implicit, distributed across CNN weights.
- **Training signal:** supervised $\ell_2$ (or $\ell_1$) on pairs; patches from the same grid as everyone else.
- **Inference:** milliseconds per image.
- **Expected failures:** collapse under kernel mismatch (SRMD, CVPR 2018), hallucination, instability under small perturbations (Antun et al., PNAS 2020; Gandikota et al., ICIP 2022 for deblurring).
- **Role:** the data-only pole; the arm against which the value of physics-awareness is measured (M1 vs M3).

---

## 3. M2 — Classical-ML, physics-aware

Solves a MAP inverse problem with an explicit forward model and an explicit, **learned but shallow** prior. Learning lives in the prior, not in the inversion.

### 3.1 Formulation

$$
\hat{x} = \arg\min_x \; \tfrac{1}{2}\lVert Ax - y\rVert_2^2 + \lambda R(x),
$$

where $A$ is the known convolution operator (FFT, circular boundary) and $R$ is the prior.

### 3.2 Solvers (inversion engines)

| Solver | Theoretical basis | Update principle | Needs |
|---|---|---|---|
| **Wiener** | Linear MMSE under a Gaussian signal prior; Tikhonov regularization | Closed form in Fourier: $\hat{X} = \kappa^* Y / (\lvert\kappa\rvert^2 + \lambda)$ | $k, \sigma$ |
| **Richardson–Lucy** | Maximum likelihood for Poisson observations; multiplicative EM | $x \leftarrow x \odot A^\top\!\left(y/(Ax)\right)$, normalized | $k$ |
| **FISTA** (Beck & Teboulle, TIP 2009) | Accelerated proximal gradient; $O(1/k^2)$ convergence | Gradient step on data term + proximal step of $R$ | $k, \sigma$ |
| **ADMM / HQS** | Variable splitting; alternating closed-form data update + prior prox + dual/penalty update | $x$-step in FFT; $z$-step = prox; $\mu$ update | $k, \sigma$ |

### 3.3 Priors (regularizers)

| Prior | Theoretical basis | Prox / inner step | Role |
|---|---|---|---|
| **Tikhonov** | Gaussian prior $\lVert x\rVert^2$; quadratic | Linear (Wiener) | Limit case |
| **TV** (Rudin et al., 1992) | Sparse gradient prior $\lVert\nabla x\rVert_1$; piecewise-constant model | Chambolle dual projection (FISTA) or soft-threshold split (ADMM) | Classical baseline |
| **EPLL-GMM** (Zoran & Weiss, ICCV 2011) | Patch distribution = mixture of Gaussians; conditional Gaussian ⇒ per-component Wiener filter | Responsibilities $\gamma_k$, pick $\arg\max_k$, apply $(\Sigma_k+\sigma^2 I)^{-1}(\Sigma_k y+\sigma^2\mu_k)$ | **Primary M2** |
| **GMM with $K$ components** | Same, but $K$ controls prior capacity | Same, $K \in \{1,4,16,64,200\}$ | **Prior-depth ablation** |
| **K-SVD** (Aharon et al., 2006) | Overcomplete dictionary + sparse coding $\lVert\alpha\rVert_0$; OMP pursuit, SVD dictionary update | Sparse-code prox (OMP/ISTA) inside ADMM | Optional, time-boxed |

> [!tip] Why the GMM prior is the right choice
> Natural image patch statistics are heavy-tailed and multimodal, so a single Gaussian prior over-smooths. Locally normalized patches are approximately Gaussian, and a finite Gaussian mixture captures the multimodal structure — the theory behind EPLL, and the reason its inner step is a closed-form Wiener filter. The $K$-sweep turns that theory into a measurable rung of the prior-depth axis.

### 3.4 Combinations actually run

| Tier | Combination | Purpose |
|---|---|---|
| Core | **FISTA + TV** | Non-learned physics-aware baseline |
| Core | **ADMM/HQS + EPLL (K = 200)** | Primary M2 |
| Ladder | **ADMM + EPLL, $K \in \{1,4,16,64,200\}$** | Prior-depth ablation ($K{=}1$ = single-Gaussian limit) |
| Sanity | **Wiener, Richardson–Lucy** | "Correct physics, dumb/no prior" lower bounds |
| Optional | **ADMM + K-SVD** | Alternative shallow learned prior; drop first if schedule slips |

### 3.5 Characteristics
- **Inputs:** $y$, **true $k$**, **true $\sigma$**.
- **Inversion:** iterative optimization (hundreds to thousands of iterations; seconds–minutes/image).
- **Prior:** explicit and shallow; learned once on clean patches, outside the solver.
- **Expected failures:** staircase/ringing on natural texture; brittle to wrong kernels.
- **Role:** the "physics but shallow prior" pole. Compared with M3 it isolates prior depth; compared with M1 it isolates physics-awareness.

### 3.6 Training and tuning (locked)

- **GMM prior.** Trained once on clean BSDS500 **train** patches ($8\times8$, DC removed, $K$ components, EM). It never sees degraded images; $\sigma$ enters only at inference through the MAP denoiser. The prior is fixed across runs and is not tuned on val or test.
- **TV weight $\lambda$.** Tuned per noise level on BSDS500 val (five scalars) from a fixed grid; the selected values are logged. This is condition-specific hyperparameter selection and is declared.
- **EPLL weight $\lambda$.** Tuned per noise level on val, initialized at $\lambda=\max(\sigma^2,10^{-6})\beta_{\max}$; the GMM prior itself is not retrained.
- **No checkpoint selection.** M2 solves each image independently; solver and hyperparameters are fixed after val tuning. Test images are never used for tuning.

---

## 4. M3 — Unrolled control, physics-aware (the bridge)

Unrolls a fixed number of proximal iterations into a network, keeping $A$ inside the loop as a differentiable closed-form module. They are **fixed-depth approximations of the MAP solver**, not converged solutions.

### 4.1 Algorithms (the unrolling genealogy)

| Rung | Method | Theoretical basis | What is learned | What stays fixed |
|---|---|---|---|---|
| 1 | **LISTA** (Gregor & LeCun, ICML 2010) | Unrolled ISTA for sparse coding: $\alpha_{t+1} = S_\theta(W_1 y + W_2\alpha_t)$; ~20× fewer iterations than FISTA at equal error | $W_1, W_2$, threshold | Forward model |
| 2 | **ISTA-Net** (Zhang & Ghanem, CVPR 2018) | Unrolled ISTA with learned analysis/synthesis transforms $F,\tilde F$; proximal = soft-threshold in the learned domain; symmetry loss $\lVert\tilde F(F(x))-x\rVert^2$ | CNN transforms, step size, thresholds | Data term |
| 3 | **USRNet** (Zhang, Van Gool, Timofte, CVPR 2020) | HQS unrolling of $\min_x \tfrac12\lVert y-(x\otimes k)\downarrow_s\rVert^2 + \lambda R(x)$; 8 iterations | ResUNet prior + MLP hyperparameters | Closed-form FFT data module |

### 4.2 USRNet data step (deblurring, $s=1$)

$$
x_{t+1} = \mathcal{F}^{-1}\!\left(\frac{\kappa^*\mathcal{F}(y) + \mu_t \mathcal{F}(z_t)}{\lvert\kappa\rvert^2 + \mu_t}\right),
\qquad
z_{t+1} = \mathrm{Denoiser}\!\left(x_{t+1};\sqrt{\lambda/\mu_t}\right).
$$

### 4.3 Training regime (locked)

- One model across $P_1 \times$ all noise levels, with $k$ and $\sigma$ as inputs.
- Same checkpoint rule as M1: mean val PSNR over the fixed 60-cell grid on the 10-image subset; no per-condition selection.
- AdamW, L2 loss, batch 8, lr $10^{-4}$, 1k warmup + cosine, gradient clipping 1.0, 30k iterations maximum, validation every 2k, early stopping patience 4.
- If training from scratch exceeds the GPU budget, initialise from the official USRNet SISR checkpoint and fine-tune for 10k iterations; log this as a documented deviation.

### 4.4 Characteristics
- **Inputs:** $y$, **$k$**, **$\sigma$**.
- **Inversion:** fixed-depth unrolled optimization (8 iterations for USRNet).
- **Prior:** learned deep CNN, applied inside the loop.
- **Inference:** feed-forward, ~ms–s.
- **Why it is the control:** it has M2's physics and M1's deep prior, so any M1–M2 difference is attributable to physics-awareness or prior depth rather than both at once.
- **The genealogy is itself an ablation:** LISTA → ISTA-Net → USRNet isolates learned matrices, then a learned proximal transform, then a full denoiser + hyperparameter module.

> [!warning] Implementation caveats
> - USRNet's released weights are for **super-resolution**. For deblurring, set $s=1$ and **retrain/fine-tune** on the same PSF/noise protocol; do not use SISR weights as-is.
> - LISTA is designed for sparse coding; adapting it to a convolutional $A$ means unrolling FISTA with learnable step sizes/thresholds — budget it as implementation work, not "off the shelf."
> - ISTA-Net was designed for compressive sensing; it must be adapted to the convolution forward operator and stated as such.

---

## 5. Master comparison

| Property | **M1 — Modern (data-only)** | **M2 — Classical-ML (physics-aware)** | **M3 — Unrolled control (physics-aware)** |
|---|---|---|---|
| Estimator | Amortized posterior mean | MAP | Fixed-depth MAP approximation |
| Mathematics | Deep regression | Convex optimization / EM / dictionary learning | Unrolled optimization |
| Family | Direct deep regression | Iterative solver + learned shallow prior | Deep unfolding |
| Knows $k$? | No (implicit in data) | Yes (explicit) | Yes (explicit input) |
| Knows $\sigma$? | No | Yes | Yes |
| Inversion | Single learned pass | Iterative optimization | Fixed-depth unrolled optimization |
| Prior | Implicit deep CNN | Explicit shallow (GMM / sparsity) | Learned deep CNN inside the loop |
| Algorithms | NAFNet (primary), DeepDeblur (historical), U-Net (minimal) | FISTA/ADMM + EPLL-GMM (primary), K-SVD (optional); TV, Wiener, RL as limits | USRNet (primary); LISTA → ISTA-Net (ablation ladder) |
| Training | End-to-end on $(x,y)$ pairs | Prior learned offline; solver not trained | End-to-end through the unrolled iterations |
| Inference | ~ms/image | seconds–minutes/image | ~ms–s/image |
| Expected failure | Kernel mismatch, hallucination, instability | Staircase/ringing; needs correct $k$ | Between the two; still sensitive to unseen kernels |
| Role | Data-only pole | Physics + shallow-prior pole | Bridge / control |

### 5.1 Model and artifact inventory

The word "model" means something different in each arm. Only M1 and M3 train
end-to-end networks; M2 learns a prior and runs a per-image solver.

| Arm | Approach | Trained? | Artifacts |
|---|---|---|---|
| **M1** | NAFNet (primary) | one model on $P_1$ × all noise levels | **1** |
| M1 | U-Net (minimal baseline) | one model | 1 |
| M1 | DeepDeblur (historical) | optional, not implemented | 0 |
| M1 | per-$\sigma$ specialists (ablation) | fine-tuned from the NAFNet checkpoint | 5 |
| **M2** | Wiener | no training | 0 |
| M2 | Richardson–Lucy | no training | 0 |
| M2 | FISTA-TV | no training; $\lambda$ tuned per $\sigma$ on val | 0 |
| M2 | ADMM-EPLL (primary) | one GMM prior, shared across images and $\sigma$ | **1** |
| M2 | GMM $K$-sweep ($K=1,4,16,64,200$) | one prior per $K$ | 5 total |
| M2 | K-SVD | optional, not implemented | 0 |
| **M3** | USRNet (primary) | one model on $P_1$ × all noise levels; $k,\sigma$ inputs | **1** |
| M3 | ISTA-Net | one model | 1 |
| M3 | LISTA | one model | 1 |

**Counts**

- **Core comparison: 3 trained artifacts** — NAFNet, the EPLL prior ($K=200$), USRNet.
- **Planned full study: 15** — 2 M1 networks + 5 per-$\sigma$ specialists + 5 GMM priors + 3 M3 networks.
- **With DeepDeblur and K-SVD: 17.**

**What "one model" means**

- M1 and M3 train **once** on all 12 $P_1$ kernels × 5 noise levels. They are not retrained per PSF or per $\sigma$.
- M1 never receives $k$ or $\sigma$; M3 receives both as inputs.
- M2 has no network. The GMM prior is trained once on clean BSDS500 train patches and shared across images and noise levels; the solver runs per image. The per-noise $\lambda$ values are five scalars per method, not models.
- The $K$-sweep and the per-$\sigma$ specialists are ablations, not the primary arms.

**Expected checkpoint layout**

```
checkpoints/
  m1/
    nafnet/{run.json,log.jsonl,best.pt,last.pt}
    unet/{run.json,log.jsonl,best.pt,last.pt}
    nafnet_sigma{0,1,5,15,25}/{best.pt,run.json}   # ablation, fine-tuned
  m2/
    epll_gmm_K{1,4,16,64,200}.npz                   # priors; K=200 is primary
  m3/
    usrnet/{run.json,log.jsonl,best.pt,last.pt}
    ista_net/{run.json,log.jsonl,best.pt,last.pt}
    lista/{run.json,log.jsonl,best.pt,last.pt}
```

Evaluation artifacts land in `artifacts/eval/` (CSV, summary JSON, metadata JSON).

---

## 6. Hypotheses, stated over the three arms

- **H1 — Physics-awareness pays off as information degrades.** The M1–M3 gap is small at low noise with the correct kernel and widens with noise, texture complexity, and kernel mismatch. (Mechanism: M3's data module still uses the true $k$, so it degrades gracefully; M1 has nothing to fall back on.)
- **H2 — Prior depth matters most on texture.** On piecewise-constant content M2 is competitive; on natural texture the deep-prior arms (M1, M3) win. The $K$-sweep tests the same claim from the classical side.
- **H3 — A-awareness buys stability.** M1 is unstable under tiny perturbations while the physics-aware arms resist them (Antun et al., PNAS 2020; tested here in the convolution regime). This is the study's headline result.

> [!note] H3 implementation
> `scripts/stability_test.py` runs a **transfer attack**: white-box PGD on M1 (or M3 with `--attack-arm m3`) finds a small observation perturbation, then every available arm is evaluated on the same perturbed inputs. This tests whether a perturbation that breaks the data-only model also breaks the physics-aware arms. Per-method white-box attacks for M2 need a gradient-enabled FISTA/EPLL and are a follow-up; the current result must be labelled as a transfer attack.

---

## 7. Datasets, splits, and protocol (locked)

### 7.1 Datasets

| Tier | Dataset | Size / split | Role |
|---|---|---|---|
| **1 (core)** | **BSDS500** (Arbeláez et al., TPAMI 2011) | 500 images: **200 train / 100 val / 200 test** | Controlled synthetic comparison |
| **2 (stretch)** | **GoPro** (Nah et al., CVPR 2017) | 3,214 pairs: **2,103 train / 1,111 test**, no kernels | Blind deblurring; out-of-protocol stress test unless kernel estimation is added |

Dataset setup is a single opt-in entry point:

```bash
python3 scripts/setup_datasets.py --datasets bsds500 --write-protocol
```

`--datasets` accepts `bsds500`, `gopro`, `levin` (comma-separated); `gopro` and
`levin` require `--gopro-archive`/`--levin-archive` (or a direct URL) and are
never downloaded implicitly. `--check` inspects the local layout without
downloading. `protocol.json` / `manifest.json` record the $P_1$/$P_2$
membership, sampling policy, checkpoint rule, and learned-arm hyperparameters.

**Not used:** Set5 / Set14 / Urban100 / Kodak (SR benchmarks). DIV2K is allowed only as an optional *data-scale ablation* applied to **all** learned priors.

### 7.2 Locked preprocessing decisions

| Decision | Choice | Rationale |
|---|---|---|
| BSDS version | **BSDS500**, standard directory split | Resolves the proposal's ambiguous "BSDS300/500"; matches the TPAMI citation |
| Color space | **Luminance only** (Rec.601), core protocol | EPLL/K-SVD are grayscale; keeps all arms comparable |
| Training data | **BSDS500 train only**, for every learned component (GMM prior, CNN weights, unrolled nets) | Prevents cross-arm training-data confounds and GMM leakage |
| Validation | **BSDS500 val only** for $\lambda$ tuning and early stopping | Same val for every arm; test never used for model selection |
| Test | **BSDS500 test**, full images; 256×256 centre crops as secondary | Circular degradation makes full-image FFT evaluation exact |
| Patch protocol | Split **by image first**, then crop 128×128 patches | Patches from one image never straddle splits |
| Augmentation | Flips + 90° rotations, identical for all learned priors | Removes augmentation as a confound |
| PSF library | Gaussian $\sigma_k \in \{0.7,1.2,1.6,2.0\}$; 8 procedural linear motion kernels (length × angle); defocus radii $\{3,5,7,9\}$ | Kernels generated in code, no download; official Levin kernels optional, outside the protocol |
| Noise | $\sigma \in \{0,1,5,15,25\}/255$, additive white Gaussian, clipped to $[0,1]$ | Clipping is part of the observation model for every arm |
| Boundary | Circular / FFT | Required by the closed-form data modules |

### 7.3 Protocol invariants (must be identical across arms)

| Item | Setting |
|---|---|
| Input | $y$ only (M1); $y, k, \sigma$ (M2, M3) |
| Normalization | Identical across methods; **never** inside the forward model (noise is absolute) |
| Metrics | PSNR, SSIM, LPIPS; data-consistency residual $\lVert A\hat{x}-y\rVert_2$; kernel-mismatch PSNR drop; PGD stability |
| Statistics | Paired Wilcoxon signed-rank; pre-register 1–2 primary endpoints; Holm–Bonferroni for the rest |
| Reproducibility | Fixed seeds; split manifest saved with every run; released protocol |

### 7.4 Implementation status

| Step | Module | Status |
|---|---|---|
| Data setup | `scripts/setup_datasets.py` | ✅ implemented (opt-in downloads; BSDS500 source verified) |
| Protocol fields | `protocol/config.py` manifest (PSF sets, sampling, checkpoint, hyperparameters) | ✅ implemented + tested |
| 1 | `operators/fft_conv.py` | ✅ implemented + tested |
| 2 | `psf/library.py` | ✅ implemented + tested |
| 3 | `data/degrade.py`, `data/datasets.py`, `protocol/` | ✅ implemented + tested |
| 3 | `metrics/` (PSNR, SSIM, consistency) | ✅ implemented + tested; LPIPS optional |
| 4–6 | classical arm (`wiener`, `richardson_lucy`, `fista_tv`, `admm_epll`, `gmm`) | ✅ implemented + tested |
| 7 | direct arm (`direct/nafnet.py`, `direct/unet.py`, `training/`, `scripts/train_m1.py`, `scripts/eval_m1.py`) | ✅ implemented + tested; M1/M2 interim results in `report/m1_m2_comparison.md` |
| 8 | unrolled arm (`unrolled/usrnet.py`, `unrolled/ista_net.py`, `unrolled/lista.py`, `scripts/train_m3.py`) | ✅ implemented + tested (torch-guarded); training pending |
| 9 | unified evaluation (`evaluation/`, `scripts/evaluate.py`) with paired per-cell baseline | ✅ implemented + tested |
| 10 | interim report (`report/m1_m2_comparison.md`, `report/generated/`) | ✅ M1/M2 results; M3 and H3 pending |
| H3 | stability (`metrics/stability.py`, `scripts/stability_test.py`) | ✅ implemented (transfer attack); run pending |

---

## 8. Diagnosis → prescription (the follow-up)

| Arm | Diagnosed weakness | What a follow-up method would need to fix |
|---|---|---|
| M1 | Kernel mismatch, hallucination, instability | Restore data-consistency without giving up the deep prior |
| M2 | Staircase/ringing on texture, brittleness to wrong $k$ | A deeper (or multimodal) prior while keeping the solver's physics |
| M3 | Cost, residual sensitivity to unseen kernels | Cheaper/adaptive physics module; better prior routing |

**Candidate follow-up (future work, not in this study's core):** a physics-aware network whose prior is a *mixture* — e.g. patch-GMM-routed experts, or a GSM-scale-driven spatially varying regularization strength — targeting the exact failure mode the three-arm comparison identifies.

---

## 9. Optional / stretch (not core)

| Method | Basis | Why optional |
|---|---|---|
| **Deep Image Prior** (Ulyanov et al., CVPR 2018) | Untrained network as implicit prior; early stopping | Per-image optimization, expensive; discussion only |
| **GSM-scale spatially varying $\lambda$** | Local scale variable $\hat{z}$ estimates local SNR | Follow-up experiment; not needed for the core table |
| **Patch-GMM-routed experts** | Mixture of experts gated by GMM responsibilities | Future-work proposal, motivated by the comparison |

---

## 10. Implementation order (proposed module map)

```
src/
  operators/fft_conv.py      # forward/adjoint convolution, PSF apply
  psf/library.py             # Gaussian / motion / defocus PSF generation
  data/degrade.py            # y = Ax + noise, circular boundary
  data/datasets.py           # BSDS500 loader, patch sampling
  protocol/config.py         # locked protocol + manifest
  protocol/splits.py         # splits, validation subset
  protocol/observations.py   # deterministic eval cells
  metrics/                   # PSNR, SSIM, LPIPS, consistency
  metrics/stability.py       # L2 projection + PGD attack for H3
  evaluation/grid.py         # torch-free grid evaluation, baseline, summaries
  training/
    sampler.py               # P1 x noise sampler (k, sigma per patch)
    trainer.py               # AdamW, cosine, early stopping, checkpoints
    evaluation.py            # M1/M3 model restorers
  methods/classical/
    wiener.py                # sanity baseline
    richardson_lucy.py       # sanity baseline
    fista_tv.py              # core classical
    tv.py                    # TV operators + ADMM prox
    admm_epll.py             # core classical-ML + K-sweep
    gmm.py                   # EPLL prior (EM, MAP denoiser)
    ksvd.py                  # optional, not implemented
  methods/direct/
    unet.py                  # minimal M1
    nafnet.py                # primary M1
    deepdeblur.py            # historical M1, optional, not implemented
  methods/unrolled/
    lista.py                 # ladder rung 1
    ista_net.py              # ladder rung 2
    usrnet.py                # primary M3
scripts/
  setup_datasets.py          # opt-in dataset setup + protocol manifests
  train_gmm.py               # EPLL prior training
  train_m1.py                # M1 training
  train_m3.py                # M3 training
  eval_m1.py                 # M1-only evaluation
  evaluate.py                # unified M1/M2/M3 evaluation with baseline
  stability_test.py          # H3 transfer attack (PGD on M1, evaluate all arms)
  run_classical.py           # one-image classical demo
```

Build order:

1. `operators/fft_conv.py` — forward/adjoint, PSF apply.
2. `psf/library.py` + `data/degrade.py` — PSF generation, noise, dataset.
3. `metrics/` + `protocol/splits.py` — harness and fixed protocol.
4. `classical/wiener.py`, `classical/rl.py` — sanity baselines (no training).
5. `classical/fista_tv.py` — first real baseline.
6. `classical/admm_epll.py` — GMM training (EM) + ADMM loop; add the $K$-sweep.
7. `direct/unet.py` → `direct/nafnet.py` — M1.
8. `unrolled/usrnet.py` — M3; then `lista.py`, `ista_net.py` for the ladder.
9. `evaluation/grid.py` + `scripts/evaluate.py` — paired baseline and the
   cross-arm comparison table.

After step 9 the full grid comparison is runnable, including the per-cell
no-restoration baseline.

---

## 11. Implementation constraints

- **Framework.** PyTorch for learned arms; NumPy/SciPy for classical solvers and GMM/EM.
- **USRNet retraining.** Official weights are SISR; deblurring requires fine-tuning at $s{=}1$, not scratch training.
- **LISTA adaptation.** The convolutional operator requires re-deriving the iteration with learnable step sizes/thresholds.
- **Compute discipline.** The core grid runs one representative per arm (NAFNet, FISTA+EPLL, USRNet); genealogy rungs use a reduced grid.
- **Headline M1.** NAFNet is the primary data-only model; DeepDeblur (2017) is the historical rung.
- **K-SVD** is optional and is dropped first if the schedule tightens.
- **Multiplicity.** Dozens of Wilcoxon tests require Holm–Bonferroni correction or pre-registered endpoints.
- **Blind vs non-blind.** Tier 1 provides true $k$ to M2/M3; GoPro is blind and is treated as an out-of-protocol stress test unless kernel estimation is added.

---

## 12. References

1. Rudin, Osher, Fatemi. *Nonlinear total variation based noise removal algorithms.* Physica D, 1992.
2. Beck, Teboulle. *Fast gradient-based algorithms for constrained TV denoising and deblurring.* IEEE TIP, 2009.
3. Gregor, LeCun. *Learning fast approximations of sparse coding (LISTA).* ICML, 2010.
4. Zhang, Ghanem. *ISTA-Net: Interpretable optimization-inspired deep network for image compressive sensing.* CVPR, 2018.
5. Zhang, Van Gool, Timofte. *Deep unfolding network for image super-resolution (USRNet).* CVPR, 2020.
6. Zoran, Weiss. *From learning models of natural image patches to whole image restoration (EPLL).* ICCV, 2011.
7. Aharon, Elad, Bruckstein. *K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation.* IEEE TSP, 2006.
8. Roth, Black. *Fields of experts: A framework for learning image priors.* CVPR, 2005.
9. Nah, Kim, Lee. *Deep multi-scale CNN for dynamic scene deblurring (DeepDeblur).* CVPR, 2017.
10. Chen, Chu, Zhang, Sun. *Simple baselines for image restoration (NAFNet).* ECCV, 2022.
11. Ulyanov, Vedaldi, Lempitsky. *Deep image prior.* CVPR, 2018.
12. Antun, Renna, Poon, Adcock, Hansen. *On instabilities of deep learning in image reconstruction and the potential costs of AI.* PNAS, 2020.
13. Levin, Weiss, Durand, Freeman. *Understanding and evaluating blind deconvolution algorithms.* CVPR, 2009.
14. Zhang, Zuo, Gu, Zhang. *Learning deep CNN denoiser prior for image restoration (IRCNN).* CVPR, 2017.
15. Zhang, Zuo, Zhang. *Learning a single convolutional super-resolution network for multiple degradations (SRMD).* CVPR, 2018.
16. Arbeláez, Maire, Fowlkes, Malik. *Contour detection and hierarchical image segmentation (BSDS).* IEEE TPAMI, 2011.
17. Gandikota, Chandramouli, Moeller. *On adversarial robustness of deep image deblurring.* ICIP, 2022.
18. Scetbon, Elad, Milanfar. *Deep K-SVD denoising.* IEEE TIP, 2021.
19. Wainwright, Simoncelli. *Scale mixtures of Gaussians and the statistics of natural images.* NIPS, 1999.
20. Portilla, Strela, Wainwright, Simoncelli. *Image denoising using scale mixtures of Gaussians in the wavelet domain.* IEEE TIP, 2003.

---

Status: active — implementation reference for the three-arm study. [[Project Overview]] gives the summary; this document is the full method record.
