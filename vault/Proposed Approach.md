---
title: Proposed Approach — Three-Arm Comparison
date: 2026-10-19
updated: 2026-11-15
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

> [!todo] Phase 2 and Phase 3 (after the mid-evaluation)
> In response to the reviewer, Phase 2 adds a synthetic training-image pipeline shared by all arms and a wider kernel library (protocol v2), and trains M3. Sections 0–11 describe the **v1** protocol used for the mid-evaluation results; the v2 changes are specified in **§12**; their code is implemented and the runs are pending ([[Phase 2 Run Checklist]]). Phase 3, the method proposal (K-NAFNet), is **§12.9** and is also not implemented.

Related: [[Project Overview]] · Proposal PDF: `proposal.pdf`

> [!note] Scope
> This document is the technical specification for the three-arm comparison summarized in [[Project Overview]]. Each method is defined by its objective, solver, prior, what is learned, what is fixed, and its role in the study.

---

## 0. Shared foundation (all methods use these)

| Component | Basis |
|---|---|
| Forward model | $y = Ax + \varepsilon$, $\varepsilon \sim \mathcal{N}(0,\sigma^2)$ |
| Operator | $A$ = circular convolution with PSF $k$; diagonalized by FFT: $A = \mathcal{F}^{-1}\mathrm{diag}(\kappa)\mathcal{F}$, $\kappa = \mathcal{F}k$ |
| PSF library | **v1:** Gaussian $\sigma_k \in \{0.7, 1.2, 1.6, 2.0\}$; procedural linear motion kernels; defocus disks. **v2 (planned, §12.3):** 7 families incl. anisotropic Gaussian, random-trajectory camera shake, real recorded kernels, Airy/Moffat |
| Noise grid | $\sigma \in \{0, 1, 5, 15, 25\}/255$ |
| Data | BSDS500 splits (200 train / 100 val / 200 test), 128×128 patches, circular boundary. **v2 (planned, §12.2):** training pool = 200 real + ~2,000 synthetic images (`syn-v1`), identical for every arm; val/test stay real |
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

> [!note] Phase 2 artifacts (planned)
> Phase 2 adds, per training-data and protocol version: NAFNet learning-curve runs (50 / 100 / 200 real images), NAFNet / U-Net / GMM-200 trained on real + `syn-v1`, two NAFNet controls (synthetic only; real + DIV2K), the three M3 networks, and the v2 retraining of every arm. Checkpoints will be namespaced by version, e.g. `checkpoints/v2/m1/nafnet_syn/`. See §12.4.

---

## 6. Hypotheses, stated over the three arms

- **H1 — Physics-awareness pays off as information degrades.** The M1–M3 gap is small at low noise with the correct kernel and widens with noise, texture complexity, and kernel mismatch. (Mechanism: M3's data module still uses the true $k$, so it degrades gracefully; M1 has nothing to fall back on.)
- **H2 — Prior depth matters most on texture.** On piecewise-constant content M2 is competitive; on natural texture the deep-prior arms (M1, M3) win. The $K$-sweep tests the same claim from the classical side.
- **H3 — A-awareness buys stability.** M1 is unstable under tiny perturbations while the physics-aware arms resist them (Antun et al., PNAS 2020; tested here in the convolution regime). This is the study's headline result.

> [!note] H3 implementation
> See §6.1. The current script runs a **transfer attack**; a per-method white-box M2 attack is a follow-up.

> [!note] Phase 2 hypotheses
> Two hypotheses are added for Phase 2 (§12.4): **H4 (data)**, that more training images help the deep arms in distribution but not under kernel mismatch, and **H5 (kernel diversity)**, that more training kernel families narrow M1's gap on unseen parameters but not on unseen families.

### 6.1 H3 stability experiment (implemented)

**Protocol.** The current implementation is a *transfer attack*:

1. White-box PGD on the attack arm (M1 by default, or M3 with `--attack-arm m3`) finds a perturbation $\eta$ with $\lVert\eta\rVert_2 \le \epsilon\lVert y\rVert_2$ that maximizes the reconstruction error.
2. Every available arm (M1, M2, optional M3) is evaluated on the same $y+\eta$.
3. Sweep $\epsilon$ over a fixed image/PSF/σ subset. $\sigma=0$ is excluded (inverse crime; see §7).

This tests whether a perturbation that breaks the data-only model also breaks the physics-aware arms. It does **not** find M2's own worst-case perturbation; a per-method white-box M2 attack requires a gradient-enabled FISTA/EPLL and is a follow-up.

**Command.**

```bash
python3 scripts/stability_test.py --data-root data \
  --m1-checkpoint checkpoints/m1/nafnet/best.pt \
  --gmm checkpoints/m2/epll_gmm_K200.npz \
  --psf-set P1 --sigmas 1,15 --eps 0.005,0.01,0.02 --limit 10
```

**Key options.**

| Option | Default | Meaning |
|---|---|---|
| `--psf-set` | `P1` | `P1`, `P2_interp` or `P2_extrap` |
| `--psf-limit` | `2` | first N kernels of the set |
| `--sigmas` | `1,15` | noise levels in `/255`; σ=0 is rejected |
| `--limit` | `10` | first N images of the split |
| `--eps` | `0.005,0.01,0.02` | relative L2 budgets |
| `--steps` | `20` | PGD iterations |
| `--step-scale` | `2.0` | step = scale × radius / steps |
| `--attack-arm` | `m1` | `m1` or `m3` |
| `--m1-checkpoint` | `checkpoints/m1/nafnet/best.pt` | M1 checkpoint |
| `--m3-checkpoint` | none | include M3 in the evaluation |
| `--m2-method` | `admm_epll` | `admm_epll` or `fista_tv` |
| `--gmm` | none | EPLL prior; required for `admm_epll` |
| `--lambda-values` | none | one value or one per noise level |
| `--m2-steps` | 10 (EPLL) / 100 (FISTA) | solver iterations |
| `--stride` | `1` | EPLL patch stride |
| `--out` | `artifacts/stability/...` | CSV path |

**Outputs.** `*_summary.json` gives mean clean PSNR, adversarial PSNR, drop, $\lVert\eta\rVert/\lVert y\rVert$, SSIM and residual by (method, σ, ε). `*_metadata.json` records the checkpoints, attack settings, protocol and git commit.

**Interpretation.** A positive `drop` at a small ε that is much larger for M1 than M2/M3 supports H3. Because this is a transfer attack, report it as such; it is not a per-method worst-case bound.

**Result (transfer attack, 10 images).** At $\sigma\ge5/255$, ADMM-EPLL loses 2–17× less PSNR and loses less in 99–100% of cells. At $\sigma=1/255$ the picture splits: on $P_1$ EPLL still loses less in 92% of cells, but on $P_2$ it loses less in only 38–41%, because the weak $\sigma=1$ λ amplifies the perturbation. EPLL's absolute PSNR after the attack remains higher than NAFNet's everywhere. H3 is therefore supported at $\sigma\ge5$ and only in distribution at $\sigma=1$; M3 is needed to attribute this to A-awareness rather than prior depth. Details: `report/study_report.md` §7.

---

## 7. Datasets, splits, and protocol (locked)

### 7.1 Datasets

| Tier | Dataset | Size / split | Role |
|---|---|---|---|
| **1 (core)** | **BSDS500** (Arbeláez et al., TPAMI 2011) | 500 images: **200 train / 100 val / 200 test** | Controlled synthetic comparison |
| **1b (Phase 2, planned)** | **`syn-v1`** synthetic images from a pretrained latent-diffusion model (§12.2) | ~2,000 after QC; **training only** | Training-pool expansion shared by every learned component; never in val/test |
| **1c (Phase 2 control, planned)** | **DIV2K** train (real) | 800 images; training only | Separates "more data" from synthetic-domain effects |
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

**Not used:** Set5 / Set14 / Urban100 / Kodak (SR benchmarks). DIV2K is allowed only as an optional *data-scale ablation* applied to **all** learned priors; Phase 2 uses it as the real-data control (§12.4, 2A-3).

### 7.2 Locked preprocessing decisions

| Decision | Choice | Rationale |
|---|---|---|
| BSDS version | **BSDS500**, standard directory split | Resolves the proposal's ambiguous "BSDS300/500"; matches the TPAMI citation |
| Color space | **Luminance only** (Rec.601), core protocol | EPLL/K-SVD are grayscale; keeps all arms comparable |
| Training data | **v1: BSDS500 train only**, for every learned component (GMM prior, CNN weights, unrolled nets). **v2 (planned):** BSDS500 train + `syn-v1`, same mixture ($p_\text{real}=0.5$) for every learned component | Prevents cross-arm training-data confounds and GMM leakage |
| Validation | **BSDS500 val only** for $\lambda$ tuning and early stopping | Same val for every arm; test never used for model selection |
| Test | **BSDS500 test**, full images; 256×256 centre crops as secondary | Circular degradation makes full-image FFT evaluation exact |
| Patch protocol | Split **by image first**, then crop 128×128 patches | Patches from one image never straddle splits |
| Augmentation | Flips + 90° rotations, identical for all learned priors | Removes augmentation as a confound |
| PSF library | Gaussian $\sigma_k \in \{0.7,1.2,1.6,2.0\}$; 8 procedural linear motion kernels (length × angle); defocus radii $\{3,5,7,9\}$ | Kernels generated in code, no download; official Levin kernels optional, outside the protocol |
| Noise | $\sigma \in \{0,1,5,15,25\}/255$, additive white Gaussian, clipped to $[0,1]$; **v2 (planned):** then quantized to 8 bits | Clipping is part of the observation model for every arm; quantization removes the σ = 0 inverse crime |
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
| 7 | direct arm (`direct/nafnet.py`, `direct/unet.py`, `training/`, `scripts/train_m1.py`, `scripts/eval_m1.py`) | ✅ implemented + tested; M1/M2 complete comparison in `report/study_report.md` |
| 8 | unrolled arm (`unrolled/usrnet.py`, `unrolled/ista_net.py`, `unrolled/lista.py`, `scripts/train_m3.py`) | ✅ implemented + tested (torch-guarded); training pending |
| 9 | unified evaluation (`evaluation/`, `scripts/evaluate.py`) with paired per-cell baseline | ✅ implemented + tested |
| 10 | consolidated report (`report/study_report.md`, `report/generated/`) | ✅ M1/M2 + H2/H3/LPIPS results; M3 pending |
| H3 | stability (`metrics/stability.py`, `scripts/stability_test.py`) | ✅ implemented + run (transfer, 10 images); report §7 |
| Overfit check | `scripts/analyze_overfit.py`, `report/nafnet_overfitting.pdf` | ✅ run: no image memorisation; kernel-distribution overfitting |
| Phase 2 | M3 fixes, synthetic pipeline, kernel library v2, mixture sampler, batched and differentiable M2 solvers (§12.6) | ✅ implemented + tested; runs pending ([[Phase 2 Run Checklist]]) |

---

## 8. Diagnosis → prescription (the follow-up)

| Arm | Diagnosed weakness | What a follow-up method would need to fix |
|---|---|---|
| M1 | Kernel mismatch, hallucination, instability | Restore data-consistency without giving up the deep prior |
| M2 | Staircase/ringing on texture, brittleness to wrong $k$ | A deeper (or multimodal) prior while keeping the solver's physics |
| M3 | Cost, residual sensitivity to unseen kernels | Cheaper/adaptive physics module; better prior routing |

**Candidate follow-up (future work, not in this study's core):** a physics-aware network whose prior is a *mixture* — e.g. patch-GMM-routed experts, or a GSM-scale-driven spatially varying regularization strength — targeting the exact failure mode the three-arm comparison identifies.

**Phase 3 proposal.** The primary follow-up is **K-NAFNet**: a NAFNet backbone conditioned on the known kernel and σ through a kernel encoder and FiLM, followed by one closed-form FFT data-consistency step. It is specified in §12.9, with the patch-GMM-routed and GSM-scale variants kept as alternatives. The comparison→method logic is deliberate: the ablation asks whether mismatch robustness comes from data coverage, kernel conditioning, unrolling, or data consistency.

---

## 9. Optional / stretch (not core)

| Method | Basis | Why optional |
|---|---|---|
| **Deep Image Prior** (Ulyanov et al., CVPR 2018) | Untrained network as implicit prior; early stopping | Per-image optimization, expensive; discussion only |
| **GSM-scale spatially varying $\lambda$** | Local scale variable $\hat{z}$ estimates local SNR | Follow-up experiment; not needed for the core table; alternative in §12.9.6 |
| **Patch-GMM-routed experts** | Mixture of experts gated by GMM responsibilities | Future-work proposal, motivated by the comparison; alternative in §12.9.6 |

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

## 12. Phase 2 plan and Phase 3 proposal (after the mid-evaluation)

> [!abstract] Summary
> The mid-evaluation reviewer raised two points: **(R1)** M1 may be data-hungry or overfit, and **(R2)** three kernel families are too few. Phase 2 answers both without breaking the controlled comparison. **(1)** A synthetic training-image pipeline (pretrained latent diffusion), used identically by every learned component of every arm. **(2)** Protocol **v2** with a wider kernel library, including real recorded camera-shake kernels. Validation and test stay real BSDS500 images. The v1 results reported at the mid-evaluation are frozen and remain the reference point. **Status: code implemented and tested (2026-11-15); runs pending, see [[Phase 2 Run Checklist]].**

> [!important] Alignment with the project goal
> Phase 2 is **not** the proposed method. Its job is to **finish and stress-test the three-arm comparison**: train M3, test data hunger, widen kernel coverage, and fix the σ = 0 protocol. The method proposal is **Phase 3 (§12.9)**, designed from the v1 diagnosis in §8 and the Phase 1/2 results.

### 12.0 Reviewer feedback and our response

| # | Reviewer point | Evidence we already have | Phase 2 response |
|---|---|---|---|
| R1a | NAFNet may have **overfit** | No image memorisation: on identical conditions the train−test gap in gain is −0.04 dB for NAFNet vs −0.08 dB for the EPLL control (`report/nafnet_overfitting.pdf`, `scripts/analyze_overfit.py`). Validation PSNR never declines (best at 98k of 100k iterations). What it does overfit is the **training kernels** (gain +5.38 → +2.73 / +2.49 dB on $P_2$). | Report the check; attack the real failure with kernel variety (R2). |
| R1b | NAFNet is **data-hungry** | Untested. More iterations would not help (flat curve); NAFNet beats the smaller U-Net by 0.7 dB on the same data. | Measure it directly (real-data learning curve), then add synthetic images to the training set of **all** arms (§12.2). |
| R2 | Only **three kernel families** | v1 has isotropic Gaussian, linear motion and defocus disks. | Protocol v2 with 7 families, procedural training pools and real recorded kernels (§12.3). |
| — | Carried over from mid-evaluation | M3 not trained; σ = 0 is an inverse crime; H3 is a transfer attack only. | Fix and train M3 first; quantize observations in v2; white-box M2 attack stays a follow-up. |

### 12.1 Principles (non-negotiable for a fair comparison)

1. **Same training data for every learned component of every arm.** M1 network weights, the M2 GMM prior and M3 network weights are all fitted on the same image pool with the same real/synthetic mixing ratio. Methods with no learned part (Wiener, Richardson–Lucy, FISTA-TV) are unaffected; this is stated in every table.
2. **Validation and test stay real.** Synthetic images never enter validation, test, checkpoint selection or λ tuning. The 10-image val subset and the 200-image test split are unchanged.
3. **Versioned protocols, never mixed.** v1 = mid-evaluation protocol (frozen). v2 = expanded kernels + quantized observations. Every table states its protocol version and training-data version.
4. **Every change is an ablation before it becomes a default.** Synthetic data and kernel variety are each measured against the v1 baseline, then adopted for v2.
5. **Pre-registered endpoints** (§12.4) are fixed before any Phase 2 run.

### 12.2 Synthetic training data pipeline (`syn-v1`)

**Goal.** Enlarge the training pool from 200 real images to about 2,200 (200 real + ~2,000 synthetic), with a scale point at ~5,000 if the learning curve is still rising.

**Generator choice.**

| Option | Diversity | Realism | Feasibility | Decision |
|---|---|---|---|---|
| Pretrained latent diffusion, text-to-image (Stable Diffusion 1.5 / SDXL-Turbo via `diffusers`) | High: prompt-controlled scenes | Photographic with good prompts; tends to smooth textures and shallow depth of field | Runs on the RTX 3090 Ti; fixed seeds make it reproducible | **Primary** |
| Same model, image-to-image variations of BSDS **train** images (strength 0.3–0.5) | Medium; stays close to BSDS content | High | Same | Secondary, at most 25% of `syn-v1` |
| GAN trained on our 200 images | Very low (200 images is far too few) | Poor | Not viable | Rejected |
| Pretrained domain GANs (e.g. StyleGAN on LSUN) | Narrow (one category per model) | Good within the domain | Feasible | Rejected for diversity |
| **Real extra data** (DIV2K, 800 train images) | High | Real | Download only | **Control** (separates "more data" from "synthetic artefacts") |

**Pipeline steps.**

1. **Prompt bank** mirroring BSDS500 content: animals, people, landscapes, architecture, plants, water, vehicles, objects and close-up textures, crossed with lighting/viewpoint modifiers and the suffix "sharp, in-focus photograph". Negative prompt: blur, bokeh, depth of field, motion blur, text, watermark, illustration, painting, 3D render. Shallow depth of field matters most: an already-blurred background would make the "sharp" target wrong.
2. **Generation** at the model's native resolution with one fixed seed per image. A manifest records model id and revision, prompt, negative prompt, seed, steps, guidance scale and scheduler.
3. **Preprocessing identical to BSDS500:** anti-aliased downsampling to 481×321 (landscape) or 321×481 (portrait), Rec. 601 luminance, lossless PNG. Whether to apply JPEG compression to match BSDS's JPEG statistics is decided once and recorded.
4. **Quality control** (`scripts/check_synthetic.py`, planned):
   - **sharpness:** reject images whose high-frequency energy (variance of the Laplacian) is below the 5th percentile of BSDS train;
   - **degenerate content:** reject flat or low-entropy images, and images with text or watermarks;
   - **near-duplicates** inside `syn-v1` (perceptual hash);
   - **leakage:** nearest-neighbour check of every synthetic image against BSDS **val and test** using a perceptual hash and a deep embedding (CLIP/DINOv2); reject above a fixed similarity threshold;
   - **statistics report** comparing `syn-v1` with BSDS train: gradient-magnitude histogram, radial power-spectrum slope, and the H2 texture score distribution.
5. **Freeze.** SHA-256 per image, dataset name `syn-v1`, manifest stored with the protocol; never regenerated silently. Expected rejection rate 15–30%, so roughly 2,600 images are generated to keep 2,000.

**Mixing.** The training sampler draws a real image with probability $p_\text{real}$ and a synthetic one otherwise. The primary setting is $p_\text{real}=0.5$; $p_\text{real}=1.0$ (real only, = v1) and $p_\text{real}=0$ (synthetic only, a domain-gap diagnostic) are ablations.

> [!note] Go/no-go for the synthetic pipeline
> Generate `syn-v1` **only if** 2A-1 shows NAFNet is still improving from 100 to 200 real images **and** the DIV2K control improves over real-only. If the real-data curve is flat, skip the synthetic pipeline and spend the budget on kernel variety, M3 and Phase 3. This keeps the plan aligned with the comparison→method goal instead of turning into a data-generation project.

**How each arm uses it.**

| Arm | Component | Change |
|---|---|---|
| M1 | NAFNet, U-Net, per-σ specialists | Training patches drawn from the mixture |
| M2 | EPLL GMM prior (K-sweep too) | Fitted on 8×8 patches drawn from the same mixture; patch budget raised from 100k to 200k |
| M2 | Wiener, Richardson–Lucy, FISTA-TV | No learned part; unchanged |
| M2 | λ (FISTA-TV, ADMM-EPLL) | Re-tuned on the real val subset whenever the prior changes |
| M3 | USRNet, ISTA-Net, LISTA | Same sampler and mixture as M1 |

**Dependencies (to add):** `diffusers`, `transformers`, `accelerate`, `imagehash`, `open_clip_torch` (or DINOv2). The model weights are a one-time download (about 2–7 GB) and their licence (CreativeML OpenRAIL-M / SDXL licence) must be cited.

### 12.3 Kernel variety: protocol v2

v1 has three families: isotropic Gaussian, linear motion and defocus disks. v2 has seven:

| # | Family | Model and parameters | What it represents | Set |
|---|---|---|---|---|
| 1 | Isotropic Gaussian (v1) | $\sigma_k$ on a 0.2 grid in $[0.6, 2.4]$; interp between grid points | Mild optical softness | train / interp |
| 2 | **Anisotropic Gaussian** | $\sigma_x \ne \sigma_y$ on a grid in $[0.6, 3]$, rotation $\theta$ at multiples of 15°; interp between grid points | Astigmatism, directional softness | train / interp |
| 3 | Linear motion (v1) | length $\{5,9,13,17,21\}$ px at multiples of 15°; interp at lengths $\{7,11,15,19\}$ and angles offset by 7.5° | Straight camera or object motion | train / interp |
| 4 | **Random-trajectory camera shake** | Boracchi & Foi (2012) random-walk trajectories (as used by DeblurGAN); shake intensity on a grid, training seeds; interp at intermediate intensities with held-out seeds | Realistic hand shake, curved paths | train / interp |
| 5 | Defocus disk (v1) | radius 2–10 px | Out-of-focus lens | **extrap** |
| 6 | **Real recorded camera shake** | Levin et al. (2009), 8 kernels (loader exists); Sun et al. (2013); Köhler et al. (2012), 12 kernels | Measured real blur, not procedural | **extrap** |
| 7 | **Diffraction / turbulence** | Airy disk; Moffat profile (heavy tails) | Small apertures, atmospheric blur | **extrap** |

**Set design.**

- **$P_1$ training pool:** families 1–4 on their *training grids* (table above), about 100 kernels per family, about 400 in total (v1 had 12). The sampler draws one per patch, so the network sees far more kernel variety than in v1.
- **$P_1$ test:** 24 kernels taken from the training pool (6 per family), as in v1, where the in-distribution test kernels are the training kernels.
- **$P_2$-interp:** 24 kernels from families 1–4 with parameters *between* the training grid points (and held-out seeds for trajectories). Same kinds of blur, settings never seen; the same logic as v1.
- **$P_2$-extrap:** families 5–7: unseen kinds of blur, including real measured kernels. Defocus stays here so v1 and v2 remain comparable.
- **Kernel caps:** normalised to sum 1, support at most 31×31.
- **Freeze:** all kernels saved as arrays with their parameters and seeds (`artifacts/protocol/v2/psfs.npz` + JSON, SHA-256).

**Observation fix (from the σ = 0 inverse crime).** v2 quantizes every observation to 8 bits after noise and clipping, so σ = 0 becomes an honest condition instead of an exact inversion.

### 12.4 Experiments, hypotheses and endpoints

**Order.** Phase 2A answers the reviewer quickly on the frozen v1 protocol; Phase 2C produces the final v2 results.

| Phase | Experiment | Runs | Answers |
|---|---|---|---|
| **2A-1** | Real-data learning curve: NAFNet on 50 / 100 / 200 real images (v1) | 2 new trainings | Is NAFNet data-hungry? (R1b) |
| **2A-2** | Synthetic augmentation on v1: NAFNet, U-Net, GMM-200 on real + `syn-v1` ($p_\text{real}=0.5$) | 2 trainings + 1 prior | Does more data help, and which arm? |
| **2A-3** | Controls: NAFNet on synthetic only; NAFNet on real + DIV2K | 2 trainings | Domain gap; synthetic vs real extra data |
| **2B** | Fix and train M3 on v1 (USRNet, ISTA-Net, LISTA) | 3 trainings | The missing control arm |
| **2C-1** | Build protocol v2 (kernels, quantized observations, manifests) | — | — |
| **2C-2** | Train all arms on v2 $P_1$ pool with the data setting chosen in 2A | NAFNet, U-Net, GMM-200, USRNet (+ ladder) | Final three-arm comparison |
| **2C-3** | Kernel-diversity ablation: NAFNet trained on v1 kernels, tested on v2 | 0 (reuse) | Does kernel variety reduce M1's mismatch gap? |
| **2C-4** | Re-run H2 (texture), H3 (stability, incl. M3 as attacker and target), LPIPS on v2 | — | Hypotheses on v2 |
| **2C-5** | **Frozen-v1 coverage ablation:** NAFNet trained on the v2 dense/continuous $P_1$ pool, evaluated on the **frozen v1 $P_2$-interp and $P_2$-extrap** sets; compare v1 M1, v2 M1 and v2 M3 | 0 new trainings (reuse 2C-2) | Does kernel coverage alone fix the original mismatch? This is the direct test of the dense-angle idea (the earlier "M1-dense" arm). The 15° training grid and 7.5° interp offsets make it a held-out-angle test by construction. |
| **2C-6** | Re-run the $K$-sweep and the per-σ specialists on v2 | 5 priors + 5 specialists | Do the prior-depth and σ ablations survive the wider protocol? |
| **2C-7** | **White-box H3 on M2:** a gradient-enabled FISTA/EPLL (reduced iterations) and per-method PGD | 1 implementation + a small run | Is M2 also unstable to its *own* worst-case perturbation, or only to M1-crafted ones? |

**New hypotheses.**

- **H4 (data).** More training images help the deep arms (M1, M3) in distribution more than the shallow M2 prior, which already saturates in the K-sweep. More data does **not** change kernel-mismatch robustness. *Falsified if* NAFNet's $P_2$ gain retention rises substantially with data alone.
- **H5 (kernel diversity).** Training on more kernel families narrows M1's gap on unseen *parameters* ($P_2$-interp) but not on unseen *families* ($P_2$-extrap); the physics-aware arms stay ahead on unseen families and real kernels. *Falsified if* NAFNet trained on the v2 pool matches the physics-aware arms on $P_2$-extrap. The direct coverage check is 2C-5: a v2-trained NAFNet evaluated on the **frozen v1 $P_2$** sets.

**Pre-registered endpoints.**

| ID | Endpoint | Test |
|---|---|---|
| E1 | ΔPSNR on v1 $P_1$ test, real + `syn-v1` vs real only, for NAFNet and for ADMM-EPLL (σ > 0) | Paired Wilcoxon over 200 images, Holm |
| E2 | NAFNet gain retention (gain on $P_2$ / gain on $P_1$), v1 kernels vs v2 pool, on v2 test | Paired Wilcoxon |
| E2b | NAFNet gain retention on the **frozen v1 $P_2$** sets, v2-trained vs v1-trained | Paired Wilcoxon |
| E3 | Three-arm PSNR on v2 $P_1$, $P_2$-interp, $P_2$-extrap (σ > 0) | As in Phase 1 |

### 12.5 Compute budget (single shared RTX 3090 Ti)

| Item | Estimate |
|---|---|
| Synthetic generation (~2,600 images) + QC | 2–6 GPU-h |
| 2A: 6 M1 trainings (≈2.8 h NAFNet, 0.5 h U-Net) + GMM priors + v1 re-evaluation | ≈20–25 GPU-h |
| 2B: M3 fixes, training and v1 evaluation | ≈10–15 GPU-h |
| 2C: v2 training (all arms) | ≈15–20 GPU-h |
| 2C: v2 evaluation grid (~72 kernels × 5 σ; M2 solvers batched across cells, 100-image subset for FISTA-TV) | ≈40–60 GPU-h |
| 2C: H3, LPIPS, texture on v2 | ≈10 GPU-h |
| **Total** | **≈100–135 GPU-h** (planning estimate), about 1.5–2.5 weeks of wall-clock on a shared GPU. Re-estimated from measured speeds after implementation: **≈90 GPU-h without synthetic data, ≈105 with** ([[Phase 2 Run Checklist]]) |

The v2 evaluation grid is the cost driver: without batching the M2 solvers it would be about 55 GPU-h for FISTA-TV alone. Batching the torch solvers across cells is therefore part of the implementation work.

### 12.6 Implementation (code complete, 2026-11-15; runs pending)

All Phase 2 code below is implemented, unit-tested (153 tests) and smoke-tested on the GPU. What remains is running it: see **[[Phase 2 Run Checklist]]** for every run, its command and its measured time.

| Area | Implemented in |
|---|---|
| M3 fixes | `unrolled/usrnet.py`: pads to a multiple of 8 and returns the last prior output; `unrolled/ista_net.py`, `unrolled/lista.py`: residual on the gradient step with a zero-initialised last layer, so each phase starts as a plain gradient step; the 3 `assert_close` tests fixed. Smoke runs: USRNet 25.4 dB, ISTA-Net 25.8 dB after 400 iterations (previously a crash and 4.7–12.5 dB) |
| Kernels | `psf/library.py`: `anisotropic_gaussian_kernel`, `trajectory_kernel` (Boracchi & Foi), `airy_kernel`, `moffat_kernel`, `fit_support` (31×31 cap), `load_kernel_dir` (Levin/Sun/Köhler), `protocol_psf_sets("v2")`. v1 is unchanged and pinned by a SHA-256 test |
| Protocol | `protocol/config.py`: `protocol_version`, per-version defaults, `real_kernel_dirs`, `quantize_observations`, `psf_sets()`, kernel-set digest in the manifest; `protocol/observations.py`: 8-bit quantization |
| Training data | `data/training_pool.py` (one pool builder and one set of options for `train_m1.py`, `train_m3.py`, `train_gmm.py`: `--train-images`, `--extra-images`, `--p-real`, `--require-manifest`); `data/synthetic.py` (BSDS-size normalisation, manifest-verified loading, nested subsets, QC primitives) |
| Sampler / trainer | `training/sampler.py`: real/extra mixture, family-balanced kernels, quantized observations (v1 batches bit-identical); `training/trainer.py`: logs train PSNR and validation PSNR on σ > 0 |
| Synthetic pipeline | `scripts/generate_synthetic.py` (336-prompt bank, fixed seeds, img2img from train only, ≤25%), `scripts/check_synthetic.py` (sharpness, entropy, duplicates, pHash/CLIP leakage vs val+test, statistics report, frozen manifest) |
| Setup | `scripts/setup_datasets.py`: `sun2013`, `kohler2012`, `div2k`; `--write-protocol --version v2` writes the kernels as arrays with SHA-256 |
| Evaluation | `scripts/evaluate.py`: `--protocol`, `--real-kernels`, batched torch M2 solvers (`--m2-batch`); `evaluation/grid.py`: batch API and quantization |
| White-box H3 (2C-7) | `methods/classical/torch_backend.py`: differentiable `fista_tv_torch_diff` / `admm_epll_torch_diff`; `scripts/stability_test.py --attack-arm m2` with a reduced-iteration surrogate (20×10 FISTA-TV, 10 EPLL) |
| Tests | `tests/test_phase2.py` (18 tests) and new M3 tests |

**Decisions made during implementation** (all recorded in the protocol manifest):

- **v2 training pool: 290 kernels**: 10 isotropic Gaussian, 120 anisotropic, 60 motion, 100 shake. Because the families are unequal in size, v2 samples kernels **family-balanced** (family first, then a kernel). `condition_sampling` records this; v1 stays `uniform_independent`.
- **Shake intensity** is mapped to an irregularity of at most 0.03. This was calibrated visually against recorded kernels: higher values produce unrealistic scribbles.
- **$P_1$ test** kernels are a seeded random draw of 6 per family from the pool (spread over angles and widths); **$P_2$-extrap** without downloads is defocus (4) + Airy (3) + Moffat (3), and real kernels are added with `--real-kernels` before v2 is frozen.
- **FISTA-TV cost is set by the FFT size, not by batching or precision:** cuFFT is ≈2.5× slower at 481×321 than at 480×320; batching gives ≈1.2× and float32 nothing. The v2 budget therefore keeps FISTA-TV on a 100-image subset.
- **2C-5 caveat:** the v2 Gaussian training grid contains σ_k = 1.4 and 1.8, two of the three Gaussians in the *v1* $P_2$-interp set; report or exclude them in the frozen-v1 coverage test.

### 12.7 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Synthetic images are smoother or "painterly" (domain gap) and bias the priors | Sharpness/statistics QC, $p_\text{real}=0.5$, synthetic-only diagnostic, DIV2K real-data control |
| Generated images resemble BSDS test images (leakage) | Hash and embedding nearest-neighbour check against val and test; rejected images listed in the manifest |
| Kernel pool makes M1 look better only because test kernels are "easier" | $P_1$ test and $P_2$ sets frozen before any v2 training; report the blurred-input baseline per set |
| Real kernel datasets unavailable or licence-restricted | Levin kernels already supported; Sun/Köhler are optional additions to $P_2$-extrap |
| v2 evaluation cost | Batched solvers, image subset for FISTA-TV, M1/M3 on the full grid |
| Shared GPU | One job at a time, memory guard, low priority (as in Phase 1) |

### 12.8 Timeline

| Week | Work |
|---|---|
| 1 | M3 fixes and v1 training; synthetic pipeline; learning curve (2A-1) |
| 2 | Generate and QC `syn-v1`; 2A-2 and 2A-3 runs; build v2 kernels and protocol |
| 3 | Train all arms on v2 |
| 4 | v2 evaluation, H2/H3/LPIPS reruns, final report and presentation |

### 12.9 Phase 3: from diagnosis to a proposed method (K-NAFNet)

> [!abstract] Summary
> Phase 2 finishes and stress-tests the three-arm comparison. Phase 3 uses that diagnosis to propose **one** better method. The diagnosis is specific: **M1 has a deep prior but is kernel-blind; M2 has the right physics but a shallow prior; M3 is designed to have both but pays for unrolling.** The proposal is **K-NAFNet**: the same NAFNet backbone, conditioned on the known kernel and noise level through a small kernel encoder and FiLM modulation, followed by one closed-form FFT data-consistency step. It is designed to keep M1's in-distribution accuracy, borrow M2's mismatch robustness, and cost one forward pass plus one FFT solve. **Status: proposed, not implemented; it starts only after Phase 2 produces the v2 results.**

#### 12.9.1 What the diagnosis requires

| Evidence (v1, frozen) | Design requirement | K-NAFNet element |
|---|---|---|
| NAFNet loses half its gain on unseen motion angles (+5.38 → +2.73 / +2.49 dB); unseen Gaussian widths cost only −0.7 dB | The model must be told *which* blur it is inverting, not infer it from a finite training set | Kernel encoder → FiLM conditioning |
| Per-σ specialists add at most 0.25 dB | One model should cover σ, not five checkpoints | σ embedding |
| NAFNet is unstable under small perturbations and can hallucinate | The output must remain consistent with the observation | One closed-form FFT data-consistency step |
| GMM capacity saturates ($K=1\to4$ +1.8 dB, then +0.27, +0.11, +0.02) | The prior must be deeper than a patch GMM | NAFNet backbone (17.1 M) |
| M2 costs 0.8–2.1 s/image and needs λ tuning | Amortise inference into one network pass | One forward pass + one FFT solve |
| The M2 family's TV variant has staircase/ringing on texture; M1 is not better on high texture either | Replace the shallow prior without losing physics | Deep prior + DC, texture evaluated explicitly |
| USRNet pays 8 unrolled iterations with a small prior | Separate what *conditioning* buys from what *unrolling* buys | One DC step primary; 4-step unrolled ablation |

#### 12.9.2 Architecture

**Backbone.** NAFNet exactly as M1: width 32, `enc_blocks=(1,1,1,28)`, `middle_blocks=1`, `dec_blocks=(1,1,1,1)`, 36 NAFBlocks, 17.1 M parameters. Keeping the backbone identical makes the comparison a test of *conditioning and data consistency*, not of capacity.

**Kernel encoder.** The kernel is spatially uniform and small (≤31×31), so a global embedding is sufficient. A small CNN maps $k$ to $e_k\in\mathbb{R}^{64}$:

$$
k\in\mathbb{R}^{31\times31}\;\to\;\text{Conv}_{3\times3}(1\to16)\to\text{GELU}\to\text{Conv}_{3\times3}(16\to32)\to\text{GAP}\to\text{Linear}(32\to64).
$$

The encoder is trained jointly with the backbone; it has about 0.02 M parameters.

**Noise encoder.** $\sigma$ spans 1–25/255, so encode $\log\sigma$ with Fourier features and a two-layer MLP to $e_\sigma\in\mathbb{R}^{16}$.

**Conditioning.** Concatenate $c=[e_k;e_\sigma]\in\mathbb{R}^{80}$. Blocks at the same width share a FiLM layer (four scales), so the modulation is scale-specific but not block-specific:

$$
h_{\ell+1}=h_\ell+\bigl(1+\gamma_{s(\ell)}(c)\bigr)\,B_\ell(h_\ell)+\beta_{s(\ell)}(c),
$$

where $B_\ell$ is the NAFBlock residual branch, $s(\ell)$ is the scale of block $\ell$, and $\gamma_s,\beta_s\in\mathbb{R}^{C_s}$ come from one linear layer per scale. Initialise $\gamma_s=\beta_s=0$, so training starts from an unconditional NAFNet and learns how much conditioning to use. The FiLM layers add about 0.08 M parameters. Total ≈ **17.2 M**, essentially M1.

**Output and data consistency.** The network predicts a residual, $\hat{x}_0=y+f_\theta(y,c)$. One closed-form FFT step then projects it onto the observation-consistent set:

$$
\hat{x}=\arg\min_x \tfrac12\lVert Ax-y\rVert_2^2+\frac{\rho}{2}\lVert x-\hat{x}_0\rVert_2^2
\quad\Longleftrightarrow\quad
\hat{X}=\frac{\overline{K}\,Y+\rho\,\hat{X}_0}{|K|^2+\rho},
$$

with $\rho>0$ selected on validation as a function of $\sigma$ (a single scalar per noise level, following the HQS data module used by USRNet). This is the same update as the $x$-step of ADMM/HQS, but applied once and amortised. $\rho\to\infty$ recovers the raw network output; $\rho\to0$ recovers the (unstable) inverse filter. The DC step is differentiable, so it can be trained through.

**Unrolled variant (ablation).** Repeat $x_{t+1}=\mathrm{DC}\bigl(x_t+f_\theta(x_t,c)\bigr)$ for 2–4 steps, sharing weights. This tests whether the gain comes from conditioning or from iterating the physics; it is not the primary model because it multiplies inference cost.

#### 12.9.3 Training protocol

| Item | Setting |
|---|---|
| Data | Protocol v2 $P_1$ pool (families 1–4), same 200 real images as M1-v2, plus `syn-v1` **only if** Phase 2A adopts it |
| Conditions | One model over all kernels and $\sigma>0$; no per-kernel or per-σ checkpoints |
| Patches | 128×128 random crops, rot90 + flip, kernel and σ drawn per patch, as M1 |
| Kernel jitter | With probability 0.1, perturb the sampled kernel (angle ±7.5°, length ±2 px, width ±0.3 px) so the encoder sees a continuous manifold rather than 400 discrete points |
| Loss | L2 on the final $\hat{x}$ after DC, plus $0.1\times$L2 on $\hat{x}_0$ |
| Optimiser | AdamW, lr $2\times10^{-4}$, wd $10^{-4}$, 1k warm-up + cosine, grad clip 1.0 |
| Budget | 100k iterations, batch 16, ≈3–5 GPU-h on the RTX 3090 Ti |
| Init | From scratch on the same v2 data as M1-v2; warm-starting from M1 is an ablation, not the default |
| Selection | Same 10-image real val subset and 60-cell $P_1$ grid as M1, never per-condition |

The kernel jitter and the wrong-kernel test in §12.9.5 are the safeguards against the failure mode this study already measured: a network that ignores the condition and memorises the training pool.

#### 12.9.4 Hypotheses and falsification

| ID | Claim | Target | Falsified if |
|---|---|---|---|
| **P1** | In distribution, conditioning costs nothing | Within 0.2 dB of M1-v2 on v2 $P_1$ | Loses >0.3 dB |
| **P2** | Kernel conditioning closes the unseen-parameter gap | ≥2 dB above M1 and within 1 dB of M2 on $P_2$-interp | Still >2 dB behind M2 |
| **P3** | Conditioning transfers to unseen *families* | Beats M1 on $P_2$-extrap (defocus, real, Airy/Moffat) | No gain over M1 |
| **P4** | One DC step buys stability | Smaller H3 drop than M1, approaching M2 | H3 drop unchanged |
| **P5** | A deep prior fixes M2's texture deficit without hallucination | Positive K-NAFNet−EPLL on high-texture images | Inherits M1's texture behaviour |
| **P6** | The condition is actually used | Wrong kernel at test hurts more than the true kernel; zeroing $e_k$ hurts $P_2$ but not $P_1$ | No difference, i.e. the network ignored $c$ |

P3 is the honest stress test. A kernel encoder sees shape, not a family label, so it *may* extrapolate; if it does not, the conclusion is that unseen families need explicit iteration (M3) or a broader prior, and that is a publishable negative result rather than a failure of the study.

#### 12.9.5 Ablations and evaluation

| ID | Variant | Question |
|---|---|---|
| A0 | M1-v2 (NAFNet, no $k$) | Baseline: deep prior without physics |
| A1 | $k$ concatenated as a second input channel | Is a simple spatial concatenation enough? |
| A2 | Kernel encoder + FiLM | Does global conditioning work? |
| A3 | A2 + σ embedding | Does σ conditioning add anything to one model? |
| A4 | A3 + one DC step | **Primary:** does explicit consistency help? |
| A5 | A4 + 2–4 unrolled DC steps | Is iteration worth the cost? |
| A6 | A4 without kernel jitter | Does jitter prevent pool memorisation? |
| A7 | A4 with the wrong kernel at test | Is the condition used? |
| A8 | A4 with $e_k=0$ at test | How much comes from conditioning vs the prior? |

Evaluation uses the v2 grid and the same statistics as Phase 1: PSNR/SSIM, gain retention, LPIPS, data-consistency residual, H3, texture, inference time and parameter count, paired Wilcoxon over 200 images with Holm correction. The headline table has four rows: **M1-v2, M2, M3-v2, K-NAFNet**, on $P_1$, $P_2$-interp, $P_2$-extrap and real kernels.

#### 12.9.6 Alternatives considered

| Alternative | Idea | Why it is not primary | When it becomes primary |
|---|---|---|---|
| **PnP-EPLL + CNN denoiser** | Keep ADMM, replace the GMM Wiener step with a small CNN denoiser | Loses the classical-ML identity; essentially DPIR; still iterative | If K-NAFNet fails on unseen families and M3 is too expensive |
| **Patch-GMM-routed experts** | GMM responsibilities route between expert heads or set a local λ | More complex; routing is noisy at high σ | If texture is the dominant failure and one deep prior over-smooths |
| **GSM-scale spatially varying λ** | Local scale variable sets $\lambda(x)$ | A refinement, not a standalone method | As a plug-in on M2 or K-NAFNet |
| **A properly trained USRNet** | More unrolled iterations, larger prior | Already M3; expensive per image | If unrolling beats conditioning in A5 |

The patch-GMM-routed and GSM-scale ideas in §8–§9 remain the "mixture prior" branch. K-NAFNet is preferred as the primary proposal because it changes one axis at a time, reuses the M1 code and budget, and its ablations answer the study's central question directly: *is mismatch robustness bought by data coverage, kernel conditioning, unrolling, or data consistency?*

#### 12.9.7 Decision gate, budget and risks

**Decision gate.** Phase 3 starts after 2C-5.

- If **2C-5** (v2-trained NAFNet evaluated on the frozen v1 $P_2$ sets) already closes the gap, kernel **coverage** is sufficient. K-NAFNet's remaining contribution is the DC step, σ conditioning and cheaper inference; report both and recommend the cheaper option.
- If **2C-5** does not close the gap, K-NAFNet is the main proposal.
- If **K-NAFNet also fails on $P_2$-extrap**, the study's conclusion becomes stronger: unseen *families* require explicit iterative physics (M3), not just conditioning. Report that as the answer, not as a failed method.

**Budget.** ≈8–12 GPU-h total: 3–5 h for the primary training, 2–3 h for A1–A3 and A5–A6, 3–4 h for evaluation. One week to ten days. One seed for the headline; a second seed only if the P2 gain is below 0.5 dB.

| Risk | Mitigation |
|---|---|
| FiLM is ignored and the model behaves like M1 | A7/A8 mechanism checks; kernel jitter during training |
| Kernel encoder overfits the v2 training pool | Jitter, frozen v1 $P_2$ evaluation, real kernels in $P_2$-extrap |
| The DC step hurts when $k$ is wrong (blind setting) | Non-blind only, as in M2/M3; blind is out of scope |
| Gains appear only in distribution | Report as a negative result; the comparison still stands |
| One seed, small differences | Second seed if the headline gain is <0.5 dB; paired tests over 200 images |
| Scope creep into a new paper | One model, one table, one figure; no new dataset or architecture family |

**Positioning.** This is not a claim of a new architecture family. SRMD conditions on a degradation map, USRNet unrolls HQS with a CNN prior, IRCNN/DPIR use a CNN denoiser inside a plug-and-play solver, and NAFNet is a strong modern prior. K-NAFNet is the **controlled synthesis** the three-arm comparison points to: a modern deep prior, an explicit kernel condition, and one data-consistency step, evaluated under the same protocol as the arms. Its scientific value is the ablation, not state-of-the-art PSNR: it tells us which ingredient buys mismatch robustness.

---

## 13. References

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
21. Zhang, Li, Zuo, Zhang. *Plug-and-play image restoration with deep denoiser prior (DPIR).* IEEE TPAMI, 2022.

---

Status: active — implementation reference for the three-arm study. [[Project Overview]] gives the summary; this document is the full method record.
