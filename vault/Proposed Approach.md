---
title: Proposed Approach — Three-Arm Comparison
date: 2026-10-19
tags:
  - project
  - smai
  - deblurring
  - methods
  - design
aliases:
  - Three Arms
  - Method Design
  - Three-Arm Study
status: draft
---

# Proposed Approach — Three-Arm Comparison

> [!abstract] In one line
> Three **established** methods, one controlled protocol: **M1** (data-only CNN), **M2** (physics-aware classical-ML solver), **M3** (physics-aware unrolled network, the control). No new architecture in the core study — the comparison itself is the contribution, and its failure signature motivates the follow-up method.

Related: [[Project Idea]] · Proposal PDF: `proposal.pdf`

---

## 1. The design principle: two independent axes

Every method in the literature answers two *separate* questions. Keeping them separate is what makes the three-arm design interpretable:

| Axis | Question | Levels |
|---|---|---|
| **Physics-awareness** | Does the method consume the blur kernel $k$ and noise level $\sigma$, and enforce $Ax \approx y$? | no (M1) / yes (M2, M3) |
| **Prior depth** | How much structure does the model assume about natural images, and how much does it learn? | fixed (TV) → shallow learned (GMM, K-SVD) → deep learned (CNN) |

A two-arm comparison (data-only CNN vs classical solver) **confounds** these axes: M1 differs from M2 in both physics-awareness *and* prior depth. The third arm fixes that — **M3 is physics-aware like M2 but deep like M1**, so it isolates the two effects.

> [!note] Terminology (fixes earlier label drift)
> - **M1** = Modern / data-only. **Never** sees $k$ or $\sigma$.
> - **M2** = Classical-ML / physics-aware. **FISTA/ADMM solver + EPLL-GMM prior**, given true $k$, $\sigma$.
> - **M3** = Unrolled control / physics-aware. **USRNet** family, given $k$, $\sigma$.
>
> USRNet is **not** M1, and LISTA is **not** M2. USRNet and LISTA both belong to the unrolled family (M3).

---

## 2. M1 — Modern ML, data-only ($A$-agnostic)

Learns a direct mapping $g_\theta(y) \to x$ from (blurry, sharp) pairs. The degradation is implicit in the data; the network must infer blur and noise by itself.

### Algorithms tried

| Algorithm | Type | Role | What it contributes to the comparison |
|---|---|---|---|
| **NAFNet** (Chen et al., ECCV 2022) | U-Net + NAFBlocks (SimpleGate, simplified channel attention) | **Primary M1** | Strongest practical data-only baseline; 33.69 dB on GoPro |
| **DeepDeblur** (Nah et al., CVPR 2017) | Multi-scale CNN with residual blocks | Historical rung | The original kernel-free deblurring network; shows the family's starting point |
| U-Net regression | Plain encoder–decoder | Minimal baseline (optional) | Establishes how much of the result is "just capacity" |

### Characteristics
- **Inputs:** $y$ only. No $k$, no $\sigma$.
- **Inversion:** a single learned forward pass.
- **Prior:** implicit, distributed across CNN weights.
- **Training signal:** supervised $\ell_2$ (or $\ell_1$) on pairs; patches sampled from the same PSF/noise grid.
- **Inference:** milliseconds per image.
- **Expected failures:** collapse under kernel mismatch (SRMD, CVPR 2018), hallucination, instability under small perturbations (Antun et al., PNAS 2020; Gandikota et al., ICIP 2022 for deblurring specifically).
- **Role in the study:** the "data-only" pole. It is the arm against which the value of physics-awareness is measured (M1 vs M3).

---

## 3. M2 — Classical-ML, physics-aware

Solves a MAP inverse problem with an explicit forward model and an explicit, **learned but shallow** image prior. This is the classical-ML family: learning lives in the prior, not in the inversion.

### Formulation
$$
\hat{x} = \arg\min_x \; \tfrac{1}{2}\|Ax - y\|_2^2 + \lambda R(x),
$$
where $A$ is the known convolution operator (FFT, circular boundary) and $R$ is the prior.

### Solvers tried
| Solver | Role | Notes |
|---|---|---|
| **FISTA** (Beck & Teboulle, TIP 2009) | **Primary solver** | Accelerated proximal gradient; the standard TV-deblurring solver |
| ADMM / HQS | Alternative solver | Required to plug in the EPLL/GMM patch prior (splitting between data and prior) |
| Wiener filter | Sanity baseline | Closed-form, no prior learning; lower bound on what "correct physics, dumb prior" gives |
| Richardson–Lucy | Sanity baseline | Classical non-regularized deconvolution; shows noise amplification |

### Priors tried
| Prior | Type | Role | Notes |
|---|---|---|---|
| **TV** (Rudin–Osher–Fatemi, 1992) | Fixed, hand-crafted | Non-learned limit | Staircase artifacts on texture |
| **EPLL** — 200-component GMM (Zoran & Weiss, ICCV 2011) | Shallow learned | **Primary M2 prior** | 8×8 patches, DC removed, EM on ~2M BSDS patches; inner step = component selection + Wiener filter |
| GMM with $K$ components | Shallow learned | **Ablation ladder** | $K \in \{1, 4, 16, 64, 200\}$; $K{=}1$ is the "naive Gaussian" claim |
| **K-SVD** dictionary (Aharon et al., 2006) | Shallow learned | Optional alternative | Sparse coding prior; time-boxed, drop first if schedule slips |

### Characteristics
- **Inputs:** $y$, **true $k$**, **true $\sigma$**.
- **Inversion:** iterative optimization (hundreds to thousands of iterations; seconds–minutes per image).
- **Prior:** explicit and shallow; learned once on clean patches, outside the solver.
- **Expected failures:** staircase/ringing on natural texture; brittle to wrong kernels (needs correct $k$).
- **Role in the study:** the "physics but shallow prior" pole. Compared with M3 it isolates prior depth; compared with M1 it isolates physics-awareness.

> [!tip] Why the GMM prior is the right choice
> Natural image patch statistics are heavy-tailed and multimodal, so a single Gaussian prior over-smooths. Normalized/local-scale-conditioned patches are approximately Gaussian, and a finite Gaussian mixture captures the multimodal structure — this is exactly the theory behind EPLL and the reason its inner step is a closed-form Wiener filter. The $K$-sweep turns that theory into a measurable rung of the prior-depth axis.

---

## 4. M3 — Unrolled control, physics-aware (the bridge)

Unrolls a fixed number of iterations of a proximal optimization into a network, keeping the forward operator $A$ inside the loop as a differentiable, closed-form module. Physics stays in; the prior (and sometimes the hyperparameters) become learned.

### Algorithms tried (the unrolling genealogy)

| Rung | Algorithm | What is learned | What stays fixed | Role |
|---|---|---|---|---|
| 1 | **LISTA** (Gregor & LeCun, ICML 2010) | Iteration matrices + scalar shrinkage | Linear forward model | Historical rung; ~20× fewer iterations than FISTA at equal error |
| 2 | **ISTA-Net** (Zhang & Ghanem, CVPR 2018) | CNN analysis/synthesis transforms in the proximal step | Data term | Mid rung; replaces hand-crafted prox with a learned CNN |
| 3 | **USRNet** (Zhang, Van Gool, Timofte, CVPR 2020) | ResUNet prior + MLP hyperparameters | Closed-form FFT data module | **Primary M3**; 8 iterations, $k$ and $\sigma$ as inputs |

### Characteristics
- **Inputs:** $y$, **$k$**, **$\sigma$**.
- **Inversion:** fixed-depth unrolled optimization (8 iterations for USRNet), not a converged solver.
- **Prior:** learned deep CNN, applied inside the loop.
- **Inference:** feed-forward, ~ms–s depending on depth/backbone.
- **Why it is the control:** it has M2's physics and M1's deep prior. Any M1–M2 difference can therefore be attributed to physics-awareness or prior depth rather than to both at once.
- **The genealogy is itself an ablation:** LISTA → ISTA-Net → USRNet isolates the contribution of learned matrices, then a learned proximal transform, then a full denoiser + hyperparameter module.

> [!warning] Implementation caveats
> - USRNet's released weights are for **super-resolution**. For deblurring, set the scale factor to 1 and **retrain/fine-tune** on the same PSF/noise protocol; do not use SISR weights as-is.
> - LISTA is designed for sparse coding; adapting it to a convolutional $A$ means unrolling FISTA with learnable step sizes/thresholds — budget it as implementation work, not "off the shelf."
> - ISTA-Net was designed for compressive sensing; it must be adapted to the convolution forward operator and stated as such.

---

## 5. Master comparison

| Property | **M1 — Modern (data-only)** | **M2 — Classical-ML (physics-aware)** | **M3 — Unrolled control (physics-aware)** |
|---|---|---|---|
| Family | Direct deep regression | Iterative solver + learned shallow prior | Deep unfolding |
| Knows $k$? | No (implicit in data) | Yes (explicit) | Yes (explicit input) |
| Knows $\sigma$? | No | Yes | Yes |
| Inversion | Single learned pass | Iterative optimization | Fixed-depth unrolled optimization |
| Prior | Implicit deep CNN | Explicit shallow (GMM / sparsity) | Learned deep CNN inside the loop |
| Algorithms | NAFNet (primary), DeepDeblur (historical), U-Net (minimal) | FISTA/ADMM + EPLL-GMM (primary), K-SVD (optional); TV, Wiener, RL as limits | USRNet (primary); LISTA → ISTA-Net (ablation ladder) |
| Training | End-to-end on $(x,y)$ pairs | Prior learned offline; solver not trained | End-to-end through the unrolled iterations |
| Inference | ~ms/image | seconds–minutes/image | ~ms–s/image |
| Expected failure | Kernel mismatch, hallucination, instability | Staircase/ringing; needs correct $k$ | Between the two; still sensitive to unseen kernels |
| Role in study | Data-only pole | Physics + shallow-prior pole | Bridge / control |

---

## 6. Hypotheses, stated over the three arms

- **H1 — Physics-awareness pays off as information degrades.** The M1–M3 gap is small at low noise with the correct kernel and widens with noise, texture complexity, and kernel mismatch. (Mechanism: M3's data module still uses the true $k$, so it degrades gracefully; M1 has nothing to fall back on.)
- **H2 — Prior depth matters most on texture.** On piecewise-constant content M2 is competitive; on natural texture the deep-prior arms (M1, M3) win. The $K$-sweep in M2 tests the same claim from the classical side.
- **H3 — A-awareness buys stability.** M1 is unstable under tiny perturbations while the physics-aware arms resist them (Antun et al., PNAS 2020; to be tested in the convolution regime). This is the study's headline result.

---

## 7. Protocol invariants (must be identical across arms)

| Item | Setting |
|---|---|
| Data | BSDS500 standard splits (200 train / 100 val / 200 test); 128×128 patches |
| PSF library | Gaussian $\sigma_k \in \{0.7, 1.2, 1.6, 2.0\}$; Levin motion kernels; defocus disks |
| Noise | $\sigma \in \{0, 1, 5, 15, 25\}/255$ |
| Boundary | Circular / FFT, stated explicitly |
| Normalization | Identical across methods; **never** inside the forward model (noise is absolute) |
| Metrics | PSNR, SSIM, LPIPS; data-consistency residual $\|A\hat{x}-y\|_2$; kernel-mismatch PSNR drop; PGD stability |
| Statistics | Paired Wilcoxon signed-rank; pre-register 1–2 primary endpoints; Holm–Bonferroni for the rest |
| Reproducibility | Fixed seeds; released protocol |

---

## 8. Diagnosis → prescription (the follow-up)

The comparison is designed to produce a *failure signature*, not just a leaderboard:

| Arm | Diagnosed weakness | What a follow-up method would need to fix |
|---|---|---|
| M1 | Kernel mismatch, hallucination, instability | Restore data-consistency without giving up the deep prior |
| M2 | Staircase/ringing on texture, brittleness to wrong $k$ | A deeper (or multimodal) prior while keeping the solver's physics |
| M3 | Cost, residual sensitivity to unseen kernels | Cheaper/adaptive physics module; better prior routing |

**Candidate follow-up (future work, not in this study's core):** a physics-aware network whose prior is a *mixture* — e.g. patch-GMM-routed experts, or a GSM-scale-driven spatially varying regularization strength — targeting the exact failure mode the three-arm comparison identifies.

---

## 9. Decisions and open risks

- [ ] **USRNet retraining for deblurring ($s{=}1$)** — official weights are SISR; plan fine-tuning, not scratch training.
- [ ] **LISTA adaptation** to convolutional $A$ — allocate implementation time.
- [ ] **Compute discipline** — do not run all algorithms on the full grid. Core grid: one representative per arm (NAFNet, FISTA+EPLL, USRNet). Genealogy rungs (DeepDeblur, LISTA, ISTA-Net) on a reduced grid only.
- [ ] **Headline M1 = NAFNet**, not DeepDeblur (2017); use DeepDeblur only as the historical rung.
- [ ] **K-SVD is optional** — drop first if the schedule slips.
- [ ] **Multiplicity** — dozens of Wilcoxon tests will manufacture false positives without correction.
- [ ] **Blind vs non-blind scope** — Tier 1 gives true $k$ to M2/M3; GoPro (Tier 2) is blind, so either add kernel estimation or label it an out-of-protocol stress test.

---

## 10. References

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

Status: draft — design note for the revised proposal. The 3-page proposal keeps the summary; this note is the full method record.
