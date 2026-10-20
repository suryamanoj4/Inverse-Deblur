---
title: Project Overview — Inverse Function Learning for Image Deblurring
date: 2026-08-30
updated: 2026-10-20
tags:
  - project
  - deblurring
  - inverse-problems
aliases:
  - Deblurring Study
status: active
---

# Project Overview — Inverse Function Learning for Image Deblurring

> [!abstract] Summary
> Image deblurring is an ill-posed inverse problem. The project studies it as **inverse function learning**: learn $g_\theta$ minimizing $\mathbb{E}\lVert g_\theta(y)-x\rVert^2$ over pairs from the known degradation model $y = Ax+\varepsilon$. The central question is whether $g_\theta$ should be built around the physics of $A$ or learned purely from data. The answer comes from a controlled three-arm comparison under one protocol, plus two measurements the literature lacks: kernel-mismatch robustness curves and a cross-family stability analysis.

## 1. Problem statement

A sharp image $x$ is observed through blur and noise:

$$
y = Ax + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,\sigma^2),
$$

where $A$ is convolution with a point-spread function (PSF) $k$ and $\varepsilon$ is sensor noise. Convolution is ill-conditioned: its spectrum decays to near zero at high frequencies, so high-frequency content is destroyed and naive inversion amplifies noise. Reconstruction therefore requires a prior, and the choice of prior is the main design decision in deblurring.

## 2. Research question

The literature contains four families: classical solvers with hand-crafted priors (TV), classical-ML priors learned from patches (EPLL/GMM, K-SVD), unrolled networks that keep $A$ in the loop (LISTA, ISTA-Net, USRNet), and direct data-only networks (DeepDeblur, NAFNet). They differ along two independent axes:

| Axis | Question |
|---|---|
| **Physics-awareness** | Does the method consume $k$ and $\sigma$ and enforce $Ax \approx y$? |
| **Prior depth** | Fixed, shallow learned, or deep learned? |

Existing results are reported under different kernels, noise levels and evaluation protocols, so the families are not directly comparable. **The project asks: under one controlled protocol, when does physics-awareness matter, and when does prior depth matter?**

## 3. Approach

Three arms, one protocol:

| Arm | Family | Physics | Representative methods |
|---|---|---|---|
| **M1** | Modern, data-only | No $k$, no $\sigma$ | NAFNet (primary), DeepDeblur (historical) |
| **M2** | Classical-ML | True $k$, $\sigma$ | FISTA/ADMM + EPLL-GMM (primary); K-SVD optional |
| **M3** | Unrolled control | True $k$, $\sigma$ | USRNet (primary); LISTA → ISTA-Net ladder |

M3 is the control: it is physics-aware like M2 and deep like M1, so the three-arm design separates the two axes instead of confounding them. The full specification — objectives, solver/prior matrix, and implementation order — is in [[Proposed Approach]].

The primary M1 is **one model** trained on $P_1$ (4 Gaussian + 8 motion kernels) across all five noise levels; it never sees $k$ or $\sigma$. Kernel mismatch is evaluated on $P_2$-interp (unseen parameters of the same families) and $P_2$-extrap (defocus disks, an unseen family). Per-noise specialists are an ablation only.

## 4. Contributions

1. A controlled cross-family comparison of inverse-function learners for deblurring under one degradation protocol (shared PSF library, noise grid, boundary handling, splits and metrics).
2. Quantitative kernel-mismatch robustness curves: train on PSF set $P_1$, evaluate on $P_2$, and measure the PSNR drop for data-only versus physics-aware methods.
3. A cross-family stability analysis for deblurring, transferring the Antun et al. (PNAS 2020) perturbation protocol to the convolution regime.
4. An unrolling-genealogy ablation (LISTA → ISTA-Net → USRNet) that isolates what each learned component contributes.

## 5. Scope

**In scope**

- Non-blind deblurring on synthetically degraded BSDS500 (Tier 1).
- The three arms above, including the $K$-component GMM sweep and the unrolling ladder.
- Kernel-mismatch robustness and perturbation stability.

**Stretch**

- GoPro blind deblurring (Tier 2). Without a kernel-estimation stage this is an out-of-protocol stress test, not part of the controlled comparison.

**Out of scope**

- Chasing state-of-the-art PSNR on real-world benchmarks.
- Unknown real-world degradations (Tier 3).
- A new architecture in the core study; any follow-up method is motivated by the comparison and reported separately.

## 6. Evaluation

Tier 1 uses BSDS500 with the locked PSF sets: $P_1$ (4 Gaussian + 8 motion kernels) for training and in-distribution evaluation, $P_2$-interp (unseen parameters of the same families) and $P_2$-extrap (defocus disks, an unseen family) for kernel mismatch. Noise grid $\sigma \in \{0,1,5,15,25\}/255$, circular boundary handling, luminance images. Metrics: PSNR, SSIM, LPIPS, data-consistency residual, kernel-mismatch PSNR drop, and perturbation stability; paired Wilcoxon tests with multiplicity control. Checkpoint selection uses mean val PSNR over the fixed grid on a 10-image seeded subset of the sorted val list (IDs logged with the protocol); never per-condition selection. The locked protocol is in [[Proposed Approach]] §7.

## 7. Deliverables

- A reproducible degradation and evaluation protocol.
- The three-arm results table, kernel-mismatch curves, and stability figure.
- Source code for every method in the comparison.
- A written report describing the protocol, results, and the failure signature that motivates future work.

## 8. Key references

- Rudin, Osher, Fatemi (1992). Nonlinear total variation based noise removal.
- Beck, Teboulle (2009). Fast gradient-based algorithms for TV denoising and deblurring.
- Zoran, Weiss (2011). EPLL: learning patch models for whole-image restoration.
- Aharon, Elad, Bruckstein (2006). K-SVD dictionary learning.
- Gregor, LeCun (2010). LISTA: learning fast approximations of sparse coding.
- Zhang, Ghanem (2018). ISTA-Net.
- Zhang, Van Gool, Timofte (2020). USRNet.
- Nah, Kim, Lee (2017). DeepDeblur and the GoPro dataset.
- Chen et al. (2022). NAFNet.
- Antun et al. (2020). Instabilities of deep learning in image reconstruction.
- Levin et al. (2009). Blind deconvolution benchmark and motion kernels.
- Arbeláez et al. (2011). BSDS500.
- Zhang et al. (2018). SRMD: multiple-degradation super-resolution.
- Gandikota et al. (2022). Adversarial robustness of deep image deblurring.

Full bibliography: [[Proposed Approach]] §12.

## 9. Status

Active. The shared foundation (FFT operator, PSF library, degradation pipeline, splits, metrics) is implemented and tested. Next: the classical arm (Wiener, Richardson–Lucy, FISTA-TV, ADMM-EPLL).
