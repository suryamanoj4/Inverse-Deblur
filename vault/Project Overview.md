---
title: Project Overview — Inverse Function Learning for Image Deblurring
date: 2026-08-30
updated: 2026-11-28
tags:
  - project
  - deblurring
  - inverse-problems
aliases:
  - Deblurring Study
status: complete
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

**Phase 2 (after the mid-evaluation; done, see §9)** — as planned below, except that the synthetic pipeline was not used (the go/no-go gate closed), real recorded kernels became a separate set $P_2$-real, and the white-box H3 attack on M2 (2C-7) was only a small run that is not in the reported results.

- A synthetic training-image pipeline (pretrained latent diffusion, ~2,000 images after quality and leakage checks), used identically by every learned component of every arm; real val/test unchanged.
- Protocol v2: seven kernel families instead of three, including random-trajectory camera shake and real recorded kernels; 8-bit quantized observations.
- Training and evaluating M3.
- A frozen-v1 coverage ablation (2C-5): v2-trained NAFNet on the v1 $P_2$ sets, which tests whether kernel coverage alone fixes the original mismatch (endpoint E2b).
- The GMM $K$-sweep and per-σ specialists on v2 (2C-6), and a white-box H3 attack on M2 (2C-7), so the prior-depth and stability conclusions are not tied to v1 or to a transfer attack.
- Details in [[Proposed Approach]] §12.

**Phase 3 (after Phase 2; done, run as a placement ladder, see §9)** — the plan as written before Phase 2:

- **K-NAFNet:** the M1 NAFNet backbone conditioned on the true kernel and σ through a kernel encoder and FiLM, plus one closed-form FFT data-consistency step.
- Motivated directly by the diagnosis: M1's deep prior is kernel-blind, M2's physics is paired with a shallow prior, and M3 pays for unrolling.
- Evaluated against M1/M2/M3 on the v2 grid, with ablations for kernel concatenation vs FiLM, σ conditioning, one DC step vs unrolled steps, kernel jitter, and wrong-kernel/zero-condition mechanism checks.
- Decision gate: if 2C-5 shows that dense training already closes the gap, the cheaper fix is preferred; otherwise K-NAFNet is the main proposal. Specification in [[Proposed Approach]] §12.9.

**Stretch**

- GoPro blind deblurring (Tier 2; not run). Without a kernel-estimation stage this is an out-of-protocol stress test, not part of the controlled comparison.

**Out of scope**

- Chasing state-of-the-art PSNR on real-world benchmarks.
- Unknown real-world degradations (Tier 3).
- A new architecture in the core study; any follow-up method is motivated by the comparison and reported separately.

**Compute scale**

The core comparison trains three artifacts (NAFNet, the EPLL prior, USRNet); the full study including the $K$-sweep and the per-σ ablation trains about fifteen. The inventory and checkpoint layout are in [[Proposed Approach]] §5.1.

## 6. Evaluation

Tier 1 uses BSDS500 with the locked PSF sets: $P_1$ (4 Gaussian + 8 motion kernels) for training and in-distribution evaluation, $P_2$-interp (unseen parameters of the same families) and $P_2$-extrap (defocus disks, an unseen family) for kernel mismatch. Noise grid $\sigma \in \{0,1,5,15,25\}/255$, circular boundary handling, luminance images. Metrics: PSNR, SSIM, LPIPS, data-consistency residual, kernel-mismatch PSNR drop, and perturbation stability; paired Wilcoxon tests with multiplicity control. Checkpoint selection uses mean val PSNR over the fixed grid on a 10-image seeded subset of the sorted val list (IDs logged with the protocol); never per-condition selection. The locked protocol is in [[Proposed Approach]] §7.

Phase 2 evaluates the same metrics on protocol v2, adds two hypotheses (H4: more data helps the deep arms in distribution but not under mismatch; H5: more training kernel families narrow M1's gap on unseen parameters but not on unseen families) and four pre-registered endpoints (E1, E2, E2b, E3; [[Proposed Approach]] §12.4). 2C-5 tests the v2-trained NAFNet on the frozen v1 $P_2$ sets; 2C-7 replaces the transfer-only H3 with a per-method white-box attack.

## 7. Deliverables

- A reproducible degradation and evaluation protocol.
- The three-arm results table, kernel-mismatch curves, and stability figure.
- Source code for every method in the comparison.
- A written report describing the protocol, results, and the failure signature that motivates future work.
- Phase 2: the v2 kernel library, frozen with the protocol. (The synthetic training set `syn-v1` was not generated: the go/no-go gate closed.)
- Phase 3: a pre-registered kernel-placement ladder (no kernel, input maps, FiLM, one data step with and without FiLM, 4-stage unrolled) on NAFNet and U-Net, a kernel-swap probe, spectral-loss and linear-filter analyses, and a 2×2 mismatch-aware training ablation, showing which ingredient buys mismatch robustness.
- Final deliverables: `report/final_report.pdf` and `report/final_presentation.pptx`.

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

Full bibliography: [[Proposed Approach]] §13.

## 9. Status

**Complete (November 2026).** All three phases have been run. Final results: `report/final_report.pdf` and `report/final_presentation.pptx`; the Phase 1 report is `report/study_report.md`.

**Phase 1 (mid-evaluation, protocol v1, M1 vs. M2).** NAFNet and ADMM-EPLL are tied on the training kernels (27.48 vs. 27.43 dB); on unseen kernels NAFNet loses half its gain and ADMM-EPLL wins every cell, by up to 6.4 dB. Kernel mismatch is supported; the noise and texture parts of H1 and all of H2 are not; H3 holds at σ ≥ 5 under a transfer attack. NAFNet does not memorise its training images but does overfit the training kernels (`report/nafnet_overfitting.pdf`).

**Phase 2 (reviewer feedback).** NAFNet is not data-hungry: with 50, 100 and 200 training images $P_1$ PSNR spreads over 0.08 dB and the unseen-kernel drop stays, so the synthetic-data gate closed and `syn-v1` was not generated. M3 was fixed and trained: USRNet is +0.27 dB over ADMM-EPLL on v1 $P_1$ with the same small $P_2$ drop (−0.43 vs. −0.41 dB; NAFNet −2.76). Protocol v2 (four training families; defocus, ring and ghost as unseen families) and real blur (Levin real captures, Sun et al.) were added. On v2 USRNet (3 seeds) reaches 27.16 / 27.57 dB on $P_1$ / unseen families; on provided real blur EPLL beats the standard USRNet. Run record: [[Phase 2 Run Checklist]].

**Phase 3 (where the kernel enters a network).** Pre-registered in [[Phase 3 Pre-registration]], with the outcome of each prediction. On unseen families (3 seeds): no kernel 21.97 ± 0.18 dB, input maps 20.78, FiLM 17.23, one data step without FiLM 25.05 ± 0.05, K-NAFNet (FiLM + one data step) 23.45 ± 0.41, 4-stage unrolled 27.14 ± 0.38, USRNet 27.57 ± 0.06. A kernel-swap probe shows the conditioned nets use the kernel as a lookup key; through a data step it acts as a forward model. Training USRNet with realistic boundaries fixes Levin's largest kernels. The K-NAFNet proposal of [[Proposed Approach]] §12.9 was run as one rung of this ladder; its pre-registered target failed.
