---
title: Project Idea — Deblurring as Inverse Function Learning
date: 2026-08-30
tags:
  - project
  - smai
  - deblurring
  - ideation
  - active
aliases:
  - Project Idea
  - Deblurring Ideation
status: in-progress
---

# Project Idea — Deblurring as Inverse Function Learning

> [!abstract] The idea in one line
> Treat deblurring as the task of **learning the inverse function of a known blur operator** ($g_\theta(y) \approx A^{-1}y$), and run a controlled comparison of **classical ML vs modern deep ML** — with ~70% of the work on the modern side (deep unrolling + direct CNNs + stability analysis) and ~30% on classical baselines (TV, EPLL, dictionary priors).

---

## 1. The problem

Sharp image $x$ → blur → observation:

$$
y = Ax + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

$A$ = convolution with a PSF kernel $k$ (Gaussian = defocus, line/trajectory = motion blur). Convolution is a linear operator whose eigenvalues decay to ~0 at high frequencies, so $A$ is **ill-conditioned**: naive inversion amplifies noise into garbage. Information (high-frequency content) is genuinely destroyed — black+white pixels blur to gray with no way back.

**The ML formulation**: learn $g_\theta: \mathcal{Y} \to \mathcal{X}$ minimizing $\mathbb{E}\lVert g_\theta(y) - x\rVert^2$ over $(x, y)$ pairs from the known generative model. Every method below is a different answer to that one supervised regression question.

---

## 2. The theme: classical ML vs modern ML (70/30)

The methods sort along one axis — **how much the model knows about $A$** — which makes a perfect SMAI narrative:

| # | Family | Prior | Knows $A$? | Modern share |
|---|---|---|---|---|
| 1 | **Handcrafted** — TV/Tikhonov, Wiener, Richardson–Lucy | Fixed math ($\|\nabla x\|_1$) | Yes | baseline |
| 2 | **Classical ML** — FoE, EPLL/GMM, K-SVD dictionaries | *Learned shallow* prior from data | Yes | baseline |
| 3 | **Deep unrolling, A-aware** — LISTA → ISTA-Net → USRNet | Learned prior inside a truncated optimization algorithm; $A$ stays in the loop (FFT data module, closed-form) | Yes (kernel + noise fed as inputs) | **core modern** |
| 4 | **Deep direct, A-agnostic** — DeepDeblur, DeblurGAN, NAFNet | Pure CNN regression; PSF only implicit in data | No | **core modern** |
| 5 | **Zero-shot** — Deep Image Prior | Architecture is the prior, zero training data | Yes | discussion |

**The story**: Family 1 = "the math knows $A$ but the prior is dumb" (staircase artifacts). Family 2 = "the prior gets learned, still shallow". Families 3–4 = "the prior and the inverse get learned together" (the 70%). Family 5 = "a neural network *without* any learning" — a striking control.

**Why unrolling is the centerpiece**: you take a classical iteration (ISTA / half-quadratic splitting), unroll ~8–10 steps into a network, keep $A$ as a differentiable FFT module, and replace the hand-crafted proximal operator with a learned CNN. The network learns an *implicit data-driven prior* ("hair textures, smooth gradients") instead of TV's "images are cartoons". This is the intellectual heart of the project and where most of the implementation effort goes.

---

## 3. Possible approaches / solution stack

**Classical (30%)**
- Wiener filter, Richardson–Lucy (trivial, sanity baselines)
- TV-ADMM / FISTA (the flagship classical baseline; $\lambda$ tuned per noise)
- EPLL — 200-component GMM patch prior learned with EM (strongest classical-ML prior; outperformed all generic priors on deblurring in 2011)
- K-SVD dictionary + sparse deblurring (classical-ML route via sparse coding)

**Modern (70%)**
- LISTA (unrolling origin — learned matrices + shrinkage)
- ISTA-Net/ISTA-Net+ (learned CNN proximal: $F(x) = B\,\mathrm{ReLU}(Ax)$)
- USRNet (flagship: data module = closed-form FFT, prior module = ResUNet denoiser, hyper-param module = MLP; single model handles any kernel + noise because kernel and $\sigma$ are *inputs*)
- Direct CNN (small UNet regression) — the A-agnostic contrast
- Deep Image Prior (zero-shot control)
- **Stability test** (Antun et al. 2020 protocol) — perturbation-based evaluation, not just PSNR

**Tiering** (how the project scales):
- **Tier 1 (core)**: non-blind — all families on synthetic data (BSDS300/500, PSF library: Gaussian / motion / defocus × noise $\sigma \in \{0,1,5,15,25\}$). Controlled; every difference attributable to one variable.
- **Tier 2 (stretch)**: blind — GoPro dataset (3214 real blurry/sharp pairs, no kernels released). Classical kernel estimation ($\mathrm{MAP}_k$, never joint $\mathrm{MAP}_{x,k}$ — it collapses to the delta kernel) + non-blind restore; modern: estimated kernel → USRNet, or kernel-free DeepDeblur/NAFNet.
- **Tier 3**: unknown/real-world degradation — future work only.

**Metrics**: PSNR, SSIM, LPIPS; data-consistency residual $\|A\hat{x} - y\|$; Levin kernel-error ratio (blind tier); paired Wilcoxon tests across test images.

---

## 4. Key references (verified against primary sources, 2026-08-30)

**Classical + classical-ML priors**
- Rudin, Osher, Fatemi 1992 — TV origin. 📚
- Beck & Teboulle 2009 (FISTA) — solver for TV deblurring. 📚
- Jalalzai 2016, arXiv:1402.0091 — **proof** that TV staircasing always occurs (even noise-free, worse with noise). ✅
- Chan, Marquina, Mulet 2000 — staircase effect documented. ✅
- Roth & Black 2005/2009 — Fields of Experts, learned MRF filters. ✅
- Zoran & Weiss 2011 (EPLL) — GMM patch prior, SOTA generic prior on deblurring at the time. ✅ (PyTorch: github.com/icbcbicc/PyTorch_EPLL)
- Aharon, Elad, Bruckstein 2006 (K-SVD) + Zhang et al. 2011 (sparse blind deblurring). ✅

**Modern**
- Gregor & LeCun 2010 (LISTA, ICML) — unrolling origin; **~20× fewer iterations than FISTA at equal error** (18–35 FISTA iterations ≈ 1 LISTA layer). ✅
- Zhang & Ghanem 2018 (ISTA-Net, CVPR) — CNN-in-the-proximal design. ✅ (official PyTorch code)
- Zhang, Van Gool, Timofte 2020 (USRNet, CVPR) — kernel+noise as inputs, closed-form FFT data module, 8 iterations. ✅ (github.com/cszn/USRNet)
- Zhang et al. 2017 (IRCNN, CVPR) — CNN denoiser as prior; direct CNNs **collapse on kernel mismatch**. ✅
- Nah et al. 2017 (DeepDeblur + GoPro dataset, CVPR) — 3214 pairs (2103/1111), kernel-free real blur. ✅
- Chen et al. 2022 (NAFNet, ECCV) — SOTA 33.69 dB on GoPro. ✅
- Kupyn et al. 2018 (DeblurGAN). 📚
- Ulyanov et al. 2018 (Deep Image Prior, CVPR) — untrained net as prior. ✅ (github.com/DmitryUlyanov/deep-image-prior)
- **Antun, Renna, Poon, Adcock, Hansen 2020 (PNAS)** — deep reconstruction is unstable in-domain: tiny invisible perturbations → severe artifacts; classical solvers unaffected. ✅ — the scientific backbone of the "gotcha" section.

**Blind (Tier 2)**
- Levin, Weiss, Durand, Freeman 2009/2011 — naive joint MAP fails (prefers delta kernel); benchmark (32 images); error-ratio metric (>2 = visually implausible). ✅
- Fergus et al. 2006 (variational Bayes blind deconvolution) — classical blind baseline. 📚

> [!todo] Verify before final report
> Restormer (Zamir et al., CVPR 2022); Schuler et al., "Learning to Deblur" (TPAMI 2016).

---

## 5. The gap — where the novelty targets

Existing literature is fragmented: classical papers benchmark against classical; unrolling papers benchmark within their own family; SOTA papers (NAFNet/Restormer) chase PSNR tables and don't run classical baselines at all. **No single study runs all five families on one controlled degradation protocol.** That is the opening:

1. **Systematic cross-family comparison along the $A$-awareness axis** (the organizing contribution). Same PSF library, same noise grid, same boundary conditions, same test set, statistical tests — produces the first clean "when is math enough, when is learning needed" map. Benchmark-style novelty (the spirit of NAFNet's "Simple Baselines").

2. **Kernel-mismatch robustness curves** (new quantitative result). IRCNN showed SRCNN/VDSR break on PSF shift, but nobody has published the systematic curve: train on PSF set $P_1$, sweep test PSFs $P_2$, plot PSNR drop for direct CNN vs USRNet vs TV. Predict: TV and USRNet (both $A$-aware) degrade gracefully, direct CNN drops hard. This is a clean, publishable figure.

3. **In-domain instability for deblurring** (transfer of Antun et al. to the convolution regime). Their tests were MRI/CT (subsampled Fourier/Radon). Deblurring's $A$ is a *different regime* — full-rank, ill-conditioned — and data-consistency is cheap to enforce, so the question "are unrolled $A$-aware nets more stable than direct CNNs?" is genuinely open. A small perturbation experiment (projected gradient ascent, ~50 lines) that shows direct CNNs hallucinate artifacts while unrolled nets resist them (because $A$ is baked in) would be the strongest result in the report.

4. **Component-wise ablation of the unrolling genealogy** (interpretability contribution). Nobody has isolated exactly what each learned component buys: scalar shrinkage (LISTA) → learned transform (ISTA-Net) → full denoiser + hyper-params (USRNet). A clean ablation turns the "history of the field" into one training run per rung — great for both the report and the 70%-modern emphasis.

**Positioning**: we are *not* trying to beat NAFNet's PSNR (that's a compute war). We claim: the first controlled cross-family study for deblurring, plus two new quantitative results (kernel-mismatch curves, deblurring-specific stability), plus a unified unrolling ablation. Classical ML (TV, EPLL, dictionaries) provides the baselines; modern ML (unrolling, direct CNNs, stability analysis) carries the results — the required 70/30 split.

---

## 6. Corrections that keep the idea accurate (learned in the research pass)

- **LISTA does not contain a CNN** — it learns matrices + scalar shrinkage. The "CNN in the thresholding step" design is ISTA-Net/USRNet. Cite LISTA for *unrolling*, ISTA-Net/USRNet for *learned CNN priors*. (The project's centerpiece should therefore be ISTA-Net/USRNet, with LISTA as the historical rung.)
- **Speedup is ~20×, not "100×"** (LISTA's own measurement), and unrolled nets are feed-forward approximations of the fixed point, not "converged iterations" (ISTA-Net: 10 phases, USRNet: 8).
- **The generalization gotcha is stronger than "faces→cars"**: instability is provable *in-domain* (PNAS 2020). Keep the OOD demo for the talk; make the perturbation test the scientific one.
- **The TV↔NN gap is governed by noise × kernel correctness together**, not noise alone (IRCNN evidence).
- State boundary handling (circular/Fourier) — it changes results.

---

## 7. Presentation story (for later)

Blurry image → TV result (sharp edges, staircase stripes) → EPLL (smoother, still blocky) → unrolled USRNet (crisp textures, kernel fed as input) → ground truth. Then the twist: feed the trained direct CNN a slightly perturbed input — artifacts appear; feed USRNet — it resists, because the math ($A$) is still inside it. *"The learned inverse is powerful; the question is what keeps it honest."*

---
Status: ideation — next step is a spec (repo skeleton + Phase 1 protocol) if the idea is accepted.