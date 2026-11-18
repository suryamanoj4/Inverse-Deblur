# Diffusion and generative posterior-sampling solvers for non-blind deblurring (2021–2026), compared with DPIR/USRNet and supervised networks

Scope note: about 15 tool calls. Numbers marked **[verified]** were read from a fetched primary paper page (arXiv HTML or ar5iv). Statements marked **[unverified / prior knowledge]** come from the researcher's background knowledge and were not re-fetched in this session; check them before quoting. No arXiv IDs are given for papers whose IDs could not be confirmed.

### Standard benchmark protocol (the "DPS protocol")
Almost every zero-shot diffusion solver from 2022 to 2025 reuses one setup: FFHQ 256x256 and ImageNet 256x256, 100 validation images (DPS itself used 1k), a **61x61 Gaussian kernel with std 3.0**, **one 61x61 random motion kernel with intensity 0.5** (Chung et al. generator), and additive noise **σ_y = 0.05** on images in [0,1]. DiffPIR states that "the same [motion] kernel is used across all experiments." — [DiffPIR, ar5iv](https://ar5iv.labs.arxiv.org/html/2305.08995); [DAPS, arXiv HTML](https://arxiv.org/html/2407.01521) **[verified]**

Two consequences follow. (a) The motion "family" is effectively a single fixed kernel, so these benchmarks do not test generalisation across kernels. (b) σ=0.05 is heavy noise, which favours strong generative priors on perceptual metrics.

---

## Q1. Do diffusion solvers beat DPIR/USRNet on PSNR, or only on perceptual metrics?

### Takeaway
Under the DiffPIR paper's own comparison (FFHQ/ImageNet, σ=0.05), **DPIR beats DiffPIR on Gaussian-deblur PSNR by about 0.4–1.1 dB**. DiffPIR beats DPIR on motion-deblur PSNR by only 0.16–0.41 dB, while it wins by a large margin on FID and LPIPS (FID roughly halved, LPIPS about 0.45 → 0.24). Newer samplers (DAPS, about 2024) report about 29.2–29.7 dB on FFHQ. That is roughly 1.4–3.3 dB above DPIR's 27.79/26.41 dB as reported by DiffPIR, but the two numbers come from different papers. **No source in this session found USRNet evaluated under the diffusion protocol.**

### Cited Findings
**DiffPIR (Zhu, Zhang, Liang, Cao, Wen, Timofte, Van Gool; CVPR-W NTIRE 2023), Table 1, σ_n=0.05, 100 images.** Values are PSNR / FID / LPIPS. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.08995); [CVF](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.pdf); [GitHub](https://github.com/yuanzhi-zhu/DiffPIR) **[verified]**

| FFHQ | NFEs | Gauss PSNR | Gauss FID | Gauss LPIPS | Motion PSNR | Motion FID | Motion LPIPS |
|---|---|---|---|---|---|---|---|
| DiffPIR | 100 | 27.36 | 59.65 | 0.236 | 26.57 | 65.78 | 0.255 |
| DPS | 1000 | 25.46 | 65.57 | 0.247 | 23.31 | 73.31 | 0.289 |
| DDRM | 20 | 25.93 | 101.89 | 0.298 | – | – | – |
| DPIR | ">>20" | **27.79** | 123.99 | 0.450 | 26.41 | 146.44 | 0.467 |

| ImageNet | NFEs | Gauss PSNR | Gauss FID | Gauss LPIPS | Motion PSNR | Motion FID | Motion LPIPS |
|---|---|---|---|---|---|---|---|
| DiffPIR | 100 | 22.80 | 93.36 | 0.355 | 24.01 | 124.63 | 0.366 |
| DPS | 1000 | 19.58 | 138.80 | 0.434 | 17.75 | 184.45 | 0.491 |
| DDRM | 20 | 22.33 | 160.73 | 0.427 | – | – | – |
| DPIR | ">>20" | **23.86** | 189.92 | 0.476 | 23.60 | 210.31 | 0.489 |

- DiffPIR leaves DDRM out of motion deblurring "because it supports only separable kernels." — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.08995) **[verified]**
- The DiffPIR paper says DPIR "tends to generate blurry images" while DiffPIR keeps detail. It also says PSNR is log-linear in NFEs. — [DiffPIR PDF](https://arxiv.org/pdf/2305.08995) (via search snippet)
- The DiffPIR README says the authors also compared against USRNet and found the diffusion sampling framework "more systematic" for splitting the data and prior subproblems. No USRNet numbers are given. — [GitHub](https://github.com/yuanzhi-zhu/DiffPIR) (via search summary)

**DAPS (Decoupled Annealing Posterior Sampling; Zhang et al., 2024), Table 1, FFHQ/ImageNet 256, 100 images, σ=0.05, 61x61 kernels.** Values are PSNR / LPIPS. — [arXiv HTML 2407.01521](https://arxiv.org/html/2407.01521) **[verified]**

| Method | FFHQ Gauss | ImageNet Gauss | FFHQ Motion | ImageNet Motion |
|---|---|---|---|---|
| DAPS-1K | 29.19 / 0.165 | 26.15 / 0.253 | 29.66 / 0.157 | 27.86 / 0.196 |
| DPS | 25.87 / 0.219 | 20.31 / 0.397 | 24.52 / 0.246 | 18.96 / 0.423 |
| DDRM | 24.93 / 0.239 | 21.26 / 0.443 | – | – |
| DDNM | 28.20 / 0.216 | 28.06 / 0.278 | – | – |
| DCDP | 27.50 / 0.304 | – | 25.08 / 0.364 | – |
| FPS-SMC | 26.54 / 0.253 | 23.91 / 0.387 | 27.39 / 0.227 | 24.52 / 0.326 |
| DiffPIR | 27.36 / 0.236 | 22.80 / 0.355 | 26.57 / 0.255 | 24.01 / 0.366 |
| LatentDAPS | 27.93 / 0.234 | 25.05 / 0.345 | 27.00 / 0.283 | 26.83 / 0.296 |
| PSLD (latent) | 23.27 / 0.316 | 25.86 / 0.390 | 22.31 / 0.336 | 20.85 / 0.511 |
| ReSample (latent) | 26.39 / 0.255 | 25.97 / 0.254 | 27.41 / 0.198 | 26.94 / 0.227 |

- DAPS has no DPIR or USRNet row. Its DiffPIR row copies DiffPIR's own numbers exactly, so mixing values from the two tables (DAPS 29.19 vs DPIR 27.79 on FFHQ Gaussian) is plausible but not a same-run comparison. — [DAPS](https://arxiv.org/html/2407.01521) **[verified]**
- DPS numbers differ between papers (FFHQ Gaussian: 25.46 in DiffPIR vs 25.87 in DAPS). Baseline re-runs are not consistent across papers. — [DiffPIR](https://ar5iv.labs.arxiv.org/html/2305.08995); [DAPS](https://arxiv.org/html/2407.01521) **[verified]**
- Latent-diffusion solvers (PSLD, ReSample) are no better than pixel-space ones on PSNR. PSLD is the weakest on FFHQ (22–23 dB). — [DAPS Table 1](https://arxiv.org/html/2407.01521) **[verified]**
- Later work reports DiffPIR's FFHQ motion-deblur PSNR falling from 23.10 dB (σ=0.01) to 20.69 dB (σ=0.10), with no DPIR row. This setting differs from DiffPIR's own reported 26.57, which shows that numbers are not stable across reimplementations. — [arXiv 2605.12573 via search snippet](https://arxiv.org/pdf/2605.12573) **[snippet only, unverified]**

**DPS (Chung, Kim, McCann, Klasky, Ye; ICLR 2023 spotlight).** "Manifold constrained gradient" guidance with "no strict measurement consistency projection step". It handles Gaussian/Poisson noise and nonlinear (non-uniform) deblurring. — [arXiv 2209.14687](https://arxiv.org/abs/2209.14687) **[verified, abstract only]**

**RED-diff (Mardani et al., ICLR 2024; arXiv 2305.04391).** Variational: it minimises ‖y − f(μ)‖² plus a score-based regulariser with Adam over about 1000 steps. Its only deblurring experiment in the fetched text is DPS-style **nonlinear** deblurring on ImageNet. The extraction returned DPS PSNR 6.4 vs RED-diff 45.00 and LPIPS 0.73 vs 0.0012. These values look implausible (possibly misparsed) and should be checked against the PDF. Per-step time: DPS 0.23 s, ΠGDM 0.24 s, DDRM 0.1 s, RED-diff 0.05 s (inpainting table). — [arXiv HTML 2305.04391](https://arxiv.org/html/2305.04391) **[fetched; deblur numbers suspicious]**

**Flow-matching solvers (2024–2025).** PnP-Flow (Martin et al., ICLR 2025) reports CelebA deblurring at 34.51 dB / SSIM 0.940, vs PnP-GS 33.65, OT-ODE 32.63, PnP-Diff 32.49 and D-Flow 31.07. A continuous-time follow-up reports 35.13 dB. PnP-Flow is said to run 5–10x faster than other flow-matching restoration methods. No DPIR row was found. — [emergentmind summary of 2410.02423](https://www.emergentmind.com/papers/2410.02423); [arXiv 2512.04283](https://arxiv.org/pdf/2512.04283) **[secondary summaries; numbers unverified against PDF]**

**Consistency-model solvers.** CoSIGN (Zhao, Song, Shen; ECCV 2024) uses a CM prior with ControlNet soft guidance plus an optimisation step for hard constraints, at 1–2 NFEs. It was tested on LSUN Bedroom (inpainting, 4x SR, *nonlinear* deblurring, sparse-view CT), against DPS/MCG (about 1000 NFEs) and DDRM (20 NFEs). The authors note that PSNR/SSIM "strictly penalize hallucination." MACS (arXiv 2510.02208) uses Gaussian blur std 5 at σ_y=0.05 on LSUN. PnP-CM (arXiv 2509.22736) reports Gaussian deblurring at about 4 NFEs (2 NFEs "meaningful"). — [CoSIGN arXiv 2407.12676](https://arxiv.org/html/2407.12676v1); [MACS](https://arxiv.org/pdf/2510.02208); [PnP-CM](https://arxiv.org/html/2509.22736v1) **[search snippets; no PSNR numbers obtained]**

**Supervised / conditional diffusion deblurrers. These are blind: they take only the blurry image, not the kernel.**
- *Deblurring via Stochastic Refinement* (DvSR; Whang, Delbracio, Talebi, Saharia, Dimakis, Milanfar; CVPR 2022, arXiv 2112.02475). A deterministic predictor followed by a conditional diffusion refiner, for *blind* deblurring. It claims large perceptual gains, is "competitive on distortion metrics such as PSNR," and is cheaper than standard diffusion because of predict-and-refine. — [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Whang_Deblurring_via_Stochastic_Refinement_CVPR_2022_paper.html); [arXiv](https://arxiv.org/abs/2112.02475v2) **[abstract-level]**. GoPro numbers were not obtained (gap).
- *HI-Diff* (Chen et al., NeurIPS 2023 spotlight, arXiv 2305.12966). A Restormer-based regression network with a diffusion model in a compact latent prior space. Blind. GoPro 33.33 dB / 0.964 (repo), +0.25 dB over Stripformer and +0.41 dB over Restormer. RealBlur-R 41.01 and RealBlur-J 33.70 (repo). — [arXiv](https://arxiv.org/pdf/2305.12966); [GitHub](https://github.com/zhengchen1999/HI-Diff) **[search summary]**
- I2SB (Liu et al., ICML 2023) and InDI (Delbracio & Milanfar, TMLR 2023) are supervised degraded-to-clean bridge or iterative models trained per degradation. They do not take the kernel as input. **[unverified / prior knowledge]**

### Inferences
- PSNR verdict: against DPIR, diffusion solvers *as reported by the diffusion papers themselves* are roughly at parity on PSNR (DiffPIR −0.4 dB Gaussian, +0.2 dB motion on FFHQ). They win by a large margin on FID/LPIPS. Only the 2024+ samplers (DAPS) claim clear PSNR wins (+1.4 to +3.3 dB over DPIR's DiffPIR-reported numbers), and only on single-domain priors (FFHQ faces) at σ=0.05.
- On ImageNet (diverse content), DPIR still beats DiffPIR on Gaussian-deblur PSNR (23.86 vs 22.80), and DPS is about 4 dB worse. A domain-specific diffusion prior drives most of the FFHQ gains.
- None of these papers runs DPIR at low noise (σ ≤ 0.01) with the same kernels. Typical natural-image non-blind deblurring benchmarks (Levin, BSD68 with Levin kernels, σ = 2.55/255 or 7.65/255) are almost absent from the diffusion literature.

### Gaps
- No USRNet number under the diffusion protocol was found. No diffusion-solver number on BSD68 or Set12 with the Levin/Sun kernels (the USRNet/DPIR protocol) was found.
- DiffPIR noiseless (σ=0) Table 2 could not be parsed.
- ΠGDM (Song, Vahdat, Mardani, Kautz; ICLR 2023) deblurring numbers were not retrieved.
- FPS (Dou & Song, ICLR 2024) is included only via the DAPS table. Its own paper was not fetched.
- 2025–2026 flow and rectified-flow solvers (e.g., FlowDPS, FlowChef, PnP-Flow on FFHQ) had no FFHQ deblur numbers retrieved.

---

## Q2. Which solvers use an exact closed-form FFT data-consistency step (as in DiffPIR's HQS step)?

### Takeaway
**DiffPIR** is the clear, documented case. Its data subproblem for deblurring is the closed-form FFT (circular-convolution) solution x₀ = F⁻¹[(conj(Fk)·Fy + ρ_t·F z₀)/(|Fk|² + ρ_t)] with ρ_t = λσ_n²/σ̄_t², which is the same step as DPIR/USRNet. DDRM and DDNM instead use **SVD / pseudo-inverse** projections. These are exact, but DDRM's implementation is limited to separable kernels. DPS, FPS, PSLD and ReSample use gradient or optimisation-based consistency.

### Cited Findings
- DiffPIR Appendix B.2: x₀ = F⁻¹((conj(F(k))F(y) + ρ_t F(z₀)) / (conj(F(k))F(k) + ρ_t)), assuming circular convolution, with ρ_t = λσ_n²/σ̄_t². When no closed form exists, it falls back to a one-step gradient x̂₀ ≈ x₀ − (σ̄_t²/(2λσ_n²))∇‖y − H(x₀)‖². — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.08995) **[verified]**
- DPS uses gradient guidance through the network's x̂₀ estimate ("manifold constrained gradient"), with "no strict measurement consistency projection step." — [arXiv 2209.14687](https://arxiv.org/abs/2209.14687) **[verified]**
- DDRM: in practice only separable kernels are supported (per DiffPIR's exclusion of it from motion deblurring). — [ar5iv DiffPIR](https://ar5iv.labs.arxiv.org/html/2305.08995) **[verified]**. That DDRM works in the SVD spectral domain of H (Kawar et al., NeurIPS 2022, arXiv 2201.11793) is **[unverified / prior knowledge]**.
- DDNM: an SVD-based default, plus a "simplified" mode without SVD. Deblur modes are `deblur_uni`, `deblur_gauss` and `deblur_aniso`, with **no motion** mode. Noise is handled with DDNM+ (`--sigma_y`), 100 sampling steps by default, and optional "time-travel" resampling. — [DDNM GitHub](https://github.com/wyhuai/DDNM) **[verified]**. DDNM's mechanism is range-null-space decomposition: x̂ = A†y + (I − A†A)x̄₀ (Wang, Yu, Zhang; ICLR 2023, arXiv 2212.00490) **[unverified / prior knowledge]**.
- RED-diff: gradient (Adam) on ‖y − f(μ)‖² plus a score regulariser. No Jacobian of the score network is needed. — [arXiv HTML 2305.04391](https://arxiv.org/html/2305.04391) **[verified]**
- DAPS: decoupled annealing with Langevin MCMC on the measurement likelihood (100 Langevin steps per level). This is iterative gradient-based, not closed form. — [DAPS](https://arxiv.org/html/2407.01521) **[verified]**
- CoSIGN: soft guidance through a ControlNet that conditions on the measurement, plus an optimisation-based hard constraint. — [CoSIGN](https://arxiv.org/html/2407.12676v1) **[snippet]**
- ΠGDM uses a pseudo-inverse–guided vector-Jacobian product. ReSample enforces hard data consistency by optimisation in latent space, then resamples. PSLD adds a "gluing" term to latent DPS. — **[unverified / prior knowledge]**

### Inferences
- In the diffusion literature, an HQS-style closed-form FFT proximal step is mainly associated with DiffPIR. DiffPIR also shares authors (Kai Zhang, Radu Timofte) with DPIR/USRNet, which use the same step. A proposal reusing this step is methodologically close to DiffPIR, DPIR and USRNet. Novelty has to come from elsewhere, for example the learned prior, how the kernel is used to condition the network, the number of stages, or robustness.
- The FFT step assumes circular boundaries, and DiffPIR states this assumption explicitly. Boundary handling is a known weak point, shared with USRNet/DPIR.

### Gaps
- Whether DAPS, FPS or later solvers (e.g., "DiffPIR-style" 2025–2026 variants) replace their likelihood step with an FFT proximal operator for deblurring was not confirmed.

---

## Q3. Evaluation on unseen kernel families (defocus, ring, double exposure) or kernel mismatch?

### Takeaway
**No zero-shot diffusion solver paper found in this session evaluates on unseen kernel families such as disk/defocus, ring or double exposure.** The standard protocol tests one Gaussian kernel and one fixed motion kernel. Operator mismatch has been studied only recently (2025–2026), and those studies find that mismatched kernels bias the posterior. Several groups propose uncertainty-aware or blind extensions rather than family-level generalisation tests.

### Cited Findings
- The protocol uses a single 61x61 Gaussian kernel (std 3.0) and a single 61x61 motion kernel ("the same kernel is used across all experiments"). — [DiffPIR ar5iv](https://ar5iv.labs.arxiv.org/html/2305.08995) **[verified]**
- "Diffusion Posterior Sampling with Uncertain Forward Models" (arXiv 2610.29102): fixing kernel parameters at estimated values pushes their error into the reconstruction. The paper's uncertainty-aware likelihood (UAL) cuts geometric-mean MSE by 9–20% (Gaussian deblur) and by 51–92% (motion deblur, when samplers use a population-mean kernel). — [arXiv 2610.29102](https://arxiv.org/abs/2610.29102) **[search snippet]**
- arXiv 2602.00176 (Posterior Continuation with Noise-Conditioned Frequency Exposure) includes an operator-mismatch table on FFHQ motion deblurring and says a perturbed blur operator yields a biased posterior. The numbers were not retrieved. — [arXiv 2602.00176](https://arxiv.org/html/2602.00176) **[snippet]**
- Prior mismatch: with an FFHQ prior applied to AFHQ, DPS performs about the same as TV on motion deblurring and worse than TV on nonlinear deblurring. — [arXiv 2512.08022](https://arxiv.org/html/2512.08022) **[snippet]**
- Blind with a wrong kernel prior: AFHQ PSNR is 21.72 (mismatched motion-kernel prior) and 18.53 (mismatched Gaussian prior). — [arXiv 2503.21501](https://arxiv.org/pdf/2503.21501) **[snippet]**
- Blind diffusion that sidesteps fixed kernels: Kernel-Diff (ECCV 2024, arXiv 2312.02319) diffuses over kernels conditioned on y and is guided by a non-blind solver. Fast Diffusion EM (Laroche et al., WACV 2024) uses EM with a non-blind diffusion step. — [Kernel-Diff](https://arxiv.org/pdf/2312.02319); [FastEM WACV](https://openaccess.thecvf.com/content/WACV2024/papers/Laroche_Fast_Diffusion_EM_A_Diffusion_Model_for_Blind_Inverse_Problems_WACV_2024_paper.pdf) **[snippet]**
- Deep Equilibrium Diffusion Restoration evaluates zero-shot on Gaussian and anisotropic kernels, not disk kernels. — [arXiv 2311.11600](https://arxiv.org/pdf/2311.11600) **[snippet]**

### Inferences
- Because zero-shot solvers receive the kernel at test time through the data term, they are in principle *family-agnostic*: the prior never sees the kernel. That is a structural argument against the "unseen family" failure mode of kernel-conditioned trained networks. It has not been demonstrated on defocus, ring or double-exposure kernels in the sources found, so an experiment doing this would fill a gap.
- Kernel mismatch (wrong or perturbed kernel) is an active 2025–2026 topic, but it is evaluated with perturbations of Gaussian/motion kernels, not with cross-family shifts.

### Gaps
- No source found evaluates diffusion solvers vs DPIR/USRNet on disk/defocus, ring, or double-exposure kernels.
- No source found reports the robustness of a supervised conditional diffusion deblurrer to kernel-family shift.

---

## Q4. Inference cost relative to one-pass networks and 8-stage unrolled nets; kernel-conditioned diffusion

### Takeaway
Zero-shot diffusion solvers cost about **20–1000 NFEs** of a large U-Net (DDRM 20, DiffPIR 100, DPS 1000, DAPS-1K about 1000 score evaluations plus Langevin steps). Consistency and flow solvers reduce this to **1–4 NFEs**. USRNet runs 8 unrolled stages, each with a small ResUNet and an FFT step. That is about 8 small-network evaluations in one feed-forward pass, so it is orders of magnitude cheaper, though no source here measures this directly. **No diffusion network found in this session conditions the denoiser on the blur kernel itself for non-blind deblurring.** Kernel-conditioned diffusion exists only in blind settings, where the kernel is the diffused variable (Kernel-Diff).

### Cited Findings
- NFEs reported: DiffPIR 100, DPS 1000, DDRM 20, DPIR ">>20" (per DiffPIR's table). — [ar5iv DiffPIR](https://ar5iv.labs.arxiv.org/html/2305.08995) **[verified]**
- DAPS-1K uses 4 ODE NFEs × 250 annealing levels (about 1000 NFEs) and DAPS-4K uses 10 × 400. Each level adds 100 Langevin steps (cheap, with no network call). Wall-clock for 10 FFHQ images on Gaussian deblur: 64.7 s (Gaussian approximation) vs 462.3 s (score estimation), on an A100 80GB. — [DAPS](https://arxiv.org/html/2407.01521) **[verified]**
- Per-step time and max batch size: DPS 0.23 s / 15, ΠGDM 0.24 s / 15, DDRM 0.1 s / 25, RED-diff 0.05 s / 30. DPS uses 1000 steps and ΠGDM 100. — [RED-diff](https://arxiv.org/html/2305.04391) **[verified]**
- CoSIGN: 1–2 NFEs. PnP-CM: about 4 NFEs. — [CoSIGN](https://arxiv.org/html/2407.12676v1); [PnP-CM](https://arxiv.org/html/2509.22736v1) **[snippet]**
- DvSR's predict-and-refine design makes sampling much cheaper than a standard diffusion model. — [arXiv 2112.02475](https://arxiv.org/abs/2112.02475v2) **[abstract]**
- Kernel-conditioned diffusion: Kernel-Diff models ε_θ(k_t, y, t), i.e. a diffusion over *kernels* conditioned on the blurry image, for blind deconvolution. — [arXiv 2312.02319](https://arxiv.org/pdf/2312.02319) **[snippet]**. A targeted search ("kernel-conditioned diffusion model non-blind deblurring") found no non-blind diffusion network that takes k as a conditioning input.

### Inferences
- At 100 NFEs of a 256x256 ADM U-Net (DiffPIR) vs USRNet's 8 small ResUNet stages, a one-pass or 8-stage unrolled net is likely 1–3 orders of magnitude faster per image. This is an estimate, not a measured ratio.
- The space between "trained network that takes k explicitly" (USRNet takes k, σ and the scale factor as inputs) and "diffusion prior that sees k only through the data step" (DiffPIR) looks unoccupied for *generative* models. Physics-conditioned or kernel-conditioned diffusion denoisers for non-blind deblurring appear not to exist as published work, at least in this search. Treat this as a tentative novelty signal and confirm it with a targeted check (e.g., Google Scholar "kernel-aware diffusion", "degradation-conditioned diffusion deconvolution").

### Gaps
- No direct wall-clock comparison of DiffPIR/DPS vs USRNet/DPIR was found.
- 2025–2026 one-step distilled solvers for deblurring (e.g., distilled DPS variants, rectified-flow solvers) were not covered in depth.
- Surveys found: Daras et al., "A Survey on Diffusion Models for Inverse Problems" (arXiv 2410.00083, 2024), with a taxonomy table covering DPS, ΠGDM, DDRM, latent methods and blind forward models — [arXiv HTML](https://arxiv.org/html/2410.00083v1); and "Diffusion Models for Image Restoration and Enhancement: A Comprehensive Survey" (arXiv 2308.09388) — [arXiv](https://arxiv.org/pdf/2308.09388). Neither was mined for further deblur numbers.
