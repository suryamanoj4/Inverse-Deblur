# Phase 3 literature review: kernel placement under family shift (C1) and realistic-forward-model training (C2)

Date: 2026-11-22. Scope: about 35 web searches and fetches, plus full-text reads of 12 PDFs or HTML pages. This builds on, and does not repeat, `research_notes/Non blind deblurring novelty survey/*.md` and `reports/Non blind deblurring novelty survey.md`. Items already covered there (SRMD, SFTMD, Kaufman & Fattal, IJCV 2022 Chen et al., DPIR/USRNet structure, Shimron, Blur2Blur, SRGA) are referenced but not re-cited in detail.

Tags:
- **[V]**: checked in the primary text this session (PDF or HTML full text).
- **[S]**: from a search snippet, an abstract, or a secondary page only.
- **[V\*]**: from a fetched full-text page, read through a summariser with quotes; numbers not cross-checked against a PDF.

Absence claims are bounded by this budget. Absence of evidence is not proof of absence.

---

## Q1 (claim C1): Do kernel-conditioned networks degrade on out-of-distribution kernels, even below an unconditioned network? Does a data-consistency step fix this?

### Takeaway
I found **no paper that reports C1's specific direction**. In that direction, concatenation or FiLM kernel conditioning improves in-distribution accuracy but makes held-out kernel *families* worse than the same backbone with no kernel at all. Adjacent literatures give evidence pointing both ways:
- **Explains the mechanism (supports C1):**
  - Conditioned SR nets fed a wrong kernel produce systematic over-sharpening/ringing or over-smoothing artifacts (IKC).
  - Unrolled solvers fed a mismatched operator fail too (Gilton: about 9 dB drop for MoDL).
  - Kernel-agnostic networks trained on varied blur nearly match kernel-conditioned ones in distribution (KBPN).
- **Points the other way:** in denoising, the noise-*conditioned* FFDNet degrades *less* than blind DnCNN above its training range (CDLNet).

So "conditioning hurts OOD below unconditioned" is not a general law the field already knows. In the non-blind deblurring setting with held-out kernel families, it appears unreported. The second half, that one FFT data step recovers OOD performance and full unrolling recovers more, agrees qualitatively with:
- the inverse-problems consensus that keeping the operator in an explicit data term helps under operator change (Guan et al. 2024: robust loop-unrolling 34.4 dB vs direct inverse 24.0 dB);
- the algorithmic-alignment argument of Xu et al.

No retrieved paper measures this as a controlled placement ablation on one backbone. The closest "injection-level study" (arXiv 2610.24599, Oct 2026) is blind, in-distribution only, and conditions on a severity map rather than a kernel.

### Cited findings
**Kernel/degradation-conditioned nets under mismatch (SR, denoising)**
- **IKC / SFTMD** (Gu et al., CVPR 2019). Kernel mismatch in a kernel-conditioned SR net gives regular, asymmetric artifacts. A kernel input smoother than the truth gives over-smoothing. A sharper one gives over-sharpening and ringing ("unnatural ringing artifacts caused by over-enhancing high-frequency edges"); see the Fig. 2 caption, "SR sensitivity to the kernel mismatch". Test kernels (Gaussian8) lie inside the training width ranges, and no out-of-family test is reported. [V] [ar5iv 1904.03377](https://ar5iv.labs.arxiv.org/html/1904.03377)
  - *Relevance:* an unseen-family kernel embedding may behave like a "wrong kernel" input, which predicts ringing and lower PSNR than a kernel-agnostic net.
- **Kernelized Back-Projection Networks** (Yoshida et al., arXiv 2302.08478v3). In a preliminary experiment with MSE-only training for both models, a non-kernel DBPN trained on variously blurred images is "superior or comparable to IKC". The abstract states: "non-blind SR trained simply with various blur kernels exhibits comparable performance as those with the degradation model". The test is in distribution only (isotropic Gaussians). [V] [arXiv 2302.08478](https://arxiv.org/abs/2302.08478)
- **CMDSR** (Yin et al., arXiv 2104.03926v5). On a "Severe" set (anisotropic Gaussian, σ_N = 50), SRMD scores 22.43 and RCAN+ (blind, bicubic-trained) scores 16.32 on Set5. **Caveat:** the set is OOD for CMDSR, not for SRMD, which was trained with anisotropic kernels. It is therefore **not** evidence about conditioned-net OOD behaviour, and should not be cited as such. [V] [arXiv 2104.03926](https://arxiv.org/abs/2104.03926)
- **Bias-free CNNs** (Mohan, Kadkhodaie, Simoncelli, Fernandez-Granda, ICLR 2020):
  - Denoisers "systematically overfit to the noise levels" in training and fail far outside them.
  - Removing biases fixes noise-level extrapolation.
  - But "BF-CNNs can generalize outside the training range for noise levels for a fixed blur level, but do not outperform CNN when generalizing to unseen blur levels."

  This is one of very few statements about blur-level extrapolation. [V] [arXiv 1906.05478](https://arxiv.org/abs/1906.05478)
- **CDLNet** (Janjušević et al., arXiv 2112.00913; trained on σ∈[1,20] or [20,30]):
  - Above the training range: "Even FFDNet-B*, which takes the noise-level as input, fails to generalize, although with a smaller performance drop."
  - Colour above the training range: DnCNN-B* fails, FFDNet-B* "significantly drop[s]".
  - Below the training range: "FFDNet-B* and DnCNN-B* are able to generalize better than CDLNet-B".
  - Visually, FFDNet-B* "produces unwanted artifacts" at σ = 50.

  Here the conditioned net is less bad OOD than the unconditioned one, the **opposite** of C1. It is a useful contrast: a scalar noise level extrapolates along one dimension, whereas a kernel embedding leaves the training manifold in many directions. [V] [arXiv 2112.00913](https://arxiv.org/pdf/2112.00913)
- **FFDNet** (Zhang et al., TIP 2018). With a mismatched noise map, a level set too low leaves residual noise and one set too high over-smooths; the best result comes at the matched level. [S] [ar5iv 1710.04026](https://ar5iv.labs.arxiv.org/html/1710.04026)
- **AdaFM / CResMD** (He et al., CVPR 2019 / ECCV 2020). They modulate between trained start and end levels by interpolation only, and I found no extrapolation result. A later NAS paper reports that CResMD is "susceptible to artifacts when the restoration type of the task vector doesn't match the degradation type". [S] [AdaFM 1904.08118](https://ar5iv.labs.arxiv.org/html/1904.08118), [CResMD 1912.05293](https://arxiv.org/pdf/1912.05293), [2012.11225](https://arxiv.org/pdf/2012.11225)
- **Severity-Channel Conditioning: An Injection-Level Study** (Al Shafi et al., arXiv 2610.24599, 17 Oct 2026). It compares input concatenation, global FiLM-style modulation, spatial modulation and a contrastive objective on one diffusion deblurrer (iPhoneBlur):
  - Concatenation gives +0.07 dB (31.23 → 31.30), and "feature modulation ... do[es] not exceed the baseline".
  - The conditioning is an oracle severity map (blind, target-derived), not a kernel.
  - There is no OOD tier.
  - It reports three seeds and explicitly disclaims robustness to retraining.

  This is the **closest framing competitor** ("where to inject conditioning"), but not a substantive overlap. [V abstract] [arXiv 2610.24599](https://arxiv.org/abs/2610.24599)
- **All-in-one restoration** papers (DAIR, Adaptive Blind AiO, MIRAGE) assert that explicit degradation prompts or labels give "brittle assumptions that weaken generalization to unseen ... degradations". These are argued, not measured, as a controlled ablation. [S] [2505.18679](https://arxiv.org/html/2505.18679v2), [2411.18412](https://arxiv.org/html/2411.18412v2)

**Placement of operator knowledge and robustness under operator shift (inverse problems)**
- **Gilton, Ongie & Willett, "Model Adaptation for Inverse Problems in Imaging"** (arXiv 2012.00139v2):
  - MRI with MoDL (unrolled, with a DC step): naively swapping A0→A1 gives "catastrophic results (a roughly 9 dB drop in reconstruction PSNR)".
  - Deblurring: 7×7 motion at 10° → 20°.
  - Fig. 9: one net trained on many blur kernels "sacrifices performance on all blurs", both seen and unseen. "Training on multiple blur kernels simultaneously incurs a performance penalty as the number of blur kernels used in training increases."
  - *Relevance:* unrolled nets are **not automatically robust**. C1's USRNet robustness relies on multi-kernel training plus an exact operator, so reviewers will ask to see that interaction. [V] [arXiv 2012.00139](https://arxiv.org/pdf/2012.00139)
- **Hu, Shoushtari, ..., Kamilov, "Robustness of Deep Equilibrium Architectures to Changes in the Measurement Model"** (arXiv 2211.00531):
  - DEQ priors trained with a *mismatched* operator still beat operator-agnostic AWGN (PnP) priors, by 1.84 dB in CS-MRI and 0.23 dB in Gaussian-kernel SR.
  - "a severe performance drop due to the usage of an inaccurate measurement model during inference".
  - No end-to-end baseline. [V\*] [ar5iv 2211.00531](https://ar5iv.labs.arxiv.org/html/2211.00531)
- **Guan et al., "Solving Inverse Problems with Model Mismatch using Untrained Neural Networks within Model-based Architectures"** (arXiv 2403.04847). Blind deblurring with a mismatched Gaussian kernel (true variance 7, estimates 1–7): direct inverse network 24.00 dB, robust loop-unrolled net (trained with inexact operators) 34.38 dB, their adaptive DEQ 37.81 dB. This is the closest numerical analogue to "unrolled ≫ direct under operator shift". [V\*] [ar5iv 2403.04847](https://ar5iv.labs.arxiv.org/html/2403.04847)
- **Antun et al., PNAS 2020.** Learned reconstructions are unstable to tiny perturbations and structural changes, and sometimes "more samples may yield poorer performance". [S] [PMC7720232](https://pmc.ncbi.nlm.nih.gov/articles/PMC7720232)
- **Jalal et al., NeurIPS 2021.** End-to-end CS-MRI methods show "severe degradation in reconstruction quality when applied out-of-distribution" (sampling pattern, anatomy); an operator-agnostic score prior is robust. [S] [arXiv 2108.01368](https://arxiv.org/abs/2108.01368)
- **Knoll et al., MRM 2019.** Variational-network generalisation: SNR mismatch hurts substantially; heterogeneous training helps. [S] [PMC6240410](https://pmc.ncbi.nlm.nih.gov/articles/PMC6240410)
- **Darestani, Chaudhari & Heckel, ICML 2021.** This study shifts the image distribution, not the operator family (already covered in the prior survey). Trained and un-trained methods "suffer very similarly". [prior survey]
- **Xu et al., "How Neural Networks Extrapolate", ICLR 2021.** ReLU MLPs become linear away from the training support and "do not extrapolate most nonlinear functions". Extrapolation succeeds when the right nonlinearity is encoded in the architecture. This gives a theoretical framing for C1: the FFT data step hard-codes the kernel-dependent inverse, while FiLM must learn it from the training families. [S] [arXiv 2009.11848](https://arxiv.org/abs/2009.11848)

### Closest prior art (ranked)
1. **Gilton et al. 2021**: multi-kernel training without conditioning costs accuracy on seen and unseen kernels, and unrolled MoDL collapses under naive operator swap. It is the nearest controlled operator-shift study, but uses no kernel conditioning and tiny 7×7 kernels.
2. **IKC (Gu et al. 2019)**: documents the artifact mechanism of conditioned networks given a mismatched kernel, but not held-out families.
3. **Guan et al. 2024; Hu et al. 2022**: model-based architectures vs direct or operator-agnostic ones under operator mismatch. They are blind or adaptive, with no FiLM/concat comparison.
4. **CDLNet 2021; Mohan et al. 2020**: conditioning vs no conditioning under noise-level extrapolation. The direction is opposite to C1, which should be discussed.
5. **KBPN 2023**: unconditioned ≈ conditioned in distribution.
6. **Severity-channel injection study 2026**: injection-location framing, in distribution and blind.
7. **Chen et al. IJCV 2022** (prior survey): supervised nets do not generalise to unseen non-motion kernels, but there is no conditioned vs unconditioned contrast.

### Verdict on novelty
- **C1a**, "conditioning helps in distribution (+0.9 dB)": **known**, the expected SRMD/SFTMD/Kaufman result.
- **C1b**, "concat/FiLM conditioning makes unseen-family performance *worse than no conditioning*" (22.2 → 20.8 / 18.2 dB): **novel as a reported measurement** in non-blind deblurring, as far as these searches reach. The mechanism is anticipated (IKC mismatch artifacts; Xu et al. extrapolation). Contrary evidence exists for scalar noise conditioning (CDLNet), which makes the result more interesting, not less.
- **C1c**, "one FFT data step recovers it (23.7) and full unrolling gives 27.6": **partially known**. The qualitative direction is consensus (Guan; Hu; MRI DC literature). The controlled same-backbone placement ladder (none / concat / FiLM / FiLM+1 DC / unrolled) with held-out families was not found.

### Gaps and risks
- C1b might be an artifact of one seed or one family (for example, the ghost kernel alone). The FiLM drop of −4.0 dB is much larger than the concat drop of −1.4 dB, so seed variance and per-family breakdowns are mandatory.
- The unconditioned NAFNet's 22.2 dB OOD may be close to the PSNR of the *blurred input* or a conservative output (Blur2Blur reports that NAFNet on unseen blur outputs nearly the input). If so, "conditioned is worse" may mean "conditioned attempts inversion and rings", which IKC predicts. Report blurred-input PSNR and a Wiener/EPLL floor per family.
- I did not verify whether DAN, UDVD or KOALAnet report kernel-family OOD. The MANet and KernelGAN+SRMD literature was not opened.
- No 2025–2026 non-blind "kernel placement" paper was found. A cited-by search on SFTMD and Kaufman & Fattal (2025–2026) is still the main residual risk.

---

## Q2 (claim C2): Is training a USRNet-style solver on half realistic-forward-model batches (linear conv + boundaries + edgetaper) plus kernel perturbation (≤10% noise on dilated support) prior art?

### Takeaway
**Both ingredients are published. The combination and the reported failure are only partly covered.**
- **Kernel perturbation:** Nan & Ji (CVPR 2020) Algorithm 1 synthesises kernel errors by adding Gaussian noise on a *dilated kernel contour*, plus Bernoulli dropout, blur, and off-support noise. That is essentially C2's perturbation. They also train and test with **edgetaper** to simulate practical boundaries.
- **Boundary failure:** Eboli et al. (ECCV 2020) show that FFT-based HQS on Levin kernels fails specifically on the largest kernels, ker-4 and ker-8, unless boundaries are handled. This is the same failure pattern C2 reports for standard USRNet.
- **Real captures:** Levin et al. (2009) state that their blurred images are real captures. The "ground-truth" kernels are least-squares fits, and "the shift-invariant blur assumption ... is often violated". This explains why both boundary realism and kernel perturbation help on Levin even with GT kernels.

What I did not find:
- a USRNet/DPIR-style FFT-unrolled solver trained with a mixed circular/realistic forward model;
- a report that official USRNet or DPIR collapse on the two largest Levin kernels when run on the real captures with GT kernels;
- a 2×2 attribution of boundary realism vs kernel noise.

### Cited findings
- **Nan & Ji, "Deep Learning for Handling Kernel/model Uncertainty in Image Deconvolution"** (CVPR 2020) [V] ([CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Nan_Deep_Learning_for_Handling_Kernelmodel_Uncertainty_in_Image_Deconvolution_CVPR_2020_paper.html)):
  - Algorithm 1: "M = where(k > 0) ... M = dilate(M) ... k̂ = n1 ⊙ M + k ... Bernoulli random sampling ... Blurring the kernel ... Add noise off the contour". They train on about 140k noisy kernels, half synthetic and half returned by blind methods.
  - Ablation (Levin, estimated kernels): removing the synthetic kernels costs "about 0.7 dB" (for example 30.92 → 30.06 with [7]-estimated kernels).
  - Boundary: "we use 'edge-taper' ... for simulating the boundary condition of practical blurring". Against Vasu they follow Vasu's setting, "which keeps the boundary information".
  - Gains on Lai's "large kernels" with estimated kernels: "about 2 dB".
  - Their Sun et al. test set is described as "640 images, generated by 80 clear images and the same truth kernels from [25]", i.e. **synthetic convolution**.
  - **Difference from C2:** an uncertainty-specific TLS-unrolled network, tested on *estimated* kernels, with no FFT-HQS/USRNet and no GT-kernel-on-real-capture evaluation.
- **Vasu, Maligireddy & Rajagopalan** (CVPR 2018) [V] ([CVF PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Vasu_Non-Blind_Deblurring_Handling_CVPR_2018_paper.pdf)):
  - Synthetic noisy kernels: centroid-preserving, smooth, non-negative noise restricted by "the mask for selecting only the valid region over the neighborhood of the ground truth kernel", then low-pass filtered. This is again noise on a neighbourhood of the support.
  - The CNN post-processes a conventional deconvolution.
- **Eboli, Sun & Ponce, "End-to-end Interpretable Learning of Non-blind Image Deblurring"** (ECCV 2020) [V] ([arXiv 2007.01769](https://arxiv.org/abs/2007.01769)):
  - "FFT ... is only truly valid under periodic conditions at the boundaries, which are never respected by real images."
  - Table 1 (TV-ℓ1 HQS on 160 VOC images × 8 Levin kernels, 2% noise): HQS-FFT averages 20.69 dB with no padding, 25.53 with replicate, 25.96 with FDN padding, against CHQS 26.27.
  - "For kernels 1, 2, 3 and 5, the four methods yield comparable results ... FFT-based methods are significantly worse on the other four." "FDN padding greatly improves HQS-FFT results on larger kernels ... 'ker-4' and 'ker-8'."
  - Learned LCHQS on Levin/Sun reaches 35.15/32.93 dB.
  - **This is direct prior evidence that C2's "largest two kernels" effect is a boundary effect of FFT data steps.**
- **Kruse, Rother & Schmidt (FDN, ICCV 2017).** A "simple, yet effective, boundary adjustment" for FFT-based learned deconvolution, because the circular assumption is "almost never satisfied". [S] ([CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Kruse_Learning_to_Push_ICCV_2017_paper.html)). The PDF could not be fetched.
- **Levin, Weiss, Durand & Freeman** (CVPR 2009) [V] ([PDF](https://people.csail.mit.edu/billf/papers/deconvLevinEtalCVPR09.pdf)):
  - Real hand-shake captures of a planar scene on a tripod, Z-rotation locked.
  - Kernel obtained by solving "for a non-negative kernel k minimizing ‖k⊗x−y‖²" against a sharp reference.
  - 4 windows of 255×255 were cropped from each of 8 frames, so boundaries carry real out-of-window content.
  - "the shift-invariant blur assumption made by most algorithms is often violated"; "wider kernels result in larger deconvolution error even with the true kernel".
- **Anger, Delbracio & Facciolo, "Modeling Realistic Degradations in Non-blind Deconvolution"** (ICIP 2018) [V] ([ar5iv 1806.01097](https://ar5iv.labs.arxiv.org/html/1806.01097)):
  - Boundaries: "handled by padding the image and considering the data fitting term only on valid pixels".
  - Models saturation, gamma and quantisation; figure examples show about +2.4 to +3.0 dB from realistic modelling.
  - Variational, not learned, with no kernel error. A DWDN reviewer cited it as missing related work ([NeurIPS 2020 reviews](https://proceedings.neurips.cc/paper_files/paper/2020/file/0b8aff0438617c055eb55f0ba5d226fa-Review.html)) [V].
- **USRNet (Zhang et al., CVPR 2020).** "Because Eq. (7) is based on the assumption of circular boundary condition, a proper boundary handling for the real LR image is generally required." [S] ([arXiv 2003.10428](https://arxiv.org/pdf/2003.10428)). The README claims that a 25×25-trained model "generalizes well to much larger kernel size" [S].
- **Sanghvi et al., photon-limited unrolling** (arXiv 2110.15314). Reports artifacts "as the circular boundary conditions may not apply"; fixed by reflect-padding to double size and centre-cropping. [S] ([arXiv](https://arxiv.org/pdf/2110.15314))
- **Chaganova et al., "Applied aspects of modern non-blind image deconvolution methods"** (Computer Optics 48(4), 2024) [V\*] ([cyberleninka](https://cyberleninka.ru/article/n/applied-aspects-of-modern-non-blind-image-deconvolution-methods)):
  - Benchmark of USRNet, DWDN, KerUnc (Nan & Ji) and Wiener; kernel-error robustness tested by Nan & Ji's protocol.
  - "USRNet (DM = 0.46 for motion blur ...) [is among] the least robust"; KerUnc "handles blur kernel errors best".
  - **Independent evidence that official USRNet is fragile to kernel error.**
- **DWDN** (Dong, Roth & Schiele, NeurIPS 2020). Robust to inaccurate kernels even when trained on GT kernels, about +0.7 dB over a kernel-error method. Trained with GT, noisy and mixed kernels. [S] ([arXiv 2103.09962](https://arxiv.org/pdf/2103.09962))
- **Guan et al. 2024.** "Robust LU" (loop-unrolled, trained with inexact forward models) is a named baseline: 34.38 dB vs a direct inverse at 24.00. So training an unrolled solver on perturbed operators is an established baseline. [V\*] ([ar5iv 2403.04847](https://ar5iv.labs.arxiv.org/html/2403.04847))
- **Ren et al., "Partial Deconvolution With Inaccurate Blur Kernel"** (TIP 2018). Uses only the reliable Fourier entries of an inaccurate kernel, via an E-M-estimated partial map. [S] ([PDF](https://csdwren.github.io/papers/Partial%20Deconvo_TIP.pdf)). Ren et al. NeurIPS 2018 (GLRA) is in the prior survey.
- **Quan, Chen, Zheng & Ji, ECCV 2022.** Unsupervised NBD with kernel-perturbation ensembles for kernel-error robustness. [S] ([ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2222_ECCV_2022_paper.php))

### Closest prior art (ranked)
1. **Nan & Ji 2020**: the same kernel-perturbation recipe (noise on dilated support) plus edgetaper boundary simulation. It differs in architecture (TLS unrolling, not FFT-HQS), in evaluation (estimated kernels, not GT kernels on real captures) and in the mixing scheme.
2. **Eboli et al. 2020**: the same diagnosis, that FFT data steps fail on Levin's largest kernels because of boundaries, and the fix of padding/edgetaper or a non-FFT solver. Its learned model (LCHQS) is trained on BSD with synthetic kernels, so the "realistic-boundary" training idea is close.
3. **Vasu et al. 2018**: noisy-kernel training restricted to the kernel neighbourhood.
4. **Guan et al. 2024**: "robust LU" trained with inexact operators as a standard baseline.
5. **Chaganova et al. 2024**: USRNet is least robust to kernel error. This supports C2's motivation.
6. **Anger et al. 2018; FDN 2017**: realistic boundary/acquisition modelling in deconvolution.

### Verdict on novelty
- **C2 is partially novel at best; most of its ingredients are known.**
- The perturbation scheme is close to Nan & Ji's Algorithm 1. The boundary realism follows FDN, Eboli and Anger. The large-kernel failure of FFT solvers is documented by Eboli.
- Defensible residue:
  1. the specific finding, on Levin's *real captures with GT kernels*, that a USRNet-style FFT-unrolled solver gains +4.1 dB, concentrated on the two largest kernels where standard USRNet and EPLL collapse;
  2. the mixed circular/realistic batch recipe applied to an exact-FFT unrolled net, with its in-distribution cost reported;
  3. attribution between boundary realism and kernel perturbation, if the 2×2 ablation is run.
- Frame it as "closing the circular-boundary and kernel-error gap of FFT-unrolled solvers on real captures", citing Nan & Ji and Eboli as the origin of the components.

### Gaps and risks
- **Sun et al. is synthetic** (per Nan & Ji's description, 80 images × 8 Levin kernels convolved). The +0.85 dB there cannot come from real-capture mismatch. It must come from boundary handling (and possibly noise), or from regularisation by kernel noise. Confirm how the project's Sun images were generated (linear "valid" vs "same" convolution), because this changes the interpretation. [Sun 2013 generation details still UNVERIFIED.]
- The **cheapest rival explanation** is test-time boundary handling. Standard USRNet plus test-time edgetaper or reflect-padding (the KAIR/Sanghvi fix) might recover much of the +4.1 dB without retraining. This must be run.
- I did not verify DPIR's own Levin boundary handling or whether official DPIR runs collapse on ker-4/ker-8. Levin kernel sizes per index are UNVERIFIED (reported range 13–27 px).
- Vasu and Nan & Ji Levin numbers use *estimated* kernels and are not directly comparable to C2's GT-kernel protocol.

---

## Q3: What experiments would reviewers demand?

### Takeaway
The reviews of the closest published paper (DWDN, NeurIPS 2020, [reviews](https://proceedings.neurips.cc/paper_files/paper/2020/file/0b8aff0438617c055eb55f0ba5d226fa-Review.html) [V]) and current norms visible in a 2026 injection study (seeds, retraining robustness, a second backbone; [2610.24599](https://arxiv.org/abs/2610.24599) [V abstract]) point to five recurring demands:
1. **Attribution ablations.** DWDN reviewers asked "whether the improvement comes from the multi-scale module or the Wiener module" and whether "Ours w/o Wiener" receives the kernel.
2. **Fair retraining of baselines on the same data.** "Only three learning-based methods were retrained on the same dataset."
3. **Wider noise levels**, up to 5%.
4. **Realistic degradations** (non-linear CRF, saturation; Anger et al. cited as missing).
5. **Novelty relative to existing Wiener/FFT steps.** "The Wiener deconvolution has been proposed already."

For a controlled-study paper, reviewers additionally expect variance across seeds, a second architecture, and mechanism probes.

### Ranked experiments
1. **C2 attribution and the cheap-fix baseline.** Run a 2×2 of {circular, realistic-boundary batches} × {no, with kernel perturbation}. Add official pretrained USRNet and DPIR with test-time edgetaper, reflect-padding and FDN padding on Levin real, with per-kernel tables. This shows the +4.1 dB is not obtainable by test-time padding, and isolates boundary from kernel-error effects (Eboli, Nan & Ji). It also needs the in-distribution cost on P1.
2. **Seeds and paired statistics for C1.** Use ≥3 seeds per placement variant. Report per-family breakdowns (disc / ring / ghost separately, plus P2_interp) with paired per-image bootstrap CIs. The C1b claim needs to hold in most families and to exceed seed spread. FiLM's 18.2 vs concat's 20.8 suggests high variance or one dominant family.
3. **Kernel-swap / counterfactual probes on the conditioned nets.** On P2_extrap, feed the true kernel, the nearest training-family kernel (by embedding or spectral distance), a delta kernel, and a shuffled kernel. Also plot PSNR drop against embedding distance from the training set, and against the spectral-zero overlap of each family. This distinguishes "embedding extrapolates badly and acts like a wrong kernel" (the IKC mechanism, which predicts ringing) from "the conditioned net learned a shortcut that ignores the image". Add trivial floors: blurred-input PSNR and Wiener/EPLL per family.
4. **Second backbone and matched capacity for C1.** Repeat none / concat / FiLM / FiLM+DC on a second trunk (for example a DRUNet- or Restormer-style U-Net) at matched parameters and FLOPs. Include SFT (spatial) vs FiLM (global), since SFTMD is the canonical modulation baseline. Without this, reviewers will say the result is NAFNet-specific; LayerNorm/SimpleGate interacting with FiLM is a plausible confound.
5. **Training-coverage control (leave-one-family-out).** Rotate which family is held out, or add one OOD family to training, and show the conditioned-vs-unconditioned gap shrinks as coverage grows. This links to Gilton's finding that multi-kernel training trades off seen and unseen accuracy, and shows C1b is about extrapolation, not a broken conditioning pathway. Optionally add the Danielyan non-motion kernels for comparability with Chen et al. IJCV 2022.

Also likely requested, lower priority:
- noise levels 1/3/5%;
- the Lai et al. real set with large kernels;
- runtime and parameters;
- DPIR and DiffPIR as kernel-agnostic-prior controls on P2_extrap;
- an explicit novelty table citing Nan & Ji, Eboli, IKC, Gilton and Guan.
