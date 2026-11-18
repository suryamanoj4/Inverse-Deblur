# Closed-form (Wiener / FFT / Tikhonov) deconvolution + learned network hybrids for non-blind deblurring, 2013–2026

Scope: methods that pair an explicit closed-form deconvolution step with a learned network, plus MRI data-consistency (DC) precedents. These notes feed a novelty check of the proposed method: a NAFNet predicts x0 from (y, k, sigma), then ONE closed-form DC step X = (conj(K)Y + rho X0)/(|K|^2 + rho), with rho set per sigma and the whole thing trained end-to-end (loss on the post-DC output plus 0.1 on x0).

Search budget note: about 25 tool calls. Primary sources (arXiv HTML/PDF, CVF) were read for DWDN, INFWIDE, USRNet, DPIR, DiffPIR and Schlemper 2017. Everything else rests on abstracts or search snippets only, and is labelled as such.

## Q1. Catalogue: architecture, data, kernels, noise, PSNR

### Takeaway
The closed-form step that the proposal uses, X = F^-1[(conj(F(k))F(y) + alpha F(z)) / (|F(k)|^2 + alpha)], is already the data module in USRNet (CVPR 2020) and DPIR (TPAMI 2021). It is also the per-step data proximal in DiffPIR (2023). USRNet already learns alpha as a function of sigma, using an MLP trained end-to-end through the FFT. The Wiener-first family (Schuler 2013, Son & Lee 2017, DWDN 2020/21, INFWIDE 2022/23, MultiWienerNet 2022) puts the closed-form step BEFORE the network, and sets its regularisation from image statistics rather than learning it.

### Cited Findings

**Schuler, Burger, Harmeling, Schölkopf, CVPR 2013, "A Machine Learning Approach for Non-blind Image Deconvolution" (Wiener + MLP)**
- Two-step method. A regularised Fourier-domain inverse comes first, then a learned network removes the coloured noise and artifacts that the inversion amplifies. It is trained on a large natural-image set and beats the prior SOTA on many synthetically blurred images. A real out-of-focus photo is shown. — [CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Schuler_A_Machine_Learning_2013_CVPR_paper.html)
- Zhang et al. 2017 describe the network as an MLP that removes noise and artifacts after deconvolution. — [arXiv 1611.06495](https://arxiv.org/pdf/1611.06495.pdf)
- Placement: closed-form step BEFORE the net, image domain, 1 step, fixed regulariser (UNVERIFIED: exact regulariser weight and patch-MLP size not read).
- In DWDN's Levin table, MLP scores below FNBD (34.81 dB at 1% noise). — [DWDN arXiv v2](https://arxiv.org/html/2103.09962v2)

**Son & Lee, ICCP 2017, "Fast Non-blind Deconvolution via Regularized Residual Networks with Long/Short Skip-Connections"**
- A Wiener filter runs first as a "pre-deconvolution". The resulting noise and ringing are argued to be largely independent of kernel shape, so a residual net with long/short skips removes them. A penalty on abnormal conv weights is added to improve robustness to untrained or inaccurate kernels. — [POSTECH record](https://remotecenter.postech.ac.kr/handle/2014.oak/42195); [code](https://github.com/posgraph/coupe.cnn-deconvolution)
- Placement: Wiener BEFORE the net, image domain, 1 step. Benchmark numbers not retrieved (gap).

**Zhang, Pan, Lai, Lau, Yang, CVPR 2017, "Learning Fully Convolutional Networks for Iterative Non-blind Deconvolution"**
- The problem is split into gradient-domain denoising by a CNN and a deconvolution step guided by the learned gradients. This is iterated in a multi-stage framework. — [arXiv 1611.06495](https://arxiv.org/pdf/1611.06495); [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_Fully_Convolutional_CVPR_2017_paper.html)
- UNVERIFIED: whether the deconvolution step is solved in closed form with FFT. It probably is a quadratic solve with gradient priors, but the snippets do not say. In DWDN's Levin table this method appears as "FCN", below FNBD.

**Kruse, Rother, Schmidt, ICCV 2017, FDN ("Learning to Push the Limits of Efficient FFT-based Image Deconvolution")**
- The simple regulariser in FFT-based deconvolution is replaced by a CNN-based one. The method stays FFT-efficient and adds a boundary-adjustment step for the circular-convolution assumption. It reports SOTA on two standard non-blind benchmarks. — [CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Kruse_Learning_to_Push_ICCV_2017_paper.html)
- Placement: multiple stages of closed-form FFT updates interleaved with CNNs, image domain, trained stage-wise/end-to-end. UNVERIFIED: exact update formula, number of stages, and whether the weights are noise-conditioned. The PDF fetch failed (403/404).
- Reported by others: Levin 1/3/5% = 34.05 / 29.77 / 27.94 dB, Sun 1% = 32.63 dB — [DWDN v2](https://arxiv.org/html/2103.09962v2). DPIR shows FDN well below DPIR, e.g. Leaves at 1% noise with the 17×17 kernel: FDN 27.04 vs DPIR 35.19 dB — [DPIR arXiv v2](https://arxiv.org/html/2008.13751v2).

**Xu, Ren, Liu, Jia, NeurIPS 2014, "Deep Convolutional Neural Network for Image Deconvolution"**
- Not retrieved in this session. UNVERIFIED, from background knowledge: it uses separable kernel-inversion (pseudo-inverse) convolutions followed by a denoising CNN, and it is trained per kernel. This makes it a non-explicit-FFT, Wiener-like learned inverse. Treat as a gap.

**Ren et al. 2018, "generalized low-rank" deconvolution**
- Not retrieved (gap).

**Nan & Ji, CVPR 2020, KerUnc ("Deep Learning for Handling Kernel/model Uncertainty in Image Deconvolution")**
- Starts from an error-in-variables model to handle kernel error. Unrolls a total-least-squares estimator with NN-learned priors, and outperforms prior methods when kernels are noisy (estimated by blind methods). — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Nan_Deep_Learning_for_Handling_Kernelmodel_Uncertainty_in_Image_Deconvolution_CVPR_2020_paper.html)
- A later paper describes the residual prior as a dual-path U-Net. — [arXiv 2210.05361](https://ar5iv.labs.arxiv.org/html/2210.05361)
- Not a Wiener method per se. UNVERIFIED: whether its data step is FFT closed form.

**USRNet: Zhang, Van Gool, Timofte, CVPR 2020, "Deep Unfolding Network for Image Super-Resolution". Most important precedent.**
- Data module D, Eq. 7: z_k = F^-1[(1/alpha_k)(d − conj(F(k)) ⊙_s ((F(k)d)⇓s) / ((conj(F(k))F(k))⇓s + alpha_k))], with d = conj(F(k))F(y↑s) + alpha_k F(x_{k−1}). The paper notes that "Eq. (7) also works for the special case of deblurring when s = 1", where it reduces exactly to (conj(K)Y + alpha X)/(|K|^2 + alpha). D has no trainable parameters. — [USRNet arXiv 2003.10428](https://arxiv.org/abs/2003.10428) (PDF text read)
- Hyper-parameter module: [alpha, beta] = H(sigma, s). It is a 3-layer MLP (64 hidden units, ReLU, ReLU, Softplus, +1e-6) that outputs alpha_k (data weight) and beta_k (denoiser noise level) for each of K = 8 iterations, trained end-to-end with L1 loss through the FFT data module. The authors argue that letting the weights vary with sigma and s gives a gain over fixed lambda and mu_k. — same source
- Order: x0 = nearest-neighbour upsampled y (that is, y itself for s = 1), then D → P → D → P …, ending with the prior module P (ResUNet). The final output is the prior output, not the DC output. — same source
- Training: DIV2K + Flickr2K, anisotropic Gaussian kernels plus motion kernels from Boracchi & Foi, kernel size 25×25, noise sigma in [0, 25], s in {1, 2, 3, 4}. The paper explicitly "does not consider the deblurring experiments", so no s = 1 PSNR is reported. Evaluation is on BSD68 SR. — same source
- Fig. 6: beta decreases with iteration and increases with scale and noise. alpha's behaviour with sigma is plotted. — same source

**DPIR: Zhang, Li, Zuo, Zhang, Van Gool, Timofte, TPAMI 2021, "Plug-and-Play Image Restoration with Deep Denoiser Prior"**
- Deblurring data step, Eq. 9: x_k = F^-1[(conj(F(k))F(y) + alpha_k F(z_{k−1})) / (conj(F(k))F(k) + alpha_k)]. This is literally the proposed DC formula. — [DPIR arXiv v2](https://arxiv.org/html/2008.13751v2)
- alpha_k = lambda sigma^2 / sigma_k^2, with sigma_k log-spaced from 49 down to sigma, K = 8, lambda = 0.23, z_0 = y. The rho per sigma is hand-set, not learned, and the denoiser is not trained through the FFT. — same source
- Test: Set6 with Levin kernels #2 (17×17) and #4 (27×27), sigma = 2.55 and 7.65. With the 17×17 kernel at sigma 2.55, DPIR scores 32.05 / 35.82 / 33.38 / 34.26 / 35.19 / 34.21 (C.man, House, Monarch, Butterfly, Leaves, Starfish), against IRCNN 31.69 / 35.04 / 32.71 / 33.13 / 33.51 / 33.15 and FDN 29.09 / 29.75 / 29.13 / 28.06 / 27.04 / 28.12. DPIR gains 0.2–2 dB over IRCNN while using 8 instead of 30 iterations. — same source

**DWDN: Dong, Roth, Schiele, NeurIPS 2020; extended TPAMI version "DWDN" (DWDN+), arXiv 2103.09962**
- Feature extractor (1 conv + 3 res blocks, M = 16 channels). Each feature map gets a Wiener deconvolution G_i = K^T / (K^T K + s_i^n / s_i^x) in the Fourier domain. The NSR is NOT learned: s_i^x is the std of the blurry feature and s_i^n is the variance of the feature minus its mean-filtered version. A multi-scale (L = 2) cascaded encoder-decoder refinement follows (3 cascades in DWDN+), with L1 loss per scale, trained end-to-end. — [DWDN v2](https://arxiv.org/html/2103.09962v2)
- Training: about 400 BSD + 4,744 Waterloo images, 256² crops, Schmidt et al. synthetic kernels 13×13 to 35×35, Gaussian noise 0–5% in one model. — same source
- Levin (4 images × 8 kernels), 1/3/5%: DWDN 36.90 / 32.77 / 30.77 and DWDN+ 37.27 / 33.09 / 31.08, against FNBD 34.81 / 30.63 / 27.93, IRCNN 34.33 / 30.04 / 28.51 and FDN 34.05 / 29.77 / 27.94. Sun 1%: DWDN 34.05, DWDN+ 34.62, IRCNN 33.57. BSD-100 synthetic 1%: DWDN 31.74, DWDN+ 32.02, FNBD 30.92. Tables for saturation and JPEG are included. — same source
- The comparison list does NOT include USRNet or DPIR. — same source

**Dong, Roth, Schiele, CVPR 2021, "Learning Spatially-Variant MAP Models for Non-Blind Image Deblurring"**
- Learns both the data term and the regulariser of a MAP model, with networks predicting per-pixel (spatially-variant) filters. Reported to substantially beat SOTA, with a spatially-invariant baseline SIMAP in the supplement. — [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Dong_Learning_Spatially-Variant_MAP_Models_for_Non-Blind_Image_Deblurring_CVPR_2021_paper.html)
- UNVERIFIED: whether the inner solve is closed-form FFT (spatially-variant weights generally prevent a pure FFT solve, so it is probably CG or iterative). Numbers not retrieved.

**INFWIDE: Zhang, Cheng, Suo, Bian, Dai, IEEE TIP 2023 (arXiv 2207.08201)**
- Two branches. Image space: a ResUNet "enhancement" step estimates a clean, unclipped blurry image, then Wiener deconvolution gives x̂1. Feature space: Wiener on feature maps, then the FSRM refine module gives x̂2. A multi-scale cross residual fusion module (XRFM) fuses them. The NSR = sigma_n²/sigma_s² comes from image statistics (not learned). — [ar5iv 2207.08201](https://ar5iv.labs.arxiv.org/html/2207.08201)
- There is NO data-consistency step at inference. A "reblurring loss" ||x̂ * k − y_gt-blur||_1 is used only in training and adds about 0.23 dB (INFWIDE) and about 0.13 dB (DWDN). — same source
- Data: NightShot (3,000 train / 200 test), Schmidt kernels 13–35 px, a physics-based low-light noise model (Poisson shot, dark current, Gaussian read noise, row streaks), saturation ×1.2 then clipping. — same source
- NightShot test, K = 4 / 8 / 16: INFWIDE 26.51 / 25.45 / 24.14 against DWDN 25.74 / 24.82 / 23.60. Hu et al. (154 images): INFWIDE 29.76 / 28.67 / 27.57 against DWDN 28.78 / 27.89 / 26.95. There are no USRNet or DPIR baselines. — same source

**MultiWienerNet: Yanny, Monakhova, Shuai, Waller, Optica 2022, "Deep learning for fast spatially varying deconvolution"**
- Multiple differentiable Wiener filters, initialised from measured PSFs, each deconvolve the image. A CNN fuses the intermediates to handle spatially-varying (aberration) blur. It is 625–1600× faster than iterative spatially-varying methods and beats shift-invariant DL. — [project page](https://waller-lab.github.io/MultiWienerNet); [AMiner](https://www.aminer.cn/pub/627cdac95aee126c0f4a4729/deep-learning-for-fast-spatially-varying-deconvolution)
- Known limitation: it depends on how representative the initial PSFs are. — [arXiv 2403.06439](https://arxiv.org/html/2403.06439v1)

**DiffPIR: Zhu et al., CVPRW (NTIRE) 2023**
- At each diffusion step the model predicts x0^(t), then solves the closed-form data proximal x̂0 = argmin ||y − Hx||² + rho_t ||x − x0^(t)||². For deblurring this is x0 = F^-1[(conj(F(k))F(y) + rho_t F(z0)) / (|F(k)|² + rho_t)] (App. B.2, Eq. 28), with rho_t = lambda (sigma_n / sigmā_t)². Structurally this is "network predicts x0, then one closed-form DC step with rho tied to sigma", repeated over sampling steps and training-free. — [ar5iv 2305.08995](https://ar5iv.labs.arxiv.org/html/2305.08995)
- Test: FFHQ / ImageNet 256², 61×61 Gaussian (std 3) and motion kernels, sigma_n = 0.05. PSNR on FFHQ Gaussian: DiffPIR 27.36 vs DPIR 27.79. On FFHQ motion: 26.57 vs 26.41. On ImageNet motion: 24.01 vs 23.60. — same source

**Other Wiener hybrids (snippet-level only)**
- NBD-GAP (2022): Wiener-filtered pairs of blurry images make non-blind deblurring a denoising problem that needs no clean targets, and it targets unseen images and kernels. — [arXiv 2209.09498](https://arxiv.org/pdf/2209.09498)
- "Noise Blind Deep Residual Wiener Deconvolution Network" (Digital Signal Processing, 2025): reformulates Wiener deconvolution as two successive operations, deconvolution then denoising. Numbers not retrieved. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1051200425003264)
- "Deep SNR-estimation for Wiener filter-based non-blind deconvolution" (2025): a network estimates the Wiener SNR term. This is relevant to learning the regulariser, but details are UNVERIFIED. — [ResearchGate](https://www.researchgate.net/publication/389677452_Deep_SNR-estimation_for_Wiener_filter-based_non-blind_deconvolution)
- "The Secrets of Non-Blind Poisson Deconvolution" (2023): reports that decoupling deconvolution with a Wiener step using the known kernel beats joint deconvolve-and-denoise networks. Snippet only. — [arXiv 2309.03105](https://arxiv.org/html/2309.03105)

### Inferences
- The proposed DC formula, learning rho as a function of sigma, and end-to-end training through the FFT step are each individually already in USRNet (H(sigma, s) → alpha_k, end-to-end, s = 1 supported). The DC formula with rho tied to sigma is in DPIR (fixed schedule) and DiffPIR (x0-prediction then closed-form proximal). None of these components alone can be claimed as novel.
- What remains distinguishing is the combination: (a) a strong single-shot restorer (NAFNet) conditioned on (y, k, sigma) does all the prior work, (b) exactly ONE closed-form DC step comes LAST, so the output is the DC projection rather than a prior/denoiser output (unlike USRNet and DPIR, which end in the denoiser), and (c) a dual loss on post-DC and pre-DC outputs. In MRI terms this is a single CNN + DC cascade (n_c = 1) of Schlemper type, transplanted to deblurring with a kernel-conditioned modern backbone. That is a defensible but incremental framing. It should be cited as "USRNet with K = 1 and order reversed / Schlemper-style DC head".
- Expect a reviewer to ask for direct comparisons with USRNet run at s = 1 (an official checkpoint exists for s = 1), DPIR and DWDN, and an ablation that removes the DC step and fixes versus learns rho.

### Gaps
- Xu et al. 2014, Ren et al. 2018 (generalized low-rank), FDN's exact formula and noise conditioning, and Zhang 2017's FFT usage: not verified in primary text.
- No retrieved paper explicitly matches "NAFNet/Restormer + Wiener/FFT DC" for non-blind deblurring. The searches "Wiener NAFNet" and "FFT deconvolution transformer non-blind" found no hits. This is absence of evidence from a limited search, not proof of absence.
- No deblurring PSNR for USRNet at s = 1 appears in the USRNet paper itself. Third-party numbers were not retrieved (a third-party benchmark repo exists: [non-blind-deconvolution-benchmark](https://github.com/OlgaChaganova/non-blind-deconvolution-benchmark), not read).

## Q2. Which methods put a DC/projection step AFTER the network, and who learns rho as a function of sigma?

### Takeaway
DC-after-network appears in unrolled and PnP schemes (USRNet, DPIR, FDN-style stages, DiffPIR) and in MRI cascades (Schlemper). Of these, only DiffPIR and Schlemper end with a DC step after a prediction, and in the deblurring ones the final output comes from the network rather than DC. Methods that learn rho as a function of sigma: USRNet (an MLP), and Schlemper (lambda = q/sigma, trainable). DPIR and DiffPIR set rho from sigma analytically. The Wiener-first family (Schuler, Son & Lee, DWDN, INFWIDE, MultiWienerNet) applies the closed-form step BEFORE the network, with an NSR estimated from statistics.

### Cited Findings
- USRNet: D → P alternation over K = 8 steps, with the output after the final P. alpha_k and beta_k = H(sigma, s) are learned end-to-end. — [USRNet](https://arxiv.org/abs/2003.10428)
- DPIR: alpha_k = lambda sigma²/sigma_k², fixed lambda = 0.23, denoiser trained separately (PnP). — [DPIR](https://arxiv.org/html/2008.13751v2)
- DiffPIR: x0 prediction then closed-form proximal with rho_t = lambda (sigma_n/sigmā_t)², lambda hand-tuned at about 7–12. — [DiffPIR](https://ar5iv.labs.arxiv.org/html/2305.08995)
- Schlemper et al. 2017/18 (IEEE TMI, deep cascade of CNNs for dynamic MR): the DC layer is s_rec(j) = s_cnn(j) for j ∉ Ω and (s_cnn(j) + lambda s0(j))/(1 + lambda) for j ∈ Ω. Here "lambda = q/sigma must be adjusted accordingly". Lambda "can be fixed or made trainable". In the noise experiments it was initialised at 0.025, trained, and rose to about 0.5. Cascade D5-C10, trained end-to-end through DC. — [arXiv 1704.02422 HTML](https://arxiv.org/html/1704.02422v2)
- The Schlemper DC layer is an analogue of the proposed DC step (k-space masking playing the role of |K|², and lambda ~ 1/sigma). — same source
- DWDN and INFWIDE: Wiener first, NSR from feature or image statistics. INFWIDE has only a training-time reblur loss, no inference DC. — [DWDN](https://arxiv.org/html/2103.09962v2); [INFWIDE](https://ar5iv.labs.arxiv.org/html/2207.08201)

### Inferences
- The closest "network then single DC with sigma-dependent rho" precedents are (i) Schlemper's DC layer (MRI, trainable lambda ~ q/sigma), (ii) DiffPIR's per-step proximal (training-free), and (iii) USRNet's D module (learned alpha(sigma), but placed before the denoiser). The proposal is a straightforward deblurring instance of the MRI "CNN + DC" block with n_c = 1.

### Gaps
- MoDL (Aggarwal et al., TMI 2019) and VarNet (Hammernik et al., MRM 2018; E2E-VarNet, Sriram et al. 2020) were not retrieved. From background knowledge (UNVERIFIED): MoDL uses a CG-solved DC with a learned lambda shared across iterations, and VarNet uses gradient-step DC with learned step sizes. Both are conceptual precedents.
- KerUnc's and Dong 2021's inner solvers were not verified.

## Q3. Do any test unseen kernel families (defocus, ring/annulus, double exposure) relative to training?

### Takeaway
No retrieved method reports a held-out kernel-family protocol. DWDN, INFWIDE and FDN train and test on Schmidt/Levin/Sun-type motion kernels. USRNet trains on Gaussian + motion kernels. DPIR tests on two Levin motion kernels. Robustness tests cover inaccurate kernels, saturation and JPEG, not new families. The proposal's unseen-family test is therefore a plausible differentiator, with the caveat that the search was limited.

### Cited Findings
- DWDN train and test kernels both come from Schmidt-style synthetic motion kernels and the Levin/Sun 8 kernels. No explicit unseen-family test appears in the portion read. — [DWDN v2](https://arxiv.org/html/2103.09962v2)
- DWDN is reported to be robust to inaccurate kernels, by about 0.7 dB over a prior method, even when trained with GT kernels. — snippet via [search on DWDN](https://arxiv.org/pdf/2103.09962)
- INFWIDE: train and test both use Schmidt random kernels (NightShot), plus the Hu et al. 11 kernels. — [INFWIDE](https://ar5iv.labs.arxiv.org/html/2207.08201)
- USRNet: train on anisotropic Gaussian + Boracchi-Foi motion kernels. Test on 12 kernels (Gaussian + 4 motion from Levin/Boracchi). — [USRNet](https://arxiv.org/abs/2003.10428)
- Son & Lee add a weight regulariser explicitly to improve robustness to untrained or inaccurate kernels. — [POSTECH](https://remotecenter.postech.ac.kr/handle/2014.oak/42195)
- NBD-GAP targets generalisation to unseen images and kernels. — [arXiv 2209.09498](https://arxiv.org/pdf/2209.09498)
- A survey snippet notes that many deep models degenerate to near-identity on out-of-domain blur operators. — via [search result list](https://arxiv.org/pdf/2504.14664) (UNVERIFIED attribution)
- Schuler 2013 demonstrates a real out-of-focus image, which is qualitative only. — [CVF](https://openaccess.thecvf.com/content_cvpr_2013/html/Schuler_A_Machine_Learning_2013_CVPR_paper.html)

### Inferences
- Because USRNet's and DPIR's data module is parameter-free, they are argued to generalise better ("complete decoupling between data term and prior term" — USRNet). This is the main argument for the proposed DC step on unseen families, and the proposal should test it directly against USRNet and DPIR on ring, defocus and double-exposure kernels.

### Gaps
- No quantitative cross-family (e.g. motion → disk/annulus) result was found for any Wiener hybrid.

## Q4. Reported gains over USRNet, DPIR and DWDN, and on which benchmarks

### Takeaway
Within this family, DWDN beats FDN, IRCNN and FNBD by about 2–3 dB on Levin (e.g. 36.90 vs 34.81 dB at 1%) but never compares to USRNet or DPIR. INFWIDE beats DWDN by about 0.5–1 dB on low-light sets, also without USRNet or DPIR. DPIR beats IRCNN and FDN on Set6. DiffPIR roughly ties DPIR on PSNR. No retrieved hybrid reports a head-to-head gain over USRNet in non-blind deblurring.

### Cited Findings
- Levin, DWDN+ vs DWDN vs FNBD vs IRCNN at 1%: 37.27 / 36.90 / 34.81 / 34.33. At 5%: 31.08 / 30.77 / 27.93 / 28.51. — [DWDN v2](https://arxiv.org/html/2103.09962v2)
- Sun 1%: DWDN+ 34.62, DWDN 34.05, IRCNN 33.57, FDN 32.63. BSD-100 1%: DWDN+ 32.02, DWDN 31.74, FNBD 30.92, IRCNN 30.63. — same
- INFWIDE vs DWDN, NightShot K = 4: 26.51 vs 25.74. Hu et al. K = 4: 29.76 vs 28.78. — [INFWIDE](https://ar5iv.labs.arxiv.org/html/2207.08201)
- DPIR vs IRCNN, Set6 at 1% with the 17×17 kernel: for example Leaves 35.19 vs 33.51, Butterfly 34.26 vs 33.13. — [DPIR](https://arxiv.org/html/2008.13751v2)
- DiffPIR vs DPIR (sigma 0.05): FFHQ Gaussian 27.36 vs 27.79, ImageNet motion 24.01 vs 23.60. — [DiffPIR](https://ar5iv.labs.arxiv.org/html/2305.08995)
- Köhler and GoPro do not appear in the retrieved tables for these methods. GoPro is a blind-dynamic-scene benchmark; DWDN includes DMPHN, a GoPro-trained method, as a baseline only. — [DWDN v2](https://arxiv.org/html/2103.09962v2)

### Inferences
- A clean, same-protocol comparison of USRNet (s = 1), DPIR and DWDN on Levin, Sun and BSD at 1/3/5% noise appears to be missing from the primary papers. The proposal can supply it. Since DWDN at Levin 1% (36.90) exceeds typical PnP numbers, DWDN is likely the strongest baseline on Levin-style motion kernels. UNVERIFIED: no direct DWDN-vs-DPIR table was found.

### Gaps
- Köhler benchmark numbers for this family were not found.
- 2024–2026 NAFNet/Restormer + Wiener non-blind methods: none found. The 2025 DSP residual-Wiener paper and the 2025 deep-SNR paper are the only recent hits, and their numbers were not retrieved.
- Third-party USRNet deblurring numbers were not retrieved.
