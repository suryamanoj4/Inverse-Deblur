# Deep unrolled / unfolded optimisation and plug-and-play (PnP) methods for non-blind deblurring, 2016–2026

Proposed method under novelty check: unrolled HQS, exact closed-form FFT data step at every stage, ONE shared-weight NAFNet prior, FiLM-conditioned on (kernel embedding, sigma, stage index), T in {1,2,4,8}.

Verification legend:
- **[V]** = verified this session from the primary paper's full text (pdftotext of the arXiv PDF) or from its arXiv abstract/HTML.
- **[S]** = taken from a search-engine summary or secondary page, not checked against the primary text.
- **[U]** = from prior knowledge, NOT verified this session. The URL is the paper's canonical arXiv/venue page as best known; check before citing.

## Q1. Catalogue: splitting scheme, prior architecture, weight sharing, conditioning, stages, budget, data, numbers vs USRNet/DPIR

### Takeaway
The USRNet/DPIR template is: HQS + closed-form FFT data step + a U-Net prior (ResUNet/DRUNet) shared across stages and conditioned only by concatenating a constant noise-level map. Stage dependence enters through scalar hyper-parameters (USRNet: an MLP of (sigma, scale) gives per-stage alpha_k, beta_k; DPIR: a hand-set log-spaced sigma_k schedule). No catalogued deblurring method feeds the kernel into the prior. Later unfolding papers mostly compare against DWDN/FDN/IRCNN rather than USRNet/DPIR, and they report on bespoke test sets. That makes head-to-head numbers scarce.

### Cited Findings

**USRNet (Zhang, Van Gool, Timofte, CVPR 2020): the main template [V]**
- Unfolds HQS into K iterations, alternating a data module D (closed-form FFT solution) and a prior module P. "K is empirically set to 8" — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428)
- Weight sharing: "in order to reduce the number of parameters, all the prior modules share the same architecture and same parameters." — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428)
- Prior is ResUNet: a U-Net with 4 scales (64/128/256/512 channels), 2 residual blocks per scale, 2x2 strided conv down and transposed conv up. "ResUNet takes the concatenated z_k and noise level map as input". The kernel is not an input to the prior. — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428)
- Stage/noise conditioning is a hyper-parameter module H: [alpha, beta] = H(sigma, s), with 3 FC layers (64 hidden, ReLU, ReLU, Softplus). It outputs per-stage scalars that weight the data step (alpha_k) and set the prior's noise-map level (beta_k). The authors call it a "slide bar". It is in effect a tiny hypernetwork that emits scalars, not features. — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428)
- Training: DIV2K + Flickr2K; anisotropic Gaussian kernels plus motion kernels (following Boracchi & Foi); L1 loss; Adam, batch 128, 96x96 HR patches, LR 1e-4 halved every 4e4 iterations down to 3e-6; 4x V100 GPUs, "about two days". — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428)
- Ablation on number of stages (closest existing T ablation): a 5-iteration USRNet loses "about 0.1dB on Gaussian blur kernels and 0.3dB on motion blur kernels" against K=8. No shared-vs-unshared ablation is reported. — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428)
- Claims at least about 1 dB average PSNR gain over IRCNN (30 iterations) despite using far fewer iterations. Evaluation is SR-centred; s=1 (deblurring) is supported by the same model. — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428)

**DPIR (Zhang, Li, Zuo, Zhang, Van Gool, Timofte; TPAMI 2021; extension of IRCNN): the PnP reference [V]**
- The DRUNet prior is plugged into HQS. DRUNet is a bias-free U-Net + ResNet with 4 scales that "takes an additional noise level map as input". The noise map is a concatenated constant channel, and DRUNet is a Gaussian denoiser that is not trained for deblurring. — [arXiv 2008.13751](https://arxiv.org/abs/2008.13751)
- Deblurring settings: K = 8 iterations, sigma_K = sigma, z_0 = y. sigma_k is log-spaced from 49 down to sigma, and mu_k = lambda / sigma_k^2 with lambda fixed to 0.23 (range [0.19, 0.55] found by grid search). The data step is closed-form FFT. — [arXiv 2008.13751](https://arxiv.org/abs/2008.13751)
- Training data for the denoiser: 400 BSD + 4,744 Waterloo Exploration + 900 DIV2K + Flickr2K. LR 1e-4 halved every 100k iterations, 16 patches per batch. — [arXiv 2008.13751](https://arxiv.org/abs/2008.13751)
- Deblurring test set: "Set6" (3 grayscale + 3 color images), using two of the 8 Levin kernels (17x17 and 27x27) at noise sigma = 2.55 (1%) and 7.65 (3%). Compared against EPLL, FDN, DMPHN, IRCNN and IRCNN+, not USRNet. "Although DPIR has 8 iterations rather than 30 iterations of IRCNN, it has a PSNR gain of 0.2dB~2dB over IRCNN". Against IRCNN+ (the same 8-iteration HQS but with the IRCNN denoiser), DPIR is significantly better, "which indicates that the denoiser plays a vital role in plug-and-play IR". This is direct evidence that prior strength matters at a fixed T. — [arXiv 2008.13751](https://arxiv.org/abs/2008.13751)
- Ablation inside DRUNet: removing the noise-level-map input lowers PSNR, and bias-free design improves generalisation to unseen noise levels. — [arXiv 2008.13751](https://arxiv.org/abs/2008.13751)

**IRCNN (Zhang, Zuo, Gu, Zhang, CVPR 2017) [U except as cited]**
- HQS PnP with a bank of 25 small dilated-CNN denoisers (one per noise interval), 30 iterations for deblurring. No kernel conditioning. DPIR describes itself as "an extended version of IRCNN (CVPR17)". — [arXiv 1704.03264](https://arxiv.org/abs/1704.03264) [U]; lineage statement [V] from [arXiv 2008.13751](https://arxiv.org/abs/2008.13751)

**Eboli, Sun, Ponce (ECCV 2020): "End-to-end interpretable learning of non-blind image deblurring" [V]**
- HQS with 1 to 5 stages. The x-update is not FFT. Instead it uses preconditioned Richardson fixed-point iterations (CPCR), whose preconditioner is the approximate inverse filter of the known kernel. A CNN embeds this, so the kernel enters through the data-step filters, not the prior. The proximal step is a learned CNN. — [arXiv 2007.01769](https://arxiv.org/abs/2007.01769); [ECVA PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620307.pdf)
- The authors claim CPCR beats FFT and CG in accuracy and/or speed, and handles boundaries better than FFT with edgetaper. Training uses BSD500 images and random 41x41 synthetic kernels. They evaluate on uniform and non-uniform blur and claim robustness to high noise. — [arXiv 2007.01769](https://arxiv.org/abs/2007.01769)
- Note for novelty: Eboli argues against the exact FFT data step, which the proposed method relies on, because of boundary artefacts.

**DWDN (Dong, Roth, Schiele, NeurIPS 2020; TPAMI extension) [S/U]**
- Wiener deconvolution in a learned feature space, followed by a multi-scale encoder-decoder refinement. It estimates noise from the blurry features. The kernel enters through feature-space Wiener deconvolution, not unrolled HQS. — [arXiv 2103.09962](https://arxiv.org/abs/2103.09962) [S]
- It is used as the main non-blind baseline by DECUN and other unrolling papers (see below). — [arXiv 2402.12872](https://arxiv.org/html/2402.12872) [V]

**DECUN: "Deep, convergent, unrolled HQS for image deconvolution" (Zhao, Li, Zhang, Monga, Eldar; IEEE TCI, accepted 2024) [V]**
- Unrolled HQS. The prior is not a deep network: each layer has C=4 learned filters plus soft-thresholding. Weights are partly shared: D_l = D_bar + xi_l E_l, with shared D_bar and beta_bar and per-layer residuals pushed to fixed points so that convergence holds. L=30 layers, with an ablation over 10/15/20/25/30 showing diminishing gains after 30. 1,117 parameters. No noise, kernel or stage conditioning. — [arXiv 2402.12872](https://arxiv.org/html/2402.12872)
- Data: 800 MS-COCO images; 300 linear motion kernels or 10k nonlinear kernels; noise 1%. Tested on BSD500 (100 images; unseen linear and 100 unseen nonlinear kernels) and VOC2012. PSNR on BSD linear: 33.24 vs DWDN 31.70. BSD nonlinear: 30.27 vs DWDN 30.22. VOC nonlinear: 31.36 vs DWDN 31.20. No USRNet or DPIR comparison. — [arXiv 2402.12872](https://arxiv.org/html/2402.12872)
- Correction: one search summary claimed a "HQS unrolling paper analysing NAFNet layer weights". In the DECUN full text, NAFNet does not appear. Treat that snippet as a hallucinated or misattributed search summary. — [arXiv 2402.12872](https://arxiv.org/html/2402.12872)

**DUBLID (Li, Tofighi, Geng, Monga, Eldar; "Deep algorithm unrolling for blind image deblurring", IEEE TSP 2020) [S]**
- Blind, not non-blind: it unrolls a total-variation (TV) style blind deconvolution, with learned filters, kernel estimation and a final non-blind step. It is listed among examples of unrolled HQS. — [arXiv 1902.03493](https://arxiv.org/abs/1902.03493)

**Photon-limited non-blind deblurring via algorithm unrolling (Sanghvi, Gnanasambandam, Chan; IEEE TCI 2022) [S]**
- Unrolled PnP-ADMM for Poisson noise. On Levin (photon-limited) it reports 23.48 dB vs DPIR 22.09, DWDN 22.85, PURE-LET 20.88, RGDN 19.80. USRNet is not reported. The search summary wrongly labelled this "DUBLID". — [ResearchGate](https://www.researchgate.net/publication/355730567_Photon_Limited_Non-Blind_Deblurring_Using_Algorithm_Unrolling)

**DSDNet: "Efficient non-blind image deblurring with discriminative shrinkage deep networks" [S]**
- Listed on ResearchGate with an HQS-style learned-shrinkage design. Architecture, numbers and venue were not verified this session. — [ResearchGate](https://www.researchgate.net/publication/390150849_Efficient_Non-Blind_Image_Deblurring_with_Discriminative_Shrinkage_Deep_Networks)

**Convergent PnP with provable denoisers [S]**
- Hurault, Leclaire, Papadakis, "Gradient Step Denoiser for convergent PnP" (ICLR 2022). The denoiser is D = I - grad g_theta, with g_theta parameterised by a DRUNet-like network. On CBSD10 deblurring it "closely follows DPIR in terms of PSNR for low noise level but performs equally or better for higher noise levels". The authors note DPIR uses 8 HQS iterations with fast-decreasing sigma and "can asymptotically diverge". — [arXiv 2110.03220](https://arxiv.org/abs/2110.03220); [GitHub GSPnP](https://github.com/samuro95/GSPnP)
- Prox-DRUNet, "Proximal denoiser for convergent PnP" (ICML 2022): DPIR's PSNR degrades over long runs (2000 iterations on "starfish"), whereas the provable variant is stable. — [arXiv 2201.13256](https://arxiv.org/abs/2201.13256); [PMLR](https://proceedings.mlr.press/v162/hurault22a/hurault22a.pdf)
- Primal-dual PnP (2025): DRUNet is not firmly nonexpansive, yet "achieves higher PSNR values than the proposed method in deblurring at certain noise levels". — [arXiv 2501.03780](https://arxiv.org/abs/2501.03780)
- A 2026 contractive-denoiser paper claims it "often surpasses ... DnCNN and Restormer in deblurring" within PnP. DRUNet is absent from that comparison. — [arXiv 2603.26168](https://arxiv.org/html/2603.26168v1)

**Deep equilibrium (DEQ) [U]**
- Gilton, Ongie, Willett, "Deep equilibrium architectures for inverse problems in imaging" (IEEE TCI 2021). It runs PnP/proximal-gradient/ADMM-style iterations to a fixed point with a single shared network, so the effective T is unbounded. Deblurring is among its tasks. Its numbers vs USRNet/DPIR were not found. — [arXiv 2102.07944](https://arxiv.org/abs/2102.07944) [U]
- "Deep equilibrium learning of explicit regularizers" (2023) is a related DEQ line that cites GS-PnP. — [arXiv 2303.05386](https://arxiv.org/abs/2303.05386) [S]

**RED / DeepRED [U]**
- Romano, Elad, Milanfar, "RED" (SIAM J. Imaging Sci. 2017): the regulariser is x^T (x - D(x)). Solvers include steepest descent, ADMM and fixed-point. — [arXiv 1611.02862](https://arxiv.org/abs/1611.02862) [U]
- Mataev, Elad, Milanfar, "DeepRED" (ICCVW 2019): RED combined with a Deep Image Prior. — [arXiv 1903.10176](https://arxiv.org/abs/1903.10176) [U]

**Other kernel-aware non-blind networks (not unrolled HQS)**
- Ren et al., "Deep non-blind deconvolution via generalized low-rank approximation" (NeurIPS 2018): uses the kernel's low-rank decomposition (SVD of the pseudo-inverse) to initialise network weights, so one network handles arbitrary kernels. This is a form of kernel conditioning through weights. — [NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/0aa1883c6411f7873cb83dacb17b0afc-Paper.pdf) [S]
- Vasu, Maligireddy, Rajagopalan (CVPR 2018): trained on noisy or estimated kernels to handle kernel error. — [CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Vasu_Non-Blind_Deblurring_Handling_CVPR_2018_paper.pdf) [S]
- A 2025 frequency-domain method fuses kernel information into a deblurring net in the Fourier domain. It is blind/kernel-prior based and is the closest search hit to "prior conditioned on the kernel", but it is not SFT or FiLM. — [arXiv 2504.14664](https://arxiv.org/html/2504.14664v1) [S]
- Fang et al., "A robust non-blind deblurring method using deep denoiser prior" (CVPRW/NTIRE 2022): HQS PnP with a texture-preserving denoiser. — [CVF](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Fang_A_Robust_Non-Blind_Deblurring_Method_Using_Deep_Denoiser_Prior_CVPRW_2022_paper.pdf) [S]
- Event-guided deblurring (arXiv 2608.24223, 2026): an HQS unfolding "inspired by USRNet and DPIR". It is trained on perturbed kernels and uses under 1M parameters. It compares to USRNet-tiny on estimated kernels (26.35 dB), not on the standard Levin protocol. — [arXiv 2608.24223](https://arxiv.org/html/2608.24223) [S]

### Inferences
- Within non-blind deblurring, the proposed method's skeleton (HQS, FFT data step, one shared U-Net prior, noise map, T around 8) is exactly USRNet at s=1. The novelty has to come from: (a) the NAFNet backbone, (b) FiLM conditioning, and especially (c) feeding a kernel embedding into the prior. USRNet's H is the nearest precedent for (sigma, stage) conditioning, but it emits only per-stage scalars, not feature-wise FiLM.
- Reported comparisons are fragmented. DPIR used Set6 with 2 Levin kernels, USRNet used SR benchmarks, and DECUN used BSD/VOC against DWDN. Any "beats USRNet/DPIR" claim will need the authors to re-run both under one protocol.

### Gaps
- Primary-text verification was not completed for DWDN, Gilton DEQ, IRCNN, DUBLID, DSDNet, KXNet, DCDicL, ProxUnroll, "Unfolded deep kernel estimation", and pixel-adaptive weight methods. Downloads timed out or the call budget ran out, so these entries are [U] or [S].
- A curated 2022–2026 leaderboard of unfolding methods with Levin/Sun/BSD68 numbers vs both USRNet and DPIR was not found. No paperswithcode leaderboard for non-blind deblurring was located.
- KXNet (blind SR, ECCV 2022, kernel-explicit unfolding) and DCDicL (deep convolutional dictionary learning, CVPR 2021) are known from memory as unfolding precedents. Their details were not verified. — KXNet [arXiv 2209.10305](https://arxiv.org/abs/2209.10305) [U]

## Q2. Has anyone used NAFNet/NAFBlocks as the prior inside unrolled HQS/ADMM for deblurring? Has anyone conditioned the unrolled prior on the KERNEL via FiLM/SFT/attention?

### Takeaway
No paper was found that uses NAFNet as the prior in unrolled HQS/ADMM for non-blind deblurring. The one apparent hit was a misattributed search snippet, which DECUN's full text disproves. No non-blind deblurring unfolding work was found that conditions the prior on the kernel via FiLM/SFT. Kernel-conditioned modulation does have clear precedents in blind SR (SFT/kernel-PCA). In spectral compressive imaging, DAUHST conditions each stage on estimated degradation parameters, but they are scalars and enter through a noise level.

### Cited Findings
- The proposed method's backbone: NAFNet (Chen et al., ECCV 2022), an activation-free U-Net with SimpleGate and simplified channel attention. — [arXiv 2204.04676](https://arxiv.org/abs/2204.04676) [U]
- The only NAFNet-plus-HQS hit was a search summary of a "ResearchGate entry" said to analyse NAFNet layer weights. DECUN's full text contains no mention of NAFNet. — [arXiv 2402.12872](https://arxiv.org/html/2402.12872) [V]
- PnP with NAFNet or Restormer denoisers: no head-to-head PnP deblurring study against DRUNet was found. A 2025 remote-sensing PnP paper says NAFNet and Restormer "over-smooth some regions". On its own denoising table, NAFNet trails DRUNet on CBSD68 (22.86 vs 25.99) while Restormer leads (27.09). This is a single domain-specific source. — [MDPI Remote Sensing 2025](https://www.mdpi.com/2072-4292/18/18/3171) [S]
- USRNet and DPIR condition the prior only on a noise map. Neither feeds the kernel to the prior: the kernel enters only through the FFT data step. — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428); [arXiv 2008.13751](https://arxiv.org/abs/2008.13751) [V]
- Mechanism precedent from another inverse problem, spectral compressive imaging (CASSI). DAUHST (Cai et al., NeurIPS 2022) has an estimator E that maps (y, Phi) to per-stage alpha_k, beta_k, which are "fed into each stage". They "adaptively scale the linear projection and provide the noise level information for the denoising network". Ablation: replacing the estimated alpha with learnable parameters hurts. beta_k grows with iterations as noise falls. The prior is a half-shuffle transformer. — [arXiv 2205.10102](https://arxiv.org/abs/2205.10102); [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f621c2ead473ca36763696b712ffda01-Abstract.html) [S, consistent across arXiv/NeurIPS pages]
- Mechanism precedents from SR, not verified this session:
  - SRMD (CVPR 2018) concatenates a PCA-projected kernel map with the input. — [arXiv 1712.06116](https://arxiv.org/abs/1712.06116) [U]
  - IKC (CVPR 2019) injects a kernel code via SFT layers. — [arXiv 1904.03377](https://arxiv.org/abs/1904.03377) [U]
  - DAN (NeurIPS 2020) unfolds alternating kernel/image estimation, with the kernel code conditioning the restorer. — [arXiv 2010.02631](https://arxiv.org/abs/2010.02631) [U]
  - KXNet (ECCV 2022) is an explicit-kernel unfolding for blind SR. — [arXiv 2209.10305](https://arxiv.org/abs/2209.10305) [U]

### Inferences
- "Shared NAFNet prior + FiLM on (kernel embedding, sigma, stage)" inside exact-FFT HQS for non-blind deblurring appears unclaimed in what was searched. Its individual ingredients are not new:
  - shared prior with an FFT data step: USRNet
  - stage/sigma hyper-conditioning: USRNet's H, DAUHST
  - kernel-code modulation: IKC/SRMD/DAN in SR
- The defensible novelty is the combination, applied to non-blind deblurring: kernel-aware feature modulation of the prior, plus an ablation over T in {1,2,4,8}. Reviewers will likely cite IKC/DAN (kernel-SFT) and DAUHST (degradation-aware unfolding) as precedents. These should be acknowledged as mechanism precedents.

### Gaps
- Not exhaustively searched: arXiv 2023–2026 for "NAFBlock unfolding", "kernel-conditioned unrolled", or "FiLM deep unfolding" in MRI/CT. Learned primal-dual and hypernetwork-MRI variants (e.g., per-stage hypernetworks) are known to exist but were not verified here.
- Whether any 2024–2026 "transformer unfolding for deblurring" (Restormer-in-the-loop) paper conditions on the kernel was not found. Absence of evidence is not proof of absence.

## Q3. More stages vs a stronger prior; shared vs unshared weights

### Takeaway
The few published data points suggest returns from more stages diminish quickly once T is around 5–8 with a strong prior, while a stronger prior at a fixed T gives large gains. No catalogued deblurring paper reports a clean shared-vs-unshared ablation with a deep prior.

### Cited Findings
- USRNet: going from K=8 to K=5 costs only about 0.1 dB (Gaussian kernels) and about 0.3 dB (motion kernels). — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428) [V]
- USRNet with 8 iterations beats IRCNN with 30 iterations by at least about 1 dB on average, so a stronger, end-to-end-trained prior outweighs extra iterations. — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428) [V]
- DPIR vs IRCNN+ (same 8-iteration HQS, different denoiser): DPIR is significantly better, "the denoiser plays a vital role in plug-and-play IR". DPIR with 8 iterations also beats IRCNN with 30 iterations by 0.2–2 dB. — [arXiv 2008.13751](https://arxiv.org/abs/2008.13751) [V]
- DPIR shows the sigma_k/alpha_k schedules for K = 8, 24 and 40 (Fig. 5). It uses K=8 for deblurring. — [arXiv 2008.13751](https://arxiv.org/abs/2008.13751) [V]
- DECUN (weak filter prior): gains diminish after L=30 layers. Its partial sharing (shared base plus per-layer residual) is motivated by convergence: per-layer-learned parameters lose the iterative algorithm's convergence. — [arXiv 2402.12872](https://arxiv.org/html/2402.12872) [V]
- Eboli et al. vary 1–5 HQS stages. Per-stage numbers were not extracted this session. — [arXiv 2007.01769](https://arxiv.org/abs/2007.01769) [V for the stage range]
- PnP long-run behaviour: DPIR's PSNR peaks at a few iterations and degrades over long runs, while convergent GS/Prox denoisers stay stable. Short-T truncation is part of why DPIR performs well. — [arXiv 2201.13256](https://arxiv.org/abs/2201.13256); [arXiv 2110.03220](https://arxiv.org/abs/2110.03220) [S]

### Inferences
- A T in {1,2,4,8} sweep with a fixed strong (NAFNet) prior is a useful contribution in itself. Published evidence on T is thin (USRNet's single 5-vs-8 point). Expect a saturating curve, with motion kernels benefiting more from extra stages than Gaussian kernels, per USRNet.
- USRNet argues that sharing the prior removes the need for intermediate-output losses. A shared-vs-unshared comparison at matched parameters or FLOPs would fill a real gap.

### Gaps
- No paper found that ablates shared vs unshared deep priors in deblurring unfolding. Possibly in MRI unrolling (e.g., MoDL, [arXiv 1712.02862](https://arxiv.org/abs/1712.02862) [U], which shares weights), but this was not verified.

## Q4. Robustness to unseen kernel FAMILIES (not just unseen kernels from the same family)

### Takeaway
Most methods test on unseen kernels drawn from the training family, or on the Levin/Sun real kernels after training on synthetic motion. No paper was found that sets up an explicit held-out-family protocol (e.g., train Gaussian+motion, test on defocus/ring), apart from partial cases.

### Cited Findings
- USRNet trains on anisotropic Gaussian plus motion kernels and tests on a "large variety" of kernels, including 4 isotropic Gaussian kernels. The tests are largely within-family. — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428) [V]
- DPIR's denoiser never sees a kernel (it is a Gaussian denoiser), so it is kernel-family agnostic by construction. It is tested on Levin motion kernels. — [arXiv 2008.13751](https://arxiv.org/abs/2008.13751) [V]
- DECUN trains and tests separately on linear vs nonlinear motion kernels. It reports both settings but no cross-family transfer. — [arXiv 2402.12872](https://arxiv.org/html/2402.12872) [V]
- Eboli et al. train on synthetic 41x41 kernels and test on the Levin real kernels and on non-uniform blur. This is a partial synthetic-to-real and uniform-to-nonuniform shift. — [arXiv 2007.01769](https://arxiv.org/abs/2007.01769) [V]
- Kernel-error robustness, rather than family shift, is addressed by training on noisy or estimated kernels. — [CVF Vasu 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Vasu_Non-Blind_Deblurring_Handling_CVPR_2018_paper.pdf); [arXiv 2608.24223](https://arxiv.org/html/2608.24223) [S]

### Inferences
- A kernel-conditioned prior risks overfitting to the training kernel families. PnP (DPIR) is immune because its prior ignores the kernel. A held-out-family test (the project's protocol v2 with ring/ghost kernels) is therefore both a novelty point and a necessary control: compare the kernel-FiLM model against a sigma-only-FiLM ablation and against DPIR on held-out families.

### Gaps
- No systematic cross-family benchmark for non-blind deblurring was located. Gilton DEQ and the GS-PnP papers may include defocus/Gaussian tests, but this was not verified.
