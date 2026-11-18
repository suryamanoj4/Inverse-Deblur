# Robustness of learned non-blind deblurring to kernel distribution shift, OOD kernels and kernel errors; evaluation protocols; GMM/EPLL-deep hybrids (2011-2026)

Scope note: about 18 search/fetch calls. Items marked **[UNVERIFIED]** come from background knowledge and were not confirmed against a primary source in this session. They are kept out of Cited Findings, or flagged where they appear.

## Q1. How do data-driven, physics-based and unrolled deblurring methods degrade on kernels outside the training distribution, and under kernel error?

### Takeaway
It is well established, and often stated, that supervised end-to-end deblurring networks generalize poorly to blur unlike their training blur, while methods that take the kernel as an explicit input (classical MAP/EPLL, plug-and-play, unrolled) are reported as more robust to *kernel type* shift. Robustness to *kernel error* (a wrong kernel) is a separate thread, handled by Vasu 2018 and Nan & Ji 2020. I found no paper that quantifies, in one controlled study, the gap between an end-to-end net (e.g. NAFNet), an unrolled net (USRNet) and an EPLL/ADMM baseline on matched in-distribution vs held-out-family kernels. So the project's "~2.8 dB vs ~0.4 dB" observation fits the literature qualitatively, but I did not find it published as a measured, controlled comparison.

### Cited Findings
- **Kernel error / uncertainty (non-blind):**
  - Nan & Ji (CVPR 2020), "Deep Learning for Handling Kernel/model Uncertainty in Image Deconvolution". They point out that most non-blind methods assume an error-free kernel, while real kernels are blind estimates. They use an error-in-variables model and unroll a total-least-squares estimator with learned priors. They report robustness to kernel and model error and gains on kernels estimated by blind methods. — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Nan_Deep_Learning_for_Handling_Kernelmodel_Uncertainty_in_Image_Deconvolution_CVPR_2020_paper.html); code [GitHub](https://github.com/ysnan/NBD_KerUnc)
  - Vasu, Maligireddy & Rajagopalan (CVPR 2018), "Non-Blind Deblurring: Handling Kernel Uncertainty with CNNs". A conventional non-blind deconvolution step is followed by a CNN that removes artifacts caused by kernel error. Training on noisy kernels generated from "a wide variety of GT kernels enabled the network to generalize better". — [CVF PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Vasu_Non-Blind_Deblurring_Handling_CVPR_2018_paper.pdf); [supp](https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/1792-supp.pdf)
  - Ren et al. (NeurIPS 2018), "Deep Non-Blind Deconvolution via Generalized Low-Rank Approximation". Kernel low-rank decomposition lets one network handle "arbitrary kernels in a unified network". This is a kernel-conditioned design aimed at generalizing across kernels. — [NeurIPS PDF](https://proceedings.neurips.cc/paper/2018/file/0aa1883c6411f7873cb83dacb17b0afc-Paper.pdf)
- **Held-out kernel TYPE evidence (closest prior art to the project's finding):**
  - Chen, Quan, Pang & Ji (IJCV 2022), "Nonblind Image Deconvolution via Leveraging Model Uncertainty in An Untrained Deep Neural Network", Sec. 4.2.2 "Non-Motion Blur Kernels". The authors take supervised models trained on **motion kernels** (FCNN, IRCNN, FDN, DMSP, RGDN, VEM, CPCR, DWDN), plus NNet trained on defocus, and test them **without retraining** on the five non-motion kernels of Danielyan et al. 2011. Test sets are Set12 and Sun et al. Their conclusion: supervised models "do not generalize well when there exist noticeable variations of blur kernels between training and test data". Classical EPLL scored 30.41 dB (Set12) and 32.20 dB (Sun). Their dataset-free method scored 31.10 / 32.51 dB. The best supervised method was 0.14 / 0.22 dB below theirs (from the "Margin" column). So on unseen kernel types, EPLL came within about 0.5 dB of the best supervised net. The comparison is qualitative: no per-image paired statistics, and no in-distribution vs OOD delta per method. — [IJCV 2022 PDF (author-hosted)](https://cpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/8/10877/files/2022/04/IJCV-2022-Deblur.pdf)
  - Blur2Blur (Pham et al., CVPR 2024) states that "supervised method NAFNet fails to handle unseen blurs, often yielding output mostly identical to the blurred inputs". NAFNet trained on GoPro scores 24.78 dB on RB2V_Street; trained on synthetic data it scores 22.10 dB. Blur2Blur adds +2.20 / +2.31 / +2.67 dB (RB2V_Street / REDS / RSBlur). This is a **blind**, real-camera-domain shift, not non-blind with a known kernel. — [arXiv 2403.16205](https://arxiv.org/pdf/2403.16205); [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Pham_Blur2Blur_Blur_Conversion_for_Unsupervised_Image_Deblurring_on_Unknown_Domains_CVPR_2024_paper.pdf)
  - Blur2seq (2025) notes that NAFNet generalizes better when trained on its dataset than on GoPro. Training-blur distribution drives NAFNet's generalization. — [arXiv 2510.20539](https://arxiv.org/pdf/2510.20539)
  - A 2025 blind-deblurring paper argues that kernel priors help OOD generalization because "blur kernels are irrelevant to the content of the image". Its OOD axis is image domain, not kernel family. — [arXiv 2504.14664](https://arxiv.org/html/2504.14664v1)
  - A 2025 unified-deblurring MoE paper builds test-only OOD sets (Real-LOLBlur, RealDOF, RSBlur), so its shift is capture conditions rather than kernel families. — [arXiv 2508.06228](https://arxiv.org/html/2508.06228v2)
- **Unrolled / kernel-conditioned designs:**
  - USRNet (Zhang, Van Gool & Timofte, CVPR 2020) takes the kernel k, scale s and noise level σ as inputs, so one model covers many degradations. I did not find a published USRNet or DPIR robustness study under kernel mismatch. — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Deep_Unfolding_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf)
  - Eboli et al. (ECCV 2020) is a learnable half-quadratic solver, "interpretable", reaching SOTA on uniform and non-uniform kernels. — [ECVA PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620307.pdf)
  - Rosenbaum & Weiss (NeurIPS 2015) note, per a reviewer summary, that a generative-prior approach "avoids the need for re-training for each task", unlike discriminative methods that need retraining "for each blur kernel in de-blurring". This is an early statement of the kernel-generalization advantage of explicit priors. — [NeurIPS reviews](https://papers.neurips.cc/paper_files/paper/2015/file/596f713f9a7376fe90a62abaaedecc2d-Reviews.html)
- **Forward-model shift in inverse problems generally:**
  - Gilton, Ongie & Willett, "Model Adaptation for Inverse Problems in Imaging" (arXiv 2012.00139; IEEE TCI 2021 **[venue UNVERIFIED]**). Learned reconstructions degrade substantially when the test forward model differs even slightly from training. They propose adapting a trained network to a changed operator, including blur-kernel changes with a learnable 7x7 kernel. In Willett's talk slides, one MRI drift case gives 21 dB without adaptation, 30 dB with adaptation and 31 dB with a matched model. That figure is from slides, not the paper. — [arXiv](https://arxiv.org/pdf/2012.00139); [slides](https://ani.stat.fsu.edu/GFDW/TalksPdfs/Rebecca_Willett.pdf)
  - Darestani, Chaudhari & Heckel (ICML 2021), "Measuring Robustness in Deep Learning Based Compressive Sensing". Trained and un-trained methods tuned to a dataset "suffer very similarly from distribution shifts", and both are vulnerable to adversarial perturbations. DL methods improve performance "without compromising robustness" relative to traditional methods. The shift studied is **image/data** distribution, not operator family. — [PMLR](https://proceedings.mlr.press/v139/darestani21a.html); [arXiv](https://arxiv.org/abs/2102.06103)
  - Genzel, Macdonald & März (TPAMI 2022, DOI 10.1109/TPAMI.2022.3148324), "Solving Inverse Problems with Deep Neural Networks – Robustness Included?". They examine robustness of DL for underdetermined inverse problems in response to earlier instability claims. — [arXiv 2011.04268](https://arxiv.org/abs/2011.04268)
  - Antun et al., "On instabilities of deep learning in image reconstruction and the potential costs of AI", PNAS 2020. Confirmed only as a reference entry. — [Bohrium listing of Genzel refs](https://www.bohrium.com/paper-details/solving-inverse-problems-with-deep-neural-networks-robustness-included/813038275026485251-2429). I recall its content as instabilities under tiny perturbations, small structural changes, and changes in sampling. **[content UNVERIFIED this session]**
  - Gandikota, Chandramouli & Moeller (ICIP 2022), "On Adversarial Robustness of Deep Image Deblurring". Small imperceptible perturbations substantially degrade SOTA deblurring nets. Blind deblurring networks were the most vulnerable. Training with noise was not sufficient for robustness in deep Wiener deconvolution. — [arXiv 2210.02502](https://arxiv.org/abs/2210.02502)
  - Super-resolution analogue: Liu, Zhao, Gu, Qiao & Dong, "Evaluating the Generalization Ability of Super-Resolution Networks" (TPAMI). They propose the SRGA index and the PIES benchmark, argue "image quality is not equal to generalization ability", and find that DAN (a blind-SR kernel-estimation method) "can only well generalized within mild degradation region". — [arXiv 2205.07019](https://arxiv.org/pdf/2205.07019)

### Inferences
- Qualitative consensus: end-to-end supervised nets degrade on unseen blur types, while explicit-kernel and prior-based methods degrade less. Rosenbaum & Weiss 2015, Chen et al. IJCV 2022, Blur2Blur 2024 and Gilton et al. 2020 all point this way. The project's NAFNet −2.8 dB vs USRNet / ADMM-EPLL −0.4 dB therefore **confirms an expected direction**. It is not a surprising new phenomenon.
- What appears novel:
  - a controlled, *non-blind* comparison of the three families (end-to-end, unrolled, classical patch-prior) on the *same* image set (BSDS500);
  - a shared training-kernel pool;
  - per-method *in-distribution vs OOD drop*, rather than absolute PSNR on OOD only;
  - paired per-image statistics.
- Chen et al. IJCV 2022 is the closest prior art and should be cited and contrasted. It tests motion-trained supervised nets on Danielyan's non-motion kernels, but its results come as absolute PSNR tables without a per-method ID baseline on the same images, and it includes neither NAFNet nor USRNet.
- NAFNet's failure in Blur2Blur is a blind real-blur-domain shift. It should not be cited as evidence for non-blind kernel-family shift, only as a related observation.

### Gaps
- No published kernel-mismatch robustness study of USRNet or DPIR was found (searched directly). Absence is not proof; a more exhaustive search of citing papers is advisable.
- Could not verify whether Dong et al. DWDN (NeurIPS 2020) or Eboli 2020 report explicit unseen-kernel-family experiments.
- Not checked: Shimron et al.'s analysis covers MRI only. Whether any deblurring paper explicitly discusses the "inverse crime" of using the same kernel generator for training and test was not established.

## Q2. Does any prior work systematically evaluate held-out kernel PARAMETERS vs held-out kernel FAMILIES (train: Gaussian / anisotropic Gaussian / linear motion / camera shake; test: defocus disc, ring/annulus, ghost/double exposure) with paired statistics?

### Takeaway
I found **no** prior paper with a three-tier protocol: train pool, then interpolated/held-out parameters, then entirely new kernel families. I also found none that adds ring/annulus or ghost/double-exposure kernels as OOD families, or that reports paired per-image tests. Existing benchmarks (Levin 2009, Sun 2013, Köhler 2012, Lai 2016) supply fixed kernel sets, not train/test family splits. Danielyan 2011's non-motion kernels have been used ad hoc as "unseen" kernels.

### Cited Findings
- Levin et al., "Understanding and Evaluating Blind Deconvolution Algorithms" (CVPR 2009; TPAMI 2011 version "Understanding Blind Deconvolution Algorithms"). Built a ground-truth blur dataset to compare algorithms under identical settings and found shift-invariance often violated. The benchmark provides 32 test instances. — [mlanthology](https://mlanthology.org/cvpr/2009/levin2009cvpr-understanding); [DSpace@MIT](https://dspace.mit.edu/handle/1721.1/59815). The count of 4 images × 8 kernels is consistent with 32 instances but was not directly confirmed. **[8-kernel detail UNVERIFIED]**
- Lai et al. (CVPR 2016), "A Comparative Study for Single Image Blind Deblurring". Includes uniform blurred images with known GT kernels. It is a benchmark of methods, not of kernel-family generalization. — [PDF](https://faculty.ucmerced.edu/mhyang/papers/cvpr16_deblurring_benchmark.pdf)
- The IJCV 2022 paper uses the "six scenarios designed in (Danielyan et al, 2011), which contain five non-motion blur kernels" as unseen kernels for motion-trained nets. This is an ad hoc held-out-family test with no parameter-interpolation tier and no paired statistics. — [IJCV 2022 PDF](https://cpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/8/10877/files/2022/04/IJCV-2022-Deblur.pdf)
- Searches for held-out-kernel-family benchmarks (2024-2025) returned only image-domain or capture-condition OOD protocols. Examples: 2504.14664, and 2508.06228 with Real-LOLBlur/RealDOF/RSBlur test-only sets. Also cross-dataset evaluation in Cascades of Regression Tree Fields ("validate the generalization ... avoid evaluating dataset-specific mappings"). — [2504.14664](https://arxiv.org/html/2504.14664v1); [2508.06228](https://arxiv.org/html/2508.06228v2); [RTF cascades](https://arxiv.org/pdf/1404.2086)
- Shimron, Tamir, Wang & Lustig (PNAS 2022), "Implicit data crimes". Training/evaluating on processed public data yields overly optimistic results, up to 48% artificial improvement in MRI. This is the closest formal "evaluation crime" framing, transferable to "same kernel generator for train and test". — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9060447); [arXiv 2109.08237](https://ar5iv.arxiv.org/html/2109.08237)

### Inferences
- The tiered protocol (ID pool → held-out parameters → held-out families incl. ring and ghost kernels) with paired statistics appears **novel as a protocol** within non-blind deblurring. Defocus disc as an "unseen" kernel has precedent (Danielyan kernels in IJCV 2022; NNet trained on defocus). Ring/annulus and ghost/double-exposure kernels as OOD families were not found in any surveyed paper.
- The "data crime" framing (Shimron 2022) plus Gilton's "model drift" framing give a citable motivation. Testing only on kernels from the training generator is optimistic, in the same way as an inverse crime.

### Gaps
- Sun et al. 2013 (ICCP; 640 images × 8 kernels per common usage) and Köhler et al. 2012 (ECCV; 12 recorded camera-shake trajectories × 4 images per common usage) were not verified this session. **[UNVERIFIED]**
- The "inverse crime" concept (Wirgin 2004; Kaipio & Somersalo) as applied to deblurring was not verified with a source.
- An exhaustive search of 2025-2026 arXiv for "kernel-family split" protocols was not possible within budget. A parallel or later work could exist.

## Q3. GMM/EPLL patch priors combined with deep networks

### Takeaway
GMM/GSM-plus-deep hybrids exist in three main forms:
- networks that predict GMM latent states, i.e. a learned gating network (Rosenbaum & Weiss 2015);
- deep networks that learn GSM scale priors inside a MAP/unrolled solver (Huang et al. CVPR 2021 / TPAMI 2023);
- EPLL's half-quadratic framework with any learned patch denoiser plugged in.

I found no work that routes patches by GMM component to specialized deep experts for deblurring robustness. If the project does that, it appears novel, but my coverage is incomplete.

### Cited Findings
- Zoran & Weiss (ICCV 2011), EPLL. A GMM learned on clean patches is used through half-quadratic splitting for whole-image restoration. It outperformed other generic priors for denoising and deblurring at the time. — [PDF](https://people.csail.mit.edu/danielzoran/EPLLICCVCameraReady.pdf); IPOL analysis [IPOL 2018](https://www.ipol.im/pub/art/2018/242/article_lr.pdf)
- Rosenbaum & Weiss (NeurIPS 2015), "The Return of the Gating Network". A discriminative feed-forward net predicts GMM latent states (component assignments). It is comparable to the original GMM-EPLL with about two orders of magnitude speedup, and keeps generative-prior flexibility (no retraining per blur kernel). — [NeurIPS PDF](https://proceedings.nips.cc/paper_files/paper/2015/file/596f713f9a7376fe90a62abaaedecc2d-Paper.pdf)
- torchEPLL notes that EPLL can use "any patch-denoiser", so learned patch denoisers can replace the GMM in the HQS loop. — [GitHub](https://github.com/friedmanroy/torchEPLL). DeepInverse ships an EPLL demo. — [deepinv colab](https://colab.research.google.com/github/deepinv/deepinv/blob/gh-pages/auto_examples/_notebooks/optimization/demo_epll.ipynb)
- Huang, Dong, Yuan, Wu & Shi (CVPR 2021), "Deep Gaussian Scale Mixture Prior for Spectral Compressive Imaging". They replace hand-crafted GSM scale priors with a CNN-learned one inside a MAP/unrolled framework. The extended version is "Deep Gaussian Scale Mixture Prior for Image Reconstruction" (TPAMI 2023, DOI 10.1109/tpami.2023.3265103; details not opened). — [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Deep_Gaussian_Scale_Mixture_Prior_for_Spectral_Compressive_Imaging_CVPR_2021_paper.html); [arXiv 2103.07152](https://arxiv.org/abs/2103.07152); [TPAMI listing](https://www.bohrium.com/paper-details/deep-gaussian-scale-mixture-prior-for-image-reconstruction/865006158782398670-2429)
- Compressive learning of patch GMMs for denoising (SIAM J. Imaging Sci.). — [DOI](https://dx.doi.org/10.1137/21M1459812)
- Mixture-of-experts decoders for unified deblurring (2025) are an MoE-routing analogue, but routing is not GMM-based. — [arXiv 2508.06228](https://arxiv.org/html/2508.06228v2)

### Inferences
- The project's GMM/EPLL-deep hybrid should be positioned against three lines: Rosenbaum & Weiss 2015 (learned gating of a GMM), deep GSM priors (Huang 2021/2023), and PnP/unrolled methods (IRCNN/DPIR/USRNet). The claimed novelty should be the *specific routing/coupling* plus *kernel-shift robustness evaluation*, not "GMM + deep" per se.

### Gaps (unverified, not cited above)
- van den Oord & Schrauwen (NeurIPS 2014), "Factoring Variations in Natural Images with Deep Gaussian Mixture Models". **[UNVERIFIED]**
- Tang, Salakhutdinov & Hinton (ICML 2012), "Deep Mixtures of Factor Analysers". **[UNVERIFIED]**
- Parameswaran et al., accelerating GMM patch priors (TIP ~2019). **[UNVERIFIED]**
- Friedman & Weiss, posterior sampling with explicit patch priors (~2021-22, referenced by torchEPLL). **[UNVERIFIED]**
- Papyan & Elad's multi-scale EPLL. **[UNVERIFIED]**
- Meinhardt et al. (ICCV 2017), "Learning Proximal Operators". This is a denoiser-as-prox method, not GMM. **[UNVERIFIED]**
- No "GMM-guided CNN" for non-blind deblurring was located.

## Q4. What evaluation statistics are standard in this literature?

### Takeaway
None of the surveyed deblurring papers report paired significance tests: Nan & Ji 2020, Vasu 2018, IJCV 2022, Blur2Blur 2024, Lai 2016 and Levin 2009 (as surveyed). Standard practice is mean PSNR/SSIM tables, sometimes with per-kernel breakdowns and "margin" columns. Paired Wilcoxon signed-rank with Holm correction would therefore be **above the field norm** and is a legitimate methodological contribution.

### Cited Findings
- IJCV 2022 reports PSNR/SSIM means with a "Margin" column (gap to the best supervised method) and no significance testing. — [IJCV 2022 PDF](https://cpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/8/10877/files/2022/04/IJCV-2022-Deblur.pdf). A grep of the extracted text for "Wilcoxon"/"significan" found only non-statistical uses of "significantly".
- Liu et al. argue PSNR-type image quality "is not equal to generalization ability" and propose the feature-statistics index SRGA plus pairwise SRGA matrices across test sets. This is an alternative to paired testing for generalization measurement. — [arXiv 2205.07019](https://arxiv.org/pdf/2205.07019)
- Blur2Blur reports mean PSNR/SSIM per dataset only. — [arXiv 2403.16205](https://arxiv.org/pdf/2403.16205)

### Inferences
- Recommend paired per-image tests (Wilcoxon signed-rank on ΔPSNR, ID vs OOD, per method) with Holm-Bonferroni across method × split comparisons, plus bootstrap CIs on mean drops. Frame this explicitly as a protocol contribution, since the literature reports unpaired means.

### Gaps
- No systematic survey of statistical-testing practice in image restoration was found. The "no paired tests" claim rests on the handful of papers inspected, not an exhaustive audit.
