# Kernel / noise-level conditioning of restoration networks (2017–2026): novelty check for K-NAFNet

Verification legend: **[V]** = checked against the primary source in this session (paper text, arXiv HTML/PDF, or CVF page). **[R]** = from the researcher's prior knowledge of the paper, with the arXiv/CVF link given but the specific detail **not re-fetched this session**. Spot-check [R] items before quoting exact numbers. Anything marked **UNVERIFIED** is a lead, not a fact.

K-NAFNet under test: NAFNet backbone + small CNN kernel encoder (kernel -> 64-d) + log-sigma Fourier embedding, zero-initialised FiLM at every scale, **non-blind** deblurring (known kernel).

---

## Q1. Classic degradation-conditioned SR/restoration models: how is the condition injected, and is the kernel itself the condition?

### Takeaway
Since SRMD (2018), the kernel itself has been the condition: PCA-projected and then concatenated as stretched maps (SRMD, DAN), applied as SFT affine modulation in every block (SFTMD/IKC), or used to drive dynamic convolutions (UDVD, DASR, KDSR). Later work moved toward *implicit learned degradation embeddings* (DASR, KDSR, CDFormer) and *type-level prompts/text* (PromptIR, DA-CLIP, InstructIR, AutoDIR). Those prompt and text methods condition on degradation **type**, not on a kernel. Almost all of this lineage is super-resolution or all-in-one restoration, not non-blind deblurring.

### Cited Findings
**Kernel as explicit condition (mostly SR):**
- **SRMD** (Zhang, Zuo, Zhang, CVPR 2018). The blur kernel is vectorised and PCA-projected to a t-dim code (t=15), the noise level σ is appended, and the vector is stretched into H×W maps and **concatenated with the LR input at the first layer only**. Non-blind SR (kernel and σ given). Isotropic/anisotropic Gaussian kernels; tested on Set5/Set14/BSD100 with Gaussian kernels. [R] — [arXiv 1712.06116](https://arxiv.org/abs/1712.06116)
- **SFTMD + IKC** (Gu, Lu, Zuo, Dong, CVPR 2019). SFTMD is a *non-blind* SR network in which kernel maps (PCA code stretched spatially) modulate features in **every middle layer through Spatial Feature Transform (SFT) layers** (per-pixel γ, β affine). The authors criticise SRMD for injecting the kernel only at the first layer and report that SFTMD beats SRMD "by a wide margin" [V via search excerpt]. IKC adds a predictor and an iterative corrector that estimate the kernel code for blind SR. Kernels are isotropic Gaussians only (Gaussian8 test protocol). — [arXiv 1904.03377](https://arxiv.org/abs/1904.03377)
- **UDVD** (Xu, Tseng, Tseng, Lin, CVPR 2020). Degradation maps (kernel PCA code + noise) are concatenated with the input, and a **dynamic convolution** branch predicts per-pixel kernels from features plus degradation, so the condition is spatially variant. Non-blind, variational degradations. [R] — [CVF CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Unified_Dynamic_Convolutional_Network_for_Super-Resolution_With_Variational_Degradations_CVPR_2020_paper.html)
- **DAN** (Luo et al., NeurIPS 2020). Unrolled alternating Estimator/Restorer. The kernel PCA code is **stretched and concatenated** with features in conditional residual blocks. Blind SR with an estimated kernel. [R] — [arXiv 2010.02631](https://arxiv.org/abs/2010.02631)
- **MANet** (Liang et al., ICCV 2021). Only *estimates* spatially variant kernels (mutual affine convolution). SR is then done by a separate non-blind model (RRDB-SFT, i.e. SFT-conditioned on the kernel). Blind SR. [R] — [arXiv 2108.05302](https://arxiv.org/abs/2108.05302)
- **KernelGAN** (Bell-Kligler, Shocher, Irani, NeurIPS 2019). Internal-GAN kernel estimation. The estimated kernel is fed to ZSSR or other non-blind SR (e.g. SRMD/USRNet-style conditioning). Blind. [R] — [arXiv 1909.06581](https://arxiv.org/abs/1909.06581)
- **DCLS** (Luo et al., CVPR 2022). Reformulates the HR kernel into an LR-space kernel and applies a **deep constrained least-squares (Wiener-like) deconvolution in feature space**, with the kernel entering the closed-form step rather than a modulation. Blind SR. [R] — [arXiv 2202.07508](https://arxiv.org/abs/2202.07508)
- **KOALAnet** (Kim et al., CVPR 2021). Kernel-oriented adaptive local adjustment: the estimated degradation kernel drives local filter generation. Blind SR. [R] — [arXiv 2012.08103](https://arxiv.org/abs/2012.08103)
- **Kernelized Back-Projection Networks** (2023) restate that "an SR network is conditioned by a kernel in SRMD" and that the idea carries over to blind SR. [V, search excerpt] — [arXiv 2302.08478](https://arxiv.org/pdf/2302.08478)

**Implicit learned degradation embedding (no explicit kernel):**
- **DASR** (Wang et al., CVPR 2021). A contrastively trained degradation encoder produces a representation. **DA-Conv** uses it to predict depthwise conv kernels plus channel-attention modulation (two branches). The kernel is never used as input at test time. Blind SR. [V, DAConv description via search excerpt] — [arXiv 2104.00416](https://arxiv.org/abs/2104.00416)
- **KDSR** (Xia et al., ICLR 2023). A teacher sees HR+LR and a student sees LR only; both produce an implicit degradation representation (IDR) that drives **IDR-based dynamic convolution**. Blind SR. [R] — [arXiv 2211.16928](https://arxiv.org/abs/2211.16928)
- **CDFormer** (Liu et al., CVPR 2024). A diffusion model predicts a content+degradation representation that is injected into transformer SR blocks. Blind SR. [R] — [arXiv 2405.07648](https://arxiv.org/abs/2405.07648)

**Continuous level modulation (scalar blur σ / noise σ, not a kernel):**
- **AdaFM** (He et al., CVPR 2019) and **CResMD** (He et al., ECCV 2020). A degradation-level vector (e.g. Gaussian blur width, noise σ) is mapped to modulation of features or residual connections for continuous control. The condition is a scalar level, not a kernel. [R] — [AdaFM arXiv 1904.08118](https://arxiv.org/abs/1904.08118)

**Prompt / text / type conditioning (all-in-one):**
- **PromptIR** (Potlapalli et al., NeurIPS 2023). Restormer backbone. Learnable prompt components are weighted by a softmax over globally pooled features and fused into decoder features (prompt generation + interaction blocks). Implicit and type-level, covering denoise/derain/dehaze. No kernel. [R] — [arXiv 2306.13090](https://arxiv.org/abs/2306.13090)
- **DA-CLIP** (Luo et al., ICLR 2024). A CLIP controller outputs a degradation embedding and a clean-content embedding. Content goes in through cross-attention and degradation through prompt-learning modules in an IR-SDE (NAFNet-style U-Net). Type-level over 10 degradation types; motion blur is just one class. [R] — [arXiv 2310.01018](https://arxiv.org/abs/2310.01018)
- **InstructIR** (Conde, Geigle, Timofte, ECCV 2024). **NAFNet backbone**. A sentence-encoder text embedding is projected and drives Instruction Condition Blocks (channel-wise feature modulation / task routing) in encoder and decoder. Conditions on a human instruction ("remove the blur"), not on the kernel. [R] — [arXiv 2401.16468](https://arxiv.org/abs/2401.16468)
- **AutoDIR** (Jiang et al., ECCV 2024). A BIQA/CLIP stage detects the dominant degradation, and a text prompt conditions a latent diffusion restorer. Type-level. [R] — [arXiv 2310.10123](https://arxiv.org/abs/2310.10123)

### Inferences
- On the "is the kernel the condition?" axis: **yes** for SRMD, SFTMD, UDVD, DAN, MANet+RRDB-SFT, DCLS, KOALAnet; **no** (implicit embedding) for DASR, KDSR, CDFormer; **no** (degradation type or text) for PromptIR, DA-CLIP, InstructIR, AutoDIR.
- K-NAFNet's modulation design (affine γ/β from a degradation code in every block) is SFTMD-style conditioning. The differences are spatially global FiLM instead of per-pixel SFT, a learned CNN kernel encoder instead of fixed PCA, and NAFNet instead of SRResNet. The **mechanism itself is not novel**. Any novelty has to come from the combination, the task (non-blind deblurring with arbitrary, non-Gaussian kernels) and the generalisation study.

### Gaps
- Exact numbers for SRMD, SFTMD, UDVD, DAN, KDSR, CDFormer were not re-extracted this session. They are SR benchmarks (Gaussian8, DIV2KRK) and not directly comparable to deblurring anyway.

---

## Q2. Modern backbones (NAFNet, Restormer, Uformer, SwinIR) conditioned on a blur kernel for deblurring

### Takeaway
The most important prior art is **Kaufman & Fattal, "Deblurring using Analysis-Synthesis Networks Pair" (CVPR 2020)**. It conditions a U-Net on the blur kernel with an MLP that produces per-layer multipliers and biases, `r ← r ⊙ (1 + m(k)) + b(k)`. That is FiLM on the kernel, with a residual "1+" form, at every U-Net layer. It also reports a GT-kernel (non-blind) configuration. On modern backbones, the kernel-conditioned work is **blind**: UFPNet (CVPR 2023) on NAFNet uses kernel attention, and Sun et al. 2025 on a Transformer uses frequency attention. The 2025 paper does include a non-blind GT-kernel ablation. **Converse-USRNet (ICCV 2025)** maps the kernel to a **64-d embedding with an MLP**, but inside a USRNet-style unrolled model. No paper was found that FiLM-conditions **NAFNet** on a known kernel for non-blind deblurring. **KBNet is not degradation-kernel conditioning.**

### Cited Findings
- **Kaufman & Fattal 2020** (Hebrew Univ.; arXiv 2004.02956, CVPR 2020) [V, read from the paper PDF]:
  - An analysis network estimates the kernel (with cross-correlation layers). A **synthesis U-Net** deblurs using that kernel.
  - "Kernel Guided Convolution": at each U-Net layer, after the first conv, `r = r ⊙ (1 + m(k)) + b(k)`, where m and b come from a per-layer "synthesis-guiding unit". That unit is three FC layers on the **raw flattened kernel pixels**, hidden width 128, output width equal to the layer's channel count. Effect: "the blur-kernel affect[s] the U-Net's action uniformly across the entire spatial domain." The authors cite FiLM-like control units from image translation as inspiration.
  - The authors rejected predicting conv weights from k (a hypernetwork) because it would need "a large number of fairly large fully-connected layers".
  - Fig. 1 mean PSNR on 1000 images: no analysis (kernel-agnostic) 23.7; analysis+synthesis (estimated kernel) 26.6; **synthesis + GT kernel 28.28**; a kernel-specific expert 33.75.
  - Training: Open Images crops, spline-based camera-shake kernels following Chakrabarti, noise σ=0.02, kernel sizes split into three size classes ([0,31], (31,61], (61,85]). The setting is **uniform motion blur only**.
  - Source: [arXiv 2004.02956](https://arxiv.org/pdf/2004.02956)
- **UFPNet** (Fang et al., CVPR 2023, "Self-Supervised Non-Uniform Kernel Estimation with Flow-Based Motion Prior for Blind Image Deblurring"). A normalizing-flow latent represents per-pixel motion kernels, and a CNN predicts latent codes with uncertainty. A **multi-scale kernel attention module** fuses image features with the estimated kernels and "can be plugged into encoder-decoder architectures". The code is built on NAFNet/BasicSR and the module was plugged into NAFNet and MIMO-UNet. **Blind**, kernels self-supervised, evaluated on GoPro/HIDE/RealBlur. [V, abstract/contributions/repo via search; full-text numbers not retrieved because the CVF PDF returned 403] — [CVF paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_Self-Supervised_Non-Uniform_Kernel_Estimation_With_Flow-Based_Motion_Prior_for_Blind_CVPR_2023_paper.pdf); [GitHub UFPDeblur](https://github.com/Fangzhenxuan/UFPDeblur)
- **Sun, Lei, Zhang, Sun, Yang, "Frequency-domain Learning with Kernel Prior for Blind Image Deblurring"** (arXiv 2504.14664, Apr 2025; no venue listed on arXiv) [V, arXiv HTML]:
  - Frequency Integration Module (FIM): image features → Q; kernel features → K, V (conv + depthwise conv). `FA = LN(F⁻¹(F(Q)⊙F(K))) ⊙ V`, residual add. Kernel features are injected at three scales, before every encoder and decoder Transformer block.
  - Per-pixel kernels come from a U-Net self-supervised by reblur loss.
  - **Non-blind ablation** (GT kernels extracted per image and reblurred, train GoPro), PSNR: GoPro-Syn 28.48 (baseline) → 28.59 (blind) → **28.80 (non-blind)**; HIDE-Syn 27.35 → 27.73 → **27.88**; RB-R-Syn 37.95 → 38.01 → **38.08**; RB-J-Syn 29.98 → 30.44 → **30.51**.
  - Main table vs NAFNet (GoPro 34.21 vs 33.71; HIDE 32.08 vs 31.32; RealBlur-R 36.59 vs 35.50; RealBlur-J 29.01 vs 28.32).
  - Source: [arXiv 2504.14664](https://arxiv.org/html/2504.14664v1)
- **Converse2D / ConverseNet** (Huang, Liu, **Kai Zhang**, Tai, Yang, Zeng, **Lei Zhang**; ICCV 2025, "Reverse Convolution and Its Applications to Image Restoration") [V, arXiv HTML]:
  - **Converse-USRNet** uses "KernelNet", which "maps the blurred kernel to a **64-dimensional embedding** via three fully connected layers with GELU". The embedding parameterises a Converse2D (closed-form reverse convolution) layer, with λ the only other learnable parameter on that path.
  - Non-blind deblurring on BSD100/Urban100. Training uses 7×7 isotropic Gaussian kernels; testing uses 7×7 motion kernels, noise σ=2.55.
  - PSNR, five test kernels, BSD100: Conv-USRNet 32.18/31.03/37.80/31.82/34.33 vs Converse-USRNet 32.46/31.20/38.40/32.21/34.62.
  - The paper reports **no USRNet/DPIR numbers in that table** and does not frame Gaussian→motion as a generalisation study.
  - Source: [arXiv 2508.09824](https://arxiv.org/html/2508.09824); code [github.com/cszn/ConverseNet](https://github.com/cszn/ConverseNet)
- **KBNet** (Zhang et al., arXiv 2303.02881, 2023; CUHK/SenseTime). "Kernel basis attention" learns a set of *convolution kernel bases* and fuses them per pixel with predicted coefficients for adaptive spatial aggregation, inside an MFF block. It is a backbone design for denoising, deraining and deblurring and is **NOT conditioned on the degradation/blur kernel**. [V] — [arXiv 2303.02881](https://arxiv.org/pdf/2303.02881); [GitHub](https://github.com/zhangyi-3/KBNet)
- **DeepRFTv2** (arXiv 2511.21132, late 2025). The kernel is treated as a spatial feature: a Fourier Kernel Estimator turns convolution into Fourier-space multiplication and learns global feature-sized kernels without reblur supervision. **Blind** and implicit. [V, search excerpt only] — [arXiv 2511.21132](https://arxiv.org/html/2511.21132)
- **PG-ControlNet** (arXiv 2511.21043, 2025). A "physics-guided ControlNet for generative spatially varying image deblurring"; the PSF/kernel field is presumably the ControlNet condition. **UNVERIFIED details**, only the title was seen. — [arXiv 2511.21043](https://arxiv.org/pdf/2511.21043)
- **CASSI CG-unrolled network with PSF conditioning** (arXiv 2607.20138, 2026). Unrolled conjugate gradient with PSF conditioning. It notes that MIMO-UNet/Restormer "do not explicitly model the PSF". This is spectral imaging, a loose match. [V, search excerpt] — [arXiv 2607.20138](https://arxiv.org/pdf/2607.20138)
- **Li et al., ICCV 2021, "Universal and Flexible Optical Aberration Correction Using Deep-Prior Based Deconvolution"**. A **PSF-aware** deep prior network takes the aberrated image **and a PSF map** as input inside a model-based iterative deconvolution. Non-blind, spatially variant lens PSFs. The authors argue that one weight setting across varying PSFs "will either require a large model capacity or lead to an average performance". [V, search excerpt] — [arXiv 2104.03078](https://arxiv.org/abs/2104.03078)
- **SVBR-NET** (arXiv 2206.12930, 2022). A non-blind spatially varying **defocus** blur removal network; the defocus map/kernel parameters are input. [V title, details UNVERIFIED] — [arXiv 2206.12930](https://arxiv.org/pdf/2206.12930)

**Known-kernel non-blind deep deconvolution where the kernel enters through physics, not modulation** (the baselines K-NAFNet must beat):
- **USRNet** (Zhang, Van Gool, Timofte, CVPR 2020). Unfolded HQS. The kernel enters the closed-form FFT data module. A **hyper-parameter module (MLP on σ and scale)** outputs per-iteration α, β, which is hypernetwork conditioning on noise, not on the kernel. [R] — [arXiv 2003.10428](https://arxiv.org/abs/2003.10428)
- **DPIR** (Zhang et al., TPAMI 2021). DRUNet denoiser prior **with a noise-level map concatenated as an input channel**; the kernel enters through the HQS data step. Deblurring is evaluated on the 8 Levin kernels plus 2 Gaussians? [R; exact kernel list not re-verified] — [arXiv 2008.13751](https://arxiv.org/abs/2008.13751)
- **DWDN** (Dong, Roth, Schiele, NeurIPS 2020; TPAMI extension). Feature-space Wiener deconvolution with the given kernel, then a multi-scale refinement net. [V, abstract] — [arXiv 2103.09962](https://arxiv.org/pdf/2103.09962)
- **INFWIDE** (Zhang et al., TIP 2023). Image- and feature-space Wiener deconvolution for low-light/noisy non-blind deblurring. [R] — [arXiv 2207.08201](https://arxiv.org/abs/2207.08201)
- **Eboli, Sun, Ponce, ECCV 2020**, "End-to-end interpretable learning of non-blind image deblurring". Learned HQS-style solver with the kernel in the data term. [V title] — [ECCV paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620307.pdf)
- **Quan et al., TNNLS 2022**, "Non-blind image deblurring via deep learning in complex field". Designed to train "a universal CNN that can work on different kernels". [V, search excerpt] — [PDF](https://csyhquan.github.io/manuscript/22-tnnls-Non-Blind%20Image%20Deblurring%20via%20Deep%20Learning%20in%20Complex%20Field.pdf)
- **Vasu, Maligireddy, Rajagopalan, CVPR 2018**, "Non-blind deblurring: handling kernel uncertainty with CNNs". Trains on many real and synthetic *noisy* kernels to be robust to kernel error. [V, search excerpt] — [IEEE 8578443](https://ieeexplore.ieee.org/document/8578443/)
- The Kaufman & Fattal paper itself notes that Xu et al. 2014 used the SVD of the kernel pseudo-inverse to initialise 1D conv weights, and that Schuler et al. used an MLP after non-blind deconvolution. Both need **per-kernel training**. [V, Kaufman & Fattal related-work section]

### Inferences
- **Closest prior art, ranked:**
  1. Kaufman & Fattal 2020: kernel → MLP → per-layer (1+m, b) affine modulation of a U-Net, with a GT-kernel non-blind variant. This is essentially "FiLM on kernel in a U-Net".
  2. Converse-USRNet 2025: kernel → 64-d MLP embedding, but used to parameterise an analytic reverse convolution, not FiLM.
  3. SFTMD 2019: kernel code → SFT in every block, for SR.
  4. Blind kernel-attention on NAFNet/Transformer (UFPNet 2023, Sun et al. 2025).
- What remains defensible as new for K-NAFNet:
  - (a) a modern NAFNet backbone with zero-init FiLM, as opposed to a 2020 vanilla U-Net;
  - (b) a joint kernel + continuous log-σ Fourier embedding (no reviewed paper uses Fourier features of σ; USRNet/DPIR use raw σ via MLP or maps);
  - (c) a CNN encoder on the kernel image, as opposed to an MLP on flattened pixels (Kaufman & Fattal, Converse) or PCA (SRMD/SFTMD);
  - (d) explicit cross-family generalisation testing (see Q4).
  
  None of these is a fundamentally new mechanism. The write-up should cite Kaufman & Fattal and SFTMD as the mechanism's origin.

### Gaps
- UFPNet's numeric results and the exact kernel-attention equations were not retrieved (CVF 403).
- No 2022–2026 paper conditioning **NAFNet/Restormer/Uformer/SwinIR via FiLM on a known kernel for non-blind deblurring** was found after several targeted searches. That is absence of evidence from a limited search (about 15 queries), not proof. Recommended manual checks: Google Scholar "cited by" on Kaufman & Fattal 2020 and on SFTMD restricted to "deblurring"; and PG-ControlNet (2511.21043).
- Kaufman & Fattal's venue as CVPR 2020 is from memory; the arXiv v1 is dated 6 Apr 2020.

---

## Q3. FiLM/SFT conditioning on an explicit learned kernel embedding for **non-blind deblurring**

### Takeaway
Yes, at least once. Kaufman & Fattal 2020 FiLM-modulate a U-Net with an MLP embedding of the kernel and report a GT-kernel result (28.28 dB, against 23.7 dB kernel-agnostic, on their synthetic camera-shake set). It is embedded in a blind pipeline, with the GT-kernel variant presented as an upper bound. SFT conditioning (SFTMD) is non-blind but for SR with Gaussian kernels. No FiLM-on-kernel non-blind deblurring paper was found that reports results on standard benchmarks (Levin, Sun, Köhler) against USRNet/DPIR.

### Cited Findings
- Kaufman & Fattal: Eq. (1) `r = r ⊙ (1 + m(k)) + b(k)`, per-layer FC units on k. Mean PSNR on 1000 test images: GT-kernel synthesis 28.28, estimated kernel 26.6, no kernel 23.7. — [arXiv 2004.02956](https://arxiv.org/pdf/2004.02956) [V]
- SFTMD: kernel-map SFT at every layer, non-blind SR, beats SRMD's first-layer concatenation. — [arXiv 1904.03377](https://arxiv.org/pdf/1904.03377) [V, search excerpt]
- Converse-USRNet: learned 64-d kernel embedding (MLP), non-blind, but not FiLM. — [arXiv 2508.09824](https://arxiv.org/html/2508.09824) [V]
- Sun et al. 2025: a learned kernel-feature encoder fused by frequency attention, with a non-blind GT-kernel ablation giving +0.15–0.2 dB over its blind variant. Attention, not FiLM. — [arXiv 2504.14664](https://arxiv.org/html/2504.14664v1) [V]

### Inferences
- K-NAFNet's "zero-initialised FiLM" (start as identity, so the conditioned net equals plain NAFNet at init) is close to Kaufman & Fattal's `(1+m)` residual parameterisation. The zero-init/identity idea parallels AdaLN-Zero (DiT) and ControlNet zero-convs. Do not claim it as novel.
- The Kaufman & Fattal ablation shows a large gain from the kernel (+4.6 dB GT-kernel vs agnostic). K-NAFNet should report the same kind of ablation: NAFNet without kernel, NAFNet + concatenated kernel maps (SRMD-style), NAFNet + FiLM. That positions it against both mechanisms.

### Gaps
- No standard-benchmark (Levin / Sun / BSD-with-Levin-kernels) results exist for FiLM-on-kernel models in the found literature, so no direct number is available to compare against.

---

## Q4. Generalisation to kernel families unseen in training

### Takeaway
Explicit cross-family tests (e.g. train on motion/Gaussian, test on defocus disc, ring or double-exposure) are **essentially absent**. Existing "generalisation" means one of three things: cross-dataset image domain (GoPro→HIDE/RealBlur), Gaussian→motion with a small 7×7 support (Converse-USRNet, not framed as a study), or robustness to *noisy/estimated* kernels (Vasu 2018, Quan 2022, Nan & Ji 2020). This is the most promising novelty axis for K-NAFNet.

### Cited Findings
- Converse-USRNet trains on 7×7 isotropic Gaussian kernels and tests on 7×7 motion kernels, without discussing it as generalisation. — [arXiv 2508.09824](https://arxiv.org/html/2508.09824) [V]
- Sun et al. 2025: the "out-of-domain" tests are HIDE/RealBlur with GoPro-trained models, i.e. image-domain shift, not kernel-family shift. — [arXiv 2504.14664](https://arxiv.org/html/2504.14664v1) [V]
- Vasu et al. 2018 trains on large sets of real and synthetic noisy kernels to "generalize to arbitrary kernel noise". This is robustness to kernel error, not to an unseen kernel family. — [IEEE](https://ieeexplore.ieee.org/document/8578443/) [V, excerpt]
- Quan et al. 2022 tests robustness with kernels produced by existing blind kernel estimators on real inputs. — [PDF](https://csyhquan.github.io/manuscript/22-tnnls-Non-Blind%20Image%20Deblurring%20via%20Deep%20Learning%20in%20Complex%20Field.pdf) [V, excerpt]
- Kaufman & Fattal train and test only on spline camera-shake kernels, and train separate networks per kernel-size class. Generalisation is per-size and in-family. — [arXiv 2004.02956](https://arxiv.org/pdf/2004.02956) [V]
- IKC/SFTMD and most blind-SR works test only isotropic/anisotropic Gaussians (Gaussian8, DIV2KRK). [R] — [arXiv 1904.03377](https://arxiv.org/abs/1904.03377)
- The "Deep Image Deblurring: A Survey" (Zhang et al., 2022) notes that non-blind networks often must be trained for a fixed noise level, and that without outlier handling they produce ringing. [V, search excerpt] — [arXiv 2201.10700](https://arxiv.org/pdf/2201.10700)

### Inferences
- The conditioning-mechanism paradigm has never been stress-tested on disc/ring/ghost kernels. A learned embedding (PCA, MLP or CNN) may extrapolate poorly when the kernel's spectrum has zeros in places the training kernels never had (discs have Bessel zeros; rings and ghosts are high-frequency). Unrolled/physics models (USRNet, DPIR, DWDN) put the kernel into an exact FFT data term and should degrade more gracefully. That is the key comparison K-NAFNet's evaluation should make.
- Hypothesis (not verified): FiLM models may collapse to "average-kernel" behaviour on OOD families. Measuring PSNR as a function of distance of the kernel embedding from the training set would be a novel analysis.

### Gaps
- No paper found with a train-motion/Gaussian → test-disc/ring/double-exposure protocol for non-blind deblurring. Defocus-specific non-blind work (SVBR-NET) trains on defocus itself.

---

## Q5. Reported comparisons against USRNet / DPIR

### Takeaway
Few conditioning-mechanism papers compare against USRNet/DPIR on deblurring. Converse-USRNet only compares against its own Conv-USRNet variant. USRNet and DPIR remain the standard known-kernel baselines (Levin kernels + Gaussians, BSD68/Set-style images). A K-NAFNet comparison against them on identical kernels and noise levels would be useful and not redundant.

### Cited Findings
- Converse-USRNet (ICCV 2025), Table 7: no USRNet/DPIR deblurring numbers, only Conv-USRNet vs Converse-USRNet (BSD100, motion kernels, σ=2.55). — [arXiv 2508.09824](https://arxiv.org/html/2508.09824) [V]
- USRNet's hyper-parameter MLP on (σ, scale) and DPIR's concatenated σ-map are the noise-conditioning precedents for K-NAFNet's log-σ embedding. [R] — [USRNet](https://arxiv.org/abs/2003.10428), [DPIR](https://arxiv.org/abs/2008.13751)
- Kaufman & Fattal compare against classical blind methods and DeblurGAN/SRN-style networks, not USRNet/DPIR (which postdate or are concurrent with it). [V, from the paper's framing; full results table not read]

### Inferences
- Positioning: "K-NAFNet = a direct-regression conditioned network (SFTMD/Kaufman-Fattal lineage, modern backbone) vs unrolled physics models (USRNet/DPIR/DWDN), evaluated across kernel families". That framing is defensible. "First kernel-conditioned FiLM deblurring network" is **not**.

### Gaps
- Exact USRNet/DPIR deblurring PSNR on Levin/BSD68 were not re-extracted this session. Take them from the original papers.
