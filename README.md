# Inverse Function Learning for Image Deblurring
### Classical vs. Modern Machine Learning

Research proposal and design notes for a controlled study of image deblurring as an inverse-function learning problem.

**Authors:** Surya Manoj Pentakota (2026701036), Kevin Paul Reddy Janagari (2026900006) — IIIT Hyderabad

## Overview

A sharp image $x$ is observed through blur and noise:

$$
y = Ax + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2),
$$

where $A$ is convolution with a point-spread function (PSF) and $\varepsilon$ is sensor noise. Because $A$ is ill-conditioned — its eigenvalues decay to near zero at high frequencies — no stable exact inverse exists, and naive inversion amplifies noise.

We cast deblurring as **inverse function learning**: find $g_\theta$ minimizing $\mathbb{E}\lVert g_\theta(y) - x \rVert^2$ over pairs from the known degradation model. The central design question is: **should $g_\theta$ be built around the physics of $A$, or learned purely from data?** Existing methods differ along exactly this axis, but their results are reported under different protocols, so they are not directly comparable. This project runs that comparison under one controlled protocol.

## Three-arm comparison

| Arm | Family | Knows $A$? | Representative methods |
|---|---|---|---|
| **M1** | Modern, data-only | No | NAFNet (primary), DeepDeblur (historical) |
| **M2** | Classical-ML, physics-aware | Yes (true kernel + noise) | FISTA/ADMM solver + EPLL-GMM prior; K-SVD optional |
| **M3** | Unrolled control, physics-aware | Yes | USRNet (primary); LISTA → ISTA-Net genealogy |

M3 is the control that separates two otherwise confounded axes: **physics-awareness** (does the method use $k$ and $\sigma$?) and **prior depth** (fixed TV → shallow learned GMM → deep learned CNN).

## Evaluation

- **Tier 1 (core):** BSDS500, synthetically degraded with a shared PSF library (Gaussian, Levin motion, defocus) and noise grid $\sigma \in \{0, 1, 5, 15, 25\}/255$, circular boundary handling.
- **Tier 2 (stretch):** GoPro (3214 real blurry/sharp pairs) for blind deblurring.
- **Metrics:** PSNR, SSIM, LPIPS, data-consistency residual, kernel-mismatch robustness, and stability under small perturbations; paired Wilcoxon tests with multiplicity control.

## Repository layout

```
.
├── README.md            # this file
├── proposal.pdf         # submitted 3-page proposal
└── vault/               # design notes (Obsidian)
    ├── Project Idea.md
    └── Proposed Approach.md
```

LaTeX sources (`proposal/`, `*.tex`) are intentionally **not** versioned here; `proposal.pdf` is the reference artifact. The vault contains the working design record, including the full three-arm method specification.

## Status

Proposal stage. Next: implement the degradation pipeline and PSF library, then the classical baselines.

## References

See `proposal.pdf` and `vault/Proposed Approach.md` for the full bibliography.
