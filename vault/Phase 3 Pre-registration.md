# Phase 3 Pre-registration (written 2026-11-21, before any Phase 3 training)

Framing: **diagnose → explain → fix → validate.** Phase 2 diagnosed two failures:
(F1) data-only networks do not transfer to unseen kernel families, and neither more images, more
kernel families nor per-σ specialisation fixes it; (F2) the learned unrolled solver (USRNet) beats
classical solvers only when the input follows its exact circular forward model (on provided blur,
EPLL wins: Sun 32.53 vs 31.72 dB, Levin real 30.33 vs 29.56 dB, noise estimated per image).

## Experiments

**E-P (placement ladder, fix for F1).** One NAFNet backbone (M1 configuration), v2 data, M1 budget
(100k iterations, batch 16, lr 2e-4, early stopping, checkpoint on mean val PSNR), seed 0. Only where
the kernel enters changes (`src/deblur/methods/unrolled/placement.py`):
`nafnet` (none, = M1) → `nafnet_concat` (input maps) → `nafnet_film` (FiLM on every block) →
`knafnet` (FiLM + one FFT data step) → `nafnet_unrolled` (4 HQS stages, shared FiLM prior).
FiLM variants have ≈14% more parameters than M1 (19.5M vs 17.1M); recorded.

**E-M (mismatch-aware training, fix for F2).** USRNet with the `usrnet_full` budget (cap 200k,
early stopping), half of the batches from a realistic forward model (linear blur of a larger crop,
edge pad + edgetaper ×3, as in the provided-blur evaluation) and, in those batches, the given kernel
perturbed by up to 10% of its maximum on the dilated support. Checkpoint selection unchanged
(circular validation grid). Applied afterwards to the best E-P variant.

**E-X (explanation of F1).** Fit, per method, (a) one linear filter shared by all kernels and
(b) one per kernel, mapping the blurred input to the method's output (least squares in the Fourier
domain); report the fraction of output variance each explains.

**Control (already running).** Sun images blurred with the circular model at σ = 2.55/255.

**Seeds.** v2 NAFNet and v2 USRNet (full) with seeds 1 and 2.

## Predictions (stated before running)

1. **E-P.** `nafnet_concat` and `nafnet_film` gain over `nafnet` on P2_interp but keep a P2_extrap
   drop (vs P1) of more than 1 dB. `knafnet` and `nafnet_unrolled` have an extrap drop under 0.5 dB.
   `nafnet_unrolled` ≥ `knafnet` on all sets.
2. **E-M.** Mismatch-trained USRNet improves over `usrnet_full` by ≥ 0.5 dB on Sun (est) and Levin
   real (est) and loses ≤ 0.2 dB on the v2 circular grid.
3. **Control.** On circular Sun, USRNet ≥ EPLL (if not, F2 is not explained by forward-model
   mismatch, and is reported as open).
4. **E-X.** For NAFNet the shared filter explains nearly as much as per-kernel filters (difference
   < 0.02 in R²); for USRNet and EPLL the difference is larger.

## Analysis rules
Mean PSNR over σ > 0 per set; paired Wilcoxon signed-rank over (image, kernel, σ) cells with Holm
correction across the variant comparisons; mean ± std over seeds where available. Any prediction
that fails is reported as failed.

## Outcome (added 2026-11-28, after all Phase 3 runs)

The predictions above are unchanged. Numbers are mean PSNR over σ > 0 on the v2 test grid (200
images), mean over 3 training seeds for every NAFNet ladder rung and USRNet; paired tests are on
the seed-averaged models with Holm correction (`report/generated/phase3.json`,
`phase3_stats.json`). Summary slide: "Phase 3: diagnose → explain → fix → validate" in
`report/final_presentation.pptx`; full write-up in `report/final_report.pdf`.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 1a | `nafnet_concat` and `nafnet_film` gain over `nafnet` on P2_interp | 25.52 and 25.62 vs 25.31 dB; paired +0.21 / +0.31 dB, 200/200 images | Holds |
| 1b | ... but keep a P2_extrap drop (vs P1) above 1 dB | P1 → P2_extrap: concat 26.37 → 20.78, FiLM 26.40 → 17.23. Both fall below no kernel (21.97 ± 0.18): −1.19 and −4.74 dB | Holds (and worse than predicted) |
| 1c | `knafnet` extrap drop under 0.5 dB | 26.46 → 23.45 ± 0.41: drop 3.01 dB | **Fails** |
| 1c | `nafnet_unrolled` extrap drop under 0.5 dB | 27.30 (P1) → 27.14 ± 0.38 (P2_extrap) | Holds |
| 1d | `nafnet_unrolled` ≥ `knafnet` on all sets | +0.84 / +1.00 / +3.69 dB on P1 / P2_interp / P2_extrap, 200/200 images each | Holds |
| 2 | Mismatch-trained USRNet: ≥ 0.5 dB on Sun (est) and Levin real (est), ≤ 0.2 dB loss on the grid | Levin real 33.70 vs 29.56 (+4.14), Sun 32.57 vs 31.72 (+0.85); grid P1 27.00 vs 27.20 (−0.20) | Holds |
| 3 | Circular Sun: USRNet ≥ EPLL | EPLL 32.55 vs USRNet 31.78 dB | **Fails**: F2 is not explained by forward-model mismatch; reported as open |
| 4 | E-X: NAFNet shared-filter R² within 0.02 of per-kernel R² | Shared 0.13, per-kernel 0.77 (10 images); repeated on 50 images: 0.12 / 0.74 | **Fails**: NAFNet adapts to the kernel |
| — | Seeds 1, 2 for v2 NAFNet and USRNet | Run. P2_extrap: NAFNet 21.97 ± 0.18, USRNet 27.57 ± 0.06 | Done |

Of the five experiment-level predictions on the deck slide, two hold (unrolled placement,
mismatch training) and three fail (one-step K-NAFNet, circular-Sun control, E-X); the failures
are reported as failed.

**What the failures led to (runs added after this pre-registration, labelled as such):**

- **Data step without FiLM (`nafnet_dc`, 3 seeds).** P2_extrap 25.05 ± 0.05 dB, drop 0.96 dB from
  P1, +3.08 dB over no kernel and +1.60 dB over K-NAFNet (FiLM + the same step). Most of
  K-NAFNet's unseen-family drop comes from the FiLM conditioning, not from having only one step.
- **Unrolled vs USRNet.** The 4-stage unrolled NAFNet is 0.44 dB below USRNet on P2_extrap
  (27.14 vs 27.57, p < 10⁻³²; 7/200 images favour it), so placement carries most of the gain, not all.
- **Kernel-swap probe.** On unseen families FiLM scores higher with a wrong kernel than with the
  true one (true 18.20, nearest 20.43, no blur 20.67): the kernel acts as a lookup key. Through a
  data step the true kernel is worth +3 to +7 dB, like Wiener (+3.10). Repeated on 100 images,
  every "worth of the true kernel" moves by at most 0.08 dB.
- **Spectral loss.** Per unseen kernel, the gain over no-kernel NAFNet falls with the fraction of
  the spectrum with |K| < 0.1 (Spearman ρ −0.92 FiLM, −0.96 K-NAFNet, −0.97 unrolled, −0.99
  USRNet; n = 12 kernels). Spectral loss is correlated with kernel size (ρ = 0.58).
- **U-Net ladder (1 seed per variant).** Input maps −1.34, FiLM −2.68, FiLM + data step −0.31 dB
  vs no kernel on P2_extrap (p = 0.07 for the last): conditioning hurts on both backbones.
- **2×2 mismatch ablation.** Realistic boundaries alone fix Levin's two largest kernels
  (22.52 → 31.99 dB) at no grid cost; stopped at 66k iterations they still give 32.10 dB, so the
  gain is not extra training. Kernel noise alone gives less and costs about 0.1 dB in distribution.
  K-NAFNet + both: Levin real +0.99 dB over K-NAFNet, still 4.55 dB below EPLL on Sun (1 seed).
