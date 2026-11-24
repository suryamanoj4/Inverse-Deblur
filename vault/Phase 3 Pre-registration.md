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
