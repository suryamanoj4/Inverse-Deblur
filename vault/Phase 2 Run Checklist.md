---
title: Phase 2 Run Checklist
date: 2026-11-15
tags:
  - project
  - deblurring
  - runs
status: active
---

# Phase 2 Run Checklist

Everything here is **code-complete and smoke-tested**; what remains is running it. The plan and its rationale are in [[Proposed Approach]] §12. Commands assume `PYTHONPATH=src` and the project venv (`.venv/bin/python`).

Times are **single-GPU wall clock on the RTX 3090 Ti, with the GPU to ourselves**, from measured speeds: NAFNet 11 it/s, U-Net ≈55 it/s, USRNet 12 it/s, ISTA-Net ≈43 it/s, LISTA ≈80 it/s; FISTA-TV 1.7 s and ADMM-EPLL 0.65–0.8 s per image × kernel × σ cell; white-box H3 attack 8 s (FISTA-TV) / 20 s (ADMM-EPLL) per cell. When the GPU is shared, expect ×1.5–2 (Phase 1 experience).

> [!warning] FISTA-TV is the cost driver
> BSDS images are 481×321 (prime factors 13·37 × 3·107), and cuFFT is ≈2.5× slower at that size than at 480×320. Batching kernels per call gives only ≈1.2×, and float32 gives nothing. The full-image circular protocol fixes the size, so the plan evaluates FISTA-TV on a 100-image subset under v2.

## 0. Setup (≈1 h, mostly downloads)

- [x] Freeze protocol v2 (kernels + SHA-256): `python scripts/setup_datasets.py --write-protocol --version v2` (add `--real-kernels levin=DIR` etc. **before** any v2 run if real kernels are wanted; the kernel sets are frozen once v2 runs start). **1 min**
- [ ] Optional real kernels for $P_2$-extrap: `--datasets levin,sun2013,kohler2012` with `--<name>-archive` or `--<name>-url`, then point `--real-kernels PREFIX=DIR` at a folder of kernel files only. **≈15 min**
- [ ] DIV2K for the 2A-3 control: `python scripts/setup_datasets.py --datasets div2k` (3.5 GB). **≈20–40 min**

## 1. Phase 2A: answer the reviewer on the frozen v1 protocol

| ✓ | Run | Command | Time |
|---|---|---|---|
| [x] | 2A-1 NAFNet, 50 real images | `scripts/train_m1.py --data-root data --train-images 50` | 2.8 h |
| [x] | 2A-1 NAFNet, 100 real images | `scripts/train_m1.py --data-root data --train-images 100` | 2.8 h |
| [x] | 2A-1 evaluate both on the v1 test grid (200-image NAFNet run already exists) | `scripts/evaluate.py --arm m1 --checkpoint checkpoints/m1/nafnet_real50/best.pt` (and `_real100`) | 2 × ≈1 h |
| [ ] | 2A-3 NAFNet real + DIV2K | `scripts/train_m1.py --data-root data --extra-images data/div2k/DIV2K_train_HR --extra-name div2k --p-real 0.5` + evaluate | 2.8 h + 1 h |
| ⛔ | **Go/no-go gate** (spec §12.2): generate `syn-v1` only if 50 → 100 → 200 is still rising **and** DIV2K helps | — | — |
| [ ] | Generate raw images (needs `pip install diffusers transformers accelerate`, model download 2–7 GB) | `scripts/generate_synthetic.py --out data/synthetic/raw-v1 --count 2600` | 0.5 h (SDXL-Turbo) – 2.5 h (SD 1.5) |
| [ ] | QC + freeze `syn-v1` (sharpness, entropy, duplicates, BSDS val/test leakage, statistics) | `scripts/check_synthetic.py --raw data/synthetic/raw-v1 --out data/synthetic/syn-v1 --name syn-v1 --data-root data --keep 2000` (add `--embedding clip` with `open_clip_torch`) | 0.5 h CPU |
| [ ] | Manual spot check of accepted/rejected images | — | 15 min |
| [ ] | 2A-2 NAFNet real + syn | `scripts/train_m1.py --data-root data --extra-images data/synthetic/syn-v1 --require-manifest --p-real 0.5` + evaluate | 2.8 h + 1 h |
| [ ] | 2A-2 U-Net real + syn | same with `--model unet` + evaluate | 0.5 h + 0.5 h |
| [ ] | 2A-2 GMM-200 real + syn (200k patches) | `scripts/train_gmm.py --data-root data --components 200 --patches 200000 --extra-images data/synthetic/syn-v1 --require-manifest --p-real 0.5 --out checkpoints/m2/gmm_K200_syn-v1_p0.5.npz` | 15 min CPU |
| [ ] | 2A-2 re-tune EPLL λ on real val, then EPLL on the v1 test grid | λ sweep as in Phase 1 (`artifacts/eval/m2_tune_gpu`), then `scripts/evaluate.py --arm m2 --method admm_epll --m2-backend torch --stride 1 --gmm …` | 1.3 h + 5.3 h |
| [ ] | 2A-3 NAFNet synthetic only (domain-gap diagnostic) | `… --extra-images data/synthetic/syn-v1 --require-manifest --p-real 0` + evaluate | 2.8 h + 1 h |
| [ ] | Endpoint E1 (paired Wilcoxon, Holm) | analysis | minutes |

**2A total:** ≈10 GPU-h before the gate; ≈17 GPU-h more if synthetic data goes ahead.

## 2. Phase 2B: M3 on v1 (independent of the gate; can start now)

| ✓ | Run | Command | Time |
|---|---|---|---|
| [x] | USRNet | `scripts/train_m3.py --data-root data --model usrnet` | ≈1 h |
| [x] | ISTA-Net | `--model ista_net` | ≈0.3 h |
| [x] | LISTA | `--model lista` | ≈0.3 h |
| [x] | Evaluate the three on the v1 grid | `scripts/evaluate.py --arm m3 --checkpoint checkpoints/m3/<model>/best.pt` | 3 × ≈1 h |
| [ ] | M3 LPIPS (50-image subset) and H3 transfer with M3 as attacker and target | `evaluate.py … --subset 50 --sigmas 1,5,15,25 --lpips`; `stability_test.py --m3-checkpoint … [--attack-arm m3]` | ≈0.5 h + ≈4 h |

**2B total:** ≈9 GPU-h.

## 3. Phase 2C: protocol v2 (the final three-arm comparison)

Use the data setting chosen at the gate (real only, or real + `syn-v1`) for **every** learned component.

| ✓ | Run | Command | Time |
|---|---|---|---|
| [x] | NAFNet v2 | `scripts/train_m1.py --data-root data --protocol v2 [data options]` | ≈3.2 h |
| [x] | U-Net v2 | `--model unet` | ≈0.7 h |
| [x] | USRNet / ISTA-Net / LISTA v2 | `scripts/train_m3.py --data-root data --protocol v2 --model …` | ≈1.2 + 0.4 + 0.4 h |
| [ ] | **M3 full training, v2 and v1** (cap 30k → 200k, early stopping decides; all six 30k runs hit the cap) | `train_m3.py … --iterations 200000 --out checkpoints/[v2/]m3/<model>_full`, then test grids, 2C-5, LPIPS, H3 | ≈12 h train + ≈12 h eval |
| [ ] | **Real kernels (P2_real) and benchmarks**: Levin kernels on BSDS test; Levin 4 images (synthetic); Levin real captured blur (32); Sun et al. 2013 (640, 1% noise) | `evaluate.py --real-kernels levin=data/levin/kernels --psf-sets P2_real`, `--image-dir data/levin/images`; `scripts/evaluate_provided.py --triplets data/{levin,sun2013}/benchmark` | ≈20–24 h |
| [x] | GMM-200 (v2 uses the same data rule) | as in 2A-2 or the Phase 1 prior | 15 min CPU |
| [x] | λ re-tuning on real val for v2 (24 kernels × 5 σ × 10 λ) | `evaluate.py --protocol v2 --split val --subset 10 --psf-sets P1 --lambda-values …` per factor | FISTA-TV ≈5.7 h, ADMM-EPLL ≈2.3 h |
| [x] | v2 test grid, M1 and M3 (60 kernels × 5 σ × 200 images) | `evaluate.py --protocol v2 --arm m1|m3 --checkpoint …` | ≈1–2 h per model |
| [x] | v2 test grid, ADMM-EPLL (200 images) | `evaluate.py --protocol v2 --arm m2 --method admm_epll --m2-backend torch --stride 1 …` (shard with `--shard i/n`) | ≈10.5 h |
| [x] | v2 test grid, FISTA-TV (100-image subset) | `… --method fista_tv --subset 100` | ≈12 h |
| [x] | v2 test grid, Wiener and Richardson–Lucy (CPU, shard over cores) | `evaluate.py --protocol v2 --arm m2 --method wiener|richardson_lucy --shard i/6` | ≈1 h + ≈8 h CPU |
| [x] | 2C-3: v1-trained NAFNet on the v2 grid | `evaluate.py --protocol v2 --arm m1 --checkpoint checkpoints/m1/nafnet/best.pt` | ≈1.5 h |
| [x] | 2C-5: v2-trained NAFNet (and USRNet) on the **frozen v1** $P_2$ sets | `evaluate.py --protocol v1 --arm m1 --checkpoint checkpoints/v2/m1/nafnet/best.pt --psf-sets P2_interp,P2_extrap` | ≈0.5 h each |
| [ ] | 2C-6: K-sweep and per-σ specialists on v2 | `train_gmm.py --components K …`; `train_m1.py --protocol v2 --noise-over-255 S --init checkpoints/v2/m1/nafnet/best.pt`; evaluate (50-image subset for the K-sweep) | ≈25 min CPU + 5 × 0.3 h + ≈6 h |
| [x] | 2C-7: white-box H3 on M2 (reduced-iteration surrogate), small run | `stability_test.py --attack-arm m2 --m2-method fista_tv --lambda-values … --psf-set P1 --psf-limit 6 --sigmas 5,15 --limit 10` (and `admm_epll --gmm …`) | ≈1 h FISTA-TV + ≈2 h ADMM-EPLL |
| [ ] | H3 transfer on v2 (M1 attacker; M3 included) | `stability_test.py --protocol v2 --psf-set <set> --psf-limit 100 --sigmas 1,5,15,25 --limit 10 --m3-checkpoint …` | ≈6 h |
| [ ] | LPIPS on v2 (50-image subset, all methods) | `evaluate.py --protocol v2 --subset 50 --sigmas 1,5,15,25 --lpips …` | ≈8 h (FISTA-TV ≈5 h) |
| [ ] | H2 texture, E2/E2b/E3 statistics, report tables | analysis scripts | minutes |

**2C total:** ≈70 GPU-h + ≈10 CPU-h. FISTA-TV accounts for about 24 GPU-h of it (λ tuning, test grid, LPIPS, white-box run).

> [!note] 2C-5 caveat
> The v2 isotropic-Gaussian training grid (0.2 steps in [0.6, 2.4]) contains σ_k = 1.4 and 1.8, which are two of the three Gaussian kernels in the **v1** $P_2$-interp set. For the frozen-v1 coverage test, report the v1 $P_2$-interp motion kernels and $P_2$-extrap separately, or exclude those two Gaussians; otherwise part of the "held-out" set was seen in training.

## 4. Totals and order

| Block | GPU-h (exclusive GPU) | Notes |
|---|---|---|
| 2A before the gate | ≈10 | can run now |
| 2A after the gate (synthetic) | ≈17 | only if the gate passes |
| 2B (M3 on v1) | ≈9 | can run now, in parallel with 2A on the same GPU queue |
| 2C (v2) | ≈70 (+10 CPU) | after 2A/2B; the data setting comes from the gate |
| **Total** | **≈90 GPU-h without synthetic data, ≈105 with** | ≈4–4.5 days of continuous GPU; ≈7–9 days on a shared GPU |

**Recommended order:** 2B and 2A-1 first (both start now, ≈19 h) → gate → 2A-2/2A-3 if it passes → freeze v2 → 2C. Phase 3 (K-NAFNet, §12.9) starts after 2C-5 and is not implemented yet.

## 5. Results log

- **2026-11-15, 2A-1 learning curve:** NAFNet test PSNR (σ > 0) on P1 is 27.53 / 27.55 / 27.48 dB with 50 / 100 / 200 images; P2 drop −2.91 / −2.96 / −2.76 dB. The curve is flat, so the synthetic-data gate is **closed**; 2C uses real images only. DIV2K stays as a control once the download finishes (the ETH server sends about 22 KB/s).
- **2026-11-15, 2B M3 on v1:** USRNet 27.70 / 27.35 / 25.61 dB on P1 / P2-interp / P2-extrap, +0.27 / +0.27 / +0.24 dB over ADMM-EPLL (Holm p < 10⁻³⁰), P2 drop −0.43 dB (ADMM-EPLL −0.41, NAFNet −2.76). ISTA-Net 25.92 / 25.63 / 23.98; LISTA 23.99 / 23.79 / 22.75. Numbers: `scripts/analyze_phase2.py` → `report/generated/phase2.json`.
- **v2 re-frozen with distinct families** (2026-11-15, before any v2 run): training pool 218 (10 round Gaussian, 48 elliptical, 60 motion, 100 curved shake); test 24 P1 + 24 P2-interp + 12 P2-extrap (defocus, ring, ghost). Airy, Moffat, defocus r3 and thin/near-round ellipses dropped as near copies; max cross-family correlation 0.88. Contact sheet: `report/generated/v2_kernels.png`. No real kernels.
- **Queues:** `checkpoints/logs/phase2b_queue.sh` (M3 LPIPS, H3 with USRNet), then `phase2c_queue.sh` (2C-7 white-box M2, v2 training of all five networks, v2 M1/M3 test grids, 2C-3, 2C-5).
- **2026-11-16/17, v2 networks** (PSNR, σ > 0, P1 / P2-interp / P2-extrap): NAFNet 25.50 / 25.19 / 22.18, U-Net 25.00 / 24.80 / 21.62, USRNet (30k) 26.91 / 26.82 / 27.25, ISTA-Net 25.17 / 25.19 / 24.96, LISTA 23.50 / 23.52 / 22.91; blurred input 21.55 / 21.44 / 19.34. 2C-3 (v1 NAFNet on v2) 24.23 / 24.10 / 20.81. 2C-5 on the frozen v1 P2 sets: v2 NAFNet 25.83 / 23.54, v2 USRNet 27.28 / 25.56 (v1 USRNet 27.35 / 25.61; see the 2C-5 caveat).
- **2026-11-17, v2 λ selection** (10 factors, P1 val subset, 240 rows per cell): FISTA-TV unchanged from v1; ADMM-EPLL changes only at σ = 0 (factor 0.125). No grid-edge selections.
- **2026-11-18, v2 M2 test grids:** ADMM-EPLL 26.71 / 26.59 / 27.22 (200 images), Wiener 21.99 / 21.96 / 21.97, Richardson–Lucy 22.03 / 22.07 / 22.35, FISTA-TV 26.25 / 26.18 / 26.48 (100-image subset; on the same 100 images ADMM-EPLL 26.90 / 26.79 / 27.38, USRNet 27.10 / 27.01 / 27.42, NAFNet 25.73 / 25.42 / 22.36). The classical GMM prior is within 0.2 dB of the 30k USRNet everywhere and 0.03 dB on unseen families.
- **2026-11-18, kernel conditioning** (`scripts/analyze_v2_spectra.py` → `report/generated/v2_spectra.json`): across the 60 test kernels, the gain over the blurred input of the physics-aware methods falls with the fraction of near-zero frequencies (|K| < 0.1 on the 321×481 grid): Spearman ρ = −0.80 ADMM-EPLL, −0.77 FISTA-TV, −0.76 USRNet and ISTA-Net (p < 10⁻¹¹), but only −0.32 NAFNet and −0.04 U-Net. On ghost kernels (13% near-zero) USRNet gains +12.7 dB and EPLL +13.0, NAFNet +1.9; on defocus discs (96%) USRNet +4.6, NAFNet +2.7. Data-only networks apply a roughly fixed sharpening; the physics step is what generalises.
- **2026-11-18, M3 cap removed.** All six M3 runs (v1 and v2) stopped at the 30k cap, most still improving. Retraining with a 200k cap (NAFNet's sample budget at batch 8) and early stopping, as a documented deviation; the 30k results stay as the pre-registered numbers. **USRNet v2 full:** early stop at 80k, best at 72k, val PSNR **29.75 dB vs 29.30 dB** at the 30k cap (+0.46 dB). Test grids follow in `phase2f_queue.sh`; H3 and LPIPS on v2 now use the fully trained M3.
- **2026-11-18, real-kernel benchmarks prepared.** Levin et al. (2009): 8 kernels, stored rotated 180° in the original `.mat` files (verified: median residual 1.7/255 rotated vs 9.0/255 as-is); evaluated as a separate set `P2_real` so the pre-registered P2-extrap is unchanged. Sun et al. (2013): 80 grayscale images × the 8 Levin kernels, 1% noise; ground truth is the scene centre (50 px removed per side), kernels already in our convention (residual 2.55/255 = stated noise). Sun's own EPLL with true kernels scores 33.20 dB on the first 10 images under our metric. Provided-blur inputs are not circular: `scripts/evaluate_provided.py` uses edge padding + edgetaper and best-shift PSNR. On the first 8 Levin real photos, plain symmetric padding cost the 30k USRNet ~5 dB (24.7 vs 29.6 dB with edgetaper); USRNet scores 36.7 dB on the same kernels with circular synthetic blur, so unrolled networks are sensitive to real image boundaries.
- **Queues:** `phase2e_queue.sh` (λ re-tuning, v2 M2 test grids) → `phase2f_queue.sh` (M3 full training + evals, DIV2K, H3 v2, LPIPS v2, 2C-6, v1 M3 full) → `phase2g_queue.sh` (P2_real, Levin, Sun).
