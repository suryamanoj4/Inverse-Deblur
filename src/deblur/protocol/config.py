"""Locked protocol configuration for the three-arm study.

These values are the single source of truth. ``manifest()`` is written to disk
with every run, so the PSF split, sampling policy, checkpoint rule, and learned
arm hyperparameters are all recorded rather than living only in prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..psf.library import protocol_psf_sets


@dataclass(frozen=True)
class ProtocolConfig:
    """Configuration shared by every arm."""

    data_root: str
    patch_size: int = 128
    # stored as absolute fractions; 1/255 == noise level "1" on the usual grid
    noise_levels: tuple[float, ...] = (0.0, 1 / 255, 5 / 255, 15 / 255, 25 / 255)
    gaussian_sigmas: tuple[float, ...] = (0.7, 1.2, 1.6, 2.0)
    defocus_radii: tuple[float, ...] = (3.0, 5.0, 7.0, 9.0)
    levin_dir: str | None = None
    seed: int = 0
    luminance: bool = True
    # "full" evaluates the whole test image; "center_crop" is the secondary view
    eval_mode: str = "full"
    eval_crop: int = 256

    # --- learned-arm protocol -------------------------------------------------
    # single_model: one M1 across the grid; per_noise: the oracle-sigma ablation
    m1_regime: str = "single_model"
    # uniform_independent: k ~ U(P1), sigma ~ U(noise grid), per patch
    condition_sampling: str = "uniform_independent"
    # mean_val_psnr: one checkpoint chosen on the fixed validation grid
    checkpoint_rule: str = "mean_val_psnr"
    train_psf_set: str = "P1"
    eval_psf_sets: tuple[str, ...] = ("P1", "P2_interp", "P2_extrap")
    val_subset_size: int = 10
    val_subset_seed: int = 0
    optimizer: str = "adamw"
    loss: str = "l2"
    m1_batch_size: int = 16
    m3_batch_size: int = 8
    m1_iterations: int = 100_000
    m3_iterations: int = 30_000
    m1_learning_rate: float = 2e-4
    m3_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_iterations: int = 1_000
    val_interval: int = 2_000
    gradient_clip: float = 1.0
    early_stop_patience: int = 4

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if not self.noise_levels:
            raise ValueError("noise_levels must not be empty")
        if any(not 0.0 <= s <= 1.0 for s in self.noise_levels):
            raise ValueError("noise levels must be fractions in [0, 1]")
        if self.eval_mode not in ("full", "center_crop"):
            raise ValueError("eval_mode must be 'full' or 'center_crop'")
        if self.eval_mode == "center_crop" and self.eval_crop <= 0:
            raise ValueError("eval_crop must be positive in center_crop mode")

        if self.m1_regime not in ("single_model", "per_noise"):
            raise ValueError("m1_regime must be 'single_model' or 'per_noise'")
        if self.condition_sampling not in (
            "uniform_independent",
            "family_balanced",
            "stratified_noise",
        ):
            raise ValueError("unknown condition_sampling policy")
        if self.checkpoint_rule not in ("mean_val_psnr", "per_condition"):
            raise ValueError("checkpoint_rule must be 'mean_val_psnr' or 'per_condition'")
        if self.optimizer != "adamw":
            raise ValueError("the locked optimizer is AdamW")
        if self.loss not in ("l1", "l2"):
            raise ValueError("loss must be 'l1' or 'l2'")

        psf_sets = protocol_psf_sets()
        if self.train_psf_set not in psf_sets:
            raise ValueError(f"train_psf_set must be one of {tuple(psf_sets)}")
        if not self.eval_psf_sets:
            raise ValueError("eval_psf_sets must not be empty")
        unknown = set(self.eval_psf_sets) - set(psf_sets)
        if unknown:
            raise ValueError(f"unknown eval_psf_sets: {sorted(unknown)}")

        for name in (
            "m1_batch_size",
            "m3_batch_size",
            "m1_iterations",
            "m3_iterations",
            "warmup_iterations",
            "val_interval",
            "early_stop_patience",
            "val_subset_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("m1_learning_rate", "m3_learning_rate", "gradient_clip"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

    @property
    def noise_levels_over_255(self) -> tuple[int, ...]:
        """Noise levels in the usual ``/255`` notation, for reporting."""
        return tuple(round(level * 255) for level in self.noise_levels)

    def manifest(self) -> dict:
        """JSON-serialisable snapshot of the protocol, including PSF membership."""
        psf_sets = protocol_psf_sets()
        return {
            "data_root": str(Path(self.data_root)),
            "patch_size": self.patch_size,
            "noise_levels": list(self.noise_levels),
            "noise_levels_over_255": list(self.noise_levels_over_255),
            "gaussian_sigmas": list(self.gaussian_sigmas),
            "defocus_radii": list(self.defocus_radii),
            "levin_dir": self.levin_dir,
            "seed": self.seed,
            "luminance": self.luminance,
            "eval_mode": self.eval_mode,
            "eval_crop": self.eval_crop,
            "m1_regime": self.m1_regime,
            "condition_sampling": self.condition_sampling,
            "checkpoint_rule": self.checkpoint_rule,
            "train_psf_set": self.train_psf_set,
            "eval_psf_sets": list(self.eval_psf_sets),
            "val_subset_size": self.val_subset_size,
            "val_subset_seed": self.val_subset_seed,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "m1_batch_size": self.m1_batch_size,
            "m3_batch_size": self.m3_batch_size,
            "m1_iterations": self.m1_iterations,
            "m3_iterations": self.m3_iterations,
            "m1_learning_rate": self.m1_learning_rate,
            "m3_learning_rate": self.m3_learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_iterations": self.warmup_iterations,
            "val_interval": self.val_interval,
            "gradient_clip": self.gradient_clip,
            "early_stop_patience": self.early_stop_patience,
            "psf_sets": {
                name: [
                    {"name": psf.name, "kind": psf.kind, "shape": list(psf.kernel.shape)}
                    for psf in kernels
                ]
                for name, kernels in psf_sets.items()
            },
        }
