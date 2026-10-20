"""Locked protocol configuration for the three-arm study.

These values are the single source of truth. Changing any of them changes the
experiment and must be documented in the protocol manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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

    @property
    def noise_levels_over_255(self) -> tuple[int, ...]:
        """Noise levels in the usual ``/255`` notation, for reporting."""
        return tuple(round(level * 255) for level in self.noise_levels)

    def manifest(self) -> dict:
        """JSON-serialisable snapshot of the protocol."""
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
        }
