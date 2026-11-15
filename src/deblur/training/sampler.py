"""Training batches for the learned arms, following the locked protocol.

Each patch is a random crop of a training image, optionally flipped/rotated,
then degraded with ``k ~ Uniform(P1)`` and ``sigma ~ Uniform(noise grid)``
drawn independently per patch (``condition_sampling = uniform_independent``).
Degradation runs on the training device with the same forward model as
:func:`deblur.data.degrade`: circular FFT blur, additive Gaussian noise, clip
to [0, 1]. Augmentation is applied to the clean patch *before* degradation, so
it never changes the degradation distribution.

Phase 2 options (all off by default, so protocol v1 batches are unchanged):

- ``extra_images`` with ``p_real``: each patch comes from the real training
  images with probability ``p_real`` and from the extra pool (synthetic images
  or DIV2K) otherwise;
- ``kernel_groups``: one family label per kernel; the family is drawn
  uniformly first, then a kernel within it (``family_balanced``);
- ``quantize``: observations are rounded to 8-bit levels, matching the
  protocol v2 evaluation observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..operators.fft_conv import kernel_spectrum


@dataclass
class Batch:
    clean: torch.Tensor  # (B, 1, P, P)
    observed: torch.Tensor  # (B, 1, P, P)
    psf_index: torch.Tensor  # (B,)
    sigma_index: torch.Tensor  # (B,)
    kernel: torch.Tensor  # (B, P, P) complex spectrum, for physics-aware arms
    sigma: torch.Tensor | None = None  # (B,) noise levels, for physics-aware arms


class PatchSampler:
    def __init__(
        self,
        images: list[np.ndarray],
        kernels: list[np.ndarray],
        noise_levels: tuple[float, ...] | list[float],
        patch_size: int,
        batch_size: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
        augment: bool = True,
        extra_images: list[np.ndarray] | None = None,
        p_real: float = 1.0,
        kernel_groups: list[str] | None = None,
        quantize: bool = False,
    ) -> None:
        if not images:
            raise ValueError("no training images")
        if not kernels:
            raise ValueError("no training kernels")
        if not noise_levels:
            raise ValueError("no noise levels")
        for image in images:
            if image.ndim != 2:
                raise ValueError(f"expected 2D luminance images, got shape {image.shape}")
            if min(image.shape) < patch_size:
                raise ValueError(f"image {image.shape} is smaller than patch size {patch_size}")
        extra_images = list(extra_images or [])
        if not 0.0 <= p_real <= 1.0:
            raise ValueError("p_real must be in [0, 1]")
        if p_real < 1.0 and not extra_images:
            raise ValueError("p_real < 1 needs extra_images")
        for image in extra_images:
            if image.ndim != 2 or min(image.shape) < patch_size:
                raise ValueError(f"extra image {image.shape} is not 2D or is smaller than the patch")
        if kernel_groups is not None and len(kernel_groups) != len(kernels):
            raise ValueError("kernel_groups needs one label per kernel")
        self.images = [np.asarray(image, dtype=np.float32) for image in images]
        self.extra_images = [np.asarray(image, dtype=np.float32) for image in extra_images]
        self.p_real = float(p_real)
        self.quantize = quantize
        self.kernel_groups = None
        if kernel_groups is not None:
            labels = sorted(set(kernel_groups))
            self.kernel_groups = [
                np.flatnonzero(np.asarray(kernel_groups) == label) for label in labels
            ]
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.noise_generator = torch.Generator(device=self.device)
        self.noise_generator.manual_seed(seed)
        spectra = np.stack([kernel_spectrum(k, (patch_size, patch_size)) for k in kernels])
        self.spectra = torch.as_tensor(spectra.astype(np.complex64), device=self.device)
        self.noise_levels = torch.as_tensor(
            np.asarray(noise_levels, dtype=np.float32), device=self.device
        )

    def _crop(self, image: np.ndarray) -> np.ndarray:
        P = self.patch_size
        H, W = image.shape
        top = int(self.rng.integers(0, H - P + 1))
        left = int(self.rng.integers(0, W - P + 1))
        patch = image[top : top + P, left : left + P]
        if self.augment:
            patch = np.rot90(patch, k=int(self.rng.integers(4)))
            if self.rng.integers(2):
                patch = patch[:, ::-1]
        return patch

    def _clean_patches(self, B: int) -> np.ndarray:
        if self.p_real >= 1.0:
            image_index = self.rng.integers(len(self.images), size=B)
            return np.stack([self._crop(self.images[i]) for i in image_index])
        from_real = self.rng.random(B) < self.p_real
        patches = []
        for real in from_real:
            pool = self.images if real else self.extra_images
            patches.append(self._crop(pool[int(self.rng.integers(len(pool)))]))
        return np.stack(patches)

    def _kernel_indices(self, B: int) -> np.ndarray:
        if self.kernel_groups is None:
            return self.rng.integers(len(self.spectra), size=B)
        families = self.rng.integers(len(self.kernel_groups), size=B)
        return np.array(
            [self.kernel_groups[f][int(self.rng.integers(len(self.kernel_groups[f])))] for f in families]
        )

    def sample(self) -> Batch:
        B = self.batch_size
        clean = self._clean_patches(B)
        psf_index = self._kernel_indices(B)
        sigma_index = self.rng.integers(len(self.noise_levels), size=B)

        x = torch.as_tensor(clean, device=self.device)[:, None]
        psf_t = torch.as_tensor(psf_index, device=self.device)
        sigma_t = torch.as_tensor(sigma_index, device=self.device)
        spectra = self.spectra[psf_t]
        blurred = torch.fft.ifft2(torch.fft.fft2(x) * spectra[:, None]).real
        noise = torch.randn(
            blurred.shape, generator=self.noise_generator, device=self.device, dtype=blurred.dtype
        )
        sigma = self.noise_levels[sigma_t].view(B, 1, 1, 1)
        y = torch.clamp(blurred + sigma * noise, 0.0, 1.0)
        if self.quantize:
            y = torch.round(y * 255.0) / 255.0
        return Batch(
            clean=x,
            observed=y,
            psf_index=psf_t,
            sigma_index=sigma_t,
            kernel=spectra,
            sigma=self.noise_levels[sigma_t],
        )

    def state_dict(self) -> dict:
        return {
            "rng": self.rng.bit_generator.state,
            "noise_generator": self.noise_generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
        # torch.load(map_location=...) may move the state to the GPU; it must be a CPU ByteTensor
        self.noise_generator.set_state(state["noise_generator"].cpu())
