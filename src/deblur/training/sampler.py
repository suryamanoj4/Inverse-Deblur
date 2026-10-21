"""Training batches for the learned arms, following the locked protocol.

Each patch is a random crop of a training image, optionally flipped/rotated,
then degraded with ``k ~ Uniform(P1)`` and ``sigma ~ Uniform(noise grid)``
drawn independently per patch (``condition_sampling = uniform_independent``).
Degradation runs on the training device with the same forward model as
:func:`deblur.data.degrade`: circular FFT blur, additive Gaussian noise, clip
to [0, 1]. Augmentation is applied to the clean patch *before* degradation, so
it never changes the degradation distribution.
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
        self.images = [np.asarray(image, dtype=np.float32) for image in images]
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

    def sample(self) -> Batch:
        B = self.batch_size
        image_index = self.rng.integers(len(self.images), size=B)
        clean = np.stack([self._crop(self.images[i]) for i in image_index])
        psf_index = self.rng.integers(len(self.spectra), size=B)
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
