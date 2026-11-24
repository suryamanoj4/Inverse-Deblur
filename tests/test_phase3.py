"""Phase 3: kernel-placement models and the mismatch-aware sampler."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.ndimage import convolve  # noqa: E402

from deblur.methods.unrolled import build_model  # noqa: E402
from deblur.methods.unrolled.placement import kernel_from_spectrum  # noqa: E402
from deblur.operators.fft_conv import kernel_spectrum  # noqa: E402
from deblur.training.sampler import PatchSampler  # noqa: E402


def _kernel():
    k = np.zeros((9, 9))
    k[4, :] = 1
    k[2, 3] = 0.5
    return k / k.sum()


def test_kernel_from_spectrum_recovers_kernel():
    rng = np.random.default_rng(0)
    k = rng.random((15, 21))
    k /= k.sum()
    spectrum = torch.as_tensor(kernel_spectrum(k, (64, 64)), dtype=torch.complex64)[None]
    out = kernel_from_spectrum(spectrum, 31)[0].numpy()
    assert np.allclose(out[8:23, 5:26], k, atol=1e-6)
    assert out.sum() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("name,kwargs", [("nafnet_concat", {}), ("nafnet_film", {}), ("knafnet", {}),
                                         ("nafnet_unrolled", {"n_iter": 2})])
def test_placement_models_shape_and_grad(name, kwargs):
    torch.manual_seed(0)
    model = build_model(name, **kwargs)
    y = torch.rand(2, 1, 48, 40)
    spectrum = torch.as_tensor(kernel_spectrum(_kernel(), (48, 40)), dtype=torch.complex64)[None].expand(2, -1, -1)
    out = model(y, spectrum, torch.tensor([1 / 255, 5 / 255]))
    assert out.shape == y.shape
    out.mean().backward()


def test_circular_batches_unchanged_by_default():
    rng = np.random.default_rng(0)
    images = [rng.random((80, 80)).astype(np.float32)]
    a = PatchSampler(images, [_kernel()], [0.0, 5 / 255], 32, 2, seed=1).sample()
    b = PatchSampler(images, [_kernel()], [0.0, 5 / 255], 32, 2, seed=1, mismatch=0.0).sample()
    assert torch.equal(a.observed, b.observed) and a.crop == 0


def test_mismatch_batch_is_linear_convolution():
    rng = np.random.default_rng(0)
    images = [rng.random((120, 120)).astype(np.float32)]
    k = _kernel()
    batch = PatchSampler(images, [k], [0.0], 48, 1, augment=False, mismatch=1.0, mismatch_pad=20).sample()
    assert batch.crop == 20 and batch.observed.shape[-1] == 48 + 40 and batch.clean.shape[-1] == 48
    x = batch.clean[0, 0].numpy().astype(np.float64)
    y = batch.observed[0, 0].numpy()[20:-20, 20:-20]
    ref = convolve(x, k, mode="constant")
    assert np.abs(y[6:-6, 6:-6] - ref[6:-6, 6:-6]).max() < 1e-5


def test_kernel_noise_keeps_a_normalised_kernel():
    rng = np.random.default_rng(0)
    images = [rng.random((120, 120)).astype(np.float32)]
    batch = PatchSampler(images, [_kernel()], [0.0], 48, 2, mismatch=1.0, kernel_noise=0.1).sample()
    assert torch.allclose(batch.kernel[:, 0, 0].real, torch.ones(2), atol=1e-5)


def test_kernel_noise_only_keeps_circular_blur():
    rng = np.random.default_rng(0)
    images = [rng.random((120, 120)).astype(np.float32)]
    k = _kernel()
    kw = dict(augment=False, mismatch=1.0, mismatch_boundary=False)
    clean = PatchSampler(images, [k], [0.0], 48, 2, seed=3, **kw).sample()
    noisy = PatchSampler(images, [k], [0.0], 48, 2, seed=3, kernel_noise=0.1, **kw).sample()
    ref = PatchSampler(images, [k], [0.0], 48, 2, seed=3, augment=False).sample()
    assert clean.crop == 0 and clean.observed.shape == clean.clean.shape
    assert torch.equal(clean.observed, noisy.observed)
    assert torch.allclose(clean.kernel, ref.kernel, atol=1e-5)  # same spectrum convention
    assert not torch.allclose(noisy.kernel, ref.kernel, atol=1e-3)
    assert torch.allclose(noisy.kernel[:, 0, 0].real, torch.ones(2), atol=1e-5)
