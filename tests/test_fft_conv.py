import unittest

import numpy as np
from scipy.signal import convolve2d

from deblur.operators.fft_conv import fft_adjoint, fft_convolve, pad_kernel

try:  # optional
    import torch

    from deblur.operators.fft_conv import FFTConv2d, fft_convolve_torch

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_TORCH = False


class TestPadKernel(unittest.TestCase):
    def test_delta_placed_at_origin(self):
        k = np.zeros((5, 5))
        k[2, 2] = 1.0
        padded = pad_kernel(k, (9, 9))
        self.assertEqual(padded[0, 0], 1.0)
        self.assertEqual(padded.sum(), 1.0)

    def test_rejects_large_kernel(self):
        with self.assertRaises(ValueError):
            pad_kernel(np.ones((5, 5)), (4, 4))

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            pad_kernel(np.ones(5), (8, 8))


class TestFFTConv(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.random((24, 30))
        k = rng.random((5, 5))
        self.k = k / k.sum()

    def test_matches_scipy_circular_convolution(self):
        reference = convolve2d(self.x, self.k, mode="same", boundary="wrap")
        np.testing.assert_allclose(fft_convolve(self.x, self.k), reference, atol=1e-10)

    def test_delta_kernel_is_identity(self):
        k = np.zeros((5, 5))
        k[2, 2] = 1.0
        np.testing.assert_allclose(fft_convolve(self.x, k), self.x, atol=1e-12)

    def test_adjoint_inner_product(self):
        rng = np.random.default_rng(1)
        y = rng.random(self.x.shape)
        lhs = np.sum(fft_convolve(self.x, self.k) * y)
        rhs = np.sum(self.x * fft_adjoint(y, self.k))
        self.assertAlmostEqual(lhs, rhs, places=10)

    def test_batched_input(self):
        batch = np.stack([self.x, 2.0 * self.x])
        out = fft_convolve(batch, self.k)
        np.testing.assert_allclose(out[1], 2.0 * out[0], atol=1e-10)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestTorchOperator(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.x = rng.random((20, 22))
        k = rng.random((5, 5))
        self.k = k / k.sum()

    def test_matches_numpy(self):
        x = torch.as_tensor(self.x, dtype=torch.float64)
        k = torch.as_tensor(self.k, dtype=torch.float64)
        out = fft_convolve_torch(x, k).numpy()
        np.testing.assert_allclose(out, fft_convolve(self.x, self.k), atol=1e-10)

    def test_module_matches_numpy(self):
        x = torch.as_tensor(self.x, dtype=torch.float32)
        module = FFTConv2d(self.k)
        out = module(x).detach().numpy()
        np.testing.assert_allclose(out, fft_convolve(self.x, self.k), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
