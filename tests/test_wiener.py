import unittest

import numpy as np

from deblur.metrics import psnr
from deblur.methods.classical.wiener import wiener
from deblur.operators.fft_conv import fft_convolve
from deblur.psf import gaussian_kernel


class TestWiener(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).random((32, 32))
        self.k = gaussian_kernel(1.2)

    def test_delta_kernel_identity(self):
        k = np.zeros((5, 5))
        k[2, 2] = 1.0
        np.testing.assert_allclose(wiener(self.x, k, sigma=0.0), self.x, atol=1e-6)

    def test_noiseless_recovery(self):
        y = fft_convolve(self.x, self.k)
        restored = wiener(y, self.k, sigma=0.0)
        self.assertGreater(psnr(self.x, restored), 40.0)

    def test_noisy_improves_over_input(self):
        x = np.zeros((64, 64))
        x[10:30, 8:40] = 0.8
        x[36:54, 12:48] = 0.3
        rng = np.random.default_rng(1)
        y = fft_convolve(x, self.k) + 0.03 * rng.standard_normal(x.shape)
        y = np.clip(y, 0.0, 1.0)
        restored = wiener(y, self.k, sigma=0.03)
        self.assertGreater(psnr(x, restored), psnr(x, y))

    def test_explicit_lambda(self):
        y = fft_convolve(self.x, self.k)
        a = wiener(y, self.k, lam=1e-3)
        b = wiener(y, self.k, lam=1e-1)
        self.assertFalse(np.allclose(a, b))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            wiener(np.zeros((4, 4, 3)), self.k)


if __name__ == "__main__":
    unittest.main()
