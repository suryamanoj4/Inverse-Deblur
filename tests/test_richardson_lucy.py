import unittest

import numpy as np

from deblur.metrics import psnr
from deblur.methods.classical.richardson_lucy import richardson_lucy
from deblur.operators.fft_conv import fft_convolve
from deblur.psf import gaussian_kernel


class TestRichardsonLucy(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).random((32, 32))

    def test_delta_kernel_recovers_input(self):
        k = np.zeros((5, 5))
        k[2, 2] = 1.0
        restored = richardson_lucy(self.x, k, n_iter=10)
        np.testing.assert_allclose(restored, self.x, atol=1e-6)

    def test_output_is_nonnegative_and_bounded(self):
        k = gaussian_kernel(1.2)
        y = fft_convolve(self.x, k)
        restored = richardson_lucy(y, k, n_iter=20)
        self.assertGreaterEqual(float(restored.min()), 0.0)
        self.assertLessEqual(float(restored.max()), 1.0)

    def test_noiseless_blur_improves(self):
        k = gaussian_kernel(1.2)
        y = fft_convolve(self.x, k)
        restored = richardson_lucy(y, k, n_iter=50)
        self.assertGreater(psnr(self.x, restored), psnr(self.x, y))

    def test_invalid_iterations(self):
        k = gaussian_kernel(1.0)
        with self.assertRaises(ValueError):
            richardson_lucy(self.x, k, n_iter=0)


if __name__ == "__main__":
    unittest.main()
