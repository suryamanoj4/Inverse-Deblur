import unittest

import numpy as np

from deblur.metrics.consistency import data_consistency, psnr_drop
from deblur.metrics.quality import psnr, ssim
from deblur.operators.fft_conv import fft_convolve
from deblur.psf.library import gaussian_kernel


class TestPSNR(unittest.TestCase):
    def test_identical_is_infinite(self):
        x = np.random.default_rng(0).random((16, 16))
        self.assertEqual(psnr(x, x), float("inf"))

    def test_known_value(self):
        x = np.zeros((8, 8))
        y = x + 0.1
        self.assertAlmostEqual(psnr(x, y), 20.0, places=6)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            psnr(np.zeros((4, 4)), np.zeros((4, 5)))


class TestSSIM(unittest.TestCase):
    def test_identical_is_one(self):
        x = np.random.default_rng(0).random((32, 32))
        self.assertAlmostEqual(ssim(x, x), 1.0, places=6)

    def test_different_is_lower(self):
        rng = np.random.default_rng(0)
        x = rng.random((32, 32))
        y = rng.random((32, 32))
        self.assertLess(ssim(x, y), 1.0)
        self.assertGreaterEqual(ssim(x, y), -1.0)

    def test_requires_2d(self):
        with self.assertRaises(ValueError):
            ssim(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))


class TestConsistency(unittest.TestCase):
    def test_exact_pair_is_consistent(self):
        rng = np.random.default_rng(0)
        x = rng.random((24, 24))
        k = gaussian_kernel(1.0)
        y = fft_convolve(x, k)
        report = data_consistency(x, y, k)
        self.assertAlmostEqual(report.l2, 0.0, places=10)
        self.assertAlmostEqual(report.rmse, 0.0, places=10)

    def test_noise_increases_residual(self):
        rng = np.random.default_rng(0)
        x = rng.random((24, 24))
        k = gaussian_kernel(1.0)
        y = fft_convolve(x, k) + 0.05 * rng.standard_normal(x.shape)
        report = data_consistency(x, y, k)
        self.assertGreater(report.l2, 0.0)
        self.assertGreater(report.rmse, 0.0)

    def test_psnr_drop(self):
        self.assertAlmostEqual(psnr_drop(30.0, 24.0), 6.0)


if __name__ == "__main__":
    unittest.main()
