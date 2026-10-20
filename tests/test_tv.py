import unittest

import numpy as np

from deblur.metrics import psnr
from deblur.methods.classical.fista_tv import fista_tv, suggest_tv_lambda
from deblur.methods.classical.tv import divergence, gradient, tv_denoise_admm, tv_norm
from deblur.operators.fft_conv import fft_convolve
from deblur.psf import gaussian_kernel


class TestTVOperators(unittest.TestCase):
    def test_adjoint_identity(self):
        rng = np.random.default_rng(0)
        u = rng.random((8, 8))
        px = rng.random((8, 8))
        py = rng.random((8, 8))
        lhs = np.sum(gradient(u)[0] * px + gradient(u)[1] * py)
        rhs = -np.sum(u * divergence(px, py))
        self.assertAlmostEqual(lhs, rhs, places=12)

    def test_constant_has_zero_tv(self):
        self.assertAlmostEqual(tv_norm(np.full((8, 8), 0.3)), 0.0, places=12)

    def test_zero_weight_is_identity(self):
        rng = np.random.default_rng(0)
        f = rng.random((16, 16))
        np.testing.assert_array_equal(tv_denoise_admm(f, weight=0.0), f)

    def test_denoising_reduces_variation(self):
        rng = np.random.default_rng(0)
        f = rng.random((24, 24))
        u = tv_denoise_admm(f, weight=0.1, n_iter=100)
        self.assertLess(tv_norm(u), tv_norm(f))
        self.assertAlmostEqual(float(u.mean()), float(f.mean()), places=3)

    def test_large_weight_gives_constant(self):
        rng = np.random.default_rng(0)
        f = rng.random((16, 16))
        u = tv_denoise_admm(f, weight=1.0, n_iter=200)
        self.assertLess(float(u.std()), 1e-3)


class TestFISTATV(unittest.TestCase):
    def setUp(self):
        x = np.zeros((48, 48))
        x[8:24, 8:40] = 0.8
        x[28:42, 10:30] = 0.3
        self.x = x
        self.k = gaussian_kernel(1.2)

    def test_delta_kernel_identity_with_zero_lambda(self):
        k = np.zeros((5, 5))
        k[2, 2] = 1.0
        restored = fista_tv(self.x, k, lam=0.0, n_iter=10, tv_iter=5)
        np.testing.assert_allclose(restored, self.x, atol=1e-6)

    def test_noiseless_blur_improves(self):
        y = fft_convolve(self.x, self.k)
        restored = fista_tv(y, self.k, lam=1e-2, n_iter=60, tv_iter=20)
        self.assertGreater(psnr(self.x, restored), psnr(self.x, y) + 5.0)

    def test_output_bounded(self):
        y = fft_convolve(self.x, self.k)
        restored = fista_tv(y, self.k, lam=1e-2, n_iter=20, tv_iter=10)
        self.assertGreaterEqual(float(restored.min()), 0.0)
        self.assertLessEqual(float(restored.max()), 1.0)

    def test_lambda_suggestion_scales_with_noise(self):
        self.assertGreater(suggest_tv_lambda(0.1), suggest_tv_lambda(0.01))


if __name__ == "__main__":
    unittest.main()
