import unittest

import numpy as np

from deblur.data import degrade
from deblur.metrics import psnr
from deblur.methods.classical.admm_epll import admm_epll
from deblur.methods.classical.gmm import fit_gmm
from deblur.psf import gaussian_kernel


class TestADMMEPLL(unittest.TestCase):
    def setUp(self):
        x = np.zeros((24, 24))
        x[4:12, 4:20] = 0.8
        x[14:21, 5:17] = 0.3
        self.x = x
        patch_size = 6
        patches = []
        for i in range(0, x.shape[0] - patch_size + 1, 2):
            for j in range(0, x.shape[1] - patch_size + 1, 2):
                patch = x[i : i + patch_size, j : j + patch_size].ravel()
                patches.append(patch - patch.mean())
        self.gmm, _ = fit_gmm(np.asarray(patches), n_components=2, n_iter=5, seed=0)
        self.k = gaussian_kernel(1.0)

    def test_restores_and_improves(self):
        y = degrade(self.x, self.k, sigma=0.05, rng=np.random.default_rng(1))
        restored = admm_epll(
            y,
            self.k,
            self.gmm,
            sigma=0.05,
            n_iter=4,
            patch_size=6,
            stride=2,
        )
        self.assertEqual(restored.shape, self.x.shape)
        self.assertTrue(np.isfinite(restored).all())
        self.assertGreater(psnr(self.x, restored), psnr(self.x, y))

    def test_output_bounded(self):
        y = degrade(self.x, self.k, sigma=0.05, rng=np.random.default_rng(1))
        restored = admm_epll(
            y,
            self.k,
            self.gmm,
            sigma=0.05,
            n_iter=2,
            patch_size=6,
            stride=2,
        )
        self.assertGreaterEqual(float(restored.min()), 0.0)
        self.assertLessEqual(float(restored.max()), 1.0)

    def test_invalid_parameters(self):
        y = np.zeros((24, 24))
        with self.assertRaises(ValueError):
            admm_epll(y, self.k, self.gmm, n_iter=0)
        with self.assertRaises(ValueError):
            admm_epll(y, self.k, self.gmm, sigma_start=0.0)
        with self.assertRaises(ValueError):
            admm_epll(y, self.k, self.gmm, lam=0.0)

    def test_noiseless_inverts_blur(self):
        # With no noise the early, weakly weighted x-updates must undo the blur.
        y = degrade(self.x, self.k, sigma=0.0)
        restored = admm_epll(y, self.k, self.gmm, sigma=0.0, n_iter=8, patch_size=6, stride=2)
        self.assertGreater(psnr(self.x, restored), psnr(self.x, y) + 5.0)


if __name__ == "__main__":
    unittest.main()
