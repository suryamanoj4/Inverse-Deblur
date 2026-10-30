import unittest

import numpy as np

from deblur.data import degrade
from deblur.methods.classical import admm_epll, fista_tv
from deblur.methods.classical.gmm import fit_gmm
from deblur.psf import gaussian_kernel, motion_kernel

try:  # optional
    import torch

    from deblur.methods.classical.torch_backend import TorchGMM, admm_epll_torch, fista_tv_torch

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestTorchBackendMatchesNumPy(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        x = np.zeros((40, 36))
        x[6:20, 5:30] = 0.8
        x[24:34, 8:22] = 0.3
        self.x = np.clip(x + 0.02 * rng.standard_normal(x.shape), 0.0, 1.0)
        patches = []
        for i in range(0, x.shape[0] - 6 + 1, 2):
            for j in range(0, x.shape[1] - 6 + 1, 2):
                patch = self.x[i : i + 6, j : j + 6].ravel()
                patches.append(patch - patch.mean())
        self.gmm, _ = fit_gmm(np.asarray(patches), n_components=3, n_iter=5, seed=0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_fista_tv(self):
        k = motion_kernel(7, 30.0)
        y = degrade(self.x, k, sigma=0.02, rng=np.random.default_rng(1))
        ref = fista_tv(y, k, lam=0.002, n_iter=20, tv_iter=10)
        out = fista_tv_torch(y, k, lam=0.002, n_iter=20, tv_iter=10, device=self.device)
        np.testing.assert_allclose(out, ref, atol=1e-10)

    def test_admm_epll(self):
        k = gaussian_kernel(1.2)
        for stride in (1, 2, 4):
            y = degrade(self.x, k, sigma=0.02, rng=np.random.default_rng(2))
            ref = admm_epll(y, k, self.gmm, sigma=0.02, n_iter=4, patch_size=6, stride=stride)
            gmm = TorchGMM(self.gmm, device=self.device)
            out = admm_epll_torch(y, k, gmm, sigma=0.02, n_iter=4, patch_size=6, stride=stride)
            # float32 patch arithmetic on the GPU
            np.testing.assert_allclose(out, ref, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
