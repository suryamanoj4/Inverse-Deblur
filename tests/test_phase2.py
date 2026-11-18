"""Phase 2: versioned protocol, new kernel families, quantized observations,
training-pool mixing, synthetic-set quality control, batched and
differentiable M2 solvers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from deblur.data import degrade
from deblur.data.synthetic import (
    entropy,
    hamming,
    laplacian_variance,
    nested_subset,
    normalize_size,
    perceptual_hash,
    spectrum_slope,
)
from deblur.protocol import ProtocolConfig, observe, psf_sets_digest, quantize_8bit
from deblur.psf import (
    airy_kernel,
    anisotropic_gaussian_kernel,
    fit_support,
    gaussian_kernel,
    ghost_kernel,
    kernel_axes,
    max_cross_family_correlation,
    moffat_kernel,
    motion_kernel,
    protocol_psf_sets,
    ring_kernel,
    trajectory_kernel,
)

try:  # optional
    import torch

    from deblur.training import PatchSampler

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_TORCH = False

# Protocol v1 is frozen: these kernels produced the mid-evaluation results.
V1_DIGEST = "c3b1a3c59b51c9f4d300e271bbdfb9301182f305ba690ccaea320a8620b33d83"


def smooth_image(shape=(48, 64), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random(shape)
    for _ in range(3):
        x = 0.25 * (np.roll(x, 1, 0) + np.roll(x, -1, 0) + np.roll(x, 1, 1) + np.roll(x, -1, 1))
    return (x - x.min()) / (x.max() - x.min())


class TestProtocolVersions(unittest.TestCase):
    def test_v1_kernel_sets_are_frozen(self):
        sets = protocol_psf_sets()
        self.assertEqual({k: len(v) for k, v in sets.items()}, {"P1": 12, "P2_interp": 11, "P2_extrap": 4})
        self.assertEqual(psf_sets_digest(sets), V1_DIGEST)

    def test_v2_sets(self):
        sets = protocol_psf_sets("v2")
        self.assertEqual(len(sets["P1"]), 24)
        self.assertEqual(len(sets["P2_interp"]), 24)
        self.assertEqual(len(sets["P1_pool"]), 218)
        pool = {psf.name for psf in sets["P1_pool"]}
        self.assertTrue({psf.name for psf in sets["P1"]} <= pool)
        held_out = [psf.name for psf in sets["P2_interp"] + sets["P2_extrap"]]
        self.assertFalse(set(held_out) & pool)
        self.assertEqual({psf.kind for psf in sets["P2_extrap"]}, {"defocus", "ring", "ghost"})
        for psf in (p for kernels in sets.values() for p in kernels):
            self.assertAlmostEqual(psf.kernel.sum(), 1.0, places=10)
            self.assertLessEqual(max(psf.kernel.shape), 31)
            self.assertTrue(np.all(psf.kernel >= 0))
        self.assertEqual(psf_sets_digest(sets), psf_sets_digest(protocol_psf_sets("v2")))

    def test_v2_families_are_distinct(self):
        sets = protocol_psf_sets("v2")
        kernels = sets["P1_pool"] + sets["P2_interp"] + sets["P2_extrap"]
        self.assertLessEqual(max_cross_family_correlation(kernels)[0], 0.90)
        straight = max(kernel_axes(p.kernel)[0] for p in kernels if p.kind == "motion")
        curved = min(kernel_axes(p.kernel)[0] for p in kernels if p.kind == "shake")
        self.assertGreater(curved, 1.5 * straight)  # no shake path is a straight line
        for psf in kernels:
            if psf.kind == "aniso_gaussian":
                minor, major = kernel_axes(psf.kernel)
                self.assertGreater(major, 2.5 * minor)  # never close to round

    def test_v2_interp_angles_are_between_training_angles(self):
        sets = protocol_psf_sets("v2")
        for psf in sets["P2_interp"]:
            if psf.kind == "motion":
                angle = float(psf.name.split("_a")[1])
                self.assertAlmostEqual(angle % 15, 7.5)

    def test_v2_real_kernel_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "k1.npy", trajectory_kernel(15, 0.7, seed=3))
            sets = protocol_psf_sets("v2", real_kernel_dirs=(("levin", tmp),))
        self.assertEqual([psf.name for psf in sets["P2_real"]], ["levin_k1"])
        self.assertFalse(any(psf.kind == "real" for psf in sets["P2_extrap"]))
        self.assertNotIn("P2_real", protocol_psf_sets("v2"))

    def test_config_versions(self):
        v1 = ProtocolConfig(data_root="data")
        self.assertEqual((v1.train_psf_set, v1.condition_sampling, v1.quantize_observations),
                         ("P1", "uniform_independent", False))
        v2 = ProtocolConfig(data_root="data", protocol_version="v2")
        self.assertEqual((v2.train_psf_set, v2.condition_sampling, v2.quantize_observations),
                         ("P1_pool", "family_balanced", True))
        self.assertNotEqual(v1.manifest()["psf_sets_sha256"], v2.manifest()["psf_sets_sha256"])
        with self.assertRaises(ValueError):
            ProtocolConfig(data_root="data", real_kernel_dirs=(("a", "b"),))
        with self.assertRaises(ValueError):
            ProtocolConfig(data_root="data", protocol_version="v3")

    def test_quantized_observation(self):
        x = smooth_image()
        y = observe(x, gaussian_kernel(1.2), 0.0, seed=0, image_id="a", psf_name="g", quantize=True)
        np.testing.assert_allclose(y * 255.0, np.round(y * 255.0), atol=1e-9)
        exact = observe(x, gaussian_kernel(1.2), 0.0, seed=0, image_id="a", psf_name="g")
        np.testing.assert_allclose(quantize_8bit(exact), y)
        self.assertLessEqual(np.abs(exact - y).max(), 0.5 / 255 + 1e-12)


class TestKernelGenerators(unittest.TestCase):
    def test_anisotropic_gaussian(self):
        k = anisotropic_gaussian_kernel(3.0, 1.0, 0.0)
        self.assertGreater(k[k.shape[0] // 2].sum(), k[:, k.shape[1] // 2].sum())  # wide along x
        np.testing.assert_allclose(k, k[::-1, ::-1])  # point symmetric
        np.testing.assert_allclose(anisotropic_gaussian_kernel(1.5, 1.5, 30.0), gaussian_kernel(1.5))

    def test_trajectory_is_reproducible(self):
        a = trajectory_kernel(17, 0.6, seed=4)
        np.testing.assert_array_equal(a, trajectory_kernel(17, 0.6, seed=4))
        self.assertFalse(np.array_equal(a, trajectory_kernel(17, 0.6, seed=5)))
        self.assertAlmostEqual(a.sum(), 1.0)
        with self.assertRaises(ValueError):
            trajectory_kernel(17, 0.0, seed=0)

    def test_ring_and_ghost(self):
        r = ring_kernel(6, 3.6)
        c = r.shape[0] // 2
        self.assertEqual(r[c, c], 0.0)  # a hole, unlike a disc
        self.assertAlmostEqual(r.sum(), 1.0)
        g = ghost_kernel(10, 0.0, weight=0.7)
        self.assertAlmostEqual(g.sum(), 1.0)
        row = g[g.shape[0] // 2]
        self.assertAlmostEqual(row.max(), 0.7)
        self.assertEqual(int(np.count_nonzero(row)), 2)  # two separate points
        with self.assertRaises(ValueError):
            ring_kernel(3, 3)

    def test_cross_family_correlation(self):
        from deblur.psf import PSF

        line = motion_kernel(9, 30.0)
        shifted = np.pad(line, ((0, 4), (3, 0)))
        same = [PSF(name="a", kernel=line, kind="x"), PSF(name="b", kernel=shifted, kind="y")]
        self.assertAlmostEqual(max_cross_family_correlation(same)[0], 1.0, places=6)
        different = [PSF(name="a", kernel=line, kind="x"), PSF(name="c", kernel=ring_kernel(6, 3.6), kind="y")]
        self.assertLess(max_cross_family_correlation(different)[0], 0.7)

    def test_airy_and_moffat(self):
        k = airy_kernel(3.0)
        c = k.shape[0] // 2
        self.assertEqual(np.unravel_index(k.argmax(), k.shape), (c, c))
        self.assertLess(k[c, c + 3], 0.01 * k[c, c])  # dark ring at the first zero
        m = moffat_kernel(2.0, 3.0)
        self.assertAlmostEqual(m.sum(), 1.0)
        with self.assertRaises(ValueError):
            moffat_kernel(2.0, 1.0)

    def test_fit_support_keeps_centre(self):
        k = np.zeros((41, 41))
        k[20, 20] = 1.0
        k[20, 23] = 0.5
        fitted = fit_support(k, 31)
        c = fitted.shape[0] // 2
        self.assertEqual(fitted.shape, (7, 7))
        self.assertAlmostEqual(fitted[c, c], 1.0 / 1.5)
        self.assertLessEqual(max(fit_support(gaussian_kernel(8.0), 31).shape), 31)


class TestSyntheticQC(unittest.TestCase):
    def test_sharpness_entropy_and_hash(self):
        x = smooth_image((128, 160))
        blurred = degrade(x, gaussian_kernel(3.0))
        self.assertGreater(laplacian_variance(x), 5 * laplacian_variance(blurred))
        self.assertGreater(entropy(x), entropy(np.full_like(x, 0.5)))
        self.assertEqual(hamming(perceptual_hash(x), perceptual_hash(x)), 0)
        other = smooth_image((128, 160), seed=7)
        self.assertGreater(hamming(perceptual_hash(x), perceptual_hash(other)), 10)
        self.assertLess(spectrum_slope(x), 0)

    def test_normalize_size_keeps_orientation(self):
        self.assertEqual(normalize_size(np.random.rand(512, 768, 3)).shape, (321, 481, 3))
        self.assertEqual(normalize_size(np.random.rand(768, 512, 3)).shape, (481, 321, 3))
        with self.assertRaises(ValueError):
            normalize_size(np.random.rand(200, 300, 3))

    def test_nested_subsets(self):
        paths = [Path(f"img{i:03d}.png") for i in range(200)]
        small, large = nested_subset(paths, 50, seed=0), nested_subset(paths, 100, seed=0)
        self.assertEqual(len(small), 50)
        self.assertTrue(set(small) <= set(large))
        self.assertEqual(small, nested_subset(paths, 50, seed=0))


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestPhase2Sampler(unittest.TestCase):
    def test_real_extra_mixture(self):
        real = [np.full((40, 48), 0.2)]
        extra = [np.full((40, 48), 0.8)]
        delta = np.ones((1, 1))
        sampler = PatchSampler(real, [delta], (0.0,), patch_size=16, batch_size=400, seed=0,
                               extra_images=extra, p_real=0.25)
        values = sampler.sample().clean.mean(dim=(1, 2, 3)).numpy()
        self.assertAlmostEqual(float(np.mean(np.isclose(values, 0.2))), 0.25, delta=0.06)
        with self.assertRaises(ValueError):
            PatchSampler(real, [delta], (0.0,), patch_size=16, batch_size=4, p_real=0.5)

    def test_family_balanced_kernels(self):
        kernels = [np.ones((1, 1))] * 10
        groups = ["a"] * 9 + ["b"]
        sampler = PatchSampler([np.random.rand(40, 48)], kernels, (0.0,), patch_size=16,
                               batch_size=2000, seed=0, kernel_groups=groups)
        idx = sampler.sample().psf_index.numpy()
        self.assertAlmostEqual(float(np.mean(idx == 9)), 0.5, delta=0.05)

    def test_quantized_training_observations(self):
        sampler = PatchSampler([np.random.rand(40, 48)], [gaussian_kernel(1.0)], (5 / 255,),
                               patch_size=16, batch_size=4, seed=0, quantize=True)
        y = sampler.sample().observed.numpy() * 255.0
        np.testing.assert_allclose(y, np.round(y), atol=1e-4)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestBatchedAndDifferentiableSolvers(unittest.TestCase):
    def setUp(self):
        from deblur.methods.classical.gmm import fit_gmm

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x = smooth_image((40, 36))
        self.kernels = [gaussian_kernel(1.0), trajectory_kernel(9, 0.6, seed=1)]
        self.ys = [degrade(self.x, k, 0.02, rng=np.random.default_rng(i)) for i, k in enumerate(self.kernels)]
        patches = []
        for i in range(0, 34, 2):
            for j in range(0, 30, 2):
                p = self.x[i : i + 6, j : j + 6].ravel()
                patches.append(p - p.mean())
        self.gmm, _ = fit_gmm(np.asarray(patches), n_components=3, n_iter=4, seed=0)

    def test_batch_matches_single(self):
        from deblur.methods.classical.torch_backend import (
            TorchGMM, admm_epll_torch, admm_epll_torch_batch, fista_tv_torch, fista_tv_torch_batch,
        )

        batch = fista_tv_torch_batch(self.ys, self.kernels, [0.002, 0.002], n_iter=10, tv_iter=5, device=self.device)
        for y, k, b in zip(self.ys, self.kernels, batch):
            np.testing.assert_allclose(b, fista_tv_torch(y, k, 0.002, n_iter=10, tv_iter=5, device=self.device), atol=1e-12)
        gmm = TorchGMM(self.gmm, device=self.device)
        batch = admm_epll_torch_batch(self.ys, self.kernels, gmm, sigma=0.02, n_iter=3, patch_size=6, stride=2)
        for y, k, b in zip(self.ys, self.kernels, batch):
            single = admm_epll_torch(y, k, gmm, sigma=0.02, n_iter=3, patch_size=6, stride=2)
            np.testing.assert_allclose(b, single, atol=1e-4)

    def test_differentiable_solvers_give_gradients(self):
        from deblur.methods.classical.torch_backend import (
            TorchGMM, admm_epll_torch, admm_epll_torch_diff, fista_tv_torch_diff,
        )

        gmm = TorchGMM(self.gmm, device=self.device)
        # fill the factor cache under inference mode first, as the clean evaluation does
        admm_epll_torch(self.ys[0], self.kernels[0], gmm, sigma=0.02, n_iter=2, patch_size=6, stride=2)
        for forward in (
            lambda z: fista_tv_torch_diff(z, self.kernels[0], 0.002, n_iter=5, tv_iter=3),
            lambda z: admm_epll_torch_diff(z, self.kernels[0], gmm, sigma=0.02, n_iter=2, patch_size=6, stride=2),
        ):
            y = torch.as_tensor(self.ys[0], dtype=torch.float32, device=self.device)[None, None]
            y.requires_grad_(True)
            loss = (forward(y) - torch.as_tensor(self.x, dtype=torch.float32, device=self.device)).pow(2).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(y.grad).all())
            self.assertGreater(float(y.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
