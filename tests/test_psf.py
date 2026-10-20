import tempfile
import unittest
from pathlib import Path

import numpy as np

from deblur.psf.library import (
    PSF,
    build_psf_library,
    disk_kernel,
    gaussian_kernel,
    load_levin_kernels,
    motion_kernel,
    protocol_psf_sets,
)


class TestKernels(unittest.TestCase):
    def test_gaussian_normalised_and_odd(self):
        for sigma in (0.7, 1.2, 1.6, 2.0):
            k = gaussian_kernel(sigma)
            self.assertAlmostEqual(float(k.sum()), 1.0, places=12)
            self.assertEqual(k.shape[0] % 2, 1)
            self.assertEqual(k.shape[0], k.shape[1])

    def test_gaussian_peak_at_centre(self):
        k = gaussian_kernel(1.2)
        centre = k.shape[0] // 2
        self.assertEqual(np.unravel_index(int(np.argmax(k)), k.shape), (centre, centre))

    def test_disk_normalised_and_symmetric(self):
        k = disk_kernel(3.0)
        self.assertAlmostEqual(float(k.sum()), 1.0, places=12)
        np.testing.assert_allclose(k, k[::-1, ::-1], atol=1e-12)

    def test_motion_normalised(self):
        k = motion_kernel(9.0, 30.0)
        self.assertAlmostEqual(float(k.sum()), 1.0, places=12)
        self.assertGreater(int((k > 0).sum()), 1)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            gaussian_kernel(0.0)
        with self.assertRaises(ValueError):
            disk_kernel(-1.0)
        with self.assertRaises(ValueError):
            motion_kernel(0.0, 0.0)


class TestProtocolSets(unittest.TestCase):
    def test_membership_and_counts(self):
        sets = protocol_psf_sets()
        self.assertEqual(set(sets), {"P1", "P2_interp", "P2_extrap"})
        self.assertEqual(len(sets["P1"]), 12)
        self.assertEqual(len(sets["P2_interp"]), 11)
        self.assertEqual(len(sets["P2_extrap"]), 4)
        self.assertEqual({psf.kind for psf in sets["P1"]}, {"gaussian", "motion"})
        self.assertEqual({psf.kind for psf in sets["P2_interp"]}, {"gaussian", "motion"})
        self.assertEqual({psf.kind for psf in sets["P2_extrap"]}, {"defocus"})

    def test_sets_are_disjoint_by_name(self):
        names = [psf.name for kernels in protocol_psf_sets().values() for psf in kernels]
        self.assertEqual(len(names), len(set(names)))

    def test_interp_parameters_inside_train_ranges(self):
        import inspect

        defaults = {
            name: param.default
            for name, param in inspect.signature(protocol_psf_sets).parameters.items()
        }
        for family in ("gaussian_sigmas", "motion_lengths"):
            train = defaults[f"train_{family}"]
            for value in defaults[f"interp_{family}"]:
                self.assertGreater(value, min(train), family)
                self.assertLess(value, max(train), family)
                self.assertNotIn(value, train, family)
        # angles are circular (a line kernel at a and a + 180 is identical)
        train_angles = {angle % 180 for angle in defaults["train_motion_angles"]}
        for angle in defaults["interp_motion_angles"]:
            self.assertNotIn(angle % 180, train_angles)

    def test_all_normalised(self):
        for kernels in protocol_psf_sets().values():
            for psf in kernels:
                self.assertAlmostEqual(float(psf.kernel.sum()), 1.0, places=12)


class TestLibrary(unittest.TestCase):
    def test_default_library_counts(self):
        psfs = build_psf_library()
        self.assertEqual(len(psfs), 4 + 4 + 8)
        self.assertEqual({p.kind for p in psfs}, {"gaussian", "defocus", "motion"})
        for psf in psfs:
            self.assertIsInstance(psf, PSF)
            self.assertAlmostEqual(float(psf.kernel.sum()), 1.0, places=12)

    def test_levin_loader_from_npy(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "kernel_01.npy", np.eye(5))
            psfs = load_levin_kernels(tmp)
            self.assertEqual(len(psfs), 1)
            self.assertAlmostEqual(float(psfs[0].kernel.sum()), 1.0, places=12)

    def test_levin_loader_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_levin_kernels(tmp)

    def test_levin_loader_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            load_levin_kernels("/nonexistent/levin/kernels")


if __name__ == "__main__":
    unittest.main()
