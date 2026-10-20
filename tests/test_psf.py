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
