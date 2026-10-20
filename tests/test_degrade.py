import unittest

import numpy as np

from deblur.data.degrade import add_noise, center_crop, degrade, to_luminance
from deblur.operators.fft_conv import fft_convolve
from deblur.psf.library import gaussian_kernel


class TestDegrade(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.random((32, 32))
        self.k = gaussian_kernel(1.2)

    def test_noiseless_matches_operator(self):
        expected = np.clip(fft_convolve(self.x, self.k), 0.0, 1.0)
        np.testing.assert_allclose(degrade(self.x, self.k, sigma=0.0), expected, atol=1e-12)

    def test_seed_reproducible(self):
        a = degrade(self.x, self.k, sigma=0.05, rng=np.random.default_rng(7))
        b = degrade(self.x, self.k, sigma=0.05, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_add_noise_std(self):
        x = np.zeros((256, 256))
        noisy = add_noise(x, sigma=0.05, rng=np.random.default_rng(0))
        self.assertAlmostEqual(float(noisy.std()), 0.05, delta=0.005)

    def test_invalid_sigma(self):
        with self.assertRaises(ValueError):
            degrade(self.x, self.k, sigma=-0.1)
        with self.assertRaises(ValueError):
            degrade(self.x, self.k, sigma=1.5)

    def test_clipping(self):
        y = degrade(self.x, self.k, sigma=0.5, rng=np.random.default_rng(0))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)


class TestLuminance(unittest.TestCase):
    def test_pure_red(self):
        image = np.zeros((2, 2, 3))
        image[..., 0] = 1.0
        np.testing.assert_allclose(to_luminance(image), 0.299)

    def test_grayscale_passthrough(self):
        image = np.random.rand(4, 5)
        np.testing.assert_array_equal(to_luminance(image), image)

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            to_luminance(np.zeros((4, 5, 2)))


class TestCenterCrop(unittest.TestCase):
    def test_crop_values(self):
        image = np.arange(100, dtype=float).reshape(10, 10)
        cropped = center_crop(image, 4)
        self.assertEqual(cropped.shape, (4, 4))
        np.testing.assert_array_equal(cropped, image[3:7, 3:7])

    def test_crop_too_large(self):
        with self.assertRaises(ValueError):
            center_crop(np.zeros((8, 8)), 9)


if __name__ == "__main__":
    unittest.main()
