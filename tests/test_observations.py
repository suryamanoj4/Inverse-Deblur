import unittest

import numpy as np

from deblur.data import degrade
from deblur.protocol import cell_rng, observe
from deblur.psf import gaussian_kernel


class TestObservations(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).random((24, 20))
        self.k = gaussian_kernel(1.2)

    def test_deterministic(self):
        a = observe(self.x, self.k, 15 / 255, seed=0, image_id="val/1.jpg", psf_name="g")
        b = observe(self.x, self.k, 15 / 255, seed=0, image_id="val/1.jpg", psf_name="g")
        np.testing.assert_array_equal(a, b)

    def test_cells_get_different_noise(self):
        sigma = 15 / 255
        base = observe(self.x, self.k, sigma, seed=0, image_id="val/1.jpg", psf_name="g")
        for kwargs in (
            {"seed": 1, "image_id": "val/1.jpg", "psf_name": "g"},
            {"seed": 0, "image_id": "val/2.jpg", "psf_name": "g"},
            {"seed": 0, "image_id": "val/1.jpg", "psf_name": "h"},
        ):
            other = observe(self.x, self.k, sigma, **kwargs)
            self.assertFalse(np.allclose(base, other), kwargs)

    def test_matches_forward_model(self):
        sigma = 5 / 255
        expected = degrade(self.x, self.k, sigma=sigma, rng=cell_rng(0, "a", "g", sigma))
        actual = observe(self.x, self.k, sigma, seed=0, image_id="a", psf_name="g")
        np.testing.assert_array_equal(actual, expected)

    def test_noiseless_cell_is_blur_only(self):
        y = observe(self.x, self.k, 0.0, seed=0, image_id="a", psf_name="g")
        np.testing.assert_allclose(y, degrade(self.x, self.k, sigma=0.0))


if __name__ == "__main__":
    unittest.main()
