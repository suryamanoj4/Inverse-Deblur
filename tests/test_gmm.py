import tempfile
import unittest

import numpy as np

from deblur.methods.classical.gmm import GMM, fit_gmm, patch_denoise


def _two_cluster_data(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(-2.0, 0.0), scale=0.2, size=(300, 2))
    b = rng.normal(loc=(2.0, 0.0), scale=0.2, size=(300, 2))
    return np.vstack([a, b])


class TestGMM(unittest.TestCase):
    def test_single_gaussian_recovery(self):
        rng = np.random.default_rng(0)
        mean = np.array([0.5, -0.2, 0.1])
        cov = np.diag([0.3, 0.1, 0.2])
        X = rng.multivariate_normal(mean, cov, size=2000)
        model, _ = fit_gmm(X, n_components=1, reg=1e-8)
        np.testing.assert_allclose(model.means[0], mean, atol=0.05)
        np.testing.assert_allclose(model.covariances[0], cov, atol=0.05)

    def test_two_component_recovery(self):
        X = _two_cluster_data()
        model, history = fit_gmm(X, n_components=2, n_iter=30, seed=0)
        means = model.means[np.argsort(model.means[:, 0])]
        np.testing.assert_allclose(means[0], [-2.0, 0.0], atol=0.15)
        np.testing.assert_allclose(means[1], [2.0, 0.0], atol=0.15)
        self.assertGreater(len(history), 1)

    def test_responsibilities_sum_to_one(self):
        X = _two_cluster_data()
        model, _ = fit_gmm(X, n_components=2, n_iter=10, seed=0)
        gamma = model.responsibilities(X)
        self.assertEqual(gamma.shape, (len(X), 2))
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-10)

    def test_map_denoise_zero_noise_is_identity(self):
        X = _two_cluster_data()
        model, _ = fit_gmm(X, n_components=2, n_iter=10, seed=0)
        np.testing.assert_allclose(model.map_denoise(X, sigma=0.0), X, atol=1e-8)

    def test_map_denoise_reduces_error(self):
        X = _two_cluster_data()
        model, _ = fit_gmm(X, n_components=2, n_iter=20, seed=0)
        rng = np.random.default_rng(1)
        noisy = X + 0.3 * rng.standard_normal(X.shape)
        denoised = model.map_denoise(noisy, sigma=0.3)
        self.assertLess(np.mean((denoised - X) ** 2), np.mean((noisy - X) ** 2))

    def test_save_load_roundtrip(self):
        X = _two_cluster_data()
        model, _ = fit_gmm(X, n_components=2, n_iter=5, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = model.save(f"{tmp}/gmm.npz")
            loaded = GMM.load(path)
        np.testing.assert_allclose(loaded.weights, model.weights)
        np.testing.assert_allclose(loaded.means, model.means)
        np.testing.assert_allclose(loaded.covariances, model.covariances)

    def test_patch_denoise_shape(self):
        rng = np.random.default_rng(0)
        image = rng.random((16, 16))
        patches = np.stack(
            [image[i : i + 4, j : j + 4].ravel() for i in range(13) for j in range(13)]
        )
        patches = patches - patches.mean(axis=1, keepdims=True)
        model, _ = fit_gmm(patches, n_components=2, n_iter=5, seed=0)
        out = patch_denoise(image, model, sigma=0.05, patch_size=4, stride=2)
        self.assertEqual(out.shape, image.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_invalid_components(self):
        X = np.zeros((10, 2))
        with self.assertRaises(ValueError):
            fit_gmm(X, n_components=0)
        with self.assertRaises(ValueError):
            fit_gmm(X, n_components=11)


if __name__ == "__main__":
    unittest.main()
