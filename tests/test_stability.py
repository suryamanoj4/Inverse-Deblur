import unittest

import numpy as np

from deblur.metrics.stability import project_l2_ball, relative_l2, perturbation_radius

try:  # optional
    import torch

    from deblur.metrics.stability import pgd_attack_torch

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_TORCH = False


class TestProjection(unittest.TestCase):
    def test_projects_to_ball(self):
        eta = np.array([3.0, 4.0])
        projected = project_l2_ball(eta, 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(projected)), 1.0, places=12)
        np.testing.assert_allclose(projected, [0.6, 0.8])

    def test_inside_ball_is_unchanged(self):
        eta = np.array([0.3, 0.4])
        projected = project_l2_ball(eta, 1.0)
        np.testing.assert_array_equal(projected, eta)
        self.assertIsNot(projected, eta)

    def test_zero_radius(self):
        np.testing.assert_array_equal(project_l2_ball(np.ones(4), 0.0), np.zeros(4))

    def test_negative_radius(self):
        with self.assertRaises(ValueError):
            project_l2_ball(np.ones(4), -1.0)


class TestRelativeBudget(unittest.TestCase):
    def test_relative_l2(self):
        self.assertAlmostEqual(relative_l2(np.array([3.0, 4.0]), np.ones(2)), 5.0 / np.sqrt(2))
        self.assertEqual(relative_l2(np.zeros(2), np.zeros(2)), 0.0)

    def test_radius(self):
        reference = np.array([3.0, 4.0])
        self.assertAlmostEqual(perturbation_radius(reference, 0.01), 0.05)

    def test_negative_epsilon(self):
        with self.assertRaises(ValueError):
            perturbation_radius(np.ones(2), -0.1)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestPGD(unittest.TestCase):
    def test_increases_error_and_respects_budget(self):
        y = torch.full((1, 1, 8, 8), 0.5)
        target = torch.zeros_like(y)

        def forward(z):
            return 2.0 * z

        eta = pgd_attack_torch(forward, y, target, radius=0.1, steps=10)
        self.assertLessEqual(float(torch.linalg.vector_norm(eta)), 0.1 + 1e-5)
        clean = torch.nn.functional.mse_loss(forward(y), target)
        adv = torch.clamp(y + eta, 0.0, 1.0)
        adversarial = torch.nn.functional.mse_loss(forward(adv), target)
        self.assertGreater(float(adversarial), float(clean))

    def test_zero_radius_is_clean(self):
        y = torch.full((1, 1, 4, 4), 0.5)
        eta = pgd_attack_torch(lambda z: 2.0 * z, y, torch.zeros_like(y), radius=0.0)
        self.assertEqual(float(torch.linalg.vector_norm(eta)), 0.0)


if __name__ == "__main__":
    unittest.main()
