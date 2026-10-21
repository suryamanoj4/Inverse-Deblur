import unittest

import numpy as np

try:  # optional
    import torch

    from deblur.methods.unrolled import ISTANet, LearnedISTA, USRNet, build_model
    from deblur.methods.unrolled.common import (
        data_gradient,
        data_step,
        fft_convolve,
        soft_threshold,
    )

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestUnrolledCommon(unittest.TestCase):
    def test_soft_threshold(self):
        x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        out = soft_threshold(x, 0.5)
        expected = torch.tensor([-1.5, -0.5, 0.0, 0.0, 0.0, 0.5, 1.5])
        torch.testing.assert_close(out, expected)

    def test_delta_spectrum_is_identity(self):
        x = torch.rand(2, 1, 8, 8)
        spectrum = torch.ones(2, 8, 8, dtype=torch.complex64)
        torch.testing.assert_close(fft_convolve(x, spectrum), x, atol=1e-5)

    def test_data_gradient_zero_at_fixed_point(self):
        y = torch.rand(2, 1, 8, 8)
        spectrum = torch.ones(2, 8, 8, dtype=torch.complex64)
        grad = data_gradient(y, y, spectrum)
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-5)

    def test_data_step_with_delta_kernel(self):
        y = torch.rand(2, 1, 8, 8)
        spectrum = torch.ones(2, 8, 8, dtype=torch.complex64)
        mu = torch.full((2, 1, 1, 1), 1.0)
        out = data_step(y, spectrum, y, mu)
        torch.testing.assert_close(out, y, atol=1e-5)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestUnrolledModels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.y = torch.rand(2, 1, 16, 16)
        self.spectrum = torch.ones(2, 16, 16, dtype=torch.complex64)
        self.sigma = torch.tensor([0.0, 0.05])

    def test_models_preserve_shape_and_are_finite(self):
        models = [
            USRNet(n_iter=2, width=8),
            ISTANet(n_iter=2, channels=8),
            LearnedISTA(n_iter=2, hidden=8),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                out = model(self.y, self.spectrum, self.sigma)
                self.assertEqual(out.shape, self.y.shape)
                self.assertTrue(torch.isfinite(out).all())

    def test_registry(self):
        for name in ("usrnet", "ista_net", "lista"):
            self.assertIsNotNone(build_model(name))
        with self.assertRaises(ValueError):
            build_model("unknown")

    def test_usrnet_is_physics_aware(self):
        # USRNet must consume the kernel spectrum and sigma; a wrong kernel
        # should change the output.
        model = USRNet(n_iter=2, width=8).eval()
        correct = torch.ones(2, 16, 16, dtype=torch.complex64)
        wrong = torch.full((2, 16, 16), 0.5, dtype=torch.complex64)
        with torch.no_grad():
            a = model(self.y, correct, self.sigma)
            b = model(self.y, wrong, self.sigma)
        self.assertFalse(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
