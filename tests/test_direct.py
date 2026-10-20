import unittest

import numpy as np

try:  # optional
    import torch

    from deblur.methods.direct import NAFNet, UNet, build_model, pad_to_multiple, restore

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_TORCH = False


def small_nafnet():
    return NAFNet(width=8, enc_blocks=(1, 1), middle_blocks=1, dec_blocks=(1, 1))


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestDirectModels(unittest.TestCase):
    def test_shape_preserved_for_odd_sizes(self):
        x = torch.rand(2, 1, 37, 45)
        for model in (small_nafnet(), UNet(width=8, depth=2)):
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model(x).shape, x.shape)

    def test_default_nafnet_matches_paper_width32(self):
        model = build_model("nafnet")
        self.assertEqual(len(model.encoders), 4)
        self.assertEqual([len(e) for e in model.encoders], [1, 1, 1, 28])
        self.assertEqual(model.intro.in_channels, 1)
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 15e6)
        self.assertLess(n_params, 20e6)

    def test_nafblocks_start_as_identity(self):
        # beta = gamma = 0 at init, so the body reduces to intro -> ending + input
        model = small_nafnet()
        for block in model.modules():
            if hasattr(block, "beta"):
                self.assertEqual(float(block.beta.detach().abs().sum()), 0.0)
                self.assertEqual(float(block.gamma.detach().abs().sum()), 0.0)

    def test_circular_padding(self):
        x = torch.arange(12.0).reshape(1, 1, 3, 4)
        padded = pad_to_multiple(x, 4)
        self.assertEqual(padded.shape[-2:], (4, 4))
        torch.testing.assert_close(padded[..., 3, :], x[..., 0, :])
        self.assertIs(pad_to_multiple(padded, 4), padded)

    def test_restore_returns_clipped_float64(self):
        y = np.random.default_rng(0).random((20, 30))
        out = restore(UNet(width=8, depth=2), y)
        self.assertEqual(out.shape, y.shape)
        self.assertEqual(out.dtype, np.float64)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            build_model("resnet")


if __name__ == "__main__":
    unittest.main()
