import unittest

from deblur.protocol import ProtocolConfig


class TestProtocolConfig(unittest.TestCase):
    def make(self, **kwargs):
        return ProtocolConfig(data_root="/tmp/does-not-matter", **kwargs)

    def test_locked_defaults(self):
        config = self.make()
        self.assertEqual(config.m1_regime, "single_model")
        self.assertEqual(config.condition_sampling, "uniform_independent")
        self.assertEqual(config.checkpoint_rule, "mean_val_psnr")
        self.assertEqual(config.train_psf_set, "P1")
        self.assertEqual(config.eval_psf_sets, ("P1", "P2_interp", "P2_extrap"))
        self.assertEqual(config.optimizer, "adamw")
        self.assertEqual(config.loss, "l2")
        self.assertEqual(config.m1_iterations, 40_000)
        self.assertEqual(config.m3_iterations, 30_000)

    def test_manifest_logs_protocol_fields(self):
        manifest = self.make().manifest()
        for key in (
            "m1_regime",
            "condition_sampling",
            "checkpoint_rule",
            "train_psf_set",
            "eval_psf_sets",
            "val_subset_size",
            "val_subset_seed",
            "optimizer",
            "loss",
            "m1_batch_size",
            "m3_batch_size",
            "m1_iterations",
            "m3_iterations",
            "m1_learning_rate",
            "m3_learning_rate",
            "weight_decay",
            "warmup_iterations",
            "val_interval",
            "gradient_clip",
            "early_stop_patience",
            "psf_sets",
        ):
            self.assertIn(key, manifest)

    def test_manifest_psf_membership(self):
        sets = self.make().manifest()["psf_sets"]
        self.assertEqual(set(sets), {"P1", "P2_interp", "P2_extrap"})
        self.assertEqual(len(sets["P1"]), 12)
        self.assertEqual(len(sets["P2_interp"]), 11)
        self.assertEqual(len(sets["P2_extrap"]), 4)
        self.assertEqual({entry["kind"] for entry in sets["P2_extrap"]}, {"defocus"})

    def test_per_noise_regime_is_allowed_for_the_ablation(self):
        config = self.make(m1_regime="per_noise")
        self.assertEqual(config.manifest()["m1_regime"], "per_noise")

    def test_invalid_choices(self):
        with self.assertRaises(ValueError):
            self.make(m1_regime="per_psf")
        with self.assertRaises(ValueError):
            self.make(condition_sampling="random")
        with self.assertRaises(ValueError):
            self.make(checkpoint_rule="best_per_image")
        with self.assertRaises(ValueError):
            self.make(optimizer="adam")
        with self.assertRaises(ValueError):
            self.make(loss="l1_plus_ssim")
        with self.assertRaises(ValueError):
            self.make(eval_psf_sets=("P3",))
        with self.assertRaises(ValueError):
            self.make(m1_batch_size=0)
        with self.assertRaises(ValueError):
            self.make(m1_learning_rate=0.0)

    def test_noise_levels(self):
        config = self.make()
        self.assertEqual(config.noise_levels_over_255, (0, 1, 5, 15, 25))


if __name__ == "__main__":
    unittest.main()
