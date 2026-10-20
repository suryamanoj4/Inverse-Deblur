import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deblur.data import degrade
from deblur.protocol import ProtocolConfig
from deblur.psf import gaussian_kernel, motion_kernel
from deblur.psf.library import PSF

try:  # optional
    import torch

    from deblur.methods.direct import UNet
    from deblur.training import (
        PatchSampler,
        Trainer,
        TrainSettings,
        evaluate_grid,
        mean_psnr,
        model_restorer,
    )

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_TORCH = False


def images(n=3, shape=(40, 48), seed=0):
    rng = np.random.default_rng(seed)
    return [rng.random(shape) for _ in range(n)]


KERNELS = [gaussian_kernel(1.2), motion_kernel(9.0, 45.0)]


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestPatchSampler(unittest.TestCase):
    def sampler(self, **kwargs):
        defaults = dict(
            images=images(),
            kernels=KERNELS,
            noise_levels=(0.0, 15 / 255),
            patch_size=16,
            batch_size=8,
            seed=0,
        )
        defaults.update(kwargs)
        return PatchSampler(**defaults)

    def test_noiseless_matches_numpy_forward_model(self):
        batch = self.sampler(noise_levels=(0.0,), augment=False).sample()
        for i in range(batch.clean.shape[0]):
            x = batch.clean[i, 0].double().numpy()
            expected = degrade(x, KERNELS[int(batch.psf_index[i])], sigma=0.0)
            np.testing.assert_allclose(batch.observed[i, 0].numpy(), expected, atol=1e-5)

    def test_noise_level_matches_sigma(self):
        sigma = 15 / 255
        s = self.sampler(images=[np.full((40, 40), 0.5)], noise_levels=(sigma,), batch_size=64)
        batch = s.sample()
        residual = (batch.observed - batch.clean).numpy()  # blur of a constant is a no-op
        self.assertAlmostEqual(float(residual.std()), sigma, delta=0.1 * sigma)

    def test_output_range_and_shape(self):
        batch = self.sampler().sample()
        self.assertEqual(tuple(batch.observed.shape), (8, 1, 16, 16))
        self.assertGreaterEqual(float(batch.observed.min()), 0.0)
        self.assertLessEqual(float(batch.observed.max()), 1.0)

    def test_conditions_drawn_uniformly_and_independently(self):
        s = self.sampler(batch_size=4000)
        batch = s.sample()
        psf = batch.psf_index.numpy()
        sig = batch.sigma_index.numpy()
        counts = np.bincount(psf * 2 + sig, minlength=4) / len(psf)
        np.testing.assert_allclose(counts, 0.25, atol=0.03)

    def test_state_restore_reproduces_batches(self):
        s = self.sampler()
        s.sample()
        state = s.state_dict()
        first = s.sample()
        s.load_state_dict(state)
        again = s.sample()
        torch.testing.assert_close(first.observed, again.observed)
        torch.testing.assert_close(first.clean, again.clean)

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), "CUDA is not available")
    def test_state_survives_checkpoint_round_trip_on_cuda(self):
        s = self.sampler(device="cuda")
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(s.state_dict(), Path(tmp) / "state.pt")
            state = torch.load(Path(tmp) / "state.pt", map_location="cuda", weights_only=False)
        first = s.sample()
        s.load_state_dict(state)
        torch.testing.assert_close(s.sample().observed, first.observed)

    def test_rejects_small_images(self):
        with self.assertRaises(ValueError):
            self.sampler(images=[np.zeros((10, 10))])


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestEvaluation(unittest.TestCase):
    def test_grid_size_and_identity_restorer(self):
        psfs = [PSF("g", KERNELS[0], "gaussian"), PSF("m", KERNELS[1], "motion")]
        data = [(f"img{i}", x) for i, x in enumerate(images(2))]
        rows = evaluate_grid(lambda y, psf, sigma: y, data, psfs, (0.0, 5 / 255), seed=0, full_metrics=True)
        self.assertEqual(len(rows), 2 * 2 * 2)
        self.assertEqual({row["sigma_255"] for row in rows}, {0, 5})
        self.assertTrue(all(np.isfinite(row["psnr"]) for row in rows))
        self.assertIn("residual_rmse", rows[0])

    def test_mean_psnr_weights_cells_equally(self):
        rows = [
            {"psf": "a", "sigma_255": 0, "psnr": 30.0},
            {"psf": "a", "sigma_255": 0, "psnr": 30.0},
            {"psf": "a", "sigma_255": 0, "psnr": 30.0},
            {"psf": "b", "sigma_255": 0, "psnr": 20.0},
        ]
        self.assertAlmostEqual(mean_psnr(rows), 25.0)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestTrainer(unittest.TestCase):
    def test_schedule_warmup_then_cosine(self):
        s = TrainSettings(
            iterations=100,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_iterations=10,
            gradient_clip=1.0,
            val_interval=10,
            early_stop_patience=2,
        )
        self.assertAlmostEqual(s.lr_factor(0), 0.1)
        self.assertAlmostEqual(s.lr_factor(9), 1.0)
        self.assertAlmostEqual(s.lr_factor(10), 1.0)
        self.assertAlmostEqual(s.lr_factor(100), s.min_learning_rate / s.learning_rate)
        self.assertLess(s.lr_factor(60), s.lr_factor(30))

    def test_settings_from_protocol(self):
        config = ProtocolConfig(data_root=".")
        s = TrainSettings.from_protocol(config, arm="m1")
        self.assertEqual(s.iterations, config.m1_iterations)
        self.assertEqual(s.learning_rate, config.m1_learning_rate)
        self.assertEqual(TrainSettings.from_protocol(config, arm="m1", iterations=5).iterations, 5)

    def test_resume_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, iterations=5)
            with self.assertRaises(ValueError):
                self._run(tmp, iterations=10, resume=True)

    def _run(self, out_dir, iterations, resume=False, patience=10):
        torch.manual_seed(0)
        model = UNet(width=4, depth=2)
        sampler = PatchSampler(images(), KERNELS, (0.0, 5 / 255), patch_size=16, batch_size=4, seed=0)
        psfs = [PSF("g", KERNELS[0], "gaussian")]
        val = [("v0", images(1, seed=1)[0])]
        restore_fn = model_restorer(model, "cpu")

        def validate(_model):
            rows = evaluate_grid(restore_fn, val, psfs, (0.0,), seed=0)
            return mean_psnr(rows), rows

        settings = TrainSettings(
            iterations=iterations,
            learning_rate=1e-3,
            weight_decay=1e-4,
            warmup_iterations=2,
            gradient_clip=1.0,
            val_interval=5,
            early_stop_patience=patience,
            log_interval=5,
        )
        trainer = Trainer(model, sampler, validate, settings, out_dir, metadata={"model": "unet"})
        if resume:
            trainer.load(Path(out_dir) / "last.pt")
        return trainer, trainer.run()

    def test_smoke_run_writes_checkpoints_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = self._run(tmp, iterations=10)
            self.assertEqual(summary["iterations_run"], 10)
            for name in ("best.pt", "last.pt", "log.jsonl", "best_val.json"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            events = [json.loads(line)["event"] for line in open(Path(tmp) / "log.jsonl")]
            self.assertEqual(events.count("val"), 2)
            best = torch.load(Path(tmp) / "best.pt", weights_only=False)
            self.assertEqual(best["metadata"], {"model": "unet"})
            self.assertNotIn("optimizer", best)

    def test_resume_continues_from_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, _ = self._run(tmp, iterations=10)
            last = torch.load(Path(tmp) / "last.pt", weights_only=False)
            self.assertEqual(last["iteration"], 10)
            resumed, summary = self._run(tmp, iterations=10, resume=True)
            self.assertEqual(summary["iterations_run"], 10)
            self.assertEqual(resumed.best_iteration, trainer.best_iteration)

    def test_early_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, _ = self._run(tmp, iterations=50, patience=1)
            trainer.best_score = float("inf")  # nothing can improve on this
            trainer.bad_checks = 0
            self.assertTrue(trainer._validate())


if __name__ == "__main__":
    unittest.main()
