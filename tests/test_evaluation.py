import unittest

import numpy as np

from deblur.evaluation import identity_restorer, merge_baseline, summarize


class TestIdentityRestorer(unittest.TestCase):
    def test_returns_input(self):
        y = np.random.default_rng(0).random((8, 8))
        out = identity_restorer(y, None, 0.0)
        np.testing.assert_array_equal(out, y)


class TestMergeBaseline(unittest.TestCase):
    def test_gain_is_paired(self):
        rows = [
            {"image": "a", "psf": "k1", "sigma_255": 0, "psnr": 30.0},
            {"image": "a", "psf": "k1", "sigma_255": 15, "psnr": 25.0},
        ]
        baseline = [
            {"image": "a", "psf": "k1", "sigma_255": 0, "psnr": 22.0},
            {"image": "a", "psf": "k1", "sigma_255": 15, "psnr": 18.0},
        ]
        merged = merge_baseline(rows, baseline)
        self.assertEqual(merged[0]["baseline_psnr"], 22.0)
        self.assertAlmostEqual(merged[0]["gain"], 8.0)
        self.assertAlmostEqual(merged[1]["gain"], 7.0)


class TestSummarize(unittest.TestCase):
    def test_groups_by_set_kind_sigma(self):
        rows = [
            {"psf_set": "P1", "kind": "gaussian", "sigma_255": 0, "psnr": 30.0,
             "baseline_psnr": 20.0, "gain": 10.0, "ssim": 0.9, "residual_rmse": 0.01},
            {"psf_set": "P1", "kind": "gaussian", "sigma_255": 0, "psnr": 32.0,
             "baseline_psnr": 22.0, "gain": 10.0, "ssim": 0.8, "residual_rmse": 0.02},
            {"psf_set": "P1", "kind": "motion", "sigma_255": 0, "psnr": 24.0,
             "baseline_psnr": 22.0, "gain": 2.0, "ssim": 0.7, "residual_rmse": 0.03},
        ]
        summary = summarize(rows)
        self.assertEqual(len(summary), 2)
        gaussian = next(row for row in summary if row["kind"] == "gaussian")
        self.assertEqual(gaussian["n"], 2)
        self.assertAlmostEqual(gaussian["psnr"], 31.0)
        self.assertAlmostEqual(gaussian["gain"], 10.0)
        self.assertAlmostEqual(gaussian["baseline_psnr"], 21.0)


if __name__ == "__main__":
    unittest.main()
