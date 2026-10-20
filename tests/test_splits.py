import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from deblur.data.datasets import BSDS500, sample_patches, split_manifest
from deblur.protocol.splits import check_disjoint, manifest, save_manifest, standard_splits


class TestSplits(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        counts = {"train": 2, "val": 1, "test": 2}
        rng = np.random.default_rng(0)
        for split, n in counts.items():
            directory = self.root / split
            directory.mkdir()
            for i in range(n):
                array = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
                Image.fromarray(array).save(directory / f"{split}_{i}.png")

    def tearDown(self):
        self._tmp.cleanup()

    def test_standard_splits_counts(self):
        splits = standard_splits(self.root)
        self.assertEqual({k: len(v) for k, v in splits.items()}, {"train": 2, "val": 1, "test": 2})
        check_disjoint(splits)

    def test_manifest_is_relative(self):
        splits = standard_splits(self.root)
        entries = manifest(splits, root=self.root)
        self.assertEqual(len(entries["train"]), 2)
        self.assertTrue(all(not Path(p).is_absolute() for paths in entries.values() for p in paths))

    def test_save_manifest(self):
        splits = standard_splits(self.root)
        path = save_manifest(splits, self.root / "manifest.json", root=self.root)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(data), {"train", "val", "test"})

    def test_bsds_loader_luminance(self):
        dataset = BSDS500(self.root, split="train", luminance=True)
        self.assertEqual(len(dataset), 2)
        image = dataset[0]
        self.assertEqual(image.shape, (8, 8))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_split_manifest_matches(self):
        entries = split_manifest(self.root)
        self.assertEqual({k: len(v) for k, v in entries.items()}, {"train": 2, "val": 1, "test": 2})


class TestPatchSampling(unittest.TestCase):
    def test_shape_and_reproducibility(self):
        image = np.arange(32 * 40, dtype=float).reshape(32, 40)
        a = sample_patches(image, patch_size=16, count=4, rng=np.random.default_rng(0))
        b = sample_patches(image, patch_size=16, count=4, rng=np.random.default_rng(0))
        self.assertEqual(a.shape, (4, 16, 16))
        np.testing.assert_array_equal(a, b)

    def test_patch_too_large(self):
        with self.assertRaises(ValueError):
            sample_patches(np.zeros((8, 8)), patch_size=9, count=1)

    def test_invalid_count(self):
        with self.assertRaises(ValueError):
            sample_patches(np.zeros((8, 8)), patch_size=4, count=0)


if __name__ == "__main__":
    unittest.main()
