"""Allocation gradients, hard-deployment equivalence and joint workflow tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from gaussian_jscc.allocation import GaussianTierMask, expected_rate, scene_fingerprint
from gaussian_jscc.codec import GaussianCodec
from gaussian_jscc.data import prepare, to_features, to_raw, write_ply
from gaussian_jscc.route2 import load_mask, save_joint, hard_tiers
from test_gaussian_jscc import fixture


class Route2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_expected_rate_gradient_discourages_expensive_tiers(self):
        logits = torch.zeros((5, 4), requires_grad=True)
        expected_rate(logits.softmax(-1), (0, 8, 16, 32)).mean().backward()
        self.assertTrue((logits.grad[:, 0] < 0).all())
        self.assertTrue((logits.grad[:, 3] > 0).all())

    def test_prior_and_snr_conditioning(self):
        prior = torch.tensor([.2, .8])
        mask = GaussianTierMask(2, prior, snr_conditioned=True)
        ids = torch.arange(2)
        torch.testing.assert_close(mask.probabilities(ids, 10.)[:, 1:].sum(-1), prior)
        with torch.no_grad():
            mask.snr_slopes[:, 1] = 3.
        self.assertFalse(torch.allclose(mask.probabilities(ids, 0.), mask.probabilities(ids, 20.)))

    def test_saved_masks_keep_original_ply_order_and_check_identity(self):
        raw, model = fixture(n=17)
        mask = GaussianTierMask(len(raw))
        wanted = torch.arange(len(raw)) % 4
        with torch.no_grad():
            mask.logits.copy_(F.one_hot(wanted, 4) * 20.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_joint(root, "", model, mask, scene_fingerprint(raw), 0, {})
            loaded = load_mask(root / "route2.pt", raw, model, "cpu")
            torch.testing.assert_close(hard_tiers(loaded, 10.), wanted)
            with self.assertRaisesRegex(ValueError, "PLY"):
                load_mask(root / "route2.pt", raw.flip(0), model, "cpu")

    def test_mask_export_and_multi_snr_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, _ = fixture(n=12, degree=0)
            ply = root / "source.ply"
            write_ply(ply, raw, 0)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            def run(*args):
                result = subprocess.run([sys.executable, "-m", "gaussian_jscc", *map(str, args)],
                                        env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            training = root / "training"
            training.mkdir()
            raw, model = fixture(n=12, degree=0)
            mask = GaussianTierMask(len(raw), snr_conditioned=True)
            save_joint(training, "", model, mask, scene_fingerprint(raw), 0, {})
            codec, allocation = training / "codec.pt", training / "route2.pt"
            run("export-route2", "--ply", ply, "--checkpoint", codec, "--allocation", allocation,
                "--out", root / "map", "--device", "cpu")
            self.assertEqual(np.load(root / "map" / "probabilities.npy").shape, (12, 4))
            run("evaluate", "--ply", ply, "--checkpoint", codec, "--allocation", allocation,
                "--snrs", 0, 10, "--out", root / "eval", "--device", "cpu")
            self.assertEqual(len(json.loads((root / "eval" / "results.json").read_text())), 2)
            run("transmit", "--ply", ply, "--checkpoint", codec, "--allocation", allocation,
                "--out", root / "packet", "--device", "cpu")
            ply.unlink()
            allocation.unlink()
            run("decode", "--checkpoint", codec, "--packet", root / "packet",
                "--out", root / "decoded.ply", "--device", "cpu")


if __name__ == "__main__":
    unittest.main()
