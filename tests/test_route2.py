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

from gaussian_jscc.allocation import (GaussianTierMask, _tiers_from_draws,
                                      deployment_tiers, expected_rate, scene_fingerprint)
from gaussian_jscc.codec import GaussianCodec
from gaussian_jscc.data import prepare, to_features, to_raw, write_ply
from gaussian_jscc.transport import decode_metadata
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

    def test_balanced_positive_initialization_and_conservative_q0(self):
        mask = GaussianTierMask(3)
        expected = torch.tensor([[.01, .33, .33, .33]]).expand(3, -1)
        torch.testing.assert_close(mask.probabilities(torch.arange(3), 10.), expected)

    def test_ten_draw_drop_mode_and_positive_tie_break(self):
        draws = torch.tensor([[0]*10, [0]*9+[2], [1]*4+[2]*4+[0]*2,
                              [1]*5+[3]*5])
        probabilities = torch.tensor([[.99, .003, .004, .003],
                                      [.9, .01, .08, .01],
                                      [.2, .3, .4, .1],
                                      [.1, .4, .1, .4]])
        tie_random = torch.tensor([[.1, .2, .3], [.1, .2, .3],
                                   [.1, .2, .3], [.9, .1, .1]])
        torch.testing.assert_close(_tiers_from_draws(draws, probabilities, tie_random),
                                   torch.tensor([0, 2, 2, 1]))
        soft = torch.tensor([[.01, .33, .33, .33]]).expand(1000, -1)
        torch.testing.assert_close(deployment_tiers(soft, 19), deployment_tiers(soft, 19))
        self.assertGreater(len(torch.unique(deployment_tiers(soft, 19))), 1)
        counts = torch.bincount(deployment_tiers(soft, 19), minlength=4)
        self.assertLess(int(counts[1:].max() - counts[1:].min()), 100)

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
            exported = torch.from_numpy(np.load(root / "map" / "tiers.npy").astype(np.int64))
            torch.testing.assert_close(exported, hard_tiers(mask, 10.))
            run("evaluate", "--ply", ply, "--checkpoint", codec, "--allocation", allocation,
                "--snrs", 0, 10, "--out", root / "eval", "--device", "cpu")
            self.assertEqual(len(json.loads((root / "eval" / "results.json").read_text())), 2)
            run("transmit", "--ply", ply, "--checkpoint", codec, "--allocation", allocation,
                "--out", root / "packet", "--device", "cpu")
            _, packet_tiers = decode_metadata((root / "packet" / "metadata.bin").read_bytes())
            _, _, ordered_tiers = prepare(raw, model.cfg.morton_bits, exported)
            torch.testing.assert_close(packet_tiers, ordered_tiers)
            ply.unlink()
            allocation.unlink()
            run("decode", "--checkpoint", codec, "--packet", root / "packet",
                "--out", root / "decoded.ply", "--device", "cpu")


if __name__ == "__main__":
    unittest.main()
