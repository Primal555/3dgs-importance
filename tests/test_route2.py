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

    def test_dense_hard_choices_match_real_packing_with_zero_tiers(self):
        raw, model = fixture(n=16)
        model.cfg.depth = 2
        model = GaussianCodec(model.cfg)
        ordered, geometry, _ = prepare(raw, 16)
        f, xyz = to_features(ordered, geometry, model)
        for q in (torch.arange(16) % 4, torch.ones(16, dtype=torch.long),
                  torch.full((16,), 3, dtype=torch.long), torch.zeros(16, dtype=torch.long)):
            dense, seed, active = model.forward_tiers(f, xyz, F.one_hot(q, 4).float(), 10., "none")
            torch.testing.assert_close(active, (q > 0).float())
            self.assertTrue(torch.isfinite(dense).all())
            if (q > 0).any():
                packed, packed_seed = model(f[q > 0], xyz[q > 0], q[q > 0], 10., "none", return_seed=True)
                torch.testing.assert_close(dense[q > 0], packed, atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(seed[q > 0], packed_seed, atol=3e-6, rtol=3e-5)

    def test_distortion_alone_updates_four_way_logits_and_codec(self):
        raw, model = fixture(n=12)
        ordered, geometry, _ = prepare(raw, 16)
        f, xyz = to_features(ordered, geometry, model)
        mask = GaussianTierMask(len(f))
        ids = torch.arange(len(f))
        probabilities = mask.probabilities(ids, 5.)
        q = torch.arange(len(f)) % 4
        choices = F.one_hot(q, 4).float() - probabilities.detach() + probabilities
        pred, seed, active = model.forward_tiers(f, xyz, choices, 5.)
        # No rate penalty: a reconstruction objective must reach allocation.
        loss = (pred * active[:, None] - f).square().mean() + .2 * (seed - xyz).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(mask.logits.grad).all())
        self.assertTrue((mask.logits.grad.abs().sum(0) > 0).all())
        self.assertGreater(float(model.enc_out.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.position_seed.weight.grad.abs().sum()), 0.)

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

    def test_checkpoint_recomputation_of_gumbel_and_channel(self):
        from torch.utils.checkpoint import checkpoint
        raw, model = fixture(n=8)
        ordered, geometry, _ = prepare(raw, 16)
        f, xyz = to_features(ordered, geometry, model)
        mask = GaussianTierMask(len(raw))
        outputs = []
        for recompute in (False, True):
            model.zero_grad(set_to_none=True)
            mask.zero_grad(set_to_none=True)
            torch.manual_seed(11)
            def forward(a, b, logits):
                choices = F.gumbel_softmax(logits, tau=.7, hard=True)
                return model.forward_tiers(a, b, choices, 8.)
            pred, seed, gate = (checkpoint(forward, f, xyz, mask.logits, use_reentrant=False,
                                           preserve_rng_state=True) if recompute else forward(f, xyz, mask.logits))
            ((pred * gate[:, None] - f).square().mean() + (seed - xyz).square().mean()).backward()
            outputs.append((mask.logits.grad.clone(), model.enc_out.weight.grad.clone()))
        for left, right in zip(*outputs):
            torch.testing.assert_close(left, right)

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

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA mask rasterizer")
    def test_cuda_mask_renders_exact_hard_scene_and_keeps_drop_gradient(self):
        from gaussian_jscc.rendering import render
        from utils.graphics_utils import getProjectionMatrix
        raw, _ = fixture(n=8)
        raw[:, :2] = (raw[:, :2] - .5) * .2
        raw[:, 2] = torch.linspace(1.9, 2.1, 8)
        raw = raw.cuda()
        q = torch.arange(8, device="cuda") % 4
        logits = torch.zeros((8, 4), device="cuda", requires_grad=True)
        p = logits.softmax(-1)
        choices = F.one_hot(q, 4) - p.detach() + p
        active = choices[:, 1:].sum(-1)
        camera = SimpleNamespace(image_height=32, image_width=32, FoVx=1., FoVy=1.,
                                 world_view_transform=torch.eye(4, device="cuda"),
                                 full_proj_transform=getProjectionMatrix(.01, 100., 1., 1.).T.cuda(),
                                 camera_center=torch.zeros(3, device="cuda"))
        image = render(raw, camera, 3, existence=active)
        torch.testing.assert_close(image, render(raw[q > 0], camera, 3), atol=2e-5, rtol=2e-4)
        image.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[q == 0].abs().sum()), 0.)

    def test_joint_cli_export_and_multi_snr_deployment(self):
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
            run("train-route2", "--ply", ply, "--out", training, "--device", "cpu", "--attribute-only",
                "--warmup-steps", 1, "--joint-steps", 2, "--hidden", 16, "--grid-dim", 4, "--depth", 1,
                "--levels", 2, 3, "--rates", 0, 2, 4, 6, "--block-size", 8, "--condition-snr")
            log = [json.loads(row) for row in (training / "loss.jsonl").read_text().splitlines()]
            self.assertGreater(log[-1]["mask_grad_norm"], 0.)
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
