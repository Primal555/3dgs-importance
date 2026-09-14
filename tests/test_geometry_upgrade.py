"""Geometry-first architecture, physical objectives and joint replay contracts."""

import json
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from gaussian_jscc.allocation import GaussianTierMask
from gaussian_jscc.codec import CodecConfig, GaussianCodec, pack, prefix_mask, unpack
from gaussian_jscc.data import prepare, to_features, to_raw, write_ply
from gaussian_jscc.losses import physical_terms, reconstruction_loss
from gaussian_jscc.rendering import RenderReference
from gaussian_jscc.training import joint_scene_step
from gaussian_jscc.transport import load_checkpoint, model_id, receive, save_checkpoint, transmit
from test_gaussian_jscc import fixture


@contextmanager
def fake_mask_rasterizer():
    # Do not patch.dict(sys.modules): restoring the whole dict can unload lazy
    # PyTorch operator registrations and cause duplicate TORCH_LIBRARY errors.
    name = "mask_diff_gaussian_rasterization"
    previous = sys.modules.get(name)
    sys.modules[name] = ModuleType(name)
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class GeometryUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_branch_prefix_budgets_and_no_extra_coordinates(self):
        raw, model = fixture()
        q = torch.arange(len(raw)) % 4
        table = prefix_mask(torch.arange(4), model.cfg.rates)
        self.assertEqual((table[:, model.geometry_slots].sum(-1) // 2).tolist(),
                         list(model.cfg.geometry_rates))
        self.assertEqual((table[:, model.attribute_slots].sum(-1) // 2).tolist(),
                         [r-g for r, g in zip(model.cfg.rates, model.cfg.geometry_rates)])
        with tempfile.TemporaryDirectory() as directory:
            stats = transmit(model, raw, q, 10., "none", 42, Path(directory) / "packet")
            self.assertEqual(stats["geometry_complex_symbols"] + stats["attribute_complex_symbols"],
                             stats["payload_complex_symbols"])
            self.assertFalse(stats["per_gaussian_coordinates_in_metadata"])
            self.assertEqual(stats["global_geometry_floats"], 6)
            pred, seed = receive(model, Path(directory) / "packet", return_position_seed=True)
            torch.testing.assert_close(pred[:, :3], seed)
        for g in ((0, 0, 2, 3), (0, 2, 2, 3), (0, 1, 3, 2)):
            with self.assertRaises(ValueError):
                CodecConfig(rates=(0, 2, 4, 6), geometry_rates=g)

    def test_xyz_independent_of_attribute_symbols_and_decoder_context(self):
        raw, model = fixture(n=8)
        raw, geometry, _ = prepare(raw, 16)
        f, xyz = to_features(raw, geometry, model)
        q = torch.full((8,), 3, dtype=torch.long)
        symbols = model.encode(f, xyz, q, 10.).detach()
        original = model.decode(symbols, q, 10.)
        padded = unpack(symbols, q, model.cfg.rates)
        padded[:, model.attribute_slots] += 2
        changed = model.decode(pack(padded, q, model.cfg.rates), q, 10.)
        torch.testing.assert_close(original[:, :3], changed[:, :3], atol=0, rtol=0)
        self.assertFalse(torch.allclose(original[:, 3:], changed[:, 3:]))
        with torch.no_grad():
            for parameter in model.dec_blocks.parameters():
                parameter.add_(torch.randn_like(parameter))
        altered_context = model.decode(symbols, q, 10.)
        torch.testing.assert_close(original[:, :3], altered_context[:, :3], atol=0, rtol=0)

    def test_physical_identity_quaternion_sign_and_local_sensitivity(self):
        raw, model = fixture(n=8)
        raw, geometry, _ = prepare(raw, 16)
        f, _ = to_features(raw, geometry, model)
        target = f.clone().requires_grad_()
        pred = f.clone().requires_grad_()
        loss = reconstruction_loss(pred, target, geometry, model)
        self.assertLess(float(loss.detach()), 1e-10)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        flipped = raw.clone()
        flipped[:, 7:11] *= -1
        ff, _ = to_features(flipped, geometry, model)
        self.assertLess(float(physical_terms(ff, f, geometry, model)["shape"].max()), 1e-10)
        shifted = raw.clone()
        shifted[:, 0] += .001
        sf, _ = to_features(shifted, geometry, model)
        large = physical_terms(sf, f, geometry, model)["geometry"].mean()
        small_source = raw.clone()
        small_source[:, 4:7] -= 2
        small_pred = small_source.clone()
        small_pred[:, 0] += .001
        small = physical_terms(to_features(small_pred, geometry, model)[0],
                               to_features(small_source, geometry, model)[0], geometry, model)["geometry"].mean()
        self.assertGreater(float(small), float(large))
        extreme = f.clone()
        extreme[:, 4:7] = 100
        extreme.requires_grad_()
        value = reconstruction_loss(extreme, f, geometry, model)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_legacy_checkpoint_and_packet_identity_preserved(self):
        raw, new = fixture(n=8)
        config = new.cfg.to_dict()
        for key in ("architecture", "geometry_rates", "geometry_weight", "shape_weight",
                    "opacity_weight", "dc_weight", "sh_weight", "geometry_floor"):
            config.pop(key)
        legacy = GaussianCodec(CodecConfig.from_dict(config))
        self.assertEqual(legacy.cfg.architecture, "legacy")
        self.assertEqual(legacy.cfg.to_dict(), config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_checkpoint(root / "old.pt", legacy, 20000)
            restored = load_checkpoint(root / "old.pt", "cpu")
            self.assertEqual(model_id(legacy), model_id(restored))
            transmit(legacy, raw, torch.ones(8, dtype=torch.long), 10., "none", 42, root / "packet")
            torch.testing.assert_close(receive(legacy, root / "packet"), receive(restored, root / "packet"))

    def test_source_reference_frozen_cached_and_not_photograph(self):
        raw, _ = fixture(n=8)
        teacher = RenderReference(raw.requires_grad_(), 3)
        camera = SimpleNamespace(original_image=torch.zeros(3, 4, 4))
        with patch("gaussian_jscc.rendering.render", return_value=torch.ones(3, 4, 4)) as render:
            first = teacher.get(camera, "cpu")
            second = teacher.get(camera, "cpu")
        self.assertEqual(render.call_count, 1)
        self.assertFalse(first.requires_grad)
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, camera.original_image))

    def test_joint_replay_matches_checkpoint_and_full_autograd(self):
        raw, model = fixture(n=19, degree=0)
        raw, geometry, order = prepare(raw, 16, torch.arange(len(raw)))
        f, _ = to_features(raw, geometry, model)
        blocks, ids = list(f.split(8)), list(order.split(8))
        batches = [(pad_sequence(blocks[:2], batch_first=True),
                    pad_sequence(ids[:2], batch_first=True, padding_value=-1)),
                   (blocks[-1][None], ids[-1][None])]
        mask = GaussianTierMask(len(raw), torch.full((len(raw),), .6), snr_conditioned=True)
        parameters = list(model.parameters()) + list(mask.parameters())

        def distortion(scene, active):
            return ((scene[:, :3] * active[:, None] - raw[:, :3]).square().mean()
                    + .1 * (scene[:, 3] * active).square().mean())

        for kind in ("none", "awgn", "rayleigh"):
            outputs = []
            for mode in ("direct", "checkpoint", "replay"):
                model.zero_grad(set_to_none=True)
                mask.zero_grad(set_to_none=True)
                torch.manual_seed(31)
                if mode == "direct":
                    rows, gates, aux = [], [], []
                    for features, index in batches:
                        valid = index >= 0
                        choices = F.gumbel_softmax(mask.scores(index.clamp_min(0), 7.), tau=.7, hard=True)
                        padding = torch.zeros_like(choices)
                        padding[..., 0] = 1
                        choices = torch.where(valid[..., None], choices, padding)
                        pred, _, active = model.forward_tier_batches(features, features[..., :3], choices, 7., kind)
                        rows.append(to_raw(pred[valid], geometry, model))
                        gates.append(active[valid])
                        aux.append(reconstruction_loss(pred[valid], features[valid], geometry, model,
                                                       active=active[valid], reduction="sum"))
                    rate = (mask.scores(torch.arange(len(raw)), 7.).softmax(-1)
                            * torch.tensor(model.cfg.rates)).sum() / len(raw) / model.cfg.rates[-1]
                    loss = distortion(torch.cat(rows), torch.cat(gates)) + .1 * sum(aux) / len(raw) + .03 * rate
                    loss.backward()
                else:
                    loss, stats = joint_scene_step(model, mask, batches, geometry, 7., kind, distortion,
                                                    temperature=.7, beta=.03, mode=mode)
                    self.assertEqual(stats["source_gaussians"], len(raw))
                    self.assertEqual(sum(stats["sampled_tier_counts"]), len(raw))
                outputs.append((loss.detach(), [None if p.grad is None else p.grad.clone() for p in parameters],
                                torch.rand(4)))
                self.assertGreater(float(mask.logits.grad.abs().sum()), 0)
            for actual in outputs[1:]:
                torch.testing.assert_close(actual[0], outputs[0][0])
                torch.testing.assert_close(actual[2], outputs[0][2], atol=0, rtol=0)
                for x, y in zip(actual[1], outputs[0][1]):
                    self.assertEqual(x is None, y is None)
                    if x is not None:
                        torch.testing.assert_close(x, y, atol=3e-6, rtol=1e-4)

    def test_joint_render_cli_with_mocked_cuda_boundary(self):
        from gaussian_jscc.cli import main
        raw, model = fixture(n=19, degree=0)
        camera = SimpleNamespace(original_image=torch.zeros(3, 8, 8))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_ply(root / "scene.ply", raw, 0)
            save_checkpoint(root / "init.pt", model, 0)
            argv = ["gaussian_jscc", "train-route2", "--ply", str(root / "scene.ply"),
                    "--codec-init", str(root / "init.pt"), "--out", str(root / "run"),
                    "--source", "mock", "--device", "cuda", "--warmup-steps", "0", "--joint-steps", "2",
                    "--blocks-per-batch", "2", "--training-data-device", "cpu", "--profile-every", "1"]

            def render(scene, camera, degree, background, existence=None):
                if existence is not None:
                    scene = scene * existence[:, None]
                return scene[:, :3].mean(0).sigmoid()[:, None, None].expand(3, 8, 8)

            with patch.object(sys, "argv", argv), \
                    fake_mask_rasterizer(), \
                    patch("gaussian_jscc.cli.device_for", return_value=torch.device("cpu")), \
                    patch("gaussian_jscc.rendering.load_cameras", return_value=[camera]), \
                    patch("gaussian_jscc.rendering.render", side_effect=render):
                main()
            logs = [json.loads(row) for row in (root / "run" / "loss.jsonl").read_text().splitlines()]
            self.assertEqual(len(logs), 2)
            self.assertTrue((root / "run" / "codec.pt").exists())
            for row in logs:
                self.assertEqual(row["phase"], "joint_render")
                self.assertGreater(row["mask_grad_norm"], 0)
                self.assertEqual(row["codec_batches"], 2)


if __name__ == "__main__":
    unittest.main()
