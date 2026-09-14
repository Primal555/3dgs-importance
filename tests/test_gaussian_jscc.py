"""CPU tests; run: python -m unittest discover -s tests -p test_gaussian_jscc.py -v."""

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

from gaussian_jscc.codec import (CodecConfig, GaussianCodec, GridContext, channel,
                                 normalize_power, pack, prefix_mask, unpack)
from gaussian_jscc.data import (attribute_loss, load_tiers, prepare, read_ply,
                                to_features, to_raw, write_ply)
from gaussian_jscc.transport import (decode_metadata, load_checkpoint, receive,
                                     pack_tiers, save_checkpoint, transmit,
                                     unpack_tiers)


def fixture(n=17, degree=3):
    torch.manual_seed(17)
    raw = torch.randn(n, 11 + 3 * (degree + 1) ** 2) * .1
    raw[:, :3] = torch.rand(n, 3)
    raw[:, 4:7] = -3
    raw[:, 7:11] = torch.tensor([1., 0., 0., 0.])
    cfg = CodecConfig(sh_degree=degree, hidden=16, grid_dim=4, depth=1,
                      levels=(2, 3), block_size=8, rates=(0, 2, 4, 6), morton_bits=16)
    return raw, GaussianCodec(cfg)


class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prefix_lengths_and_gradients(self):
        q, rates = torch.tensor([0, 1, 2, 3]), (0, 2, 4, 6)
        z = torch.arange(48.).reshape(4, 12).requires_grad_()
        packed = pack(z, q, rates)
        self.assertEqual(packed.shape, (12, 2))
        restored = unpack(packed, q, rates)
        mask = prefix_mask(q, rates)
        torch.testing.assert_close(restored, z * mask)
        restored.sum().backward()
        torch.testing.assert_close(z.grad, mask.float())
        with self.assertRaises(ValueError):
            unpack(packed[:-1], q, rates)

    def test_power_and_awgn(self):
        z = normalize_power(torch.randn(100000, 2))
        self.assertAlmostEqual(float(z.square().sum(-1).mean()), 1., places=5)
        for snr in (0., 10., 20.):
            noise = channel(z, snr) - z
            self.assertAlmostEqual(float(noise.square().sum(-1).mean()) / 10 ** (-snr / 10),
                                   1., delta=.02)
        self.assertTrue(torch.isfinite(channel(z, 0., "rayleigh")).all())
        torch.testing.assert_close(channel(z, 0., "none"), z)

    def test_grid_permutation_and_neighbor_gradient(self):
        grid = GridContext(8, 3, (2, 3), True)
        x = torch.randn(9, 8, requires_grad=True)
        xyz = torch.rand(9, 3)
        perm = torch.randperm(9)
        original = grid(x, xyz)
        torch.testing.assert_close(grid(x[perm], xyz[perm]), original[perm], atol=1e-6, rtol=1e-5)
        original[0].square().sum().backward()
        self.assertGreater(float(x.grad[1:].abs().sum()), 0.)

    def test_encoder_and_decoder_context_receive_gradients(self):
        raw, model = fixture()
        raw, geom, _ = prepare(raw, 16)
        features, xyz = to_features(raw, geom, model)
        q = torch.arange(len(raw)) % 3 + 1
        output = model(features, xyz, q, 7.)
        attribute_loss(output, features).backward()
        for blocks in (model.enc_blocks, model.dec_blocks):
            grad = blocks[0].grid.project.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)
        self.assertEqual(output.shape, features.shape)
        with self.assertRaises(ValueError):
            model(features, xyz, q * 0, 7.)

    def test_noiseless_optimization_reduces_loss(self):
        raw, model = fixture(n=8, degree=0)
        raw, geom, _ = prepare(raw, 16)
        features, xyz = to_features(raw, geom, model)
        q = torch.full((len(raw),), 3, dtype=torch.long)
        optimizer = torch.optim.Adam(model.parameters(), lr=.005)
        losses = []
        for _ in range(35):
            optimizer.zero_grad()
            loss = attribute_loss(model(features, xyz, q, 10., "none"), features)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0] * .35)

    def test_checkpoint_recomputation_preserves_noisy_gradients(self):
        from torch.utils.checkpoint import checkpoint
        raw, model = fixture(n=8)
        raw, geom, _ = prepare(raw, 16)
        f, xyz = to_features(raw, geom, model)
        q = torch.arange(8) % 3 + 1
        gradients = []
        for recompute in (False, True):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(19)
            def forward(a, b, c):
                return model(a, b, c, 5., "awgn", return_seed=True)
            pred, seed = (checkpoint(forward, f, xyz, q, use_reentrant=False, preserve_rng_state=True)
                          if recompute else forward(f, xyz, q))
            (attribute_loss(pred, f) + .2 * torch.nn.functional.smooth_l1_loss(
                seed, f[:, :3])).backward()
            gradients.append([None if p.grad is None else p.grad.clone() for p in model.parameters()])
        for normal, recomputed in zip(*gradients):
            self.assertEqual(normal is None, recomputed is None)
            if normal is not None:
                torch.testing.assert_close(normal, recomputed)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA rasterizer")
    def test_cuda_rasterizer_backprop(self):
        import importlib.util
        if importlib.util.find_spec("diff_gaussian_rasterization") is None:
            self.skipTest("diff_gaussian_rasterization is not installed")
        from gaussian_jscc.rendering import render
        from utils.graphics_utils import getProjectionMatrix
        raw, _ = fixture(n=8)
        raw[:, :2] = (raw[:, :2] - .5) * .3
        raw[:, 2] = 2.
        raw = raw.cuda().requires_grad_()
        camera = SimpleNamespace(image_height=32, image_width=32, FoVx=1., FoVy=1.,
                                 world_view_transform=torch.eye(4, device="cuda"),
                                 full_proj_transform=getProjectionMatrix(.01, 100., 1., 1.).T.cuda(),
                                 camera_center=torch.zeros(3, device="cuda"))
        image = render(raw, camera, 3)
        self.assertEqual(image.shape, (3, 32, 32))
        image.mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0.)

    def test_ply_layout_and_feature_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for degree in (0, 3):
                raw, model = fixture(degree=degree)
                path = Path(directory) / f"degree{degree}.ply"
                write_ply(path, raw, degree)
                restored, actual_degree = read_ply(path)
                self.assertEqual(actual_degree, degree)
                torch.testing.assert_close(restored, raw)
                ordered, geometry, _ = prepare(raw, 16)
                features, _ = to_features(ordered, geometry, model)
                torch.testing.assert_close(to_raw(features, geometry, model), ordered,
                                           atol=2e-7, rtol=1e-5)

    def test_rate_map_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiers.npy"
            np.save(path, np.array([0, 1, 2, 3]))
            self.assertEqual(load_tiers(path, 4).tolist(), [0, 1, 2, 3])
            with self.assertRaises(ValueError):
                load_tiers(path, 5)
            np.save(path, np.array([1., 2., 3.]))
            with self.assertRaises(ValueError):
                load_tiers(path, 3)

    def test_tier_metadata_is_exactly_two_bits_per_gaussian(self):
        q = np.arange(10003, dtype=np.uint8) % 4
        packed = pack_tiers(q)
        self.assertEqual(len(packed), (len(q) + 3) // 4)
        np.testing.assert_array_equal(unpack_tiers(packed, len(q)), q)

    def test_packet_standalone_and_symbol_accounting(self):
        raw, model = fixture()
        q = torch.arange(len(raw)) % 4
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            save_checkpoint(directory / "codec.pt", model, 0)
            stats = transmit(model, raw, q, 10., "none", 42, directory / "packet")
            # New decoder instance; it receives neither clean attributes nor original PLY.
            receiver = load_checkpoint(directory / "codec.pt", "cpu")
            reconstructed = receive(receiver, directory / "packet")
            reconstructed_with_seed, position_seed = receive(
                receiver, directory / "packet", return_position_seed=True)
            torch.testing.assert_close(reconstructed_with_seed, reconstructed)
            self.assertEqual(position_seed.shape, (stats["retained_gaussians"], 3))
            self.assertTrue(torch.isfinite(position_seed).all())
            ordered, geom, oq = prepare(raw, 16, q)
            expected = []
            for start in range(0, len(raw), model.cfg.block_size):
                qb = oq[start:start + model.cfg.block_size]
                keep = qb > 0
                rb = ordered[start:start + len(qb)][keep]
                if not len(rb):
                    continue
                f, xyz = to_features(rb, geom, model)
                expected.append(to_raw(model(f, xyz, qb[keep], 10., "none"), geom, model))
            torch.testing.assert_close(reconstructed, torch.cat(expected))
            self.assertEqual(stats["payload_complex_symbols"], int(torch.tensor(model.cfg.rates)[q].sum()))
            self.assertEqual(stats["retained_gaussians"], int((q > 0).sum()))
            self.assertGreater(stats["total_channel_uses"], stats["payload_complex_symbols"])

    def test_all_dropped_packet_and_empty_blocks(self):
        raw, model = fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for i, q in enumerate((torch.zeros(len(raw), dtype=torch.long),
                                   torch.tensor([0] * 16 + [2]))):
                packet = directory / str(i)
                transmit(model, raw, q, 10., "awgn", 42, packet)
                decoded = receive(model, packet)
                self.assertEqual(len(decoded), int((q > 0).sum()))
                write_ply(directory / f"{i}.ply", decoded, model.cfg.sh_degree)

    def test_packet_integrity_and_model_mismatch(self):
        raw, model = fixture()
        with tempfile.TemporaryDirectory() as directory:
            packet = Path(directory) / "packet"
            transmit(model, raw, torch.ones(len(raw), dtype=torch.long), 10., "awgn", 42, packet)
            metadata = (packet / "metadata.bin").read_bytes()
            with self.assertRaisesRegex(ValueError, "CRC"):
                decode_metadata(metadata[:-1] + bytes([metadata[-1] ^ 1]))
            other = GaussianCodec(model.cfg)
            with self.assertRaisesRegex(ValueError, "do not match"):
                receive(other, packet)
            np.save(packet / "received.npy", np.zeros((1, 2), np.float32))
            with self.assertRaisesRegex(ValueError, "invalid received"):
                receive(model, packet)

    def test_cli_train_transmit_decode_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, _ = fixture(n=12, degree=0)
            source = root / "source.ply"
            write_ply(source, raw, 0)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

            def run(*args):
                result = subprocess.run([sys.executable, "-m", "gaussian_jscc", *map(str, args)],
                                        env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            training = root / "training"
            run("train", "--ply", source, "--out", training, "--device", "cpu", "--steps", 3,
                "--hidden", 16, "--grid-dim", 4, "--depth", 1, "--levels", 2, 3,
                "--rates", 0, 2, 4, 6, "--block-size", 8)
            weights = training / "codec.pt"
            run("transmit", "--ply", source, "--checkpoint", weights, "--out", root / "packet",
                "--device", "cpu", "--uniform-tier", 2)
            # Removing the source proves the independent decode command does not read it.
            source.unlink()
            run("decode", "--checkpoint", weights, "--packet", root / "packet",
                "--out", root / "decoded.ply", "--device", "cpu")
            decoded, _ = read_ply(root / "decoded.ply")
            self.assertEqual(len(decoded), 12)
            write_ply(source, raw, 0)
            run("evaluate", "--ply", source, "--checkpoint", weights, "--out", root / "eval",
                "--device", "cpu", "--snrs", 0, 10, "--tiers", 1, 2, 3)
            results = json.loads((root / "eval" / "results.json").read_text())
            self.assertEqual(len(results), 6)


if __name__ == "__main__":
    unittest.main()
