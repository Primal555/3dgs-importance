"""Numerical contracts for vectorized contexts, batched packets and replay."""

from itertools import product
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from gaussian_jscc.codec import GridContext
from gaussian_jscc.data import prepare, to_features, write_ply
from gaussian_jscc.training import codec_batch, full_scene_step
from gaussian_jscc.transport import save_checkpoint
from test_gaussian_jscc import fixture


def original_grid(grid, h, xyz, active):
    """Frozen pre-optimization algorithm, independent of the optimized forward."""
    f = grid.project(h)
    with torch.no_grad():
        retained = xyz[active.detach() > .5]
        if len(retained) == 0:
            retained = xyz
        lower = retained.amin(0)
        span = (retained.amax(0) - lower).clamp_min(1e-8)
    unit = ((xyz - lower) / span).clamp(0, 1)
    contexts = []
    for axes in grid.axes:
        for resolution in grid.levels:
            p = unit[:, list(axes)] * (resolution - 1)
            base = p.floor().long().clamp(max=resolution - 2)
            frac = p - base
            cells = f.new_zeros((resolution ** len(axes), f.shape[-1]))
            mass = f.new_zeros((len(cells), 1))
            queries = []
            for corner in product((0, 1), repeat=len(axes)):
                c = torch.tensor(corner, device=h.device)
                weight = torch.where(c.bool(), frac, 1 - frac).prod(-1, keepdim=True)
                contribution = weight * active[:, None]
                vertex = base + c
                index = sum(vertex[:, d] * resolution ** d for d in range(len(axes)))
                cells = cells.index_add(0, index, f * contribution)
                mass = mass.index_add(0, index, contribution)
                queries.append((index, weight))
            cells = cells / mass.clamp_min(1e-8)
            contexts.append(sum(cells[index] * weight for index, weight in queries))
    return torch.cat(contexts, -1)


class TrainingPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_vectorized_grid_values_and_feature_position_mask_gradients(self):
        torch.manual_seed(4)
        grid = GridContext(8, 3, (2, 3), True).double()
        h = torch.randn(13, 8, dtype=torch.double, requires_grad=True)
        xyz = torch.rand(13, 3, dtype=torch.double, requires_grad=True)
        active = (torch.arange(13) % 3 != 0).double().requires_grad_()
        results = []
        for operation in (lambda: original_grid(grid, h, xyz, active), lambda: grid(h, xyz, active)):
            output = operation()
            gradients = torch.autograd.grad(output.square().mean(),
                                            (h, xyz, active, grid.project.weight))
            results.append((output, gradients))
        torch.testing.assert_close(results[0][0], results[1][0], atol=1e-10, rtol=1e-10)
        for a, b in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)

    def make_batches(self):
        raw, model = fixture(n=19, degree=0)
        raw, geometry, _ = prepare(raw, 16)
        features, _ = to_features(raw, geometry, model)
        blocks = list(features.split(8))
        tiers = [torch.arange(len(f)) % 4 for f in blocks]
        batches = [(pad_sequence(blocks[:2], batch_first=True), pad_sequence(tiers[:2], batch_first=True)),
                   (blocks[2][None], tiers[2][None])]
        return model, geometry, blocks, tiers, batches

    def test_independent_batched_blocks_match_packed_codec_and_gradients(self):
        model, geometry, blocks, tiers, _ = self.make_batches()
        f = pad_sequence(blocks, batch_first=True)
        q = pad_sequence(tiers, batch_first=True)
        results, gradients = [], []
        for batched in (False, True):
            model.zero_grad(set_to_none=True)
            if batched:
                pred, _, _ = model.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).float(), 10., "none")
                result = pred[q > 0]
            else:
                result = torch.cat([model(a, a[:, :3], t, 10., "none")[t > 0]
                                    for a, t in zip(blocks, tiers)])
            result.square().mean().backward()
            results.append(result)
            gradients.append([p.grad.clone() if p.grad is not None else None for p in model.parameters()])
        torch.testing.assert_close(*results, atol=2e-6, rtol=2e-5)
        for a, b in zip(*gradients):
            self.assertEqual(a is None, b is None)
            if a is not None:
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)

    def test_replay_matches_full_autograd_and_checkpoint_including_noise_and_rng(self):
        for kind in ("none", "awgn", "rayleigh"):
            with self.subTest(channel=kind):
                model, geometry, _, _, batches = self.make_batches()
                # An empty fixed-rate batch must contribute neither NaN aux loss
                # nor a rendered row (real training normally filters these out).
                batches = [(batches[0][0], torch.zeros_like(batches[0][1])), *batches]
                losses, gradients, next_random = [], [], []
                # Coupled scene objective: gradients depend on other blocks too.
                def distortion(scene):
                    return scene.square().mean() + scene.mean(0).square().sum() * .1
                for mode in ("autograd", "checkpoint", "replay", "direct"):
                    torch.manual_seed(31)
                    model.zero_grad(set_to_none=True)
                    if mode == "autograd":
                        outputs = [codec_batch(model, f, q, 7., kind, geometry, .2) for f, q in batches]
                        scene = torch.cat([r for r, _ in outputs])
                        loss = distortion(scene) + .1 * sum(a * len(r) for r, a in outputs) / len(scene)
                        loss.backward()
                    else:
                        loss, _ = full_scene_step(model, batches, geometry, 7., kind, distortion,
                                                  mode=mode, profile=True)
                    losses.append(loss.detach())
                    gradients.append([p.grad.clone() if p.grad is not None else None for p in model.parameters()])
                    next_random.append(torch.rand(4))
                for index in (1, 2, 3):
                    torch.testing.assert_close(losses[0], losses[index])
                    torch.testing.assert_close(next_random[0], next_random[index], atol=0, rtol=0)
                    for a, b in zip(gradients[0], gradients[index]):
                        self.assertEqual(a is None, b is None)
                        if a is not None:
                            torch.testing.assert_close(a, b, atol=2e-6, rtol=3e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for device RNG replay")
    def test_cuda_replay_preserves_device_noise_and_gradients(self):
        model, geometry, _, _, batches = self.make_batches()
        model = model.cuda()
        results = []
        for mode in ("checkpoint", "replay", "direct"):
            torch.manual_seed(71)
            model.zero_grad(set_to_none=True)
            loss, _ = full_scene_step(model, batches, geometry, 3., "awgn",
                                      lambda scene: scene.square().mean(), mode=mode)
            results.append((loss, [p.grad.clone() if p.grad is not None else None for p in model.parameters()],
                            torch.rand(8, device="cuda")))
        for result in results[1:]:
            torch.testing.assert_close(results[0][0], result[0])
            torch.testing.assert_close(results[0][2], result[2], atol=0, rtol=0)
            for a, b in zip(results[0][1], result[1]):
                if a is not None:
                    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)

    def test_direct_executes_codec_once_per_batch_and_never_checkpoints(self):
        model, geometry, _, _, batches = self.make_batches()
        with patch.object(model, 'forward_tier_batches', wraps=model.forward_tier_batches) as forward, \
             patch('gaussian_jscc.training.checkpoint', side_effect=AssertionError('must not checkpoint')):
            _, stats = full_scene_step(model,batches,geometry,10,'awgn',
                                       lambda scene:scene.square().mean(),attr_weight=0,profile=True)
        self.assertEqual(forward.call_count,len(batches))
        self.assertEqual(stats['render_backward'],'direct')
        self.assertIn('codec_backward_seconds',stats)
        self.assertNotIn('codec_replay_backward_seconds',stats)


if __name__ == "__main__":
    unittest.main()
