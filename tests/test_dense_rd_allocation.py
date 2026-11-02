"""Restored dense masks: objective units, closed-point feedback and ending."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gaussian_jscc.allocation import GaussianTierMask
from gaussian_jscc.dense_rd_control import DeploymentPlateau
from gaussian_jscc.dense_rd_training import DenseRateMeter, NormalizedMaskedTask
from gaussian_jscc.mask_gradient import local_mask_step
from test_mask_gradient import Reference, toy_mask_render
from test_successive_allocation import small_successive


class PlateauTests(unittest.TestCase):
    def controller(self):
        return DeploymentPlateau(min_steps=2, patience=2, factor=.5, min_scale=.25,
            relative_improvement=.001, map_tolerance=.01, cost_tolerance=.002)

    def test_floor_then_stable_plateau_ends_and_improvement_resets(self):
        c = self.controller(); q = torch.ones(100, dtype=torch.long)
        self.assertEqual(c.observe(0, 1., 100., q)['action'], 'improved')
        self.assertEqual(c.observe(2, 1., 100., q)['action'], 'continue')
        self.assertEqual(c.observe(3, 1., 100., q)['action'], 'reduce_lr')
        self.assertEqual(c.scale, .5)
        c.observe(4, .98, 100., q)
        self.assertEqual(c.stale, 0)
        c.observe(5, .98, 100., q)
        self.assertEqual(c.observe(6, .98, 100., q)['action'], 'reduce_lr')
        c.observe(7, .98, 100., q)
        self.assertEqual(c.observe(8, .98, 100., q)['action'], 'stop')

    def test_moving_map_or_cost_prevents_false_convergence(self):
        for moving in ('map', 'cost'):
            c = self.controller(); q = torch.ones(100, dtype=torch.long)
            for step in range(9):
                tiers = q + (step % 2) if moving == 'map' else q
                cost = 100 + (step % 2) if moving == 'cost' else 100
                result = c.observe(step, 1., cost, tiers)
            self.assertNotEqual(result['action'], 'stop')
            self.assertEqual(c.scale, .25)

    def test_resume_matches_and_invalid_measurement_does_not_mutate(self):
        c = self.controller(); q = torch.ones(10, dtype=torch.long)
        c.observe(0, 1., 10., q); c.observe(2, 1., 10., q)
        restored = self.controller(); restored.load_state_dict(copy.deepcopy(c.state_dict()))
        self.assertEqual(c.observe(3, 1., 10., q), restored.observe(3, 1., 10., q))
        old = c.stale
        with self.assertRaises(ValueError): c.observe(4, float('nan'), 10., q)
        self.assertEqual(c.stale, old)


class DenseTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(71)
        self.raw, self.geometry, self.features, self.model = small_successive()
        self.model.eval().requires_grad_(False)

    def test_cost_bridge_restores_original_rows_and_total_normalizer(self):
        calls = []
        exact = SimpleNamespace(bits=2., details=lambda q: calls.append(q.clone()) or dict(
            coordinate_channel_uses=160, metadata_bytes=8, scale_channel_uses=12,
            scale_stream_bytes=3, total_channel_uses=300))
        scene = SimpleNamespace(raw=self.raw, order=torch.arange(15, -1, -1))
        meter = DenseRateMeter(exact, scene, dict(c_ref=640, q3_cost=dict(coordinate_channel_uses=240)))
        packet = torch.arange(16) % 4
        result = meter.details(packet)
        torch.testing.assert_close(calls[0], packet.flip(0))
        self.assertEqual(meter.normalizer, 40)
        self.assertEqual(result['allocation_uses_per_source_gaussian'], 300/16)
        self.assertEqual(result['allocation_side_uses_per_source_gaussian'], 204/16)

    def test_normalized_multiview_gradient_equals_joint_graph(self):
        from gaussian_jscc.allocation_distortion import ImageDistortion
        objective = ImageDistortion('mse_ssim', .007)
        cameras = [SimpleNamespace(factor=1.), SimpleNamespace(factor=1.4)]
        raw = torch.cat((self.raw, torch.ones(16, 1)), -1)
        task = NormalizedMaskedTask(cameras, Reference(), 1, distortion=objective, d_ref=.02)
        with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            x = raw.clone().requires_grad_(); expected = task(x); expected.backward()
            y = raw.clone().requires_grad_(); observed = task.backward_scene(y)
        torch.testing.assert_close(observed, expected.detach())
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(float(expected.detach()), task.stats['source_distortion']/.02, places=5)

    def test_every_closed_point_gets_feedback_and_codec_is_frozen(self):
        from gaussian_jscc.allocation_distortion import ImageDistortion
        from gaussian_jscc.allocation_diagnostics import AllocationCostMeter
        from gaussian_jscc.position_delivery import PositionCostMeter
        mask = GaussianTierMask(16, existence_prior=torch.full((16,), .5))
        meter = AllocationCostMeter(self.model.cfg, PositionCostMeter(self.model.cfg,
            self.geometry.normalize(self.raw[:, :3])), 16, 2.)
        task = NormalizedMaskedTask([SimpleNamespace(factor=1., source_count=16)], Reference(), 1,
            distortion=ImageDistortion('mse_ssim', .007), d_ref=.02)
        before = {k: v.clone() for k,v in self.model.state_dict().items()}
        with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            torch.manual_seed(6)
            _, stats = local_mask_step(self.model, mask, [self.features],
                [torch.arange(16).reshape(2,8)], self.geometry, 10., 'awgn', task,
                .1, meter, train_codec=False, reuse_codec=True, rate_chunk_size=16)
        self.assertGreater(stats['sampled_tier_counts'][0], 0)
        self.assertEqual(stats['sampled_q0_shadow_candidates'], stats['sampled_tier_counts'][0])
        self.assertEqual(stats['mask_image_grad_nonzero_rows'], 16)
        self.assertGreater(stats['mask_keep_image_grad_norm'], 0)
        self.assertGreater(stats['mask_tier_image_grad_norm'], 0)
        for value in mask.parameters(): self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        for k,v in self.model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
