"""CPU checks for the local allocator surrogate; real CUDA image parity is separate."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from gaussian_jscc.allocation import GaussianTierMask
from gaussian_jscc.allocation_diagnostics import AllocationCostMeter
from gaussian_jscc.mask_gradient import local_mask_step
from gaussian_jscc.position_delivery import PositionCostMeter
from gaussian_jscc.render_objective import MaskedMultiViewRenderTask
from test_progressive_prefix import progressive


def toy_mask_render(raw, camera, degree, white_background=False, existence=None):
    weight = raw.new_ones(len(raw)) if existence is None else existence
    divisor = getattr(camera, 'source_count', max(len(raw), 1))
    color = ((raw[:, :3] * weight[:, None]).sum(0) * camera.factor / divisor
             + (raw[:, 3:4] * weight[:, None]).sum() * .01 / divisor)
    return color.sigmoid()[:, None, None].expand(3, 4, 4)


class Reference:
    def get(self, camera, device):
        return torch.full((3, 4, 4), .6, device=device)


class LocalMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hard_forward_matches_existing_paired_codec_and_has_tier_gradient(self):
        _, _, f, model = progressive(8)
        q = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 1]])
        logits = torch.zeros(1, 8, 4, requires_grad=True)
        probabilities = logits.softmax(-1)
        hard = F.one_hot(q, 4).to(f)
        torch.manual_seed(19)
        expected = model.forward_tier_batches(f[None], f[None, :, :3], hard, 10, 'awgn',
                                              paired_noise=True)[0]
        torch.manual_seed(19)
        observed = model.forward_st_prefix_batches(f[None], f[None, :, :3], q,
                                                   probabilities, 10, 'awgn')
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        observed.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[:, q[0] > 0, 1:].norm()), 0.)

    def test_q0_and_positive_tiers_get_image_feedback_without_codec_update(self):
        raw, geometry, features, model = progressive(12)
        mask = GaussianTierMask(12, existence_prior=torch.full((12,), .5))
        meter = AllocationCostMeter(model.cfg, PositionCostMeter(model.cfg,
                                                               geometry.normalize(raw[:, :3])), 12, 2.)
        camera = SimpleNamespace(factor=1.)
        task = MaskedMultiViewRenderTask([camera], Reference(), 0)
        with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            torch.manual_seed(7)
            loss, stats = local_mask_step(model, mask, [features[None]],
                                          [torch.arange(12)[None]], geometry, 10, 'awgn',
                                          task, beta=.01, rate_meter=meter, train_codec=False)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(stats['sampled_tier_counts'][0], 0)
        self.assertGreater(sum(stats['sampled_tier_counts'][1:]), 0)
        self.assertTrue(torch.isfinite(mask.logits.grad).all())
        self.assertGreater(float(mask.logits.grad.norm()), 0.)
        self.assertFalse(any(p.grad is not None for p in model.parameters()))
        self.assertEqual(stats['sampled_retained_gaussians'], 12 - stats['sampled_tier_counts'][0])
        self.assertEqual(stats['sampled_q0_shadow_candidates'], stats['sampled_tier_counts'][0])
        self.assertGreater(stats['mask_image_grad_q0_norm'], 0.)
        self.assertGreater(stats['mask_keep_image_grad_norm'], 0.)
        self.assertGreater(stats['mask_tier_image_grad_norm'], 0.)
        self.assertGreater(stats['mask_image_grad_positive_norm'], 0.)
        self.assertIn('measured_allocation_uses_per_gaussian', stats)

    def test_replay_and_checkpoint_agree_on_hierarchical_gradients(self):
        raw, geometry, features, model = progressive(12)
        results = []
        for mode in ('replay', 'checkpoint'):
            mask = GaussianTierMask(12, existence_prior=torch.full((12,), .5))
            meter = AllocationCostMeter(model.cfg, PositionCostMeter(model.cfg,
                                        geometry.normalize(raw[:, :3])), 12, 2.)
            task = MaskedMultiViewRenderTask([SimpleNamespace(factor=1.)], Reference(), 0)
            with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
                torch.manual_seed(7)
                loss, stats = local_mask_step(model, mask, [features[None]],
                    [torch.arange(12)[None]], geometry, 10, 'awgn', task,
                    beta=.01, rate_meter=meter, train_codec=False, mode=mode)
            results.append((loss, mask.keep_logits.grad, mask.logits.grad))
        for first, second in zip(*results):
            torch.testing.assert_close(first, second)

    def test_joint_mode_updates_codec_and_rate_proxy_favors_lower_budget(self):
        raw, geometry, features, model = progressive(8)
        mask = GaussianTierMask(8, existence_prior=torch.full((8,), .5))
        meter = AllocationCostMeter(model.cfg, PositionCostMeter(model.cfg,
                                                               geometry.normalize(raw[:, :3])), 8, 2.)
        camera = SimpleNamespace(factor=1.)
        task = MaskedMultiViewRenderTask([camera], Reference(), 0)
        rates = torch.tensor(model.cfg.rates)
        before = float((mask.probabilities(torch.arange(8),10) * rates).sum(-1).mean().detach())
        optimizer = torch.optim.SGD(mask.parameters(), lr=.1)
        with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            torch.manual_seed(11)
            _, stats = local_mask_step(model, mask, [features[None]],
                                       [torch.arange(8)[None]], geometry, 10, 'none',
                                       task, beta=10., rate_meter=meter, train_codec=True)
        self.assertTrue(stats['codec_updated'])
        self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters()))
        optimizer.step()
        after = float((mask.probabilities(torch.arange(8),10) * rates).sum(-1).mean().detach())
        self.assertLess(after, before)

    def test_zero_mask_shadow_does_not_change_hard_deployment_image(self):
        from gaussian_jscc.training import codec_batch
        raw, geometry, features, model = progressive(8)
        mask = GaussianTierMask(8, existence_prior=torch.full((8,), .5))
        meter = AllocationCostMeter(model.cfg, PositionCostMeter(model.cfg,
                                                               geometry.normalize(raw[:, :3])), 8, 2.)
        camera = SimpleNamespace(factor=1., source_count=8)
        task = MaskedMultiViewRenderTask([camera], Reference(), 0)
        tiers = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        deployed = codec_batch(model, features[None], tiers, 10, 'none', geometry, 0,
                               compute_auxiliary=False, paired_noise=True)[0]
        target = Reference().get(camera, torch.device('cpu'))
        exact = float((toy_mask_render(deployed, camera, 0) - target).square().mean().detach())
        original_sample = mask.sample
        def fixed(indices, snr, **kwargs):
            _, presence, choices, positive = original_sample(indices, snr, **kwargs)
            positive = torch.where(tiers > 0, tiers, positive)
            return tiers, presence, choices, positive
        with patch.object(mask, 'sample', side_effect=fixed), \
             patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            _, stats = local_mask_step(model, mask, [features[None]], [torch.arange(8)[None]],
                                       geometry, 10, 'none', task, beta=0., rate_meter=meter,
                                       train_codec=False)
        self.assertAlmostEqual(stats['render_loss'], exact, places=6)
        self.assertEqual(stats['sampled_tier_counts'], [2, 2, 2, 2])

    def test_every_q0_gets_shadow_even_when_more_than_four_per_block(self):
        raw, geometry, features, model = progressive(12)
        mask = GaussianTierMask(12, existence_prior=torch.full((12,), .5))
        meter = AllocationCostMeter(model.cfg, PositionCostMeter(model.cfg,
                                                               geometry.normalize(raw[:, :3])), 12, 2.)
        task = MaskedMultiViewRenderTask([SimpleNamespace(factor=1.)], Reference(), 0)
        tiers = torch.tensor([[0]*7 + [1]*5])
        original_sample = mask.sample
        def fixed(indices, snr, **kwargs):
            _, presence, choices, positive = original_sample(indices, snr, **kwargs)
            return tiers, presence, choices, torch.where(tiers > 0, tiers, positive)
        with patch.object(mask, 'sample', side_effect=fixed), \
             patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            _, stats = local_mask_step(model, mask, [features[None]],
                                       [torch.arange(12)[None]], geometry, 10, 'none',
                                       task, beta=0., rate_meter=meter, train_codec=False)
        self.assertEqual(stats['sampled_q0_shadow_candidates'], 7)
        self.assertEqual(stats['sampled_q0_shadow_candidates'], stats['sampled_tier_counts'][0])


if __name__ == '__main__':
    unittest.main()
