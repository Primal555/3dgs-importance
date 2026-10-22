"""Step-local reuse must preserve the allocator's existing stochastic VJP."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gaussian_jscc.allocation import GaussianTierMask
from gaussian_jscc.allocation_diagnostics import AllocationCostMeter
from gaussian_jscc.mask_gradient import local_mask_step
from gaussian_jscc.model_factory import build_codec
from gaussian_jscc.position_delivery import PositionCostMeter
from gaussian_jscc.render_objective import MaskedMultiViewRenderTask
from test_mask_gradient import Reference, toy_mask_render
from test_modular_allocation import small_model


class AllocationReuseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.raw, self.geometry, self.features, old = small_model()
        old.cfg.channel_variant = 'successive'
        old.cfg.channel_power_mode = 'block'
        old.cfg.context_variant = 'both'
        self.model = build_codec(old.cfg)
        self.model.set_scene_geometry(self.geometry)
        self.model.channel_decoder.neighbor_enabled.fill_(True)

    def run_step(self, model, mask, reuse, joint, seed=6, mode='replay', batches=None, ids=None, profile=True):
        model.train()
        model.semantic_encoder.eval()
        model.semantic_encoder.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        mask.zero_grad(set_to_none=True)
        meter = AllocationCostMeter(model.cfg, PositionCostMeter(model.cfg,
            self.geometry.normalize(self.raw[:, :3])), 16, 2.)
        calls = {name: 0 for name in ('semantic_encoder', 'channel_encoder',
                                     'channel_decoder', 'semantic_decoder')}
        handles = []
        for name in calls:
            def count(module, inputs, output, name=name):
                calls[name] += 1
            handles.append(getattr(model, name).register_forward_hook(count))
        task = MaskedMultiViewRenderTask([SimpleNamespace(factor=1., source_count=16)], Reference(), 1)
        scenes = []
        def render(raw, *args, **kwargs):
            scenes.append(raw.detach().clone())
            return toy_mask_render(raw, *args, **kwargs)
        try:
            with patch('gaussian_jscc.rendering.render', side_effect=render):
                torch.manual_seed(seed)
                loss, stats = local_mask_step(model, mask, batches or [self.features],
                    ids or [torch.arange(16).reshape(2, 8)], self.geometry, 10., 'awgn', task,
                    .005, meter, train_codec=joint, channel_latent_weight=1. if joint else 0.,
                    mode=mode, reuse_codec=reuse, profile=profile)
                rng = torch.get_rng_state()
        finally:
            for handle in handles:
                handle.remove()
        gradients = {name: None if p.grad is None else p.grad.clone()
                     for name, p in model.named_parameters()}
        return loss, stats, rng, gradients, calls, scenes

    def test_frozen_and_joint_match_loss_gradients_rng_and_actual_scene(self):
        for joint in (False, True):
            with self.subTest(joint=joint):
                models = [copy.deepcopy(self.model) for _ in range(2)]
                masks = [GaussianTierMask(16, existence_prior=torch.full((16,), .5)) for _ in range(2)]
                slow = self.run_step(models[0], masks[0], False, joint)
                fast = self.run_step(models[1], masks[1], True, joint)
                torch.testing.assert_close(slow[0], fast[0], rtol=0, atol=0)
                self.assertTrue(torch.equal(slow[2], fast[2]))
                for name, grad in slow[3].items():
                    if grad is None:
                        self.assertIsNone(fast[3][name], name)
                    else:
                        torch.testing.assert_close(grad, fast[3][name], rtol=0, atol=0, msg=name)
                for a, b in zip(masks[0].parameters(), masks[1].parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                for a, b in zip(slow[5], fast[5]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertGreater(fast[1]['sampled_q0_shadow_candidates'], 0)
                self.assertEqual(slow[4]['semantic_encoder'], 7)
                self.assertEqual(fast[4]['semantic_encoder'], 2)
                self.assertEqual(fast[4]['channel_encoder'], 6 if joint else 5)
                self.assertTrue(fast[1]['allocation_reuse_enabled'])
                for key in ('codec_forward_seconds', 'codec_replay_backward_seconds',
                            'semantic_encode_seconds', 'q0_shadow_seconds', 'candidate_decode_seconds'):
                    self.assertGreaterEqual(fast[1][key], 0., key)

    def test_cache_never_survives_updates_and_optimizer_trajectory_matches(self):
        models = [copy.deepcopy(self.model) for _ in range(2)]
        masks = [GaussianTierMask(16, existence_prior=torch.full((16,), .5)) for _ in range(2)]
        optimizers = [torch.optim.Adam(list(m.parameters()) + list(a.parameters()), lr=1e-4)
                      for m, a in zip(models, masks)]
        for seed in (6, 9, 12):
            for index in range(2):
                self.run_step(models[index], masks[index], bool(index), True, seed, profile=False)
                optimizers[index].step()
            for a, b in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(masks[0].parameters(), masks[1].parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            states = [o.state_dict() for o in optimizers]
            self.assertEqual(states[0]['param_groups'], states[1]['param_groups'])
            for key, state in states[0]['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, states[1]['state'][key][name], rtol=0, atol=0)

    def test_reuse_rejects_checkpoint_recomputation(self):
        mask = GaussianTierMask(16)
        with self.assertRaisesRegex(ValueError, 'replay'):
            self.run_step(self.model, mask, True, False, mode='checkpoint')

    def test_multiple_batches_empty_retained_block_padding_and_all_positive(self):
        for closed in (True, False):
            with self.subTest(closed_block=closed):
                masks = [GaussianTierMask(16, existence_prior=torch.full((16,), .5)) for _ in range(2)]
                for mask in masks:
                    with torch.no_grad():
                        mask.keep_logits[:, 0].fill_(-100.)
                        mask.keep_logits[:, 1].fill_(100.)
                        if closed:
                            mask.keep_logits[:8, 0].fill_(100.)
                            mask.keep_logits[:8, 1].fill_(-100.)
                padded = torch.cat((self.features[1:], self.features.new_zeros(1, 8, self.features.shape[-1])), 0)
                features = [self.features[:1], padded]
                ids = [torch.arange(8)[None], torch.cat((torch.arange(8, 16)[None],
                                                        torch.full((1, 8), -1)), 0)]
                results = [self.run_step(copy.deepcopy(self.model), mask, bool(i), True,
                    batches=features, ids=ids) for i, mask in enumerate(masks)]
                torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
                self.assertTrue(torch.equal(results[0][2], results[1][2]))
                for name, gradient in results[0][3].items():
                    if gradient is None:self.assertIsNone(results[1][3][name])
                    else:torch.testing.assert_close(gradient, results[1][3][name], rtol=0, atol=0)
                for a, b in zip(masks[0].parameters(), masks[1].parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                self.assertEqual(results[1][4]['semantic_encoder'], 3 if closed else 2)

    def test_trainable_encoder_is_rejected_before_any_sampling(self):
        model = copy.deepcopy(self.model).train()
        model.semantic_encoder.eval()
        mask = GaussianTierMask(16)
        meter = AllocationCostMeter(model.cfg, PositionCostMeter(model.cfg,
            self.geometry.normalize(self.raw[:, :3])), 16, 2.)
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, 'frozen semantic encoder'):
            local_mask_step(model, mask, [self.features], [torch.arange(16).reshape(2, 8)],
                self.geometry, 10., 'awgn', lambda raw: raw.square().mean(), .005,
                meter, train_codec=True, reuse_codec=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
