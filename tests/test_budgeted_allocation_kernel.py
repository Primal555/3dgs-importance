import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from test_successive_allocation import small_successive
from test_mask_gradient import toy_mask_render, Reference
from gaussian_jscc.data import to_raw
from gaussian_jscc.render_objective import MaskedMultiViewRenderTask


class KernelTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(31)

    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.budgeted_allocation_kernel'))
        from gaussian_jscc import budgeted_allocation_kernel
        return budgeted_allocation_kernel

    def test_single_target_candidates_include_neighbor_response_and_preserve_rng(self):
        entry = self.entry(); _, geometry, f, model = small_successive()
        model.eval().requires_grad_(False)
        q = torch.tensor([[1, 2, 3, 1, 2, 1, 0, 0], [3, 1, 2, 3, 1, 0, 0, 0]])
        y = model.semantic_encoder(f, f[..., :3], q > 0)
        rng = torch.get_rng_state()
        actual = model.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).to(f), 10., 'awgn',
                                           paired_noise=True, _semantic_y=y)[0]
        expected_rng = torch.get_rng_state()
        got = list(entry.mixed_point_candidates(model, f, q, torch.tensor([0, 2]), 10., 'awgn', geometry,
                                                semantic_y=y, actual=actual, actual_rng=rng, timings=None))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertTrue(torch.equal(q[0], torch.tensor([1, 2, 3, 1, 2, 1, 0, 0])))
        self.assertEqual(len(got), 2)  # current tier is reused, not decoded again
        for tiers, delta in got:
            modified = q.clone(); modified[0, 0] = tiers[0]; modified[1, 2] = tiers[1]
            torch.set_rng_state(rng)
            pred = model.forward_tier_batches(f, f[..., :3], F.one_hot(modified, 4).to(f), 10., 'awgn',
                                              paired_noise=True, _semantic_y=y)[0]
            want = to_raw(pred.reshape(-1, pred.shape[-1]), geometry, model).reshape(*q.shape, -1)
            base = to_raw(actual.reshape(-1, actual.shape[-1]), geometry, model).reshape_as(want)
            torch.testing.assert_close(delta, (want-base)*(q>0)[..., None], rtol=0, atol=0)
        self.assertTrue(any(float(delta[0, 1:6].abs().sum()) > 0 for _, delta in got))

    def test_all_closed_points_have_gates_and_frozen_codec_stays_unchanged(self):
        entry = self.entry()
        from gaussian_jscc.allocation_mirror import MirrorTierTable, BlockVisitScheduler
        _, geometry, f, model = small_successive(); model.eval().requires_grad_(False)
        old = {k: v.clone() for k, v in model.state_dict().items()}
        ids = [torch.arange(16).reshape(2, 8)]
        table = MirrorTierTable(16, .5)
        table.keep_logits[:, 0] = 100.; table.keep_logits[:, 1] = -100.
        task = MaskedMultiViewRenderTask([SimpleNamespace(factor=1., source_count=16),
                                         SimpleNamespace(factor=2., source_count=16)], Reference(), 1)
        with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            result = entry.budgeted_mask_step(model, table, [f], ids, geometry, task,
                BlockVisitScheduler(ids), 10., 'awgn', d_ref=.1, price=.1, budget=1000,
                side_proxy=dict(xyz_per_retained=1., fixed_uses=8.), timings=None)
        self.assertEqual(result['sampled_tier_counts'], [16, 0, 0, 0])
        self.assertEqual(result['q0_feedback_rows'], 16)
        self.assertEqual(result['tier_ids'].numel(), 0)
        self.assertGreater(float(result['keep_image_costs'].norm()), 0.)
        self.assertTrue(torch.isfinite(result['keep_costs']).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, old[k], rtol=0, atol=0)

    def test_streaming_window_matches_direct_scene_mean(self):
        entry = self.entry()
        from gaussian_jscc.allocation_mirror import MirrorTierTable, BlockVisitScheduler
        _, geometry, f, model = small_successive(); model.eval().requires_grad_(False)
        ids = [torch.tensor([[5, 2, 7, 0, 8, 1, 4, 3], [6, 9, 10, 11, 15, 14, 13, 12]])]
        table = MirrorTierTable(16, .6)
        cams = [SimpleNamespace(factor=x, source_count=16) for x in (1., 2., 3., 4.)]
        results = []
        with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            for streaming in (True, False):
                task = MaskedMultiViewRenderTask(cams, Reference(), 1)
                if not streaming:
                    base = task
                    class Direct:
                        def __call__(self, raw): return base(raw)
                        stats = {}
                    task = Direct()
                torch.manual_seed(52)
                result = entry.budgeted_mask_step(model, table, [f], ids, geometry, task,
                    BlockVisitScheduler(ids), 10., 'awgn', d_ref=.1, price=0., budget=1000,
                    side_proxy=dict(xyz_per_retained=1., fixed_uses=8.), timings=None)
                results.append(result)
        for key in ('keep_costs', 'tier_costs', 'q'):
            torch.testing.assert_close(results[0][key], results[1][key])
        self.assertAlmostEqual(results[0]['render_loss'], results[1]['render_loss'], places=7)

    def test_audit_rerenders_keep_interventions_and_does_not_replace_actual_view_losses(self):
        entry = self.entry()
        from gaussian_jscc.allocation_mirror import MirrorTierTable, BlockVisitScheduler
        _, geometry, f, model = small_successive(); model.eval().requires_grad_(False)
        ids = [torch.arange(16).reshape(2, 8)]; table = MirrorTierTable(16, .5)
        outputs = []
        with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            for audit in (0, 3):
                torch.manual_seed(76)
                task = MaskedMultiViewRenderTask([SimpleNamespace(factor=1., source_count=16)], Reference(), 1)
                outputs.append(entry.budgeted_mask_step(model, table, [f], ids, geometry, task,
                    BlockVisitScheduler(ids), 10., 'awgn', d_ref=.1, price=0., budget=1000,
                    side_proxy=dict(xyz_per_retained=1., fixed_uses=8.), timings=None, audit_limit=audit))
        self.assertLessEqual(len(outputs[1]['audit']), 3)
        self.assertTrue(any(r['old_tier'] == 0 for r in outputs[1]['audit']))
        self.assertEqual(outputs[0]['view_mse'], outputs[1]['view_mse'])
        torch.testing.assert_close(outputs[0]['keep_costs'], outputs[1]['keep_costs'], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
