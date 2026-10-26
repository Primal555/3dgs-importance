import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from gaussian_jscc.allocation_mirror import MirrorTierTable
from gaussian_jscc.allocation_evidence import EvidenceScheduler
from gaussian_jscc.allocation_finite import PairedMaskedRenderTask
from gaussian_jscc.budgeted_allocation_kernel import budgeted_mask_step
from test_successive_allocation import small_successive
from test_mask_gradient import toy_mask_render, Reference


class EvidenceKernelTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def run_step(self, finite):
        torch.manual_seed(213)
        _, geometry, features, model = small_successive(); model.eval().requires_grad_(False)
        ids = [torch.arange(16).reshape(2, 8)]
        table = MirrorTierTable(16, .6)
        scheduler = EvidenceScheduler(ids, rounds=2, finite_samples=finite, memory=False)
        task = PairedMaskedRenderTask([SimpleNamespace(factor=1., source_count=16)], Reference(), 1)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
            result = budgeted_mask_step(model, table, [features], ids, geometry, task, scheduler,
                10., 'awgn', d_ref=.1, price=.1, rate_normalizer=1000,
                side_proxy=dict(xyz_per_retained=1., fixed_uses=8., step=1), timings=None)
        for k,v in model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        return result, scheduler, torch.get_rng_state()

    def test_two_rounds_are_distinct_and_do_not_repeat_codec_or_q0_base(self):
        result, scheduler, _ = self.run_step(0)
        self.assertEqual(result['tier_ids'].numel(), 4)
        self.assertEqual(result['tier_ids'].unique().numel(), 4)
        self.assertEqual(result['candidate_codec_calls'], 4)
        self.assertEqual(result['actual_codec_calls'], 1)
        self.assertEqual(int(scheduler.counts.sum()), 4)
        self.assertGreater(result['q0_feedback_rows'], 0)

    def test_finite_interventions_replace_training_costs_and_preserve_rng_and_losses(self):
        base, _, rng0 = self.run_step(0)
        corrected, scheduler, rng1 = self.run_step(4)
        self.assertTrue(torch.equal(rng0, rng1))
        self.assertEqual(base['view_mse'], corrected['view_mse'])
        self.assertGreater(corrected['finite_corrections'], 0)
        self.assertTrue(corrected['finite_feedback'])
        self.assertFalse(torch.equal(base['tier_costs'], corrected['tier_costs']))
        for row in corrected['audit']:
            if row.get('feedback_kind') == 'positive':
                j = (corrected['tier_ids'] == row['source_row']).nonzero().item()
                k, current = row['new_tier']-1, row['old_tier']-1
                got = corrected['tier_costs'][j, k]-corrected['tier_costs'][j, current]
                expected = row['measured_delta_mse']/.1 + row['rate_delta_cost']
                self.assertAlmostEqual(float(got), expected, places=10)
        self.assertGreater(int(scheduler.finite_counts.sum()), 0)

    def test_finite_keep_feedback_does_not_cap_all_closed_gate_path(self):
        torch.manual_seed(66)
        _,geometry,f,model=small_successive();model.eval().requires_grad_(False)
        ids=[torch.arange(16).reshape(2,8)]; table=MirrorTierTable(16)
        table.keep_logits[:,0]=100.;table.keep_logits[:,1]=-100.
        scheduler=EvidenceScheduler(ids,finite_samples=8)
        task=PairedMaskedRenderTask([SimpleNamespace(factor=1.,source_count=16)],Reference(),1)
        with patch('gaussian_jscc.rendering.render',side_effect=toy_mask_render):
            result=budgeted_mask_step(model,table,[f],ids,geometry,task,scheduler,10.,'awgn',
                d_ref=.1,price=0.,rate_normalizer=1000,
                side_proxy=dict(xyz_per_retained=1.,fixed_uses=8.,step=1),timings=None)
        self.assertEqual(result['q0_feedback_rows'],16)
        self.assertEqual(result['tier_ids'].numel(),0)
        self.assertGreater(result['finite_keep_corrections'],0)
        self.assertGreater(float(result['keep_costs'].norm()),0.)
        row=result['audit'][0]
        self.assertEqual(row['old_tier'],0)
        self.assertAlmostEqual(float(result['keep_costs'][row['source_row'],1]),row['measured_delta_mse']/.1,places=12)


if __name__ == '__main__': unittest.main()
