"""Evidence selection must preserve coverage, action scales and exact resume."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch


class EvidenceTests(unittest.TestCase):
    def scheduler(self, **kwargs):
        from gaussian_jscc.allocation_evidence import EvidenceScheduler
        return EvidenceScheduler([torch.arange(8).reshape(2, 4)], **kwargs)

    def test_distinct_targets_cover_invisible_points_and_resume_exactly(self):
        s = self.scheduler(rounds=2)
        q = torch.ones(8, dtype=torch.long)
        p = torch.full((8, 4), .25, dtype=torch.float64)
        influence = torch.tensor([0., 0., 0., 100., 0., 0., 0., 100.])
        for step in range(1, 5):
            rounds = s.select_rounds(q, influence, p, step)
            self.assertEqual(len(rounds), 2)
            first, second = (r[0].tolist() for r in rounds)
            self.assertTrue(all(a != b for a, b in zip(first, second)))
            ids = s.selected_ids(rounds)
            s.record(ids, step)
        self.assertTrue((s.counts > 0).all(), 'invisible points must retain basic coverage')
        state = s.state_dict(); resumed = self.scheduler(rounds=2); resumed.load_state_dict(state)
        for a, b in zip(s.select_rounds(q, influence, p, 5), resumed.select_rounds(q, influence, p, 5)):
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)

    def test_memory_uses_new_centered_image_evidence_and_keeps_rate_scale(self):
        s = self.scheduler(half_life=10.)
        ids = torch.tensor([0]); rate = torch.tensor([[0., 2., 8.]], dtype=torch.float64)
        first = s.accumulate(ids, torch.tensor([[10., 13., 16.]]), rate, 1)
        torch.testing.assert_close(first, torch.tensor([[-3., 2., 11.]], dtype=torch.float64))
        second = s.accumulate(ids, torch.tensor([[20., 20., 20.]]), rate, 11)
        torch.testing.assert_close(second, torch.tensor([[-1., 2., 9.]], dtype=torch.float64))
        # One old sample decays to half a sample, not another independent update.
        self.assertAlmostEqual(float(s.weight[0]), 1.5)
        self.assertEqual(int(s.observations[0]), 2)
        resumed = self.scheduler(half_life=10.); resumed.load_state_dict(s.state_dict())
        torch.testing.assert_close(resumed.mean, s.mean, rtol=0, atol=0)

    def test_distribution_drift_invalidates_evidence_not_random_mask_changes(self):
        s = self.scheduler(context_tv=.1)
        p = torch.full((8, 4), .25)
        s.select_rounds(torch.ones(8, dtype=torch.long), torch.ones(8), p, 1)
        s.accumulate(torch.tensor([0]), torch.tensor([[0., 1., 2.]]), torch.zeros(1, 3), 1)
        s.select_rounds(torch.full((8,), 3), torch.ones(8), p, 2)
        self.assertGreater(float(s.weight[0]), 0.)
        p[:4] = torch.tensor([.01, .97, .01, .01])
        s.select_rounds(torch.ones(8, dtype=torch.long), torch.ones(8), p, 3)
        self.assertEqual(float(s.weight[0]), 0.)

    def test_all_closed_or_partial_blocks_never_produce_fake_targets(self):
        s = self.scheduler(rounds=2)
        rounds = s.select_rounds(torch.zeros(8, dtype=torch.long), torch.zeros(8), torch.full((8, 4), .25), 1)
        self.assertTrue(all((r[0] == -1).all() for r in rounds))
        self.assertEqual(s.selected_ids(rounds).numel(), 0)
        from gaussian_jscc.allocation_evidence import EvidenceScheduler
        partial = EvidenceScheduler([torch.tensor([[0,-1,-1],[1,2,-1]])],rounds=2)
        rounds = partial.select_rounds(torch.ones(3,dtype=torch.long),torch.zeros(3),torch.full((3,4),.25),1)
        self.assertEqual(rounds[1][0][0].item(),-1)
        self.assertEqual(partial.selected_ids(rounds).unique().numel(),3)

    def test_duplicate_evidence_cannot_be_replayed_as_fresh_measurements(self):
        s=self.scheduler(); ids=torch.tensor([0]); costs=torch.zeros(1,3)
        s.accumulate(ids,costs,costs,1)
        with self.assertRaisesRegex(ValueError,'replayed'): s.accumulate(ids,costs,costs,1)

    def test_diagnostic_reports_evidence_aged_to_current_step(self):
        s=self.scheduler(half_life=10.)
        s.accumulate(torch.tensor([0]),torch.zeros(1,3),torch.zeros(1,3),1)
        q=torch.zeros(8,dtype=torch.long);q[0]=1
        self.assertAlmostEqual(s.diagnostics(q,step=11)['retained_effective_evidence_mean'],.5)

    def test_context_invalidation_clears_variance_before_demand_selection(self):
        s=self.scheduler(rounds=2);s.context_ready=True;s.context.fill_(.25)
        s.second[1]=100.
        p=torch.full((8,4),.25);p[:4]=torch.tensor([.01,.97,.01,.01])
        rounds=s.select_rounds(torch.ones(8,dtype=torch.long),
            torch.tensor([0.,.6,.9,0.,0.,0.,0.,0.]),p,1)
        self.assertEqual(int(rounds[1][0][0]),2)

    def test_correction_selection_checks_large_changes_and_near_zero_rd_actions(self):
        from gaussian_jscc.allocation_finite import correction_indices
        ids=torch.arange(3); q=torch.tensor([2,2,2])
        image=torch.tensor([[100.,0.,-2.],[20.,0.,1.],[10.,0.,-1.]])
        rate=torch.tensor([[0.,0.,0.],[-19.99,0.,0.],[0.,0.,0.]])
        chosen=correction_indices(ids,image,rate,q,torch.zeros(3,dtype=torch.long),3)
        self.assertEqual(chosen,[(0,1),(1,1),(2,3)])

    def test_finite_mse_uses_images_not_subtraction_of_float32_scene_losses(self):
        from gaussian_jscc.allocation_finite import PairedMaskedRenderTask
        class Reference:
            def get(self, camera, device): return torch.zeros(3, 1, 1, device=device)
        def render(raw, camera, *args): return raw[0, 0].expand(3, 1, 1)
        task = PairedMaskedRenderTask([SimpleNamespace()], Reference(), 0)
        base = torch.tensor([[1., 1.]], requires_grad=True)
        alt = torch.tensor([[1.0000001192092896, 1.]])
        with patch('gaussian_jscc.rendering.render', side_effect=render):
            task(base).backward()
            got = task.delta_mse(alt)
        self.assertAlmostEqual(got, 2.384185933124172e-7, places=17)
        self.assertEqual(task.stats['views_per_step'], 1)


if __name__ == '__main__': unittest.main()
