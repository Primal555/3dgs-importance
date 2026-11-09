import unittest

import torch

from gaussian_jscc.entropy_thresholds import threshold_map, candidate_maps, select_maps


class EntropyThresholdTests(unittest.TestCase):
    def test_independent_boundaries_preserve_ties_and_row_order(self):
        scores = torch.tensor([130., 90., 110., 110., 150.], dtype=torch.float64)
        self.assertEqual(threshold_map(scores, 100., 120.).tolist(), [3, 1, 2, 2, 3])
        self.assertEqual(threshold_map(scores, 110., 110.).tolist(), [3, 1, 1, 1, 3])
        with self.assertRaises(ValueError): threshold_map(scores, 120., 100.)
        with self.assertRaises(ValueError): threshold_map(torch.tensor([float('nan')]), 0., 1.)

    def test_grid_deduplicates_and_budget_uses_actual_sides(self):
        scores = torch.arange(30, dtype=torch.float64) + 90
        def cost(q):
            # Nonuniform metadata overhead makes payload-only q2 matching insufficient.
            return dict(total_channel_uses=int(torch.tensor([0,8,24,48])[q].sum())
                        + (17 if len(q.unique()) > 1 else 0))
        rows = list(candidate_maps(scores, [0., .25, .5, .75, 1.], cost))
        self.assertEqual(len(rows), len({r['tier_sha256'] for r,q in rows}))
        self.assertTrue(all(not (q == 0).any() for r,q in rows))
        self.assertTrue(any(set(q.tolist()) == {1,2,3} for r,q in rows))
        self.assertTrue(any(r['kind'] == 'q2_budget' for r,q in rows))
        for r,q in rows:
            if r['kind'] == 'q2_budget': self.assertLessEqual(cost(q)['total_channel_uses'], 30*24)

    def test_beta_changes_selection_without_changing_measurement(self):
        rows = [dict(tier_sha256='high', normalized_distortion=1., normalized_cost=1.),
                dict(tier_sha256='mixed', normalized_distortion=1.02, normalized_cost=.9),
                dict(tier_sha256='q2', normalized_distortion=1.25, normalized_cost=.63)]
        selected = select_maps(rows, [.1,.5], .63)
        self.assertEqual(selected['beta_0p1']['tier_sha256'], 'high')
        self.assertEqual(selected['beta_0p5']['tier_sha256'], 'mixed')
        self.assertEqual(selected['q2_budget']['tier_sha256'], 'q2')


if __name__ == '__main__': unittest.main()
