import importlib.util
import unittest
import torch


class MirrorTests(unittest.TestCase):
    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.allocation_mirror'))
        from gaussian_jscc import allocation_mirror
        return allocation_mirror

    def test_gain_preserves_cost_magnitude_and_unvisited_rows(self):
        entry = self.entry(); a = entry.MirrorTierTable(4); b = entry.MirrorTierTable(4)
        old = a.logits.clone()
        keep = torch.zeros(4, 2)
        a.update(keep, torch.tensor([1]), torch.tensor([[0., .01, .02]]), .1, .1, 4.)
        b.update(keep, torch.tensor([1]), torch.tensor([[0., .1, .2]]), .1, .1, 4.)
        self.assertTrue(torch.equal(a.logits[[0, 2, 3]], old[[0, 2, 3]]))
        self.assertGreater(float(b.logits[1].softmax(-1)[0]), float(a.logits[1].softmax(-1)[0]))
        self.assertTrue(torch.isfinite(a.probabilities(torch.arange(4), 10.)).all())

    def test_extreme_update_is_stable_and_invalid_update_is_atomic(self):
        table = self.entry().MirrorTierTable(2)
        table.update(torch.tensor([[0., 1e25], [1e25, 0.]]), torch.tensor([0]),
                     torch.tensor([[1e25, -1e25, 0.]]), .1, .1, 2.)
        p = table.probabilities(torch.arange(2), 10.)
        torch.testing.assert_close(p.sum(-1), torch.ones(2, dtype=p.dtype))
        before = {k: v.clone() for k, v in table.state_dict().items()}
        with self.assertRaises(ValueError):
            table.update(torch.zeros(2, 2), torch.tensor([0]), torch.full((1, 3), float('nan')), .1, .1, 2.)
        for k, v in table.state_dict().items(): self.assertTrue(torch.equal(v, before[k]))

    def test_block_visits_skip_padding_and_restore_original_row_order(self):
        cls = self.entry().BlockVisitScheduler
        batches = [torch.tensor([[3, 1, -1], [0, 2, 4]])]
        s = cls(batches); q = torch.tensor([1, 0, 3, 2, 0])
        columns, ids = s.select(q)
        self.assertEqual(columns[0].tolist(), [0, 0]); self.assertEqual(ids.tolist(), [3, 0])
        s.record(ids, 1)
        other = cls(batches); other.load_state_dict(s.state_dict())
        col2, ids2 = s.select(q); col3, ids3 = other.select(q)
        self.assertEqual(col2[0].tolist(), [0, 1]); self.assertEqual(ids2.tolist(), [3, 2])
        self.assertTrue(torch.equal(ids2, ids3)); self.assertTrue(torch.equal(col2[0], col3[0]))
        self.assertEqual(s.coverage(q)['unvisited_retained'], 1)
        empty, ids = s.select(torch.zeros(5, dtype=torch.long))
        self.assertEqual(empty[0].tolist(), [-1, -1]); self.assertEqual(ids.numel(), 0)

    def test_view_windows_cover_distinct_views_and_resume(self):
        cls = self.entry().ViewWindowScheduler
        s = cls(['a', 'b', 'c', 'd', 'e'], 73)
        first = s.next(4); self.assertEqual(len(set(first)), 4)
        other = cls(['a', 'b', 'c', 'd', 'e'], 1); other.load_state_dict(s.state_dict())
        self.assertEqual(s.next(4), other.next(4))
        with self.assertRaises(ValueError): s.next(6)


if __name__ == '__main__': unittest.main()
