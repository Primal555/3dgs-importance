import unittest
import torch

from gaussian_jscc.entropy_model import (GroupedEntropyModel, logistic_log_mass,
                                        split_blocks, fit_statistics, tier_map)


class EntropyModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_mass_and_tail_gradient(self):
        x = torch.linspace(-3, 3, 19, dtype=torch.float64)
        mean = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
        logp = logistic_log_mass(x, mean, scale, .25)
        expected = torch.sigmoid((x+.125-mean)/scale)-torch.sigmoid((x-.125-mean)/scale)
        torch.testing.assert_close(logp.exp(), expected)
        lattice = torch.arange(-200, 201, dtype=torch.float64)*.25
        self.assertAlmostEqual(float(logistic_log_mass(lattice, mean, scale, .25).exp().sum().detach()), 1., places=10)
        self.assertTrue(torch.autograd.gradcheck(lambda m, s: logistic_log_mass(x, m, s, .25), (mean, scale)))
        tail = logistic_log_mass(torch.tensor([-1e5, 1e5]), mean, scale, .25)
        grads = torch.autograd.grad(-tail.sum(), (mean, scale))
        self.assertTrue(torch.isfinite(tail).all())
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(abs(float(grads[1])), 1.)

    def test_distribution_parameters_are_causal_and_codec_inputs_detached(self):
        model = GroupedEntropyModel((2, 2, 2), hidden=16)
        x = torch.randn(8, 6, requires_grad=True)
        before = model.parameters_for_group(x, 1)
        changed = x.detach().clone(); changed[:, 2:] += 30
        after = model.parameters_for_group(changed, 1)
        for a, b in zip(before, after): torch.testing.assert_close(a, b, rtol=0, atol=0)
        changed[:, :2] += 2
        self.assertFalse(torch.equal(before[0], model.parameters_for_group(changed, 1)[0]))
        bits = model.bits(x)
        self.assertEqual(bits.shape, (8,))
        self.assertTrue((bits >= 0).all())
        bits.mean().backward()
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_block_split_statistics_and_supported_tiers(self):
        blocks = torch.arange(10).repeat_interleave(4)
        train, valid = split_blocks(blocks, .2, 42)
        self.assertFalse(set(blocks[train].tolist()) & set(blocks[valid].tolist()))
        self.assertEqual(len(train)+len(valid), len(blocks))
        y = torch.randn(40, 6)
        mean, scale = fit_statistics(y, train)
        y[valid] = 1e6
        m2, s2 = fit_statistics(y, train)
        torch.testing.assert_close(mean, m2, rtol=0, atol=0)
        torch.testing.assert_close(scale, s2, rtol=0, atol=0)
        q = tier_map(torch.tensor([0., 8., 8.01, 24., 24.01, 100.]), 1., (0, 8, 24, 48))
        self.assertEqual(q.tolist(), [1, 1, 2, 2, 3, 3])
        self.assertTrue((tier_map(torch.arange(50.), .8, (0, 8, 24, 48)) <=
                         tier_map(torch.arange(50.), 1.2, (0, 8, 24, 48))).all())
        with self.assertRaises(ValueError): tier_map(torch.tensor([float('nan')]), 1., (0, 8, 24, 48))


if __name__ == '__main__': unittest.main()
