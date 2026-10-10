import itertools
import unittest

import torch

import gaussian_jscc.allocation as allocation


class DeploymentTrainingTests(unittest.TestCase):
    def test_exact_marginals_match_enumerated_draws(self):
        self.assertTrue(hasattr(allocation, 'deployment_probabilities'))
        p = torch.tensor([[.2, .1, .3, .4]], dtype=torch.float64)
        # Three draws exercise positive ties; all-zero detection must use draw count.
        expected = torch.zeros_like(p)
        for draws in itertools.product(range(4), repeat=3):
            counts = [draws.count(q) for q in range(4)]
            if counts[0] == 3:
                q = 0
            else:
                q = max(range(1, 4), key=lambda q: (counts[q], float(p[0, q])))
            expected[0, q] += p[0, list(draws)].prod()
        actual = allocation.deployment_probabilities(p, draws=3)
        torch.testing.assert_close(actual, expected)

    def test_drop_probability_and_boundary_gradients(self):
        self.assertTrue(hasattr(allocation, 'deployment_probabilities'))
        p = torch.tensor([[.8, .1, .05, .05], [0., 1., 0., 0.]], requires_grad=True)
        result = allocation.deployment_probabilities(p)
        torch.testing.assert_close(result[:, 0], p[:, 0] ** 10)
        torch.testing.assert_close(result.sum(-1), torch.ones(2))
        (result * torch.tensor([0., 4., 12., 24.])).sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_equal_positive_ties_are_symmetric(self):
        self.assertTrue(hasattr(allocation, 'deployment_probabilities'))
        result = allocation.deployment_probabilities(torch.tensor([[.4, .2, .2, .2]]))
        torch.testing.assert_close(result[:, 1:], result[:, 1:2].expand(-1, 3))

    def test_marginals_match_actual_deployment_monte_carlo(self):
        p = torch.tensor([[.5, .18, .2, .12]])
        sampled = allocation.deployment_tiers(p.expand(60000, -1), seed=31)
        observed = torch.bincount(sampled, minlength=4).float()/len(sampled)
        torch.testing.assert_close(observed, allocation.deployment_probabilities(p)[0], rtol=0., atol=.006)

    def test_seed_replay_and_soft_keep_gradient(self):
        mask = allocation.GaussianTierMask(8, existence_prior=torch.full((8,), .5))
        self.assertTrue(hasattr(mask, 'sample_deployment'))
        indices = torch.arange(8)[None]
        first = mask.sample_deployment(indices, 10, seed=37)
        second = mask.sample_deployment(indices, 10, seed=37)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        q, presence, choices, positive = first
        torch.testing.assert_close(presence.detach(), (q > 0).float())
        torch.testing.assert_close(choices.detach().argmax(-1) + 1, positive)
        presence.sum().backward()
        self.assertTrue(torch.isfinite(mask.keep_logits.grad).all())
        self.assertGreater(float(mask.keep_logits.grad.norm()), 0.)


if __name__ == '__main__':
    unittest.main()
