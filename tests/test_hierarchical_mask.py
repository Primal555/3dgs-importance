import unittest
import torch
from gaussian_jscc.allocation import GaussianTierMask, expected_rate


class HierarchicalMaskTests(unittest.TestCase):
    def test_conditional_change_does_not_change_keep_probability(self):
        mask = GaussianTierMask(7)
        ids = torch.arange(7)
        before = mask.probabilities(ids, 10)[:, 0].detach().clone()
        with torch.no_grad():
            mask.logits[:, 2] += 9
        torch.testing.assert_close(mask.probabilities(ids, 10)[:, 0], before)

    def test_stored_noise_replays_hard_values_and_gradients(self):
        mask = GaussianTierMask(12, torch.full((12,), .5))
        ids = torch.arange(12)
        noise = -torch.empty(12, 5).exponential_().log()
        gradients = []
        outputs = []
        for _ in range(2):
            mask.zero_grad()
            q, gate, choice, positive = mask.sample(ids, 10, noise=noise)
            self.assertTrue(((gate == 0) | (gate == 1)).all())
            self.assertTrue(((choice == 0) | (choice == 1)).all())
            (gate.sum() + (choice * torch.tensor([1., 2., 4.])).sum()).backward()
            outputs.append(q)
            gradients.append(torch.cat((mask.keep_logits.grad, mask.logits.grad), -1).clone())
        torch.testing.assert_close(outputs[0], outputs[1])
        torch.testing.assert_close(gradients[0], gradients[1])

    def test_rate_gradient_updates_both_independent_branches(self):
        mask = GaussianTierMask(7)
        expected_rate(mask.probabilities(torch.arange(7), 10), [0, 8, 16, 32]).mean().backward()
        self.assertTrue((mask.keep_logits.grad[:, 1] > 0).all())
        self.assertTrue((mask.logits.grad[:, 2] > 0).all())

    def test_small_adam_epsilon_does_not_suppress_tiny_gradient(self):
        updates = []
        for eps in (1e-8, 1e-15):
            p = torch.nn.Parameter(torch.tensor([0.]))
            optimizer = torch.optim.Adam([p], lr=.01, eps=eps)
            p.grad = torch.tensor([1e-10])
            optimizer.step()
            updates.append(float(p.abs().detach()))
        self.assertGreater(updates[1], 50 * updates[0])
