import unittest
from unittest.mock import patch
import torch
import gaussian_jscc.multiscene as training


class PrefixPreservationTests(unittest.TestCase):
    def test_averages_only_codec_gradients_and_restores_rng(self):
        self.assertTrue(hasattr(training, 'preserve_prefix'))
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.full_like(model.weight, 6.)
        mask = torch.nn.Parameter(torch.zeros(1))
        mask.grad = torch.tensor([7.])
        torch.manual_seed(52)
        rng = torch.random.get_rng_state().clone()
        def anchor(*args, **kwargs):
            torch.rand(3)
            model.weight.grad.add_(2.)
            return torch.tensor(3.), {'render_loss': 3.}
        with patch('gaussian_jscc.multiscene.full_scene_step', side_effect=anchor):
            details = training.preserve_prefix(model, [], [], None, 10, 'none', None, 4, 4, 3)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[4.]]))
        torch.testing.assert_close(mask.grad, torch.tensor([7.]))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(details['prefix_anchor_tier'], 1)

    def test_anchor_cycle_does_not_reduce_learned_updates(self):
        self.assertTrue(hasattr(training, 'preserve_prefix'))
        model = torch.nn.Linear(1, 1)
        seen = []
        with patch('gaussian_jscc.multiscene.full_scene_step', return_value=(torch.tensor(1.), {'render_loss': 1.})):
            for visit in range(1, 13):
                details = training.preserve_prefix(model, [], [], None, 10, 'none', None, visit, 4, 3)
                if details['prefix_anchor_applied']:
                    seen.append(details['prefix_anchor_tier'])
        self.assertEqual(seen, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
