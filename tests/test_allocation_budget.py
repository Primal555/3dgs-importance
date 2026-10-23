import importlib.util
import math
import unittest


class BudgetTests(unittest.TestCase):
    def controller(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.allocation_budget'))
        from gaussian_jscc.allocation_budget import BudgetController
        return BudgetController(100, initial_price=.1, dual_lr=.2, ema_decay=0., **kwargs)

    def test_price_tracks_actual_budget_residual_and_restores(self):
        c = self.controller()
        self.assertAlmostEqual(c.observe(150)['price'], .2)
        self.assertAlmostEqual(c.observe(50)['price'], .1)
        self.assertEqual(c.observe(0)['price'], 0.)
        other = self.controller(); other.load_state_dict(c.state_dict())
        self.assertEqual(c.observe(175), other.observe(175))

    def test_invalid_observation_cannot_poison_controller(self):
        c = self.controller(); before = c.state_dict()
        for value in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError): c.observe(value)
            self.assertEqual(c.state_dict(), before)


if __name__ == '__main__': unittest.main()
