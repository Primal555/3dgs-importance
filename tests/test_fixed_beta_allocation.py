"""Fixed RD must never use a budget gate or turn beta=0 into adaptive control."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import test_budgeted_allocation as fixture


class ObjectiveTests(unittest.TestCase):
    def objective(self, beta=.2, d_ref=.02, c_ref=1000):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.allocation_objective'),
                             'fixed RD objective is missing')
        from gaussian_jscc.allocation_objective import FixedRateDistortion
        return FixedRateDistortion(beta, d_ref, c_ref)

    def test_hand_computed_normalization_and_zero_beta(self):
        obj = self.objective()
        self.assertAlmostEqual(obj.score(.01, 2000), .9)
        self.assertAlmostEqual(self.objective(0).score(.04, 999999), 2.)

    def test_selection_uses_actual_rd_not_smallest_mse_or_proxy(self):
        obj = self.objective(1., 1., 100)
        sharp = dict(source_mse=.1, total_channel_uses=100, expected_cost_proxy=0.)
        cheap = dict(source_mse=.3, total_channel_uses=10, expected_cost_proxy=99999.)
        self.assertLess(obj.candidate_key(cheap), obj.candidate_key(sharp))
        self.assertAlmostEqual(obj.candidate_key(cheap)[0], .4)
        self.assertLess(obj.candidate_key(dict(source_mse=.2, total_channel_uses=20)),
                        obj.candidate_key(cheap))  # equal RD: lower distortion wins

    def test_negative_nonfinite_coefficients_and_invalid_references_are_rejected(self):
        for beta in (-1., float('nan'), float('inf')):
            with self.subTest(beta=beta), self.assertRaises(ValueError): self.objective(beta)
        for d, c in ((0, 10), (1, 0), (float('nan'), 10), (1, float('inf'))):
            with self.subTest(d=d, c=c), self.assertRaises(ValueError): self.objective(.2, d, c)


class FixedBetaEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.BudgetedEntryTests.setUpClass()
        cls.source = fixture.BudgetedEntryTests.source
        cls.root = fixture.BudgetedEntryTests.root

    @classmethod
    def tearDownClass(cls): fixture.BudgetedEntryTests.tearDownClass()

    def entry(self):
        from gaussian_jscc import budgeted_allocation
        return budgeted_allocation

    def options(self, out, beta=.1):
        try:
            return self.entry().parser().parse_args([
                '--run', str(self.source), '--scene', 'truck', '--out', str(out), '--beta', str(beta),
                '--allocation-steps', '2', '--calibration-views', '1', '--window-views', '4',
                '--control-every', '1', '--validate-every', '1', '--save-every', '1', '--audit-every', '0',
                '--profile-every', '0', '--test-views', '2', '--trials', '1', '--image-views', '1',
                '--cpu-threads', '1', '--device', 'cpu', '--no-lpips'])
        except SystemExit:
            self.fail('explicit --beta must work without a communication budget')

    def test_zero_beta_dry_run_has_distinct_mode_and_never_writes(self):
        entry = self.entry(); out = self.root/'rd_dry'
        args = self.options(out, 0.); args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()): plan = entry.main(args)
        self.assertEqual(plan['version'], 'fixed_beta_allocation_v1')
        self.assertEqual(plan['settings']['beta'], 0.)
        self.assertFalse(out.exists())
        for budget in ('--budget-total-uses', '--budget-ratio-to-q3'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                entry.parser().parse_args(['--run', str(self.source), '--scene', 'truck', '--out', str(out),
                                           '--beta', '.1', budget, '1'])
        for beta in (-1., float('inf'), float('nan')):
            args.beta = beta
            with self.assertRaisesRegex(ValueError, 'beta'): entry.validate(args)

    def test_frozen_fixed_beta_resume_preserves_references_table_and_coverage(self):
        entry = self.entry()
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.root/'rd_full'; args = self.options(out)
            entry.main(args, max_updates=2)
            full = torch.load(out/'checkpoints/latest/state.pt', weights_only=True)
            dest = self.root/'rd_resume'; resumed_args = self.options(dest)
            entry.main(resumed_args, max_updates=1)
            before = torch.load(dest/'checkpoints/latest/state.pt', weights_only=True)
            resumed_args.resume = True; entry.main(resumed_args, max_updates=1)
            after = torch.load(dest/'checkpoints/latest/state.pt', weights_only=True)
            source = torch.load(self.source/'checkpoints/final/codec.pt', weights_only=True)
            self.assertEqual(after['family'], 'fixed_beta_allocation_v1')
            self.assertIsNone(after['controller'])
            reference = after['objective_reference']
            for key in ('beta', 'd_ref', 'd_ref_raw', 'd_ref_floor', 'c_ref', 'q3_cost'):
                self.assertEqual(reference[key], before['objective_reference'][key])
            self.assertEqual(reference, full['objective_reference'])
            self.assertEqual(reference['c_ref'], reference['q3_cost']['total_channel_uses'])
            self.assertNotIn('budget', reference)
            for k, v in source['state_dict'].items():
                torch.testing.assert_close(after['state_dict'][k], v, rtol=0, atol=0)
            for k, v in full['table'].items():
                torch.testing.assert_close(after['table'][k], v, rtol=0, atol=0)
            torch.testing.assert_close(after['blocks']['counts'], full['blocks']['counts'], rtol=0, atol=0)
            torch.testing.assert_close(after['latest_q'], full['latest_q'], rtol=0, atol=0)
            rows = [json.loads(s) for s in (dest/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows], [1, 2])
            for row in rows:
                self.assertEqual(row['beta'], .1); self.assertNotIn('budget', row)
                self.assertAlmostEqual(row['loss'], row['render_loss']/reference['d_ref']+
                                       .1*row['expected_cost_proxy']/reference['c_ref'])
            costs = [json.loads(s) for s in (dest/'metrics/control.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['beta']==.1 and 'residual' not in r for r in costs))
            self.assertTrue((dest/'checkpoints/best_rd/state.pt').exists())
            self.assertFalse((dest/'checkpoints/best_feasible').exists())
            self.assertFalse((dest/'checkpoints/lowest_violation').exists())

    def test_changed_beta_or_mode_cannot_resume_existing_output(self):
        entry = self.entry(); out = self.root/'rd_changed'
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
             contextlib.redirect_stdout(io.StringIO()):
            args = self.options(out); entry.main(args, max_updates=1)
            saved = (out/'checkpoints/latest/state.pt').read_bytes()
            args.resume=True; args.beta=.2
            with self.assertRaisesRegex(ValueError, 'mismatch'): entry.main(args, max_updates=1)
            args.beta=None; args.budget_total_uses=100000
            with self.assertRaisesRegex(ValueError, 'mismatch'): entry.main(args, max_updates=1)
            self.assertEqual((out/'checkpoints/latest/state.pt').read_bytes(), saved)


if __name__ == '__main__': unittest.main()
