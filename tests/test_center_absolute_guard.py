import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from gaussian_jscc.center_step_guard import directional_guarded_step, update_rejection_streak
from gaussian_jscc.center_attribute_train import train
from gaussian_jscc.center_attribute_validation import plot_run
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import model_id, load_checkpoint
from test_center_attribute import setup, args_for


class CenterAbsoluteGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def stale(self):
        p = nn.Parameter(torch.tensor(1.))
        opt = torch.optim.Adam([p], lr=.1)
        p.grad = torch.tensor(2.)
        opt.state[p] = {'step': torch.tensor(100.), 'exp_avg': torch.tensor(-10.), 'exp_avg_sq': torch.tensor(1.)}
        return p, opt

    def test_uphill_momentum_restart_preserves_variance_and_single_counter(self):
        p, opt = self.stale()
        row = directional_guarded_step(opt, lambda: p.square(), 1.)
        self.assertTrue(row['accepted'])
        self.assertTrue(row['momentum_restarted'])
        self.assertGreater(row['attempts'][0]['directional_derivative'], 0.)
        self.assertEqual(row['attempts'][0]['trials'], [])
        self.assertLess(row['attempts'][1]['directional_derivative'], 0.)
        self.assertLess(float(p.detach().square()), 1.)
        self.assertEqual(float(opt.state[p]['step']), 101.)
        self.assertAlmostEqual(float(opt.state[p]['exp_avg']), .2, places=6)
        self.assertAlmostEqual(float(opt.state[p]['exp_avg_sq']), 1.003, places=6)

    def test_full_rejection_clears_only_momentum_and_equality_is_not_progress(self):
        p, opt = self.stale()
        row = directional_guarded_step(opt, lambda: torch.tensor(1.), 1., 2)
        self.assertFalse(row['accepted'])
        self.assertTrue(row['momentum_cleared_on_reject'])
        self.assertEqual(float(p.detach()), 1.)
        self.assertEqual(float(opt.state[p]['step']), 100.)
        self.assertEqual(float(opt.state[p]['exp_avg_sq']), 1.)
        self.assertEqual(float(opt.state[p]['exp_avg']), 0.)

    def test_overshoot_backtracks_joint_parameters_without_extra_adam_steps(self):
        ps = [nn.Parameter(torch.tensor(.1)), nn.Parameter(torch.tensor(.1))]
        opt = torch.optim.Adam(ps, lr=1.)
        closure = lambda: sum(p.square() for p in ps)
        loss = closure(); loss.backward()
        row = directional_guarded_step(opt, closure, loss.detach())
        self.assertTrue(row['accepted'])
        self.assertEqual(row['scale'], .125)
        self.assertLess(row['loss_after'], row['loss_before'])
        self.assertEqual([float(opt.state[p]['step']) for p in ps], [1., 1.])
        self.assertEqual(opt.param_groups[0]['lr'], 1.)

    def test_rng_preserved_and_inactive_parameter_untouched(self):
        p = nn.Parameter(torch.tensor(.5)); inactive = nn.Parameter(torch.tensor(1.))
        opt = torch.optim.Adam([p, inactive], lr=.01)
        p.square().backward()
        rng = torch.get_rng_state().clone()
        def closure():
            torch.rand(4)
            return p.square()
        directional_guarded_step(opt, closure, .25)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(float(inactive.detach()), 1.)
        self.assertNotIn(inactive, opt.state)

    def test_streak_persistence(self):
        state = {}
        self.assertEqual(update_rejection_streak(state, {'accepted': False}, 2, 3), (False, False))
        state = json.loads(json.dumps(state))
        self.assertEqual(update_rejection_streak(state, {'accepted': False}, 2, 3), (True, False))
        self.assertEqual(update_rejection_streak(state, {'accepted': False}, 2, 3), (False, True))
        self.assertEqual(update_rejection_streak(state, {'accepted': True}, 2, 3), (False, False))

    def options(self):
        return ['--center-step-guard', '--center-attention-scope', 'self',
                '--center-readout-norm', 'affine', '--center-probe-blocks', '1', '--center-drift-every', '1']

    def test_training_exact_resume_charts_and_identical_random_architecture(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp); ply = root/'source.ply'; write_ply(ply, raw, 0)
            train(args_for(ply, root/'full', self.options()))
            train(args_for(ply, root/'plain', [x for x in self.options() if x != '--center-step-guard']))
            self.assertEqual(model_id(load_checkpoint(root/'full/codec_0.pt', 'cpu')),
                             model_id(load_checkpoint(root/'plain/codec_0.pt', 'cpu')))
            original = torch.optim.Adam.step
            calls = [0]
            def crash(o, *a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('interrupted')
                return original(o, *a, **kw)
            with patch.object(torch.optim.Adam, 'step', crash):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    train(args_for(ply, root/'resume', self.options()))
            args = args_for(ply, root/'resume'); args.resume = str(root/'resume/training_state.pt')
            train(args)
            self.assertEqual(model_id(load_checkpoint(root/'full/codec_3.pt', 'cpu')),
                             model_id(load_checkpoint(root/'resume/codec_3.pt', 'cpu')))
            read = lambda run: [json.loads(x) for x in (root/run/'loss.jsonl').read_text().splitlines()]
            for a, b in zip(read('full'), read('resume')):
                self.assertEqual(a['stats']['step_guard'], b['stats']['step_guard'])
                g = a['stats']['step_guard']
                self.assertLessEqual(g['loss_after'], g['loss_before'])
                self.assertNotIn('branches', g)
            plot_run(root/'full')
            self.assertTrue((root/'full/charts/center_step_guard.png').exists())

    def test_persistent_rejection_stops_before_later_phases(self):
        raw, _, _, _ = setup(32)
        def reject(opt, closure, baseline, *args):
            return {'accepted': False, 'scale': 0., 'loss_before': float(baseline),
                    'loss_after': float(baseline), 'momentum_restarted': True}
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'), \
                patch('gaussian_jscc.center_step_guard.directional_guarded_step', reject):
            root = Path(tmp); ply = root/'source.ply'; write_ply(ply, raw, 0)
            opts = self.options()+['--center-steps', '5', '--center-guard-warn-after', '1',
                                  '--center-guard-stop-after', '2', '--attribute-steps', '2',
                                  '--center-max-world-rmse', '100']
            train(args_for(ply, root/'run', opts))
            summary = json.loads((root/'run/summary.json').read_text())
            self.assertEqual(summary['status'], 'guard_stalled')
            self.assertEqual(summary['completed'], {'center': 2, 'attribute': 0, 'joint': 0})
            self.assertEqual(summary['center_step_guard']['skipped_updates'], 2)
            state = torch.load(root/'run/training_state_last_center.pt', weights_only=True)
            self.assertEqual(state['progress']['guard_rejection_streak'], 2)

    def test_decoupled_resume_is_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'old.pt'
            torch.save({'config': {'center_position_layout': 'centroid_residual'}}, path)
            args = args_for('unused.ply', Path(tmp)/'run'); args.resume = str(path)
            with self.assertRaisesRegex(ValueError, 'block-centroid'):
                train(args)

    def test_exception_restores_even_original_bad_momentum(self):
        p, opt = self.stale()
        def failure():
            raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            directional_guarded_step(opt, failure, 1.)
        self.assertEqual(float(p.detach()), 1.)
        self.assertEqual(float(opt.state[p]['exp_avg']), -10.)
        self.assertEqual(float(opt.state[p]['exp_avg_sq']), 1.)
        self.assertEqual(float(opt.state[p]['step']), 100.)
