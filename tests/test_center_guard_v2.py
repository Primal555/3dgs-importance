import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from gaussian_jscc.center_step_guard import (directional_guarded_step, branch_guarded_step,
                                              update_rejection_streaks)
from gaussian_jscc.center_attribute_train import train
from gaussian_jscc.center_attribute_validation import plot_run
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import model_id, load_checkpoint
from test_center_attribute import setup, args_for


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(.5))


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.center_encoder = nn.ModuleDict({n: Scalar() for n in ('common', 'local')})
        self.center_decoder = nn.ModuleDict({n: Scalar() for n in ('common', 'local')})

    def module_parameters(self):
        return {f'{side}.{branch}': list(module.parameters())
                for side, child in self.named_children() for branch, module in child.items()}


def toy():
    model = nn.Module()
    model.learned = ToyCore()
    opt = torch.optim.Adam([{'params': list(model.parameters()), 'name': 'centers'}], lr=.1)
    closure = lambda: sum(p for p in model.parameters()).square()
    return model, opt, closure


class CenterGuardV2Tests(unittest.TestCase):
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

    def test_local_gradient_recomputed_after_common_and_counters_independent(self):
        model, opt, closure = toy()
        loss = closure(); loss.backward()
        grads, baselines = [], []
        original = directional_guarded_step
        def spy(o, c, baseline, *args):
            active = [p for p in model.parameters() if p.grad is not None]
            grads.append(float(active[0].grad))
            baselines.append(float(baseline))
            return original(o, c, baseline, *args)
        with patch('gaussian_jscc.center_step_guard.directional_guarded_step', spy):
            row = branch_guarded_step(model, opt, closure, loss.detach())
        self.assertAlmostEqual(grads[0], 4., places=5)
        self.assertAlmostEqual(grads[1], 3.6, places=5)
        self.assertLess(baselines[1], baselines[0])
        self.assertLess(row['loss_after'], row['loss_before'])
        self.assertTrue(all(float(st['step']) == 1 for st in opt.state.values()))

    def test_common_rejection_does_not_block_local(self):
        model, opt, closure = toy()
        loss = closure(); loss.backward()
        old = {n: p.detach().clone() for n, p in model.named_parameters()}
        calls = [0]
        original = directional_guarded_step
        def reject_common(o, c, baseline, *args):
            calls[0] += 1
            if calls[0] == 1:
                return {'accepted': False, 'scale': 0., 'loss_before': float(baseline),
                        'loss_after': float(baseline)}
            return original(o, c, baseline, *args)
        with patch('gaussian_jscc.center_step_guard.directional_guarded_step', reject_common):
            row = branch_guarded_step(model, opt, closure, loss.detach())
        self.assertFalse(row['branches']['common']['accepted'])
        self.assertTrue(row['branches']['local']['accepted'])
        for name, p in model.named_parameters():
            if '.common.' in name:
                torch.testing.assert_close(p, old[name], rtol=0, atol=0)
                self.assertNotIn(p, opt.state)
            else:
                self.assertFalse(torch.equal(p, old[name]))

    def test_exception_in_local_restores_accepted_common(self):
        model, opt, closure = toy()
        loss = closure(); loss.backward()
        initial = copy.deepcopy(model.state_dict())
        original = torch.optim.Adam.step
        calls = [0]
        def crash(o, *a, **kw):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError('local interrupted')
            return original(o, *a, **kw)
        with patch.object(torch.optim.Adam, 'step', crash):
            with self.assertRaisesRegex(RuntimeError, 'local interrupted'):
                branch_guarded_step(model, opt, closure, loss.detach())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        self.assertEqual(len(opt.state), 0)

    def test_rejection_streaks_survive_serialization_and_reset_per_branch(self):
        state = {}
        reject = {'branches': {b: {'accepted': False} for b in ('common', 'local')}}
        self.assertEqual(update_rejection_streaks(state, reject, 2, 3), ([], []))
        state = json.loads(json.dumps(state))
        self.assertEqual(update_rejection_streaks(state, reject, 2, 3), (['common', 'local'], []))
        reject['branches']['local']['accepted'] = True
        self.assertEqual(update_rejection_streaks(state, reject, 2, 3), ([], ['common']))
        self.assertEqual(state['guard_rejection_streaks']['local'], 0)

    def options(self):
        return ['--center-position-layout', 'centroid_residual', '--center-step-guard',
                '--center-guard-mode', 'branch', '--center-max-backtracks', '8',
                '--center-attention-scope', 'self', '--center-readout-norm', 'affine',
                '--center-probe-blocks', '1', '--center-drift-every', '1']

    def test_training_resume_and_charts(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp); ply = root/'source.ply'; write_ply(ply, raw, 0)
            train(args_for(ply, root/'full', self.options()))
            original = torch.optim.Adam.step
            calls = [0]
            def crash(o, *a, **kw):
                calls[0] += 1
                if calls[0] == 3:
                    raise RuntimeError('interrupted')
                return original(o, *a, **kw)
            with patch.object(torch.optim.Adam, 'step', crash):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    train(args_for(ply, root/'resume', self.options()))
            args = args_for(ply, root/'resume'); args.resume = str(root/'resume/training_state.pt')
            train(args)
            self.assertEqual(model_id(load_checkpoint(root/'full/codec_3.pt', 'cpu')),
                             model_id(load_checkpoint(root/'resume/codec_3.pt', 'cpu')))
            full = [json.loads(x) for x in (root/'full/loss.jsonl').read_text().splitlines()]
            resumed = [json.loads(x) for x in (root/'resume/loss.jsonl').read_text().splitlines()]
            for a, b in zip(full, resumed):
                self.assertEqual(a['stats']['step_guard'], b['stats']['step_guard'])
                self.assertLessEqual(a['stats']['step_guard']['loss_after'], a['loss'])
            plot_run(root/'full')
            self.assertTrue((root/'full/charts/center_momentum_restarts.png').exists())

    def test_persistent_rejection_stops_and_exports_state_not_convergence(self):
        raw, _, _, _ = setup(32)
        def reject(model, opt, closure, baseline, *args):
            row = {'accepted': False, 'scale': 0., 'loss_before': float(baseline),
                   'loss_after': float(baseline), 'momentum_restarted': True}
            return {**row, 'mode': 'branch_directional_v2', 'branches': {b: dict(row) for b in ('common', 'local')}}
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'), \
                patch('gaussian_jscc.center_step_guard.branch_guarded_step', reject):
            root = Path(tmp); ply = root/'source.ply'; write_ply(ply, raw, 0)
            opts = self.options()+['--center-steps', '5', '--center-guard-warn-after', '1', '--center-guard-stop-after', '2']
            train(args_for(ply, root/'run', opts))
            summary = json.loads((root/'run/summary.json').read_text())
            self.assertEqual(summary['status'], 'guard_stalled')
            self.assertEqual(summary['completed'], {'center': 2, 'attribute': 0, 'joint': 0})
            self.assertEqual(summary['center_step_guard']['branches']['common']['skipped_updates'], 2)
            state = torch.load(root/'run/training_state_last_center.pt', weights_only=True)
            self.assertEqual(state['progress']['status'], 'guard_stalled')
            self.assertEqual(state['progress']['guard_rejection_streaks']['common'], 2)


if __name__ == '__main__':
    unittest.main()
