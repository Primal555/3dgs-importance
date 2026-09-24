import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from gaussian_jscc.center_soft_update import soft_adam_step, scheduled_center_lr
from gaussian_jscc.center_attribute_train import train, check_args
from gaussian_jscc.center_attribute_validation import plot_run
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import load_checkpoint, model_id
from test_center_attribute import setup, args_for


class SoftUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_warmup_is_exact_adam_even_when_loss_increases(self):
        p = nn.Parameter(torch.tensor(1.)); q = nn.Parameter(p.detach().clone())
        a = torch.optim.Adam([p], lr=.1); b = torch.optim.Adam([q], lr=.1)
        history = {}
        # Finite gradient need not decrease an arbitrary current loss; no approval.
        for _ in range(3):
            p.grad = torch.tensor(-1.); q.grad = torch.tensor(-1.)
            row = soft_adam_step(a, history); b.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            for key in a.state[p]:
                torch.testing.assert_close(a.state[p][key], b.state[q][key], rtol=0, atol=0)
            self.assertEqual(row['groups']['0']['scale'], 1.)
            self.assertFalse(row['momentum_restarted'])
        self.assertGreater(float(p.detach()), 1.)

    def test_scale_keeps_raw_history_and_adam_moments(self):
        p = nn.Parameter(torch.tensor(1.)); q = nn.Parameter(p.detach().clone())
        a = torch.optim.Adam([p], lr=.1); b = torch.optim.Adam([q], lr=.1)
        p.grad = torch.tensor(1.); q.grad = torch.tensor(1.)
        history = {'0': [.01, .01]}
        row = soft_adam_step(a, history, window=2, warmup=2, multiplier=3.)
        b.step()
        r = row['groups']['0']
        self.assertTrue(r['limited'])
        self.assertAlmostEqual(r['scale'], .03, places=5)
        self.assertAlmostEqual(float(p.detach()), .997, places=6)
        self.assertAlmostEqual(history['0'][-1], 1., places=5)
        for key in a.state[p]:
            torch.testing.assert_close(a.state[p][key], b.state[q][key], rtol=0, atol=0)

    def test_groups_are_independent_and_rng_unchanged(self):
        p = nn.Parameter(torch.tensor(1.)); q = nn.Parameter(torch.tensor(1.))
        opt = torch.optim.Adam([{'params': [p], 'name': 'encoder'},
                                {'params': [q], 'name': 'decoder'}], lr=.1)
        p.grad = torch.tensor(1.); q.grad = torch.tensor(1.)
        rng = torch.get_rng_state().clone()
        row = soft_adam_step(opt, {'encoder': [.01], 'decoder': [1.]}, window=1, warmup=1)
        self.assertLess(row['groups']['encoder']['scale'], 1.)
        self.assertEqual(row['groups']['decoder']['scale'], 1.)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_zero_reference_does_not_freeze_new_gradients(self):
        p = nn.Parameter(torch.tensor(1.)); opt = torch.optim.Adam([p], lr=.1)
        p.grad = torch.tensor(1.)
        row = soft_adam_step(opt, {'0': [0.]}, window=1, warmup=1)
        self.assertEqual(row['groups']['0']['scale'], 1.)

    def test_nonfinite_gradient_does_not_step(self):
        p = nn.Parameter(torch.tensor(1.)); opt = torch.optim.Adam([p])
        p.grad = torch.tensor(float('inf'))
        with self.assertRaises(FloatingPointError):
            soft_adam_step(opt, {})
        self.assertEqual(float(p.detach()), 1.)
        self.assertEqual(len(opt.state), 0)

    def test_corrupt_proposal_rolls_back_optimizer_weights_and_history(self):
        p = nn.Parameter(torch.tensor(1.)); opt = torch.optim.Adam([p])
        p.grad = torch.tensor(1.); opt.step()
        before = p.detach().clone(); state = copy.deepcopy(opt.state_dict()); history = {'0': [.4]}
        original = torch.optim.Adam.step
        def corrupt(o):
            original(o)
            with torch.no_grad():
                p.fill_(float('nan'))
        with patch.object(torch.optim.Adam, 'step', corrupt):
            with self.assertRaises(FloatingPointError):
                soft_adam_step(opt, history)
        torch.testing.assert_close(p, before, rtol=0, atol=0)
        for key in opt.state[p]:
            torch.testing.assert_close(opt.state[p][key], state['state'][0][key], rtol=0, atol=0)
        self.assertEqual(history, {'0': [.4]})

    def test_lr_schedule_holds_then_decays_monotonically(self):
        values = [scheduled_center_lr(1e-4, i, 5000) for i in range(1, 5001)]
        self.assertTrue(all(abs(v-1e-4) < 1e-15 for v in values[:2500]))
        self.assertAlmostEqual(values[-1], 2e-5)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_soft_and_legacy_guard_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            check_args(args_for('x', 'y', ['--center-update-policy', 'soft', '--center-step-guard']))

    def test_exact_resume_preserves_history_schedule_and_charts(self):
        raw, _, _, _ = setup(40)
        opts = ['--center-update-policy', 'soft', '--center-update-window', '2',
                '--center-update-warmup', '1', '--center-update-multiplier', '1',
                '--center-lr-schedule', 'late_cosine', '--center-readout-norm', 'affine',
                '--center-attention-scope', 'self', '--center-probe-blocks', '1', '--center-drift-every', '1']
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp); ply = root/'source.ply'; write_ply(ply, raw, 0)
            train(args_for(ply, root/'full', opts))
            original = torch.optim.Adam.step; calls = [0]
            def crash(o, *a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('interrupted')
                return original(o, *a, **kw)
            with patch.object(torch.optim.Adam, 'step', crash):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    train(args_for(ply, root/'resumed', opts))
            args = args_for(ply, root/'resumed'); args.resume = str(root/'resumed/training_state.pt')
            train(args)
            self.assertEqual(model_id(load_checkpoint(root/'full/codec_3.pt', 'cpu')),
                             model_id(load_checkpoint(root/'resumed/codec_3.pt', 'cpu')))
            read = lambda name: [json.loads(s) for s in (root/name/'loss.jsonl').read_text().splitlines()]
            for a, b in zip(read('full'), read('resumed')):
                self.assertEqual(a['stats']['soft_update'], b['stats']['soft_update'])
                self.assertNotIn('step_guard', a['stats'])
            for name in ('full', 'resumed'):
                state = torch.load(root/name/'training_state_last_center.pt', weights_only=True)
                self.assertEqual(len(state['progress']['center_update_history']['center_encoder']), 2)
            plot_run(root/'full')
            self.assertTrue((root/'full/charts/center_soft_updates.png').exists())
