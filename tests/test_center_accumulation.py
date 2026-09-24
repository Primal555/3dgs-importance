import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.utils.rnn import pad_sequence

from gaussian_jscc.center_accumulation import accumulate_center_gradients
from gaussian_jscc.center_attribute_codec import center_loss
from gaussian_jscc.center_attribute_train import train, check_args
from gaussian_jscc.center_attribute_validation import plot_run
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import load_checkpoint, model_id
from test_center_attribute import setup, args_for


class CenterAccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_point_weighted_gradients_match_combined_batch_with_partial_blocks(self):
        _, geometry, f, model = setup(24)
        model.learned.set_phase('center')
        other = copy.deepcopy(model)
        blocks = [f[:8], f[8:13], f[13:21]]
        loss, stats = accumulate_center_gradients(model, blocks, [[0], [1, 2]], geometry, .001, 'cpu')
        padded = pad_sequence(blocks, batch_first=True)
        active = torch.arange(8)[None] < torch.tensor([8, 5, 8])[:, None]
        pred = other.learned.centers(padded[..., :3], active)
        expected = center_loss(pred[active], padded[..., :3][active], geometry, .001)
        expected.backward()
        torch.testing.assert_close(loss, expected.detach())
        self.assertFalse(loss.requires_grad)
        self.assertEqual(stats['valid_points'], 21)
        self.assertEqual(stats['points_per_microbatch'], [8, 13])
        for p, q in zip(model.parameters(), other.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6)

    def test_microbatches_do_not_update_model_weights(self):
        _, g, f, model = setup(24)
        before = copy.deepcopy(model.state_dict())
        accumulate_center_gradients(model, list(f.split(8)), [[0, 1], [2, 0]], g, .001, 'cpu')
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_nonfinite_microbatch_clears_partial_gradients(self):
        _, g, f, model = setup(24)
        calls = [0]
        def fail(pred, target, geometry, tau):
            calls[0] += 1
            return center_loss(pred, target, geometry, tau) if calls[0] == 1 else pred.sum()*float('nan')
        with patch('gaussian_jscc.center_accumulation.center_loss', fail):
            with self.assertRaises(FloatingPointError):
                accumulate_center_gradients(model, list(f.split(8)), [[0], [1]], g, .001, 'cpu')
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_reject_invalid_accumulation_and_legacy_guard(self):
        for extra in (['--center-accumulation-steps', '0'],
                      ['--center-accumulation-steps', '2', '--center-step-guard']):
            with self.assertRaises(ValueError):
                check_args(args_for('x', 'y', extra))

    def test_exact_resume_after_mid_accumulation_failure_and_update_budget(self):
        raw, _, _, _ = setup(40)
        opts = ['--center-update-policy', 'soft', '--center-accumulation-steps', '2',
                '--center-lr-schedule', 'late_cosine', '--center-probe-blocks', '1',
                '--center-drift-every', '1', '--center-readout-norm', 'affine', '--center-attention-scope', 'self']
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp); ply = root/'source.ply'; write_ply(ply, raw, 0)
            train(args_for(ply, root/'full', opts))
            calls = [0]
            def crash(*a, **kw):
                calls[0] += 1
                if calls[0] == 4:
                    raise RuntimeError('microbatch interrupted')
                return center_loss(*a, **kw)
            with patch('gaussian_jscc.center_accumulation.center_loss', crash):
                with self.assertRaisesRegex(RuntimeError, 'microbatch interrupted'):
                    train(args_for(ply, root/'resumed', opts))
            args = args_for(ply, root/'resumed'); args.resume = str(root/'resumed/training_state.pt')
            train(args)
            self.assertEqual(model_id(load_checkpoint(root/'full/codec_3.pt', 'cpu')),
                             model_id(load_checkpoint(root/'resumed/codec_3.pt', 'cpu')))
            read = lambda name: [json.loads(s) for s in (root/name/'loss.jsonl').read_text().splitlines()]
            for a, b in zip(read('full'), read('resumed')):
                self.assertEqual(a['loss'], b['loss'])
                self.assertEqual(a['stats'], b['stats'])
            state = torch.load(root/'full/training_state_last_center.pt', weights_only=True)
            work = state['progress']['center_work']
            self.assertEqual(work['optimizer_updates'], 3)
            self.assertEqual(work['microbatches'], 6)
            self.assertEqual(work['sampled_blocks'], 12)
            self.assertEqual(work['processed_points'], 96)
            self.assertTrue(all(float(s['step']) == 3 for s in state['optimizer']['state'].values()))
            self.assertAlmostEqual(read('full')[-1]['lrs']['center_encoder'], 4e-5)
            plot_run(root/'full')
            self.assertTrue((root/'full/charts/center_training_effort.png').exists())
