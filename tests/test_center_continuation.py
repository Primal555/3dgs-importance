import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gaussian_jscc.center_continuation import prepare_continuation
from gaussian_jscc.center_soft_update import scheduled_center_lr


class ContinuationTests(unittest.TestCase):
    def test_actual_training_appends_updates_and_retains_adam_counter(self):
        from test_center_attribute import setup, args_for
        from gaussian_jscc.data import write_ply
        from gaussian_jscc.center_attribute_train import train
        torch.set_num_threads(1)
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as root, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(root)
            ply = root/'source.ply'
            write_ply(ply, raw, 0)
            opts = ['--center-loss', 'mse', '--center-update-policy', 'soft',
                    '--attribute-steps', '0', '--joint-steps', '0', '--min-center-steps', '99']
            train(args_for(ply, root/'old', opts))
            checkpoint = root/'old/training_state_last_center.pt'
            before = checkpoint.read_bytes()
            args = args_for(ply, root/'new', opts)
            args.resume = str(checkpoint)
            args.extend_center_steps = 2
            args.continuation_lr = 2e-5
            args.continuation_end_lr = 1e-5
            train(args)
            final = torch.load(root/'new/training_state_last_center.pt', weights_only=True)
            self.assertEqual(final['progress']['step'], 5)
            self.assertEqual(int(next(iter(final['optimizer']['state'].values()))['step']), 5)
            rows = [json.loads(x) for x in (root/'new/loss.jsonl').read_text().splitlines()]
            self.assertEqual([x['step'] for x in rows], [1, 2, 3, 4, 5])
            self.assertEqual(rows[-2]['lrs']['center_encoder'], 2e-5)
            self.assertEqual(rows[-1]['lrs']['center_encoder'], 1e-5)
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_preserves_state_and_forks_history(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root)/'old'
            source.mkdir()
            (source/'codec_selected_center_4500.pt').write_bytes(b'best')
            (source/'loss.jsonl').write_text('one\ntwo\nextra\n')
            saved = {'format': 'center_attribute_training_v2',
                     'arguments': {'center_steps': 5000, 'center_loss': 'mse'},
                     'progress': {'phase': 'center', 'phase_index': 0, 'phase_initialized': True,
                                  'phase_step': 5000, 'step': 5000, 'end_reason': 'budget_cap',
                                  'best_steps': {'center': 4500}, 'soft_history': [1, 2]},
                     'optimizer': {'param_groups': [{'lr': 2e-5}],
                                   'state': {0: {'step': torch.tensor(5000.),
                                                 'exp_avg': torch.tensor([.3]),
                                                 'exp_avg_sq': torch.tensor([.4])}}},
                     'rng': {'test': torch.tensor([42])}, 'log_counts': {'loss.jsonl': 2}}
            before = copy.deepcopy(saved)
            out = Path(root)/'new'
            result = prepare_continuation(saved, source/'training_state_last_center.pt',
                                          out, 5000, 2e-5, 1e-5)
            self.assertEqual(result['arguments']['center_steps'], 10000)
            self.assertEqual(result['arguments']['center_lr_offset'], 5000)
            self.assertIsNone(result['progress']['end_reason'])
            self.assertEqual(result['progress']['best_steps'], {'center': 4500})
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(result['optimizer']['state'][0][key],
                                           before['optimizer']['state'][0][key])
            torch.testing.assert_close(result['rng']['test'], saved['rng']['test'])
            self.assertEqual(saved['arguments']['center_steps'], 5000)
            self.assertEqual((source/'loss.jsonl').read_text(), 'one\ntwo\nextra\n')
            self.assertEqual((out/'loss.jsonl').read_text(), 'one\ntwo\n')
            self.assertEqual((out/'codec_selected_center_4500.pt').read_bytes(), b'best')
            a = result['arguments']
            rates = [scheduled_center_lr(a['center_lr'], step-a['center_lr_offset'],
                     a['center_steps']-a['center_lr_offset'], 0., a['center_lr_end_ratio'])
                     for step in range(5001, 10001)]
            self.assertAlmostEqual(rates[0], 2e-5)
            self.assertAlmostEqual(rates[-1], 1e-5)
            self.assertTrue(all(x >= y for x, y in zip(rates, rates[1:])))
            with self.assertRaises(ValueError):
                prepare_continuation(saved, source/'state.pt', out, 5000, 2e-5, 1e-5)
            with self.assertRaises(ValueError):
                prepare_continuation(saved, source/'state.pt', Path(root)/'bad', 5000, 1e-5, 2e-5)


if __name__ == '__main__':
    unittest.main()
