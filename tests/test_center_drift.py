import json
import tempfile
import unittest
from pathlib import Path

import torch

from gaussian_jscc.center_drift import CenterDrift, vector_metrics
from gaussian_jscc.center_attribute_train import train
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import load_checkpoint, model_id
from test_center_attribute import args_for, setup


class CenterDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_translation_and_variable_block_weighting(self):
        ids = torch.tensor([0, 0, 1])
        delta = torch.tensor([[1., 2, 3]]*3)
        row = vector_metrics(delta, ids)
        self.assertEqual(row['mean_xyz'], [1., 2., 3.])
        self.assertEqual(row['common_sse_fraction'], 1.)
        self.assertEqual(row['translation_removed_rmse'], 0.)
        delta = torch.tensor([[1., 0, 0], [1., 0, 0], [-2., 0, 0]])
        row = vector_metrics(delta, ids)
        self.assertEqual(row['common_sse_fraction'], 0.)
        self.assertEqual(row['block_common_sse_fraction'], 1.)
        self.assertEqual(row['block_translation_removed_rmse'], 0.)
        zero = vector_metrics(torch.zeros_like(delta), ids)
        self.assertEqual(zero['common_sse_fraction'], 0.)
        json.dumps(zero, allow_nan=False)

    def test_sse_decomposition(self):
        torch.manual_seed(2)
        d = torch.randn(19, 3, dtype=torch.float64)
        ids = torch.tensor([0]*3+[1]*7+[2]*9)
        row = vector_metrics(d, ids)
        self.assertAlmostEqual(row['rmse']**2,
            row['translation_removed_rmse']**2+row['mean_norm']**2/3, places=12)
        self.assertAlmostEqual(row['block_translation_removed_rmse']**2,
            row['rmse']**2*(1-row['block_common_sse_fraction']), places=12)
        self.assertGreaterEqual(row['block_common_sse_fraction'], row['common_sse_fraction'])

    def test_measurement_is_read_only_and_decoder_decomposition(self):
        _, geometry, features, model = setup(21)
        blocks = list(features.split(8))  # last block is padded, not counted
        with tempfile.TemporaryDirectory() as tmp:
            d = CenterDrift(model, blocks, [0, 2], [1], geometry, tmp, 2)
            model.train()
            state = torch.get_rng_state().clone()
            before_hash = model_id(model)
            before = d.capture(keep_latents=True)
            self.assertTrue(model.training)
            self.assertEqual(model_id(model), before_hash)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(len(before['xyz']), 21)
            with torch.no_grad():
                model.learned.center_decoder.readout[-1].bias.add_(.001)
            d.after_step(1, before, {}, {}, {}, 1.)
            row = json.loads((Path(tmp)/'center_drift_updates.jsonl').read_text())
            for group in row['groups'].values():
                self.assertEqual(group['encoder_under_new_decoder_update']['rmse'], 0.)
                self.assertEqual(group['update'], group['decoder_only_update'])
                self.assertGreater(group['update']['common_sse_fraction'], .999999)

    def test_instrumentation_does_not_change_training_and_retains_adam(self):
        raw, _, _, _ = setup(32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ply = root/'points.ply'
            write_ply(ply, raw, 0)
            base = ['--center-steps', '2', '--min-center-steps', '3',
                    '--center-lr', '.0001', '--center-probe-blocks', '1',
                    '--center-attention-scope', 'self', '--center-readout-norm', 'affine']
            train(args_for(ply, root/'plain', base))
            train(args_for(ply, root/'measured', base+['--center-drift-every', '1']))
            a = load_checkpoint(root/'plain'/'codec_2.pt', 'cpu')
            b = load_checkpoint(root/'measured'/'codec_2.pt', 'cpu')
            self.assertEqual(model_id(a), model_id(b))
            measured = root/'measured'
            records = [json.loads(x) for x in (measured/'center_drift_updates.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in records], [1, 2])
            self.assertTrue((measured/'center_drift'/'drift.png').exists())
            self.assertTrue((measured/'center_drift'/'xyz_000002.pt').exists())
            saved = torch.load(measured/'training_state_last_center.pt', weights_only=True)
            self.assertTrue(saved['optimizer']['state'])
            self.assertEqual(saved['progress']['step'], 2)
            for name, value in b.state_dict().items():
                torch.testing.assert_close(saved['model'][name], value, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
