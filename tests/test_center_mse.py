import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.utils.rnn import pad_sequence

from gaussian_jscc.center_attribute_codec import center_loss
from gaussian_jscc.center_accumulation import accumulate_center_gradients
from gaussian_jscc.center_attribute_train import train
from gaussian_jscc.center_attribute_validation import validate_centers, plot_run
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import model_id, load_checkpoint
from test_center_attribute import setup, args_for


class CenterMSETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_world_mse_value_gradient_target_detach_and_smoothing_unused(self):
        pred = torch.tensor([[1., 2., 3.], [2., 1., 0.]], requires_grad=True)
        target = torch.zeros_like(pred, requires_grad=True)
        geometry = SimpleNamespace(span=torch.tensor([2., 3., 4.]))
        loss = center_loss(pred, target, geometry, .001, kind='mse')
        expected = (pred.detach()*geometry.span).square().mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(loss, center_loss(pred, target, geometry, 100., kind='mse'))
        loss.backward()
        torch.testing.assert_close(pred.grad, 2*pred.detach()*geometry.span.square()/pred.numel())
        self.assertIsNone(target.grad)

    def test_zero_error_and_bad_kind(self):
        pred = torch.zeros(2, 3, requires_grad=True)
        geometry = SimpleNamespace(span=torch.ones(3))
        loss = center_loss(pred, pred.detach(), geometry, .001, kind='mse')
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))
        with self.assertRaises(ValueError):
            center_loss(pred, pred.detach(), geometry, .001, kind='other')

    def test_mse_accumulation_matches_full_point_mean(self):
        _, g, f, model = setup(24)
        model.learned.set_phase('center'); other = copy.deepcopy(model)
        blocks = [f[:8], f[8:13], f[13:21]]
        loss, _ = accumulate_center_gradients(model, blocks, [[0], [1, 2]], g, .001, 'cpu', 'mse')
        padded = pad_sequence(blocks, batch_first=True)
        active = torch.arange(8)[None] < torch.tensor([8, 5, 8])[:, None]
        pred = other.learned.centers(padded[..., :3], active)
        expected = center_loss(pred[active], padded[..., :3][active], g, .001, kind='mse')
        expected.backward()
        torch.testing.assert_close(loss, expected.detach())
        for p, q in zip(model.parameters(), other.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6)

    def test_validation_mse_equals_squared_world_rmse(self):
        _, g, f, model = setup(24)
        args = SimpleNamespace(center_loss='mse', center_smoothing=.001)
        result = validate_centers(model, [f[:8], f[8:13]], [0, 1], g, args)
        self.assertAlmostEqual(result['center_loss'], result['world_rmse']**2, places=6)
        self.assertAlmostEqual(result['world_mse'], result['world_rmse']**2, places=12)

    def test_training_resume_retains_mse_and_initialization_matches_distance(self):
        raw, _, _, _ = setup(40)
        opts = ['--center-loss', 'mse', '--center-update-policy', 'soft', '--center-accumulation-steps', '2',
                '--center-lr-schedule', 'late_cosine', '--center-readout-norm', 'affine', '--center-attention-scope', 'self']
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp); ply = root/'source.ply'; write_ply(ply, raw, 0)
            train(args_for(ply, root/'full', opts))
            train(args_for(ply, root/'distance', opts+['--center-loss', 'distance']))
            self.assertEqual(model_id(load_checkpoint(root/'full/codec_0.pt', 'cpu')),
                             model_id(load_checkpoint(root/'distance/codec_0.pt', 'cpu')))
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
                self.assertEqual(a['loss'], b['loss'])
                self.assertEqual(a['objective'], 'world_center_mse')
                self.assertEqual(a['stats'], b['stats'])
            plot_run(root/'full')
            self.assertTrue((root/'full/charts/training_by_phase.png').exists())
