import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_center_attribute import setup, args_for
from gaussian_jscc.attribute_objective import attribute_objective
from gaussian_jscc.center_attribute_validation import validate_attributes, validate
from gaussian_jscc.center_attribute_train import train, make_batches
from gaussian_jscc.data import write_ply, fit_feature_statistics, to_features
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.transport import load_checkpoint


class AttributeObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identity_xyz_invariance_target_detach_and_gradients(self):
        _, g, f, model = setup(8)
        dirs = torch.eye(3)
        loss, _, _ = attribute_objective(f, f, g, model, directions=dirs)
        self.assertLess(float(loss), 1e-12)
        pred = (f+torch.randn_like(f)*.1).detach().requires_grad_()
        target = f.detach().requires_grad_()
        loss, stats, terms = attribute_objective(pred, target, g, model, directions=dirs)
        torch.testing.assert_close(loss, sum(terms.values()).to(loss))
        self.assertAlmostEqual(float(loss.detach()), (stats['attribute_logcov_mse']+stats['attribute_response_mse'])/2, places=6)
        changed = pred.detach().clone()
        changed[:, :3] += 1000
        other, _, _ = attribute_objective(changed, target, g, model, directions=dirs)
        torch.testing.assert_close(loss, other, atol=0, rtol=0)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertEqual(float(pred.grad[:, :3].abs().sum()), 0)
        self.assertTrue(torch.isfinite(pred.grad).all())
        for cols in (slice(3, 4), slice(4, 10), slice(10, 13)):
            self.assertGreater(float(pred.grad[:, cols].abs().sum()), 0)

    def test_fixed_validation_directions_do_not_consume_rng(self):
        _, g, f, model = setup(24)
        args = args_for('none', 'none')
        blocks = list(f.split(8))
        before = torch.random.get_rng_state().clone()
        a = validate_attributes(model, blocks, [1, 2], g, args)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.randn(100)
        b = validate_attributes(model, blocks, [1, 2], g, args)
        self.assertEqual(a, b)
        self.assertEqual(a['points'], 16)

    def test_degree_three_all_attribute_heads_receive_gradients(self):
        raw, g, _, base = setup(8)
        raw = torch.cat((raw, torch.randn(8, 45)*.01), -1)
        cfg = CodecConfig.from_dict({**base.cfg.to_dict(), 'sh_degree': 3})
        model = GaussianCodec(cfg)
        fit_feature_statistics(raw, model)
        f, _ = to_features(raw, g, model)
        model.learned.set_phase('attribute')
        pred = model.reconstruct_clean(f)
        pred.retain_grad()
        loss, _, _ = attribute_objective(pred, f, g, model, directions=torch.eye(3))
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        for columns in (slice(3, 4), slice(4, 10), slice(10, 13), slice(13, None)):
            self.assertGreater(float(pred.grad[:, columns].abs().sum()), 0)
        for name, params in model.learned.module_parameters().items():
            if name.startswith('center'):
                self.assertTrue(all(p.grad is None for p in params))

    def test_b_validation_skips_rasterizer_between_image_steps(self):
        raw, g, f, model = setup(24)
        args = args_for('none', 'none')
        state = dict(step=100, phase='attribute', phase_step=10)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_validation.validate_render', side_effect=AssertionError('rasterizer called')):
            result = validate(model, list(f.split(8)), [2], make_batches(list(f.split(8)), 2),
                              raw, g, [object()], None, args, state, tmp)
        self.assertIsNone(result['render'])
        self.assertIsNotNone(result['attributes'])

    def test_actual_b_training_never_uses_render_or_updates_centers(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'), \
                patch('gaussian_jscc.center_attribute_train.render_step', side_effect=AssertionError('replay called')), \
                patch('gaussian_jscc.rendering.render', side_effect=AssertionError('rasterizer called')):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            args = args_for(root/'source.ply', root/'run', ['--attribute-steps', '3', '--center-max-world-rmse', '1000'])
            train(args)
            a = load_checkpoint(root/'run/codec_center.pt', 'cpu')
            b = load_checkpoint(root/'run/codec_6.pt', 'cpu')
            changed = []
            for name, value in a.state_dict().items():
                if 'center_' in name:
                    torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
                elif 'attribute_' in name:
                    changed.append(not torch.equal(value, b.state_dict()[name]))
            self.assertTrue(any(changed))
            rows = [json.loads(line) for line in (root/'run/validation.jsonl').read_text().splitlines()]
            scores = [r['attributes']['loss'] for r in rows if r['phase'] == 'attribute']
            summary = json.loads((root/'run/summary.json').read_text())
            self.assertEqual(summary['best']['attribute'], min(scores))
            logs = [json.loads(line) for line in (root/'run/loss.jsonl').read_text().splitlines()]
            for row in logs:
                if row['phase'] == 'attribute':
                    self.assertEqual(row['module_grad_norms']['center_encoder'], 0)
                    self.assertEqual(row['module_updates']['center_decoder'], 0)
                    self.assertEqual(set(row['objective_module_grad_norms']), {'shape', 'appearance'})
            info = json.loads((root/'run/training.json').read_text())
            self.assertFalse(set(info['heldout_blocks']) & set(info['fitted_blocks']))

    def test_old_resume_rejected_before_reading_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)/'v1.pt'
            torch.save({'format': 'center_attribute_training_v1'}, state)
            args = args_for('missing', 'missing', ['--resume', str(state)])
            with self.assertRaisesRegex(ValueError, 'v1 used render-MSE'):
                train(args)


if __name__ == '__main__':
    unittest.main()
