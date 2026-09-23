import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.center_attribute_codec import masked_point_mean, center_loss
from gaussian_jscc.center_attribute_train import train, make_batches, render_step
from gaussian_jscc.center_attribute_validation import plot_run
from gaussian_jscc.center_step_guard import guarded_step
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import save_checkpoint, load_checkpoint, model_id
from test_center_attribute import setup, args_for


class CenterDecoupledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        raw, geometry, features, old = setup(24)
        cfg = CodecConfig.from_dict({**old.cfg.to_dict(), 'center_position_layout': 'centroid_residual',
                                     'center_attention_scope': 'self', 'center_readout_norm': 'affine'})
        model = GaussianCodec(cfg)
        model.attr_mean.copy_(old.attr_mean)
        model.attr_std.copy_(old.attr_std)
        return raw, geometry, features, model

    def test_config_compatibility_and_roundtrip(self):
        self.assertNotIn('center_position_layout', CodecConfig().to_dict())
        with self.assertRaises(ValueError):
            CodecConfig(center_position_layout='centroid_residual')
        _, _, f, model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp)/'codec.pt', model, 0)
            loaded = load_checkpoint(Path(tmp)/'codec.pt', 'cpu')
            self.assertEqual(model_id(model), model_id(loaded))
            torch.testing.assert_close(model.reconstruct_clean(f), loaded.reconstruct_clean(f))

    def test_zero_mean_padding_and_parameter_path_isolation(self):
        _, _, f, model = self.model()
        x = f[:, :3].reshape(3, 8, 3)
        mask = torch.tensor([[True]*8, [True]*5+[False]*3, [False]*8])
        x[~mask] = float('nan')
        enc, dec = model.learned.center_encoder, model.learned.center_decoder
        z = enc(x*2-1, x, mask)
        c, r = dec.components(z, mask)
        pred = dec(z, mask)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(float(pred[~mask].detach().abs().sum()), 0.)
        torch.testing.assert_close(masked_point_mean(r, mask), torch.zeros_like(c), atol=1e-7, rtol=0)
        torch.testing.assert_close(masked_point_mean(pred, mask), c, atol=1e-7, rtol=0)
        parameters_common = list(enc.common.parameters())+list(dec.common.parameters())
        parameters_local = list(enc.local.parameters())+list(dec.local.parameters())
        self.assertFalse({id(p) for p in parameters_common} & {id(p) for p in parameters_local})
        with torch.no_grad():
            dec.local.readout[-1].weight.add_(torch.randn_like(dec.local.readout[-1].weight)*.03)
            enc.local.self_latent[-1].weight.add_(.01)
        z2 = enc(x*2-1, x, mask)
        c2, r2 = dec.components(z2, mask)
        torch.testing.assert_close(c2, c, atol=0, rtol=0)
        self.assertFalse(torch.allclose(r2, r))
        # Common-only edits cannot change decoded local structure either.
        with torch.no_grad():
            dec.common[-1].bias.add_(.1)
            enc.common.self_latent[-1].bias.add_(.1)
        z3 = enc(x*2-1, x, mask)
        c3, r3 = dec.components(z3, mask)
        torch.testing.assert_close(r3, r2, atol=0, rtol=0)
        self.assertFalse(torch.allclose(c3, c2))

    def test_local_encoder_translation_invariance_and_both_branches_learn(self):
        _, geometry, f, model = self.model()
        x = f[:, :3].reshape(3, 8, 3)
        mask = torch.ones(3, 8, dtype=torch.bool)
        enc = model.learned.center_encoder
        shifted = x+torch.tensor([.25, -.125, .125])
        z = enc(x*2-1, x, mask)
        moved = enc(shifted*2-1, shifted, mask)
        torch.testing.assert_close(z[..., enc.common_dim:], moved[..., enc.common_dim:], atol=2e-6, rtol=2e-5)
        model.learned.set_phase('center')
        pred = model.learned.centers(x, mask)
        center_loss(pred[mask], x[mask], geometry, .001).backward()
        for name, params in model.learned.module_parameters().items():
            nonzero = any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in params)
            self.assertEqual(nonzero, name.startswith('center'), name)

    def test_joint_direct_replay_parity(self):
        _, geometry, f, source = self.model()
        results = []
        for mode in ('direct', 'replay'):
            model = copy.deepcopy(source)
            model.learned.set_phase('joint')
            loss, _ = render_step(model, make_batches(list(f.split(8)), 2), geometry,
                                  lambda scene: (scene*.01).square().mean(), mode)
            results.append((loss, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0], results[1][0])
        for name, grad in results[0][1].items():
            torch.testing.assert_close(grad, results[1][1][name], atol=2e-6, rtol=2e-4)

    def test_partial_step_matches_one_adam_step_with_smaller_lr(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        opt = torch.optim.Adam([p], lr=4.)
        loss = p.square().sum(); loss.backward()
        result = guarded_step(opt, lambda: p.square().sum(), loss.detach())
        # Full Adam step overshoots; 1/2 can be accepted at equal loss due to
        # finite rounding, so compare to the LR actually accepted by the guard.
        q = torch.nn.Parameter(torch.tensor([1.]))
        q.grad = torch.tensor([2.])
        reference = torch.optim.Adam([q], lr=4.*result['scale'])
        reference.step()
        self.assertTrue(result['accepted'])
        self.assertLess(result['scale'], 1.)
        self.assertLessEqual(result['loss_after'], result['loss_before'])
        torch.testing.assert_close(p, q)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(opt.state[p][key], reference.state[q][key])
        self.assertEqual(opt.param_groups[0]['lr'], 4.)

    def test_reject_and_exception_restore_weights_and_adam(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        opt = torch.optim.Adam([p], lr=.1)
        p.square().sum().backward(); opt.step()
        opt.zero_grad(); p.square().sum().backward()
        old = p.detach().clone()
        state = copy.deepcopy(opt.state_dict())
        result = guarded_step(opt, lambda: torch.tensor(float('nan')), float(p.square().sum().detach()))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['scale'], 0.)
        self.assertEqual(len(result['trials']), 5)
        torch.testing.assert_close(p, old, atol=0, rtol=0)
        for key, value in state['state'][0].items():
            torch.testing.assert_close(opt.state_dict()['state'][0][key], value, atol=0, rtol=0)
        def fail():
            raise RuntimeError('probe failed')
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            guarded_step(opt, fail, 1.)
        torch.testing.assert_close(p, old, atol=0, rtol=0)
        for key, value in state['state'][0].items():
            torch.testing.assert_close(opt.state_dict()['state'][0][key], value, atol=0, rtol=0)

    def test_no_descent_rejection_restores_empty_state(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        opt = torch.optim.Adam([p], lr=.1)
        p.grad = torch.tensor([-1.])  # intentionally uphill proposal
        result = guarded_step(opt, lambda: p.square().sum(), 1.)
        self.assertFalse(result['accepted'])
        self.assertEqual(len(opt.state), 0)
        self.assertEqual(float(p.detach()), 1.)

    def test_training_and_exact_resume(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp)
            ply = root/'source.ply'
            write_ply(ply, raw, 0)
            extra = ['--center-position-layout', 'centroid_residual', '--center-step-guard',
                     '--center-attention-scope', 'self', '--center-readout-norm', 'affine',
                     '--center-probe-blocks', '1', '--center-drift-every', '1']
            train(args_for(ply, root/'full', extra))
            actual = torch.optim.Adam.step
            calls = [0]
            def crash(opt, *a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('interruption')
                return actual(opt, *a, **kw)
            with patch.object(torch.optim.Adam, 'step', crash):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    train(args_for(ply, root/'resume', extra))
            args = args_for(ply, root/'resume')
            args.resume = str(root/'resume/training_state.pt')
            train(args)
            a = load_checkpoint(root/'full/codec_3.pt', 'cpu')
            b = load_checkpoint(root/'resume/codec_3.pt', 'cpu')
            self.assertEqual(model_id(a), model_id(b))
            records = [json.loads(x) for x in (root/'full/loss.jsonl').read_text().splitlines()]
            for row in records:
                guard = row['stats']['step_guard']
                self.assertLessEqual(guard['loss_after'], guard['loss_before'])
                self.assertIn('center_encoder.common', row['module_grad_norms'])
            self.assertTrue((root/'full/center_drift/xyz_000003.pt').exists())
            plot_run(root/'full')
            self.assertTrue((root/'full/charts/center_step_guard.png').exists())
            summary = json.loads((root/'full/summary.json').read_text())
            self.assertEqual(summary['center_step_guard']['attempted_updates'], 3)


if __name__ == '__main__':
    unittest.main()
