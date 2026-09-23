import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from test_center_attribute import setup, args_for
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.transformer_decoder import BlockSelfAttention
from gaussian_jscc.center_attribute_train import train, make_batches, render_step
from gaussian_jscc.center_interaction_diagnostics import (neighbor_sensitivity, layer_features,
    fit_layer_probes, feature_cache, run)
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import load_checkpoint, model_id
from scripts.compare_center_decoders import summarize


class CenterInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pair(self):
        raw, geometry, features, source = setup(16)
        torch.manual_seed(42)
        block = GaussianCodec(source.cfg)
        rng = torch.rand(3)
        torch.manual_seed(42)
        single = GaussianCodec(CodecConfig.from_dict({**source.cfg.to_dict(), 'center_attention_scope': 'self'}))
        torch.testing.assert_close(torch.rand(3), rng, atol=0, rtol=0)
        return raw, geometry, features, block, single

    def test_config_and_matching_weights(self):
        self.assertNotIn('center_attention_scope', CodecConfig().to_dict())
        with self.assertRaises(ValueError):
            CodecConfig(center_attention_scope='self')
        *_, block, single = self.pair()
        for name, value in block.state_dict().items():
            torch.testing.assert_close(value, single.state_dict()[name], atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in block.parameters()), sum(p.numel() for p in single.parameters()))

    def test_self_attention_exact_diagonal_formula_and_gradients(self):
        a = BlockSelfAttention(16, 2)
        b = copy.deepcopy(a)
        x = torch.randn(2, 8, 16)
        active = torch.ones(2, 8, dtype=torch.bool)
        mask = torch.full((8, 8), float('-inf')).fill_diagonal_(0)
        aa, bb = a(x, active, bias=mask), b(x, active, self_only=True)
        torch.testing.assert_close(aa, bb, atol=1e-6, rtol=1e-5)
        aa.square().sum().backward()
        bb.square().sum().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=1e-4)

    def test_self_decoder_neighbor_invariance_and_padding(self):
        *_, model = self.pair()
        dec = model.learned.center_decoder
        z = torch.randn(2, 8, model.cfg.center_latent_dim)
        active = torch.tensor([[True]*6+[False]*2, [False]*8])
        z[~active] = float('nan')
        original = dec(z, active)
        changed = z.clone()
        changed[0, 1:6] += 10
        torch.testing.assert_close(dec(changed, active)[0, 0], original[0, 0], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(original).all())
        self.assertEqual(float(original[~active].detach().abs().sum()), 0)

    def test_zero_perturbation_and_decoder_isolation(self):
        _, geometry, features, _, model = self.pair()
        model.eval().requires_grad_(False)
        before = model_id(model)
        report = neighbor_sensitivity(model, list(features[:, :3].split(8)), [0], geometry.span,
                                      [0., .05], targets=2, trials=1)
        for row in report['samples']:
            if row['factor'] == 0 or row['route'] == 'decoder_only':
                self.assertLess(row['target_shift_world'], 1e-7)
        self.assertEqual(before, model_id(model))

    def test_layer_cache_probes_do_not_mutate_codec(self):
        _, geometry, features, model, _ = self.pair()
        model.eval().requires_grad_(False)
        blocks = list(features[:, :3].split(8))
        before = model_id(model)
        xf, yf, _ = feature_cache(model, blocks, [0])
        xh, yh, _ = feature_cache(model, blocks, [1])
        self.assertEqual(list(xf), ['input', 'block_1', 'tap_1', 'block_2', 'tap_2', 'block_3', 'tap_3'])
        # Same features -> identical head initializations and sampling must yield identical probes.
        report = fit_layer_probes({'a': xf['input'], 'b': xf['input']}, yf,
                                  {'a': xh['input'], 'b': xh['input']}, yh, geometry.span,
                                  'cpu', steps=3, width=8, batch_size=4)
        self.assertEqual(report['final']['a'], report['final']['b'])
        self.assertEqual(before, model_id(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_self_replay_parity(self):
        _, g, f, _, source = self.pair()
        results = []
        for mode in ('direct', 'replay'):
            model = copy.deepcopy(source)
            model.learned.set_phase('joint')
            loss, _ = render_step(model, make_batches(list(f.split(8)), 1), g,
                                  lambda scene: (scene*.01).square().mean(), mode)
            results.append((loss, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0], results[1][0])
        for name, value in results[0][1].items():
            torch.testing.assert_close(value, results[1][1][name], atol=2e-6, rtol=2e-4)

    def test_pair_and_diagnostics_end_to_end(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            (root/'experiment.json').write_text(json.dumps({'comparison': 'interaction'}))
            for case, scope in (('block_attention', 'block'), ('self_only', 'self')):
                train(args_for(root/'source.ply', root/case,
                      ['--center-attention-scope', scope, '--center-readout-norm', 'affine',
                       '--center-probe-blocks', '2', '--min-center-steps', '4']))
                restored = load_checkpoint(root/case/'codec_last.pt', 'cpu')
                self.assertEqual(restored.cfg.center_attention_scope, scope)
            report = summarize(root)
            self.assertTrue(report['audit_passed'])
            self.assertFalse(report['not_parameter_matched'])
            args = SimpleNamespace(checkpoint=str(root/'block_attention/codec_last.pt'), ply=str(root/'source.ply'),
                out=str(root/'diagnostics'), device='cpu', fit_blocks=2, heldout_blocks=1,
                targets=1, trials=1, factors=[0., .01], probe_steps=3, probe_width=8, probe_batch=4,
                probe_lr=.001, seed=42, cpu_threads=1, allow_ply_mismatch=False)
            result = run(args)
            self.assertTrue(result['weights_unchanged'])
            self.assertTrue(result['ply_fingerprint_match'])
            self.assertTrue((root/'diagnostics/diagnostics.png').is_file())
            # Mismatched source must not silently diagnose another scene.
            args.out = str(root/'bad')
            raw[0, 3] += .1
            write_ply(root/'source.ply', raw, 0)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                run(args)
            # Prove full decoder tensors, not merely encoder, are audited.
            path = root/'self_only/codec_0.pt'
            saved = torch.load(path, weights_only=True)
            saved['state_dict']['learned.center_decoder.readout.2.bias'] += 1
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, 'paired audit failed'):
                summarize(root)

    def test_self_scope_exact_resume(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            opts = ['--center-attention-scope', 'self', '--center-readout-norm', 'affine', '--min-center-steps', '4']
            train(args_for(root/'source.ply', root/'full', opts))
            original = torch.optim.Adam.step
            calls = [0]
            def crash(opt, *a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('interrupted')
                return original(opt, *a, **kw)
            with patch.object(torch.optim.Adam, 'step', crash), self.assertRaisesRegex(RuntimeError, 'interrupted'):
                train(args_for(root/'source.ply', root/'resume', opts))
            args = args_for(root/'source.ply', root/'resume')
            args.resume = str(root/'resume/training_state.pt')
            train(args)
            full, resumed = [load_checkpoint(root/d/'codec_last.pt', 'cpu') for d in ('full', 'resume')]
            self.assertEqual(resumed.cfg.center_attention_scope, 'self')
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
