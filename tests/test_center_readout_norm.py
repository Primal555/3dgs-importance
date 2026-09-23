import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from test_center_attribute import setup, args_for
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.center_attribute_codec import FeatureAffine
from gaussian_jscc.center_attribute_train import train, make_batches, render_step
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import load_checkpoint, save_checkpoint


class CenterReadoutNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pair(self):
        raw, geometry, features, baseline = setup(16)
        torch.manual_seed(42)
        baseline = GaussianCodec(baseline.cfg)
        following = torch.rand(4)
        torch.manual_seed(42)
        affine = GaussianCodec(CodecConfig.from_dict({**baseline.cfg.to_dict(), 'center_readout_norm': 'affine'}))
        torch.testing.assert_close(torch.rand(4), following, atol=0, rtol=0)
        return raw, geometry, features, baseline, affine

    def test_config_legacy_and_invalid_combinations(self):
        self.assertNotIn('center_readout_norm', CodecConfig().to_dict())
        for kwargs in ({'center_readout_norm': 'rms'}, {'center_readout_norm': 'affine'},
                       {'center_readout_norm': 'affine', 'center_decoder_kind': 'historical_light',
                        'center_latent_dim': 8, 'representation_dim': 16}):
            with self.assertRaises(ValueError):
                CodecConfig(**kwargs)
        *_, model = self.pair()
        self.assertEqual(CodecConfig.from_dict(model.cfg.to_dict()), model.cfg)

    def test_affine_keeps_mean_and_scale(self):
        layer = FeatureAffine(8)
        x = torch.randn(2, 4, 8)
        torch.testing.assert_close(layer(x), x, atol=0, rtol=0)
        torch.testing.assert_close(layer(x+3)-layer(x), torch.full_like(x, 3))
        torch.testing.assert_close(layer(2*x), 2*layer(x), atol=0, rtol=0)
        layer(x).sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)

    def test_identical_initial_weights_and_internal_norms(self):
        _, _, _, baseline, affine = self.pair()
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), sum(p.numel() for p in affine.parameters()))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, affine.state_dict()[name], atol=0, rtol=0)
        self.assertTrue(all(isinstance(n, FeatureAffine) for n in affine.learned.center_decoder.norms))
        for block in affine.learned.center_decoder.blocks:
            self.assertIsInstance(block.norm1, nn.LayerNorm)
            self.assertIsInstance(block.norm2, nn.LayerNorm)
        self.assertIsInstance(affine.learned.attribute_decoder.norm, nn.LayerNorm)

    def test_masking_gradients_and_checkpoint_roundtrip(self):
        _, _, f, _, model = self.pair()
        model.learned.set_phase('center')
        f = f.reshape(2, 8, -1).clone()
        active = torch.tensor([[True]*6+[False]*2, [False]*8])
        f[~active] = float('nan')
        xyz = model.learned.centers(f[..., :3], active)
        self.assertTrue(torch.isfinite(xyz).all())
        self.assertEqual(float(xyz[~active].detach().abs().sum()), 0)
        xyz[active].square().sum().backward()
        for module in (model.learned.center_encoder, model.learned.center_decoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in model.learned.attribute_decoder.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'codec.pt'
            save_checkpoint(path, model, 0)
            restored = load_checkpoint(path, 'cpu')
            self.assertEqual(restored.cfg.center_readout_norm, 'affine')
            torch.testing.assert_close(restored.learned.centers(f[..., :3], active), xyz)

    def test_replay_parity(self):
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

    def test_affine_resume_and_legacy_missing_option(self):
        raw, _, _, _ = setup(40)
        for mode in ('affine', 'layernorm'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
                root = Path(tmp)
                write_ply(root/'source.ply', raw, 0)
                options = ['--center-readout-norm', mode, '--center-probe-blocks', '2', '--min-center-steps', '4']
                train(args_for(root/'source.ply', root/'full', options))
                original = torch.optim.Adam.step
                calls = [0]
                def crash(opt, *a, **kw):
                    calls[0] += 1
                    if calls[0] == 2:
                        raise RuntimeError('interrupted')
                    return original(opt, *a, **kw)
                with patch.object(torch.optim.Adam, 'step', crash), self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    train(args_for(root/'source.ply', root/'resume', options))
                path = root/'resume/training_state.pt'
                if mode == 'layernorm':
                    state = torch.load(path, weights_only=True)
                    del state['arguments']['center_readout_norm']
                    torch.save(state, path)
                args = args_for(root/'source.ply', root/'resume')
                args.resume = str(path)
                # CLI default must not override saved affine mode.
                train(args)
                a, b = [load_checkpoint(root/d/'codec_last.pt', 'cpu') for d in ('full', 'resume')]
                self.assertEqual(b.cfg.center_readout_norm, mode)
                for name, value in a.state_dict().items():
                    torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
                summary = json.loads((root/'resume/summary.json').read_text())
                self.assertEqual(summary['completed'], {'center': 3, 'attribute': 0, 'joint': 0})


if __name__ == '__main__':
    unittest.main()
