import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_center_attribute import setup, args_for
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.center_attribute_codec import HistoricalLightCenterDecoder
from gaussian_jscc.multiscale_codec import MultiScaleSelfCore
from gaussian_jscc.center_attribute_train import train, make_batches, render_step
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import save_checkpoint, load_checkpoint
from scripts.compare_center_decoders import summarize


class CenterDecoderComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self):
        raw, g, f, baseline = setup(16)
        cfg = CodecConfig.from_dict({**baseline.cfg.to_dict(), 'center_decoder_kind': 'historical_light'})
        return raw, g, f, GaussianCodec(cfg)

    def test_legacy_config_and_matched_initialization_rng(self):
        self.assertNotIn('center_decoder_kind', CodecConfig().to_dict())
        with self.assertRaises(ValueError):
            CodecConfig(center_decoder_kind='historical_light')
        _, _, _, base = setup()
        torch.manual_seed(18)
        trunk = GaussianCodec(base.cfg)
        following_trunk = torch.rand(10)
        torch.manual_seed(18)
        light = GaussianCodec(CodecConfig.from_dict({**base.cfg.to_dict(), 'center_decoder_kind': 'historical_light'}))
        following_light = torch.rand(10)
        torch.testing.assert_close(following_trunk, following_light, atol=0, rtol=0)
        for name, value in trunk.state_dict().items():
            if not name.startswith('learned.center_decoder.') or name.startswith('learned.center_decoder.input.'):
                torch.testing.assert_close(value, light.state_dict()[name], atol=0, rtol=0)

    def test_matches_historical_xyz_formula(self):
        _, _, _, base = setup()
        cfg = CodecConfig.from_dict({**base.cfg.to_dict(), 'representation_dim': 32, 'center_latent_dim': 16,
                                    'center_decoder_kind': 'historical_light', 'rates': [0, 1, 2, 4]})
        light = HistoricalLightCenterDecoder(cfg)
        legacy_cfg = CodecConfig.from_dict({**cfg.to_dict(), 'representation_dim': 0, 'center_latent_dim': 0,
                                           'center_decoder_kind': 'transformer', 'decoder_attention': 'window'})
        legacy = MultiScaleSelfCore(legacy_cfg)
        legacy.dec_geometry_in.load_state_dict(light.input.state_dict())
        legacy.dec_geometry_context.load_state_dict(light.context.state_dict())
        legacy.heads['xyz'].load_state_dict(light.head.state_dict())
        legacy.context_heads['xyz'].load_state_dict(light.context_head.state_dict())
        with torch.no_grad():
            legacy.dec_geometry_gate.copy_(light.gate)
            legacy.tier.weight.zero_()
            for p in legacy.snr.parameters():
                p.zero_()
        received = torch.randn(2, 19, 8)
        active = torch.ones(2, 19, dtype=torch.bool)
        active[0, -3:] = False
        packet = torch.cat((received*active[..., None], active[..., None].expand_as(received)), -1)
        torch.testing.assert_close(light(packet, active), legacy.decode(received, active.long()*3, 0)[..., :3])

    def test_padding_updates_and_roundtrip(self):
        _, _, f, model = self.make()
        active = torch.tensor([[True]*6+[False]*2, [False]*8])
        features = f.reshape(2, 8, -1).clone()
        features[~active] = float('nan')
        model.learned.set_phase('center')
        pred = model.learned.centers(features[..., :3], active)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(float(pred[~active].detach().abs().sum()), 0)
        pred[active].square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.learned.center_decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.learned.attribute_decoder.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            save_checkpoint(path, model, 1)
            restored = load_checkpoint(path, 'cpu')
            self.assertEqual(restored.cfg.center_decoder_kind, 'historical_light')
            torch.testing.assert_close(restored.reconstruct_clean(f), model.reconstruct_clean(f))

    def test_light_replay_parity(self):
        _, g, f, source = self.make()
        results = []
        for mode in ('direct', 'replay'):
            model = copy.deepcopy(source)
            model.learned.set_phase('joint')
            loss, _ = render_step(model, make_batches(list(f.split(8)), 1), g, lambda scene: (scene*.01).square().mean(), mode)
            results.append((loss, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0], results[1][0])
        for name, value in results[0][1].items():
            torch.testing.assert_close(value, results[1][1][name], atol=2e-6, rtol=2e-4)

    def test_paired_training_audits_and_summary(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            for case in ('transformer', 'historical_light'):
                train(args_for(root/'source.ply', root/case,
                               ['--center-decoder-kind', case, '--center-probe-blocks', '2', '--min-center-steps', '999']))
            report = summarize(root)
            self.assertTrue(report['audit_passed'])
            for case in report['results']:
                self.assertEqual(report['results'][case]['steps'], 3)
                self.assertIsNotNone(report['results'][case]['last']['fitted_centers'])
            path = root/'historical_light/loss.jsonl'
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            rows[0]['stats']['sampled_blocks'][0] = -1
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, 'paired audit failed'):
                summarize(root)


if __name__ == '__main__':
    unittest.main()
