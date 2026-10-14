"""Larger channel budgets must not change the frozen semantic representation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_codec import ModularGaussianCodec
from gaussian_jscc.codec import prefix_mask


class ProgressiveBudgetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_three_positive_integer_budgets_and_old_default(self):
        self.assertEqual(ModularCodecConfig().rates, (0, 4, 12, 24))
        cfg = ModularCodecConfig(rates=(0, 8, 24, 48))
        self.assertEqual(ModularCodecConfig.from_dict(cfg.to_dict()), cfg)
        for rates in ((0, 8, 8, 48), (0, 24, 8, 48), (0, 8, 24),
                      (0, 8, 24, 48, 64), (0, 8., 24, 48), (False, 8, 24, 48)):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                ModularCodecConfig(rates=rates)

    def test_new_signal_width_prefix_invariance_and_power(self):
        cfg = ModularCodecConfig(rates=(0, 8, 24, 48), channel_variant='progressive',
                                 hidden=16, grid_dim=4, levels=(2,), channel_hidden=32)
        model = ModularGaussianCodec(cfg)
        y = torch.randn(2, 5, 48)
        q = torch.full((2, 5), 3)
        signal = model.channel_encoder(y, q > 0, 10.)
        self.assertEqual(signal.shape, (2, 5, 96))
        stages = model.channel_decoder.decode_stages(signal, torch.ones_like(signal, dtype=torch.bool), q, 10.)
        for tier in (1, 2):
            low = torch.full_like(q, tier)
            mask = prefix_mask(low.flatten(), cfg.rates).reshape_as(signal)
            decoded = model.channel_decoder(signal.masked_fill(~mask, float('nan')), mask, low, 10.)
            torch.testing.assert_close(decoded, stages[..., tier-1, :], rtol=0, atol=0)
        for start, end in zip(cfg.rates[:-1], cfg.rates[1:]):
            power = signal[..., 2*start:2*end].square().sum(-1)/(end-start)
            self.assertLessEqual(float(power.detach().max()), 1.00001)

    def test_plan_forwards_budget_and_pure_latent_objective(self):
        from test_modular_experiment import source_fixture
        from gaussian_jscc.channel_capacity_controls import parser, build_plan
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            source = source_fixture(root)
            teacher = root/'teacher.pt'; teacher.write_bytes(b'plan only')
            args = parser().parse_args(['--run', str(source), '--out', str(root/'out'),
                '--init-semantic', str(teacher), '--groups', 'progressive',
                '--rates', '8', '24', '48', '--channel-bootstrap-steps', '4000',
                '--channel-render-steps', '0'])
            plan = build_plan(args)
            self.assertEqual(plan['rates'], [0, 8, 24, 48])
            self.assertEqual(len(plan['jobs']), 1)
            command = plan['jobs'][0]['command']
            i = command.index('--rates')
            self.assertEqual(command[i+1:i+4], ['8', '24', '48'])
            self.assertEqual(plan['losses'], 'latent MSE only; render evaluation without updates')

    def test_receiver_packet_uses_new_lengths_and_unchanged_positions(self):
        from test_gaussian_jscc import fixture
        from gaussian_jscc.transport import save_checkpoint, load_checkpoint, transmit, receive
        from gaussian_jscc.data import prepare, to_features, to_raw
        from gaussian_jscc.position_delivery import delivered_positions
        cfg = ModularCodecConfig(rates=(0, 8, 24, 48), channel_variant='progressive',
                                 hidden=16, grid_dim=4, levels=(2,), channel_hidden=32,
                                 block_size=8, sh_degree=1)
        model = ModularGaussianCodec(cfg); raw, _ = fixture(19, 1); q = torch.arange(19)%4
        ordered, geometry, tiers = prepare(raw, 16, q); expected = []
        for start in range(0, 19, 8):
            f, x = to_features(ordered[start:start+8], geometry, model)
            tq = tiers[start:start+8]; pred = model(f, x, tq, 10., 'none')
            torch.testing.assert_close(pred[:, :3], delivered_positions(f[:, :3], tq, cfg), rtol=0, atol=0)
            expected.append(to_raw(pred[tq>0], geometry, model))
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); save_checkpoint(root/'codec.pt', model, 0)
            stats = transmit(model, raw, q, 10., 'none', 7, root/'packet')
            receiver = load_checkpoint(root/'codec.pt', 'cpu')
            receiver.semantic_encoder.forward = lambda *a: (_ for _ in ()).throw(AssertionError('sender access'))
            decoded = receive(receiver, root/'packet')
            torch.testing.assert_close(decoded, torch.cat(expected), rtol=1e-5, atol=1e-6)
            self.assertEqual(stats['payload_complex_symbols'], sum(cfg.rates[int(v)] for v in q))

    def test_context_wrapper_does_not_silently_drop_budget(self):
        from gaussian_jscc.pointnet_context_controls import parser, build_plan
        args = parser().parse_args(['--run', 'source', '--out', 'out', '--rates', '8', '24', '48'])
        for job in build_plan(args)['jobs']:
            command = job['command']; i = command.index('--rates')
            self.assertEqual(command[i+1:i+4], ['8', '24', '48'])

    def test_fresh_budget_training_exact_resume_frozen_teacher_and_dynamic_probe(self):
        import test_modular_experiment as fixture
        import test_render_first as rendering
        from gaussian_jscc.modular_experiment import main
        from gaussian_jscc.transport import load_checkpoint
        from gaussian_jscc.latent_prefix_probe import parser as probe_parser, main as probe_main
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render', side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras', return_value=rendering.RenderFirstTests().cameras()):
            root = Path(t); source = fixture.source_fixture(root)
            helper = fixture.ModularExperimentTests()
            teacher_args = helper.args(source, root/'teacher')
            teacher_args.channel_bootstrap_steps = teacher_args.channel_render_steps = 0
            main(teacher_args)
            teacher = root/'teacher/checkpoints/end_semantic_render/codec.pt'
            reference = load_checkpoint(teacher, 'cpu')
            from gaussian_jscc.modular_checkpoint import initialize_channel_run
            bad = ModularCodecConfig.from_dict(reference.cfg.to_dict())
            bad.rates = (0, 8, 24, 48); bad.latent_dim = 49
            with self.assertRaisesRegex(ValueError, 'config mismatch'):
                initialize_channel_run(teacher, 'cpu', cfg=bad)
            for name in ('full', 'resumed'):
                args = helper.args(source, root/name)
                args.init_semantic = str(teacher); args.freeze_semantic = True
                args.rates = [8, 24, 48]; args.channel_variant = 'progressive'; args.channel_hidden = 32
                args.channel_bootstrap_steps = 4; args.channel_render_steps = 0
                main(args, max_updates=3 if name == 'resumed' else None)
                if name == 'resumed':
                    args.init_semantic = None; args.resume = str(root/name/'checkpoints/latest/codec.pt')
                    args.rates = [4, 12, 24]
                    with self.assertRaisesRegex(ValueError, 'config'):
                        main(args)
                    args.rates = [8, 24, 48]
                    main(args)
                model = load_checkpoint(root/name/'checkpoints/final/codec.pt', 'cpu')
                self.assertEqual(model.cfg.rates, (0, 8, 24, 48))
                for key, value in reference.state_dict().items():
                    if key.startswith(('semantic_encoder.', 'semantic_decoder.')) or key in ('attr_mean', 'attr_std'):
                        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
                rows = [json.loads(line) for line in (root/name/'metrics/loss.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(r['phase'] == 'channel_bootstrap' for r in rows))
            full = load_checkpoint(root/'full/checkpoints/final/codec.pt', 'cpu')
            resumed = load_checkpoint(root/'resumed/checkpoints/final/codec.pt', 'cpu')
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            summary = probe_main(probe_parser().parse_args(['--run', str(root/'full'),
                '--out', str(root/'probe'), '--device', 'cpu', '--trials', '1']))
            self.assertEqual(len(summary['comparison']), 18)
            for row in summary['comparison']:
                if row['negative_control']:
                    self.assertEqual(row['source_psnr_drop_db'], 0.)
                    self.assertEqual(row['latent_mse_increase'], 0.)


if __name__ == '__main__':
    unittest.main()
