import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class ChannelNormalizationControlsTests(unittest.TestCase):
    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.channel_normalization_controls'),
                             'normalization control experiment is missing')
        from gaussian_jscc import channel_normalization_controls
        return channel_normalization_controls

    def setUp(self):
        torch.set_num_threads(1)

    def test_same_fresh_channel_weights_and_teacher_across_modes(self):
        c = self.entry()
        from gaussian_jscc.modular_config import ModularCodecConfig
        from gaussian_jscc.modular_codec import ModularGaussianCodec
        cfg = ModularCodecConfig(hidden=16, channel_variant='progressive', channel_hidden=16,
                                 rates=(0, 8, 24, 48))
        teacher = ModularGaussianCodec(cfg)
        saved = dict(config=cfg.to_dict(), state_dict=copy.deepcopy(teacher.state_dict()))
        models = [c.fresh_model(saved, mode, 42, .01, 'cpu') for mode in ('point', 'statistical', 'block')]
        self.assertEqual(len({c.channel_digest(m) for m in models}), 1)
        expected = c.semantic_digest(saved['state_dict'])
        self.assertTrue(all(c.semantic_digest(m.state_dict()) == expected for m in models))
        self.assertTrue(all(not p.requires_grad for m in models for p in m.semantic_encoder.parameters()))
        y = torch.randn(1, 8, 48); mask = torch.ones(1, 8, dtype=torch.bool)
        before = models[1].channel_encoder.raw_signal(y, mask, 10.)
        torch.testing.assert_close(models[0].channel_encoder.raw_signal(y, mask, 10.), before, rtol=0, atol=0)

    def test_full_population_calibration_and_accounting_ignore_padding(self):
        c = self.entry()
        from gaussian_jscc.modular_config import ModularCodecConfig
        from gaussian_jscc.modular_codec import ModularGaussianCodec
        cfg = ModularCodecConfig(hidden=16, channel_variant='progressive', channel_hidden=16,
                                 channel_power_mode='statistical', block_size=8)
        model = ModularGaussianCodec(cfg)
        cache = [dict(y=torch.randn(2, 8, 48), active=torch.tensor([[True]*8, [True]*3+[False]*5]),
                      xyz=torch.rand(2, 8, 3))]
        report = c.calibrate_and_measure(model, cache, calibrate=True)
        self.assertEqual(report['population_points'], 11)
        self.assertTrue(report['within_average_power_budget'])
        for row in report['segments']:
            self.assertLess(row['mean_complex_power'], 1.00001)
            self.assertGreater(row['mean_complex_power'], .5)
            self.assertGreaterEqual(row['symbol_papr'], 1.)
        before = copy.deepcopy(model.state_dict())
        again = c.calibrate_and_measure(model, cache, calibrate=False)
        self.assertEqual(report['segments'], again['segments'])
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_training_validation_and_exact_resume_preserve_teacher_and_adam(self):
        c = self.entry()
        import test_modular_experiment as fixture
        import test_render_first as renders
        from gaussian_jscc.modular_experiment import main as train_teacher
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render', side_effect=renders.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras', return_value=renders.RenderFirstTests().cameras()):
            root = Path(t); source = fixture.source_fixture(root)
            opts = fixture.ModularExperimentTests().args(source, root/'teacher')
            opts.channel_variant = 'progressive'; opts.channel_hidden = 16; opts.rates = [8, 24, 48]
            opts.channel_bootstrap_steps = opts.channel_render_steps = 0
            train_teacher(opts)
            args = c.parser().parse_args(['--run', str(root/'teacher'), '--out', str(root/'suite'),
                '--steps', '3', '--validate-every', '2', '--save-every', '1', '--device', 'cpu',
                '--diagnostic-blocks', '2', '--response-blocks', '1', '--trials', '2',
                '--render-trials', '1', '--validation-views', '1', '--no-lpips'])
            plan = c.build_plan(args)
            self.assertEqual(len(plan['jobs']), 6)
            self.assertEqual(plan['rates'], [0, 8, 24, 48])
            original = (root/'teacher/checkpoints/final/codec.pt').read_bytes()
            for arm in args.arms:
                c.run_arm(args, arm)
            args.out = str(root/'resumed')
            c.run_arm(args, 'statistical_awgn10', max_updates=1)
            self.assertFalse((root/'resumed/statistical_awgn10/complete.json').exists())
            args.resume_suite = True
            c.run_arm(args, 'statistical_awgn10')
            left = root/'suite/statistical_awgn10'; right = root/'resumed/statistical_awgn10'
            a = torch.load(left/'checkpoints/final/codec.pt', weights_only=True)
            b = torch.load(right/'checkpoints/final/codec.pt', weights_only=True)
            for key, value in a['state_dict'].items():
                torch.testing.assert_close(value, b['state_dict'][key], rtol=0, atol=0)
            for key, row in a['training']['optimizer']['state'].items():
                for field, value in row.items():
                    torch.testing.assert_close(value, b['training']['optimizer']['state'][key][field], rtol=0, atol=0)
            expected = c.semantic_digest(torch.load(root/'teacher/checkpoints/final/codec.pt', weights_only=True)['state_dict'])
            self.assertEqual(c.semantic_digest(a['state_dict']), expected)
            reloaded = load_checkpoint(left/'checkpoints/final/codec.pt', 'cpu')
            self.assertEqual(reloaded.cfg.channel_power_mode, 'statistical')
            rows = [json.loads(line) for line in (right/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows], [1, 2, 3])
            self.assertEqual([r['layout'] for r in rows], ['1', '2', '3'])
            self.assertTrue((right/'diagnostics/final/latent_dimensions.csv').exists())
            self.assertTrue((right/'metrics/power.jsonl').exists())
            self.assertTrue((right/'images/final/finite_noiseless_3_view0000.png').exists())
            self.assertEqual((root/'teacher/checkpoints/final/codec.pt').read_bytes(), original)
            result = c.summarize(root/'suite', args.arms)
            self.assertEqual(len(result['rows']), 18)
            self.assertTrue(result['matched_initial_channel_weights'])
            for arm in args.arms:
                history = [json.loads(s) for s in (root/'suite'/arm/'metrics/loss.jsonl').read_text().splitlines()]
                self.assertEqual([(r['batch_index'], r['layout']) for r in history],
                                 [(r['batch_index'], r['layout']) for r in rows])
            self.assertTrue((root/'suite/comparison/curves.png').exists())
            # Initial rendering may fail (e.g. missing LPIPS weights). The arm
            # must still have a resumable checkpoint before that long operation.
            args.out = str(root/'early_interruption'); args.resume_suite = False
            with patch.object(c, 'evaluate', side_effect=RuntimeError('initial evaluation interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'initial evaluation interrupted'):
                    c.run_arm(args, 'statistical_awgn10')
            self.assertTrue((root/'early_interruption/statistical_awgn10/checkpoints/latest/codec.pt').exists())
            args.resume_suite = True
            c.run_arm(args, 'statistical_awgn10')
            early = torch.load(root/'early_interruption/statistical_awgn10/checkpoints/final/codec.pt', weights_only=True)
            for key, value in a['state_dict'].items():
                torch.testing.assert_close(value, early['state_dict'][key], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
