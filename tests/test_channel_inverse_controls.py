"""Controlled inverse-training forks and finite-trial error decomposition."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class ChannelInverseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_error_decomposition_separates_global_bias_point_bias_and_noise(self):
        from gaussian_jscc.latent_error_diagnostics import error_moments
        target = torch.tensor([[0., 2.], [2., 4.]])
        finite = target + torch.tensor([1., -2.])
        noisy = torch.stack((finite + torch.tensor([1., 2.]), finite - torch.tensor([1., 2.])))
        result = error_moments(target, finite, noisy)
        self.assertAlmostEqual(result['summary']['noiseless_mse'], 2.5)
        self.assertAlmostEqual(result['summary']['noisy_mse'], 5.)
        self.assertAlmostEqual(result['summary']['pointwise_mean_error_mse'], 2.5)
        self.assertAlmostEqual(result['summary']['noise_variance'], 2.5)
        self.assertAlmostEqual(result['summary']['global_bias_mse'], 2.5)
        self.assertAlmostEqual(result['summary']['noise_induced_mean_shift_mse'], 0.)
        self.assertLess(abs(result['summary']['decomposition_residual']), 1e-12)
        self.assertEqual(result['dimensions'][0]['target_std'], 1.)
        self.assertEqual(result['dimensions'][0]['noisy_bias'], 1.)
        self.assertEqual(result['dimensions'][1]['noisy_bias'], -2.)
        self.assertEqual(result['dimensions'][1]['noise_variance'], 4.)

    def test_diagnostics_rejects_nonfinite_shape_and_single_trial(self):
        from gaussian_jscc.latent_error_diagnostics import error_moments
        y = torch.zeros(2, 3)
        for finite, noisy in ((y, y[None]), (y[:, :2], y[None].repeat(2, 1, 1)),
                              (y, y[None].repeat(2, 1, 1).fill_(float('nan')))):
            with self.assertRaises(ValueError): error_moments(y, finite, noisy)

    def test_sample_schedule_is_independent_of_noise_rng_and_excludes_padding(self):
        from gaussian_jscc.channel_inverse_controls import training_batch
        from types import SimpleNamespace
        scene = SimpleNamespace(groups=[torch.randn(2, 4, 7)]*3,
                                group_ids=[torch.tensor([[0,1,2,3],[4,5,-1,-1]])]*3)
        before = torch.get_rng_state().clone()
        a = training_batch(scene, 4, 42)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.randn(1000)
        b = training_batch(scene, 4, 42)
        self.assertEqual(a[0], b[0])
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        self.assertTrue((a[2][scene.group_ids[a[0]] < 0] == 0).all())
        self.assertTrue((a[2][scene.group_ids[a[0]] >= 0] > 0).all())

    def test_diagnostic_preserves_model_rng_and_records_actual_rates(self):
        from gaussian_jscc.latent_error_diagnostics import diagnose
        from test_modular_training import setup
        from test_modular_evaluation import scene_fixture
        model, g, f, q = setup()
        scene = scene_fixture(model, g, f)
        before = {k:v.clone() for k,v in model.state_dict().items()}; rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as t:
            report = diagnose(model, scene, Path(t), points=7, trials=3, batch_size=3, seed=42)
            self.assertTrue((Path(t)/'latent_errors.json').exists())
            self.assertTrue((Path(t)/'latent_dimensions.csv').exists())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        for key, value in before.items(): torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(report['sample_points'], 7)
        self.assertEqual([r['symbols'] for r in report['tiers']], [4,12,24])
        self.assertTrue(all(abs(r['summary']['decomposition_residual']) < 1e-12 for r in report['tiers']))

    def test_plan_checks_source_and_runs_only_two_training_conditions(self):
        from gaussian_jscc.channel_inverse_controls import parser, build_plan
        args = parser().parse_args(['--run','source','--out','out'])
        self.assertEqual(args.steps, 2000)
        with self.assertRaises(FileNotFoundError): build_plan(args)

    def test_real_continuation_retains_adam_freezes_teacher_and_resumes_exactly(self):
        import test_modular_experiment as fixture
        import test_render_first as rendering
        from gaussian_jscc.modular_experiment import main as train_teacher
        from gaussian_jscc.channel_inverse_controls import parser, build_plan, run_arm, summarize, main as controls_main
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render', side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras', return_value=rendering.RenderFirstTests().cameras()):
            root = Path(t); source = fixture.source_fixture(root); helper = fixture.ModularExperimentTests()
            options = helper.args(source, root/'teacher')
            options.channel_bootstrap_steps = options.channel_render_steps = 0
            train_teacher(options)
            options = helper.args(source, root/'channel_source')
            options.init_semantic = str(root/'teacher/checkpoints/end_semantic_render/codec.pt')
            options.freeze_semantic = True; options.train_snrs = [10.]
            options.channel_variant = 'progressive'; options.channel_hidden = 32; options.rates = [8,24,48]
            options.channel_bootstrap_steps = 4; options.channel_render_steps = 0
            train_teacher(options)
            checkpoint = root/'channel_source/checkpoints/final/codec.pt'
            saved = torch.load(checkpoint, weights_only=True)
            original_bytes = checkpoint.read_bytes()
            args = parser().parse_args(['--run',str(root/'channel_source'),'--out',str(root/'controls'), '--device','cpu',
                '--steps','4','--validate-every','2','--save-every','2', '--diagnostic-points','8',
                '--diagnostic-trials','2','--diagnostic-batch','4'])
            plan = build_plan(args)
            self.assertEqual([j['kind'] for j in plan['jobs']], ['none','awgn'])
            self.assertEqual(plan['rates'], [0,8,24,48])
            original_out = args.out
            args.out = str(root/'channel_source/forbidden_output')
            with self.assertRaisesRegex(ValueError, 'source'): build_plan(args)
            args.out = original_out
            args.arm = 'noiseless'; args.dry_run = True
            with patch('gaussian_jscc.channel_inverse_controls.run_arm', side_effect=AssertionError('dry run trained')):
                dry = controls_main(args)
            self.assertEqual(dry['jobs'][0]['arm'], 'noiseless')
            args.arm = None; args.dry_run = False
            for arm in ('noiseless','awgn10'):
                run_arm(args, arm)
                data = torch.load(root/'controls'/arm/'checkpoints/initial/codec.pt', weights_only=True)
                self.assertEqual(list(data['training']['optimizer']['state']), list(saved['training']['optimizer']['state']))
                for key, state in saved['training']['optimizer']['state'].items():
                    for name, value in state.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(value, data['training']['optimizer']['state'][key][name], rtol=0, atol=0)
                trained = load_checkpoint(root/'controls'/arm/'checkpoints/final/codec.pt','cpu')
                for key, value in saved['state_dict'].items():
                    if key.startswith(('semantic_encoder.','semantic_decoder.')) or key in ('attr_mean','attr_std'):
                        torch.testing.assert_close(value, trained.state_dict()[key], rtol=0, atol=0)
                rows = [json.loads(line) for line in (root/'controls'/arm/'metrics/loss.jsonl').read_text().splitlines()]
                self.assertEqual([r['step'] for r in rows], [5,6,7,8])
                self.assertEqual([r['continuation_step'] for r in rows], [1,2,3,4])
                self.assertTrue(all(r['channel_kind'] == ('none' if arm=='noiseless' else 'awgn') for r in rows))
            a = [json.loads(line) for line in (root/'controls/noiseless/metrics/loss.jsonl').read_text().splitlines()]
            b = [json.loads(line) for line in (root/'controls/awgn10/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([(r['layout'],r['batch_index'],r['latent_target_power']) for r in a],
                             [(r['layout'],r['batch_index'],r['latent_target_power']) for r in b])
            args.out = str(root/'resumed')
            run_arm(args, 'awgn10', max_updates=3)
            # Replay an uncheckpointed log row must not duplicate resumed updates.
            log = root/'resumed/awgn10/metrics/loss.jsonl'
            with log.open('a') as handle: handle.write(json.dumps(dict(b[-1],step=9))+'\n')
            args.resume_suite = True
            run_arm(args, 'awgn10')
            full = load_checkpoint(root/'controls/awgn10/checkpoints/final/codec.pt','cpu')
            resumed = load_checkpoint(root/'resumed/awgn10/checkpoints/final/codec.pt','cpu')
            for key, value in full.state_dict().items(): torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(len(log.read_text().splitlines()),4)
            self.assertEqual(checkpoint.read_bytes(), original_bytes)
            old_options = helper.args(root/'controls/noiseless', root/'controls/noiseless')
            old_options.resume = str(root/'controls/noiseless/checkpoints/initial/codec.pt')
            old_options.freeze_semantic = True
            old_options.semantic_bootstrap_steps = old_options.semantic_render_steps = 0
            old_options.channel_bootstrap_steps = 8; old_options.channel_render_steps = 0
            old_options.train_snrs = [10.]; old_options.color_metrics = True
            old_options.channel_variant = 'progressive'; old_options.channel_hidden = 32; old_options.rates = [8,24,48]
            with self.assertRaisesRegex(ValueError, 'channel_inverse_controls'):
                train_teacher(old_options)
            args.out = str(root/'early_interrupt'); args.resume_suite = False
            with patch('gaussian_jscc.channel_inverse_controls.diagnose', side_effect=RuntimeError('initial interrupt')):
                with self.assertRaisesRegex(RuntimeError, 'initial interrupt'): run_arm(args, 'noiseless')
            args.resume_suite = True
            run_arm(args, 'noiseless')
            self.assertTrue((root/'early_interrupt/noiseless/diagnostics/initial/latent_errors.json').exists())
            import gaussian_jscc.channel_inverse_controls as module
            args.out = str(root/'endpoint_interrupt'); args.resume_suite = False
            real_evaluation = module.evaluate_paths
            def interrupt_after_endpoint(*values, **kwargs):
                result = real_evaluation(*values, **kwargs)
                if kwargs['phase']=='final': raise RuntimeError('endpoint interrupt')
                return result
            with patch.object(module, 'evaluate_paths', side_effect=interrupt_after_endpoint):
                with self.assertRaisesRegex(RuntimeError,'endpoint interrupt'): run_arm(args,'awgn10')
            args.resume_suite = True; run_arm(args, 'awgn10')
            endpoints = [json.loads(line) for line in (root/'endpoint_interrupt/awgn10/metrics/validation.jsonl').read_text().splitlines()]
            self.assertEqual(len(endpoints),len({(r['step'],r['phase'],r['path'],r['layout']) for r in endpoints}))
            args.out = str(root/'setup_interrupt'); args.resume_suite = False
            real_scene = module.make_scene
            def interrupt_after_scene(*values, **kwargs):
                real_scene(*values, **kwargs)
                raise RuntimeError('setup interrupt')
            with patch.object(module, 'make_scene', side_effect=interrupt_after_scene):
                with self.assertRaisesRegex(RuntimeError,'setup interrupt'): run_arm(args, 'noiseless')
            args.resume_suite = True; run_arm(args, 'noiseless')
            self.assertTrue((root/'setup_interrupt/noiseless/complete.json').exists())
            report = summarize(root/'controls')
            self.assertTrue((root/'controls/comparison/psnr_curves.png').exists())
            self.assertTrue((root/'controls/comparison/latent_errors.png').exists())
            self.assertEqual(len(report['comparison']),8)


if __name__ == '__main__': unittest.main()
