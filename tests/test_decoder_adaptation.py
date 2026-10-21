"""Real codec/optimizer/resume checks; only CUDA rendering is substituted."""
import importlib.util
from dataclasses import replace
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class DecoderAdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.decoder_adaptation'),
                             'independent decoder adaptation entry is missing')
        from gaussian_jscc import decoder_adaptation
        return decoder_adaptation

    def test_optimizer_updates_only_semantic_decoder_and_restores_scope(self):
        from test_modular_training import setup
        from gaussian_jscc.modular_training import configure_phase
        from gaussian_jscc.modular_checkpoint import capture_rng, save_training_checkpoint, restore_training_checkpoint
        model, _, _, _ = setup()
        optimizer = configure_phase(model, 'channel_render', {'semantic_decoder': 1e-5},
                                    semantic_decoder_only=True)
        self.assertEqual([g['name'] for g in optimizer.param_groups], ['semantic_decoder'])
        self.assertEqual({n.split('.')[0] for n, p in model.named_parameters() if p.requires_grad},
                         {'semantic_decoder'})
        before = {n: v.clone() for n, v in model.state_dict().items()}
        sum(p.square().sum() for p in model.semantic_decoder.parameters()).backward()
        optimizer.step()
        for name, value in before.items():
            if not name.startswith('semantic_decoder.'):
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        progress = dict(phase='channel_render', step=1, semantic_decoder_only=True,
                        module_lrs={'semantic_decoder': 1e-5})
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)/'codec.pt'
            save_training_checkpoint(path, model, optimizer, progress, capture_rng(), {})
            restored, opt, _, _, _ = restore_training_checkpoint(path, 'cpu')
            self.assertEqual([g['name'] for g in opt.param_groups], ['semantic_decoder'])
            self.assertEqual({n.split('.')[0] for n, p in restored.named_parameters() if p.requires_grad},
                             {'semantic_decoder'})
            for a, b in zip(optimizer.state.values(), opt.state.values()):
                for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            configure_phase(model, 'semantic_render', {}, semantic_decoder_only=True)
        with self.assertRaises(ValueError):
            configure_phase(model, 'channel_render', {}, semantic_decoder_only=True, freeze_semantic=True)

    def test_noisy_render_updates_decoder_without_optimizing_latent_error(self):
        c = self.entry()
        from test_modular_training import setup
        from gaussian_jscc.modular_training import configure_phase
        from gaussian_jscc.perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask
        from test_render_first import RenderFirstTests, Reference, synthetic_render
        model, geometry, features, q = setup()
        from gaussian_jscc.modular_codec import ModularGaussianCodec
        model = ModularGaussianCodec(replace(model.cfg, channel_variant='progressive',
                                              channel_hidden=16, channel_power_mode='statistical'))
        q[0, -1] = 0
        configure_phase(model, 'channel_render', {'semantic_decoder': 1e-5}, semantic_decoder_only=True)
        c.set_adaptation_mode(model)
        before = {n: v.clone() for n, v in model.state_dict().items()}
        task = PerceptualMultiViewRenderTask(RenderFirstTests().cameras(), Reference(), 1, False,
                                            PerceptualImageObjective('mse_ssim', 1/3, 'cpu'))
        with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
            loss, stats = c.render_adaptation(model, [(features, q)], geometry, task)
        self.assertAlmostEqual(float(loss), stats['render_loss'], places=7)
        self.assertGreater(stats['latent_mse'], 0)
        self.assertEqual(stats['weighted_latent_loss'], 0)
        self.assertEqual(stats['aux_loss'], 0)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                            if not name.startswith('semantic_decoder.')))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.semantic_decoder.parameters()
                               if p.grad is not None), 0)
        for name, value in before.items():
            if not name.startswith('semantic_decoder.'):
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(model.channel_encoder.training)
        self.assertFalse(model.channel_decoder.training)

    def test_complete_resume_and_pending_evaluation_preserve_frozen_channel(self):
        c = self.entry()
        import test_modular_experiment as fixture
        import test_render_first as renders
        from gaussian_jscc.modular_experiment import main as train_teacher
        from gaussian_jscc.channel_normalization_controls import parser as power_parser, run_arm

        def cameras(*args, **kwargs):
            result = renders.RenderFirstTests().cameras()
            if args[-1] == 'test':
                for i, camera in enumerate(result): camera.image_name = 'test_'+str(i)
            return result

        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render', side_effect=renders.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
            root = Path(t)
            source = fixture.source_fixture(root)
            teacher = fixture.ModularExperimentTests().args(source, root/'teacher')
            teacher.channel_variant = 'progressive'; teacher.channel_hidden = 16
            teacher.rates = [8, 24, 48]
            teacher.channel_bootstrap_steps = teacher.channel_render_steps = 0
            train_teacher(teacher)
            power = power_parser().parse_args(['--run', str(root/'teacher'), '--out', str(root/'power'),
                '--steps', '3', '--device', 'cpu', '--save-every', '1', '--validate-every', '3',
                '--diagnostic-blocks', '1', '--response-blocks', '1', '--trials', '2',
                '--render-trials', '1', '--no-lpips'])
            run_arm(power, 'block_awgn10')
            source_run = root/'power/block_awgn10'
            endpoint = source_run/'checkpoints/end_channel_bootstrap/codec.pt'
            endpoint.parent.mkdir()
            shutil.copyfile(source_run/'checkpoints/final/codec.pt', endpoint)
            original_bytes = endpoint.read_bytes()
            args = c.parser().parse_args(['--run', str(source_run), '--out', str(root/'full'),
                '--steps', '4', '--device', 'cpu', '--validate-every', '2', '--save-every', '1',
                '--test-views', '2', '--render-trials', '2', '--probe-every', '1', '--no-lpips'])
            c.main(args)
            final = torch.load(root/'full/checkpoints/final/codec.pt', weights_only=True)
            parent = torch.load(endpoint, weights_only=True)
            for name, value in parent['state_dict'].items():
                if not name.startswith('semantic_decoder.'):
                    torch.testing.assert_close(value, final['state_dict'][name], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, final['state_dict'][name])
                                for name, value in parent['state_dict'].items()
                                if name.startswith('semantic_decoder.')))
            self.assertEqual([g['name'] for g in final['training']['optimizer']['param_groups']], ['semantic_decoder'])
            rows = [json.loads(x) for x in (root/'full/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['layout'] for r in rows], ['1', '2', '3', 'mixed'])
            self.assertTrue(all(r['same_input']['latent_mse_before'] == r['same_input']['latent_mse_after'] for r in rows))
            self.assertTrue(all(r['loss'] == r['render_loss'] and r['weighted_latent_loss'] == 0 for r in rows))
            self.assertEqual(set(p.name for p in (root/'full/evaluation').iterdir()), {'initial', 'adapt_000002', 'final'})
            self.assertFalse((root/'full/evaluation/adapt_000002/images').exists())
            self.assertTrue((root/'full/evaluation/final/images/final').exists())
            a = json.loads((root/'full/evaluation/initial/complete.json').read_text())
            b = json.loads((root/'full/evaluation/final/complete.json').read_text())
            self.assertEqual(a['view_names'], b['view_names'])

            args.out = str(root/'resumed')
            c.main(args, max_updates=2)
            args.resume = True
            c.main(args)
            resumed = torch.load(root/'resumed/checkpoints/final/codec.pt', weights_only=True)
            for name, value in final['state_dict'].items():
                torch.testing.assert_close(value, resumed['state_dict'][name], rtol=0, atol=0)
            for key, state in final['training']['optimizer']['state'].items():
                for field, value in state.items():
                    torch.testing.assert_close(value, resumed['training']['optimizer']['state'][key][field], rtol=0, atol=0)
            args.out = str(root/'pending'); args.resume = False
            evaluate = c.evaluate_saved
            def fail_evaluation(*values, **options):
                if options['label'] == 'adapt_000002': raise RuntimeError('render interrupted')
                return evaluate(*values, **options)
            with patch.object(c, 'evaluate_saved', side_effect=fail_evaluation):
                with self.assertRaisesRegex(RuntimeError, 'render interrupted'): c.main(args)
            args.resume = True
            c.main(args)
            repaired = torch.load(root/'pending/checkpoints/final/codec.pt', weights_only=True)
            for name, value in final['state_dict'].items():
                torch.testing.assert_close(value, repaired['state_dict'][name], rtol=0, atol=0)
            self.assertEqual(endpoint.read_bytes(), original_bytes)
            self.assertTrue(json.loads((root/'full/complete.json').read_text())['frozen_state_unchanged'])
            args.out = str(source_run/'unsafe'); args.resume = False
            with self.assertRaises(ValueError): c.build_plan(args)
            args.out = str(root/'bad_source')
            modified = torch.load(endpoint, weights_only=True)
            modified['training']['progress']['phase'] = 'channel_render'
            torch.save(modified, endpoint)
            with self.assertRaises(ValueError): c.build_plan(args)
            self.assertFalse((root/'bad_source').exists())


if __name__ == '__main__': unittest.main()
