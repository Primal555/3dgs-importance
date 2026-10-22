"""Joint B must adapt the real encoder and retain exact four-module resume."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import test_successive_joint as base
from test_successive_joint import source_run, cameras


class SuccessiveJointEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)

    def options(self, source, out):
        args = base.SuccessiveJointTests().options(source, out)
        # Setting this explicitly also exercises the runtime, not just CLI text.
        args.train_semantic_encoder = True
        return args

    def test_all_four_modules_update_and_encoder_adam_resumes_exactly(self):
        # A missing optimizer group or encoder detach must fail this test.
        from gaussian_jscc import successive_joint as entry
        from gaussian_jscc.modular_checkpoint import restore_training_checkpoint
        from test_render_first import synthetic_render
        with tempfile.TemporaryDirectory() as folder, \
             patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
            root = Path(folder)
            parent = source_run(root, context_variant='both')
            source_path = parent / 'checkpoints/final/codec.pt'
            original = source_path.read_bytes()
            source = torch.load(source_path, weights_only=True)
            result = entry.main(self.options(parent, root / 'full'))
            self.assertEqual(result['status'], 'complete')
            initial = torch.load(root / 'full/checkpoints/initial/codec.pt', weights_only=True)
            self.assertFalse(initial['training']['optimizer']['state'])
            for name, value in source['state_dict'].items():
                torch.testing.assert_close(initial['state_dict'][name], value, rtol=0, atol=0)
            rows = [json.loads(s) for s in (root / 'full/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['layout'] for r in rows], ['1', '2', '3', 'mixed'])
            for row in rows:
                for name in ('semantic_encoder', 'channel_encoder', 'channel_decoder', 'semantic_decoder'):
                    self.assertGreater(row['updates'][name]['update_norm'], 0., name)
                    self.assertEqual(row['module_lrs'][name], 1e-5)
                self.assertAlmostEqual(row['loss'], row['render_loss'] + row['weighted_latent_loss'], places=7)
                self.assertGreater(row['latent_target_power'], 0.)
                self.assertAlmostEqual(row['latent_nmse'], row['latent_mse'] / row['latent_target_power'])
                self.assertAlmostEqual(row['latent_target_std'] ** 2,
                                       row['latent_target_power'] - row['latent_target_mean'] ** 2, places=7)
            final = torch.load(root / 'full/checkpoints/final/codec.pt', weights_only=True)
            self.assertTrue(any(not torch.equal(value, final['state_dict'][name])
                                for name, value in source['state_dict'].items()
                                if name.startswith('semantic_encoder.')))
            self.assertTrue(any(not torch.equal(value, final['state_dict'][name])
                                for name, value in source['state_dict'].items()
                                if name.startswith('semantic_encoder.spatial_context.')))
            for name in ('attr_mean', 'attr_std'):
                torch.testing.assert_close(final['state_dict'][name], source['state_dict'][name], rtol=0, atol=0)
            model, opt, progress, _, _ = restore_training_checkpoint(root / 'full/checkpoints/final/codec.pt', 'cpu')
            self.assertEqual({g['name'] for g in opt.param_groups},
                             {'semantic_encoder', 'channel_encoder', 'channel_decoder', 'semantic_decoder'})
            self.assertTrue(all(p.requires_grad for p in model.semantic_encoder.parameters()))
            self.assertTrue(progress['train_semantic_encoder'])
            self.assertTrue(result['frozen_attribute_statistics_unchanged'])
            self.assertTrue(result['semantic_encoder_updated'])
            args = self.options(parent, root / 'resumed')
            entry.main(args, max_updates=2)
            args.resume = True
            entry.main(args)
            resumed = torch.load(root / 'resumed/checkpoints/final/codec.pt', weights_only=True)
            for name, value in final['state_dict'].items():
                torch.testing.assert_close(value, resumed['state_dict'][name], rtol=0, atol=0)
            self.assertEqual(final['training']['progress'], resumed['training']['progress'])
            for key, state in final['training']['optimizer']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, resumed['training']['optimizer']['state'][key][name], rtol=0, atol=0)
            self.assertEqual(source_path.read_bytes(), original)
            # An A resume must never silently load B's four-group Adam.
            wrong = base.SuccessiveJointTests().options(parent, root / 'resumed')
            wrong.resume = True
            with self.assertRaises(ValueError):
                entry.main(wrong)

    def test_replay_encoder_render_gradient_matches_checkpoint(self):
        # Removing the renderer VJP to E_s must fail even if latent MSE still trains it.
        from gaussian_jscc.successive_joint import joint_step
        from gaussian_jscc.modular_checkpoint import restore_training_checkpoint
        from gaussian_jscc.modular_training import configure_phase, MODULES
        from gaussian_jscc.modular_snr_controls import control_inputs
        from gaussian_jscc.modular_experiment import make_scene
        from gaussian_jscc.data import read_ply
        from gaussian_jscc.render_objective import MultiViewRenderTask
        from test_render_first import synthetic_render
        with tempfile.TemporaryDirectory() as folder, \
             patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
            root = Path(folder)
            parent = source_run(root, context_variant='both')
            model, _, _, _, _ = restore_training_checkpoint(parent / 'checkpoints/final/codec.pt', 'cpu')
            configure_phase(model, 'channel_render', {n: 1e-5 for n in MODULES}, train_semantic_encoder=True)
            metadata = json.loads((parent / 'training.json').read_text())
            original, _, spec, old, _ = control_inputs(metadata)
            raw, degree = read_ply(spec['ply'])
            scene = make_scene(spec, raw, degree, model, original, old, root / 'parity')
            batches = [(f, torch.where(ids >= 0, torch.arange(ids.numel()).reshape_as(ids) % 3 + 1, 0))
                       for f, ids in zip(scene.groups, scene.group_ids)]
            gradients = []
            losses = []
            for mode in ('replay', 'checkpoint'):
                model.zero_grad(set_to_none=True)
                torch.manual_seed(91)
                task = MultiViewRenderTask(scene.cameras, scene.reference, degree, scene.white_background)
                loss, _ = joint_step(model, batches, scene.geometry, task, 1., mode=mode,
                                     train_semantic_encoder=True)
                losses.append(loss)
                gradients.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
            torch.testing.assert_close(losses[0], losses[1], rtol=1e-6, atol=1e-8)
            self.assertEqual(set(gradients[0]), set(gradients[1]))
            self.assertGreater(sum(float(v.abs().sum()) for n, v in gradients[0].items()
                                   if n.startswith('semantic_encoder.')), 0.)
            for name in gradients[0]:
                torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-5, atol=1e-7)
            # Independently backpropagate ONLY image MSE through the same receiver
            # boundary: this distinguishes a renderer gradient from latent-only updates.
            from gaussian_jscc.data import to_raw
            from gaussian_jscc.position_delivery import delivered_positions
            model.zero_grad(set_to_none=True)
            raw_batches = []
            for features, q in batches:
                _, recovered, _ = model.forward_channel_batches(features, features[..., :3],
                    torch.nn.functional.one_hot(q, 4).to(features), 10., 'awgn')
                pred = model.decode_representation(recovered, q, delivered_positions(features[..., :3], q, model.cfg))
                raw_batches.append(to_raw(pred[q > 0], scene.geometry, model))
            task = MultiViewRenderTask(scene.cameras, scene.reference, degree, scene.white_background)
            task(torch.cat(raw_batches)).backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.semantic_encoder.parameters()
                                   if p.grad is not None), 0.)

    def test_current_latent_target_statistics_ignore_padding_and_do_not_backpropagate(self):
        # Raw MSE alone must not hide an encoder shrinking its target feature power.
        from gaussian_jscc.modular_training import latent_reconstruction_loss, latent_diagnostics
        y = torch.tensor([[[1., 2.], [100., 100.]]], requires_grad=True)
        yh = torch.tensor([[[3., 1.], [-100., -100.]]], requires_grad=True)
        loss, metrics = latent_reconstruction_loss(y, yh, torch.tensor([[2, 0]]))
        stats = latent_diagnostics(metrics)
        self.assertIn('latent_target_mean', stats)
        self.assertEqual(stats['latent_target_mean'], 1.5)
        self.assertEqual(stats['latent_target_power'], 2.5)
        self.assertEqual(stats['latent_target_std'], .5)
        self.assertEqual(stats['latent_nmse'], 1.)
        loss.backward()
        self.assertIsNone(y.grad)
        torch.testing.assert_close(yh.grad, torch.tensor([[[2., -1.], [0., 0.]]]))

    def test_optimizer_scope_rejects_conflicting_freeze_or_nonjoint_phase(self):
        from gaussian_jscc.modular_training import configure_phase
        from test_modular_training import setup
        for phase, extra in [('channel_bootstrap', {}), ('semantic_render', {}),
                             ('channel_render', {'freeze_semantic': True}),
                             ('channel_render', {'semantic_decoder_only': True})]:
            model, _, _, _ = setup()
            with self.assertRaises(ValueError):
                configure_phase(model, phase, {}, train_semantic_encoder=True, **extra)

    def test_b_shell_launcher_uses_b_plan_without_changing_default_a(self):
        from gaussian_jscc import successive_joint as entry
        from test_render_first import synthetic_render
        with tempfile.TemporaryDirectory() as folder, \
             patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
            root = Path(folder)
            parent = source_run(root)
            default = base.SuccessiveJointTests().options(parent, root / 'a')
            a = entry.build_plan(default)
            del default.train_semantic_encoder
            self.assertEqual(entry.build_plan(default), a)  # Old A namespaces/plans still resume.
            b = entry.build_plan(self.options(parent, root / 'b'))
            self.assertNotIn('semantic_encoder', a['trainable_modules'])
            self.assertIn('semantic_encoder', b['trainable_modules'])
            self.assertNotEqual(a['version'], b['version'])
            self.assertNotIn('train_semantic_encoder', a['settings'])
            self.assertEqual(b['settings']['train_semantic_encoder'], True)
            bash = Path('C:/softwares/Git/bin/bash.exe')
            if bash.exists():
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONUTF8='1')
                launch = subprocess.run([str(bash), 'scripts/test_successive_joint_encoder_truck.sh', parent.as_posix(),
                                         (root / 'shell').as_posix(), '--dry-run', '--device', 'cpu', '--no-lpips'],
                                        env=env, capture_output=True, text=True, encoding='utf8', timeout=30)
                self.assertEqual(launch.returncode, 0, launch.stdout + launch.stderr)
                plan = json.loads(launch.stdout[launch.stdout.index('{'):])
                self.assertIn('semantic_encoder', plan['trainable_modules'])
                self.assertEqual(plan['settings']['steps'], 2000)
                self.assertFalse((root / 'shell').exists())


if __name__ == '__main__':
    unittest.main()
