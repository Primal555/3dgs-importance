"""Small real-optimizer controls; synthetic rendering replaces CUDA only."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class JointAdamResetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_reset_discards_all_moments_and_counters_but_keeps_weights_and_groups(self):
        from test_modular_training import setup
        from gaussian_jscc.modular_training import configure_phase
        from gaussian_jscc.power_followup import joint_optimizer
        model, _, _, _ = setup()
        old = configure_phase(model, 'channel_bootstrap', {}, freeze_semantic=True)
        for group in old.param_groups:
            group['betas'] = (.7, .91); group['eps'] = 1e-7
        sum(p.square().sum() for group in old.param_groups for p in group['params']).backward()
        old.step()
        weights = {n: p.detach().clone() for n, p in model.named_parameters()}
        previous = copy.deepcopy(old.state_dict())
        reset = joint_optimizer(model, old, 1e-4, 1e-5, reset_channel_adam=True)
        self.assertEqual(len(reset.state), 0)
        self.assertEqual([g['name'] for g in reset.param_groups],
                         ['channel_encoder', 'channel_decoder', 'semantic_decoder'])
        self.assertEqual([g['lr'] for g in reset.param_groups], [1e-4, 1e-4, 1e-5])
        self.assertEqual(reset.param_groups[0]['betas'], (.7, .91))
        self.assertEqual(reset.param_groups[0]['eps'], 1e-7)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, weights[name], rtol=0, atol=0)
            self.assertIsNone(p.grad)
            self.assertEqual(p.requires_grad, not name.startswith('semantic_encoder.'))
        for key, state in previous['state'].items():
            for field, value in state.items():
                torch.testing.assert_close(value, old.state_dict()['state'][key][field], rtol=0, atol=0)
        sum(p.square().sum() for g in reset.param_groups for p in g['params']).backward()
        reset.step()
        self.assertTrue(all(float(s['step']) == 1 for s in reset.state.values()))

    def test_short_probe_records_initial_reset_and_keeps_the_source_immutable(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.joint_adam_reset'), 'short Adam-reset entry missing')
        from gaussian_jscc import joint_adam_reset as entry
        from gaussian_jscc.modular_codec import ModularGaussianCodec
        from gaussian_jscc.modular_config import ModularCodecConfig
        from gaussian_jscc.modular_checkpoint import capture_rng, save_training_checkpoint
        from gaussian_jscc.modular_experiment import make_scene, json_write
        from gaussian_jscc.modular_training import configure_phase, bootstrap_step
        from gaussian_jscc.data import read_ply
        from test_modular_experiment import source_fixture
        from test_render_first import RenderFirstTests, synthetic_render
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.load_cameras', return_value=RenderFirstTests().cameras()), \
                patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
            root = Path(t); original = source_fixture(root)
            source = root/'channel'
            model = ModularGaussianCodec(ModularCodecConfig(sh_degree=1, hidden=16, grid_dim=4, levels=(2,),
                block_size=8, decoder_window=4, channel_variant='progressive', channel_hidden=16,
                channel_power_mode='block', rates=(0, 8, 24, 48)))
            raw, degree = read_ply(root/'truck.ply')
            spec = dict(name='truck', role='train', prior='none', ply=str(root/'truck.ply'), source=str(root/'source_images'))
            metadata = json.loads((original/'training.json').read_text())
            old_record = json.loads((original/'scenes/truck/scene.json').read_text())
            scene = make_scene(spec, raw, degree, model, metadata, old_record, source)
            optimizer = configure_phase(model, 'channel_bootstrap', {}, freeze_semantic=True)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                bootstrap_step(model, scene.groups[0], torch.ones_like(scene.group_ids[0]), scene.geometry,
                               10., 'awgn', 'channel_bootstrap')
                optimizer.step()
            progress = dict(step=3, phase='channel_bootstrap', phase_step=3, phase_complete=True,
                freeze_semantic=True, module_lrs={'channel_encoder': 1e-4, 'channel_decoder': 1e-4},
                completed={'channel_bootstrap': 3, 'channel_render': 0},
                phase_budgets={'channel_bootstrap': 3, 'channel_render': 0}, train_snrs=[10.])
            path = source/'checkpoints/end_channel_bootstrap/codec.pt'
            save_training_checkpoint(path, model, optimizer, progress, capture_rng(), scene.record)
            json_write(source/'training.json', dict(manifest=str(root/'manifest.json'), source_run=str(original),
                codec_config=model.cfg.to_dict(), scene=scene.record))
            source_bytes = path.read_bytes()
            args = entry.parser().parse_args(['--run', str(source), '--out', str(root/'reset'),
                '--steps', '3', '--top-k', '3', '--device', 'cpu'])
            plan = entry.build_plan(args)
            self.assertFalse((root/'reset').exists())
            self.assertEqual(plan['optimizer_policy'], 'reset')
            done = entry.main(args)
            self.assertEqual(done['status'], 'complete')
            initial = torch.load(root/'reset/checkpoints/initial/codec.pt', weights_only=True)
            final = torch.load(root/'reset/checkpoints/final/codec.pt', weights_only=True)
            self.assertEqual(initial['training']['optimizer']['state'], {})
            # Higher-tier heads first participate when their prefix is sampled;
            # their Adam counters need not equal the outer update count.
            self.assertTrue(all(1 <= float(s['step']) <= 3 for s in final['training']['optimizer']['state'].values()))
            decoder_ids = final['training']['optimizer']['param_groups'][-1]['params']
            self.assertTrue(all(float(final['training']['optimizer']['state'][i]['step']) == 3 for i in decoder_ids))
            rows = [json.loads(x) for x in (root/'reset/steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows], [4, 5, 6])
            self.assertEqual([r['layout'] for r in rows], ['1', '2', '3'])
            self.assertTrue(all(r['after'] is not None and 'decomposition' in r for r in rows))
            self.assertGreater(rows[0]['updates']['channel_encoder']['update_norm'], 0)
            self.assertGreater(rows[0]['updates']['semantic_decoder']['update_norm'], 0)
            self.assertTrue(done['semantic_encoder_unchanged'])
            self.assertEqual(path.read_bytes(), source_bytes)
            args.out = str(root/'inherit'); args.adam_mode = 'inherit'
            entry.main(args)
            inherited = torch.load(root/'inherit/checkpoints/initial/codec.pt', weights_only=True)
            self.assertTrue(inherited['training']['optimizer']['state'])
            for name, value in initial['state_dict'].items():
                torch.testing.assert_close(value, inherited['state_dict'][name], rtol=0, atol=0)
            old_rows = [json.loads(x) for x in (root/'inherit/steps.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['training_view_indices'], old_rows[0]['training_view_indices'])
            self.assertEqual(rows[0]['before'], old_rows[0]['before'])
            self.assertEqual(rows[0]['grad_norm'], old_rows[0]['grad_norm'])
            self.assertNotEqual(rows[0]['updates']['channel_encoder']['update_norm'],
                                old_rows[0]['updates']['channel_encoder']['update_norm'])
            args.out = str(root/'failed'); args.adam_mode = 'reset'
            audit = entry.audit_joint_step
            def fail_second(*values, **options):
                if values[4] == 2: raise RuntimeError('injected nonfinite diagnostic failure')
                return audit(*values, **options)
            with patch.object(entry, 'audit_joint_step', side_effect=fail_second):
                failed = entry.main(args)
            self.assertEqual(failed['status'], 'probe_error')
            self.assertEqual(failed['updates'], 1)
            self.assertFalse((root/'failed/complete.json').exists())
            self.assertTrue((root/'failed/error.json').exists())
            self.assertEqual(path.read_bytes(), source_bytes)
            args.out = str(source/'unsafe')
            with self.assertRaises(ValueError): entry.build_plan(args)


if __name__ == '__main__': unittest.main()
