import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class PowerFollowupTests(unittest.TestCase):
    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.power_followup'), 'follow-up suite missing')
        from gaussian_jscc import power_followup
        return power_followup

    def setUp(self):
        torch.set_num_threads(1)

    def test_attribute_replacements_preserve_xyz_and_use_exact_clean_slices(self):
        self.entry()
        from gaussian_jscc.power_followup_evaluation import replace_attributes
        raw = torch.arange(3*59).reshape(3, 59).float(); clean = raw+1000
        original = raw.clone()
        for variant, lo, hi in [('color', 11, 59), ('shape', 4, 11), ('opacity', 3, 4)]:
            out = replace_attributes(raw, clean, variant)
            torch.testing.assert_close(out[:, lo:hi], clean[:, lo:hi], rtol=0, atol=0)
            torch.testing.assert_close(out[:, :lo], raw[:, :lo], rtol=0, atol=0)
            torch.testing.assert_close(out[:, hi:], raw[:, hi:], rtol=0, atol=0)
        torch.testing.assert_close(raw, original, rtol=0, atol=0)
        with self.assertRaises(ValueError): replace_attributes(raw, clean[:1], 'color')

    def test_joint_transition_retains_channel_adam_and_adds_only_decoder(self):
        c = self.entry()
        from gaussian_jscc.modular_config import ModularCodecConfig
        from gaussian_jscc.modular_codec import ModularGaussianCodec
        from gaussian_jscc.modular_training import configure_phase
        model = ModularGaussianCodec(ModularCodecConfig(hidden=16, channel_variant='progressive', channel_hidden=16))
        optimizer = configure_phase(model, 'channel_bootstrap', {'channel_encoder': 1e-4, 'channel_decoder': 1e-4}, freeze_semantic=True)
        sum(p.square().sum() for g in optimizer.param_groups for p in g['params']).backward(); optimizer.step()
        before = {p: copy.deepcopy(s) for p, s in optimizer.state.items()}
        joint = c.joint_optimizer(model, optimizer, 1e-4, 1e-5)
        for p, state in before.items():
            for key, value in state.items(): torch.testing.assert_close(joint.state[p][key], value, rtol=0, atol=0)
        self.assertEqual([g['lr'] for g in joint.param_groups], [1e-4, 1e-4, 1e-5])
        self.assertTrue(all(not p.requires_grad for p in model.semantic_encoder.parameters()))
        self.assertTrue(all(p.requires_grad and p not in joint.state for p in model.semantic_decoder.parameters()))

    def test_three_stage_run_and_boundary_resume_use_same_weights_and_test_views(self):
        c = self.entry()
        import test_modular_experiment as fixture
        import test_render_first as renders
        from gaussian_jscc.modular_experiment import main as train_teacher
        from gaussian_jscc.channel_normalization_controls import parser as power_parser, run_arm as run_power
        def cameras(*args, **kwargs):
            result = renders.RenderFirstTests().cameras()
            if args[-1] == 'test':
                for i, camera in enumerate(result): camera.image_name = 'test_'+str(i)
            return result
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render', side_effect=renders.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
            root = Path(t); source = fixture.source_fixture(root)
            opts = fixture.ModularExperimentTests().args(source, root/'teacher')
            opts.channel_variant = 'progressive'; opts.channel_hidden = 16; opts.rates = [8, 24, 48]
            opts.channel_bootstrap_steps = opts.channel_render_steps = 0
            train_teacher(opts)
            args = power_parser().parse_args(['--run', str(root/'teacher'), '--out', str(root/'power'),
                '--steps', '3', '--device', 'cpu', '--validate-every', '3', '--save-every', '1',
                '--diagnostic-blocks', '1', '--response-blocks', '1', '--trials', '2', '--render-trials', '1', '--no-lpips'])
            for mode in ('point', 'block'): run_power(args, mode+'_awgn10')
            opts = c.parser().parse_args(['--run', str(root/'power'), '--out', str(root/'full'), '--device', 'cpu',
                '--channel-steps', '3', '--joint-steps', '2', '--validate-every', '3', '--joint-validate-every', '1',
                '--save-every', '1', '--test-views', '2', '--render-trials', '2', '--trials', '2',
                '--diagnostic-blocks', '1', '--response-blocks', '1', '--no-lpips'])
            plan = c.build_plan(opts); self.assertEqual(len(plan['jobs']), 2)
            source_path = root/'power/block_awgn10/checkpoints/final/codec.pt'
            source_bytes = source_path.read_bytes()
            for mode in ('point', 'block'): c.run_worker(opts, mode)
            opts.out = str(root/'resumed')
            c.run_worker(opts, 'block', max_updates=3)
            saved = torch.load(root/'resumed/block/checkpoints/latest/codec.pt', weights_only=True)
            self.assertEqual(saved['training']['progress']['phase'], 'channel_bootstrap')
            self.assertTrue(saved['training']['progress']['phase_complete'])
            opts.resume_suite = True
            c.run_worker(opts, 'block', max_updates=1)
            c.run_worker(opts, 'block')
            a = torch.load(root/'full/block/checkpoints/final/codec.pt', weights_only=True)
            b = torch.load(root/'resumed/block/checkpoints/final/codec.pt', weights_only=True)
            for key, value in a['state_dict'].items(): torch.testing.assert_close(value, b['state_dict'][key], rtol=0, atol=0)
            for key, row in a['training']['optimizer']['state'].items():
                for field, value in row.items(): torch.testing.assert_close(value, b['training']['optimizer']['state'][key][field], rtol=0, atol=0)
            self.assertEqual(a['step'], 8)
            parent = torch.load(source_path, weights_only=True)
            for key, value in parent['state_dict'].items():
                if key.startswith('semantic_encoder.'):
                    torch.testing.assert_close(value, a['state_dict'][key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, a['state_dict'][key]) for key, value in parent['state_dict'].items() if key.startswith('semantic_decoder.')))
            initial = torch.load(root/'full/block/checkpoints/initial/codec.pt', weights_only=True)
            for key, row in parent['training']['optimizer']['state'].items():
                for field, value in row.items(): torch.testing.assert_close(value, initial['training']['optimizer']['state'][key][field], rtol=0, atol=0)
            endpoint = json.loads((root/'full/block/evaluation/initial/complete.json').read_text())
            self.assertEqual(endpoint['view_names'], ['test_0', 'test_1'])
            ablation = json.loads((root/'full/block/ablation/results.json').read_text())
            self.assertEqual(len(ablation['rows']), 24)
            rows = [json.loads(s) for s in (root/'resumed/block/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows], [4, 5, 6, 7, 8])
            self.assertEqual(source_path.read_bytes(), source_bytes)
            single = c.summarize(root/'full', ['point'])
            self.assertTrue(all(r['pair_power_matched_5pct'] is None for r in single['test_rows']))
            report = c.summarize(root/'full', ['point', 'block'])
            self.assertEqual(len(report['test_rows']), 18)
            self.assertTrue(all(isinstance(r['pair_power_matched_5pct'], bool) for r in report['test_rows']))
            self.assertTrue((root/'full/comparison/test_endpoints.png').exists())
            # Failure inside evaluation must not replay already-completed CUDA
            # optimizer updates, which need not be bitwise deterministic.
            opts.out = str(root/'failed_evaluation'); opts.resume_suite = False
            real_evaluate = c.evaluate_endpoint
            def interrupt_evaluation(*values, **kwargs):
                if values[6] == 'joint_000001': raise RuntimeError('validation interrupted')
                return real_evaluate(*values, **kwargs)
            with patch.object(c, 'evaluate_endpoint', side_effect=interrupt_evaluation):
                with self.assertRaisesRegex(RuntimeError, 'validation interrupted'):
                    c.run_worker(opts, 'block')
            failed = root/'failed_evaluation/block/checkpoints/latest/codec.pt'
            pending = torch.load(failed, weights_only=True)['training']['progress']
            self.assertEqual(pending['step'], 7)
            self.assertEqual(pending['pending_evaluation']['label'], 'joint_000001')
            opts.resume_suite = True
            c.run_worker(opts, 'block')
            repaired = torch.load(root/'failed_evaluation/block/checkpoints/final/codec.pt', weights_only=True)
            for key, value in a['state_dict'].items(): torch.testing.assert_close(value, repaired['state_dict'][key], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
