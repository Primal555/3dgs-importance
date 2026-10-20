"""Channel continuation retains Adam/RNG and never runs the parent's joint stage."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_modular_experiment as fixtures
import test_render_first as render_fixture
from test_render_first import synthetic_render


class ChannelContinuationTests(unittest.TestCase):
    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.modular_channel_continue'),
                             'missing channel-only continuation entry')
        from gaussian_jscc.modular_channel_continue import parser, main
        return parser, main

    def source(self, root):
        from gaussian_jscc.modular_experiment import main as train
        metadata = fixtures.source_fixture(root)
        opts = fixtures.ModularExperimentTests().args(metadata, root/'parent')
        opts.semantic_bootstrap_steps = 1
        opts.semantic_render_steps = 1
        opts.channel_bootstrap_steps = 2
        opts.channel_render_steps = 2
        opts.channel_variant = 'progressive'
        opts.channel_hidden = 16
        opts.rates = [8, 24, 48]
        opts.image_objective = 'mse_ssim'
        opts.semantic_decoder_lr = 1e-5
        opts.train_snrs = [10.]
        opts.test_snrs = [10.]
        opts.cpu_threads = 1
        train(opts)
        return opts

    def options(self, parser, root):
        return parser().parse_args(['--run', str(root/'parent'), '--out', str(root/'continued'),
            '--additional-steps', '2', '--device', 'cpu', '--cpu-threads', '1',
            '--validate-every', '1', '--save-every', '1'])

    def test_channel_fork_matches_uninterrupted_training_and_freezes_both_semantic_modules(self):
        parser, main = self.entry()
        from gaussian_jscc.modular_experiment import main as train
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras', return_value=render_fixture.RenderFirstTests().cameras()):
            root = Path(t)
            opts = self.source(root)
            checkpoint = root/'parent/checkpoints/end_channel_bootstrap/codec.pt'
            source = torch.load(checkpoint, map_location='cpu', weights_only=True)
            watched = [checkpoint, root/'parent/training.json', root/'parent/checkpoints/final/codec.pt']
            digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
            uninterrupted = copy.deepcopy(opts)
            uninterrupted.out = str(root/'whole')
            uninterrupted.channel_bootstrap_steps = 4
            uninterrupted.channel_render_steps = 0
            train(uninterrupted)

            args = self.options(parser, root)
            main(args, max_updates=1)
            initial = torch.load(root/'continued/checkpoints/initial/codec.pt', weights_only=True)
            self.assertEqual(initial['step'], 4)
            self.assertEqual(initial['training']['progress']['phase_step'], 2)
            self.assertEqual(initial['training']['progress']['phase_budgets'],
                             {'semantic_bootstrap': 1, 'semantic_render': 1,
                              'channel_bootstrap': 4, 'channel_render': 0})
            self.assertFalse(initial['training']['progress']['phase_complete'])
            before = source['training']['optimizer']
            after = initial['training']['optimizer']
            self.assertEqual(before['param_groups'], after['param_groups'])
            for key, state in before['state'].items():
                for field, value in state.items():
                    torch.testing.assert_close(value, after['state'][key][field], rtol=0, atol=0)
            for name, value in source['state_dict'].items():
                torch.testing.assert_close(value, initial['state_dict'][name], rtol=0, atol=0)

            with self.assertRaises(FileExistsError):
                main(args)
            args.resume = True
            args.dry_run = True
            self.assertEqual(main(args)['target_channel_step'], 4)
            self.assertFalse((root/'continued/checkpoints/final/codec.pt').exists())
            args.additional_steps = 3
            with self.assertRaisesRegex(ValueError, 'settings mismatch'):
                main(args)
            args.additional_steps = 2
            args.dry_run = False
            main(args)
            final = torch.load(root/'continued/checkpoints/final/codec.pt', weights_only=True)
            expected = torch.load(root/'whole/checkpoints/final/codec.pt', weights_only=True)
            for name, value in expected['state_dict'].items():
                torch.testing.assert_close(value, final['state_dict'][name], rtol=0, atol=0, msg=name)
                if name.startswith(('semantic_encoder.', 'semantic_decoder.')) or name in ('attr_mean', 'attr_std'):
                    torch.testing.assert_close(source['state_dict'][name], final['state_dict'][name], rtol=0, atol=0)
            rows = [json.loads(s) for s in (root/'continued/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([(r['step'], r['phase_step'], r['layout']) for r in rows],
                             [(5, 3, '3'), (6, 4, 'mixed')])
            self.assertTrue(all(r['phase'] == 'channel_bootstrap' and r['objective'] == 'latent_mse' for r in rows))
            self.assertTrue(all(r['updates']['semantic_encoder']['update_norm'] == 0 and
                                r['updates']['semantic_decoder']['update_norm'] == 0 for r in rows))
            self.assertEqual(final['training']['progress']['completed']['channel_render'], 0)
            self.assertFalse((root/'continued/checkpoints/end_channel_render').exists())
            summary = json.loads((root/'continued/metrics/phase_summary.json').read_text())
            self.assertFalse(summary['channel_bootstrap']['optimizer_reset_at_boundary'])
            self.assertTrue(summary['channel_bootstrap']['optimizer_state_inherited'])
            for p, digest in digests.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)

    def test_dry_run_and_bad_checkpoints_never_create_output(self):
        parser, main = self.entry()
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras', return_value=render_fixture.RenderFirstTests().cameras()):
            root = Path(t)
            self.source(root)
            args = self.options(parser, root)
            args.dry_run = True
            plan = main(args)
            self.assertEqual((plan['parent_channel_step'], plan['target_channel_step']), (2, 4))
            self.assertEqual((plan['parent_step'], plan['target_step']), (4, 6))
            self.assertTrue(plan['optimizer_preserved'])
            self.assertFalse((root/'continued').exists())
            args.checkpoint = str(root/'parent/checkpoints/final/codec.pt')
            with self.assertRaisesRegex(ValueError, 'completed channel_bootstrap'):
                main(args)
            bad = torch.load(root/'parent/checkpoints/end_channel_bootstrap/codec.pt', weights_only=True)
            bad['training']['optimizer'] = None
            torch.save(bad, root/'no_adam.pt')
            args.checkpoint = str(root/'no_adam.pt')
            with self.assertRaisesRegex(ValueError, 'Adam state'):
                main(args)
            args.checkpoint = None
            args.additional_steps = 0
            with self.assertRaisesRegex(ValueError, 'positive'):
                main(args)
            args.additional_steps = 2
            args.out = str(root/'parent/child')
            with self.assertRaisesRegex(ValueError, 'NEW directory'):
                main(args)
            self.assertFalse((root/'parent/child').exists())

    def test_attribute_bootstrap_fork_keeps_completed_semantic_render_count_on_resume(self):
        parser, main = self.entry()
        from gaussian_jscc.modular_experiment import main as train
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras', return_value=render_fixture.RenderFirstTests().cameras()):
            root = Path(t)
            opts = self.source(root)
            opts.out = str(root/'bootstrap_parent')
            opts.init_bootstrap = str(root/'parent/checkpoints/end_semantic_bootstrap/codec.pt')
            opts.semantic_bootstrap_steps = 0
            train(opts)
            metadata = json.loads((root/'bootstrap_parent/training.json').read_text())
            self.assertEqual(metadata['initializer_kind'], 'semantic_bootstrap')
            args = self.options(parser, root)
            args.run = str(root/'bootstrap_parent')
            main(args, max_updates=1)
            rows = [json.loads(s) for s in (root/'continued/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([(r['step'], r['phase_step'], r['phase']) for r in rows],
                             [(4, 3, 'channel_bootstrap')])
            latest = torch.load(root/'continued/checkpoints/latest/codec.pt', weights_only=True)
            self.assertEqual(latest['training']['progress']['completed']['semantic_render'], 1)


if __name__ == '__main__':
    unittest.main()
