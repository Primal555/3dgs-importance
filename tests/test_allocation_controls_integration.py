"""Real codec/optimizer/checkpoints; synthetic renderer is plumbing evidence only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.allocation_controls import main, parser as controls_parser
from gaussian_jscc.data import write_ply
from gaussian_jscc.multiscene import train, parser as training_parser
from gaussian_jscc.multiscene_experiments import evaluate, parser as evaluation_parser
from test_dense_prefix import dense
from test_render_first import synthetic_render
import test_render_first as render_fixture


class AllocationControlIntegrationTests(unittest.TestCase):
    def test_complete_suite_freezes_codec_and_matches_training_views(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, _, _, _ = dense(16)
            write_ply(root/'train.ply', raw, 0)
            (root/'images').mkdir()
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'scenes': [
                {'name': 'train', 'role': 'train', 'ply': 'train.ply', 'source': 'images'}]}))
            source = root/'source'
            source_args = training_parser().parse_args(['--manifest', str(manifest), '--out', str(source),
                '--device', 'cpu', '--bootstrap-steps', '1', '--render-steps', '1',
                '--allocation-steps', '1', '--joint-steps', '0', '--hidden', '16', '--depth', '1',
                '--block-size', '8', '--blocks-per-batch', '2', '--validation-views', '1',
                '--validation-trials', '1', '--views-per-step', '1', '--output-profile', 'compact'])
            out = root/'controls'
            args = controls_parser().parse_args(['--run', str(source), '--out', str(out),
                '--device', 'cpu', '--allocation-steps', '2', '--validate-every', '2',
                '--test-views', '1', '--trials', '1', '--image-views', '1'])

            def run_in_process(command, check):
                self.assertTrue(check)
                module = command[command.index('-m')+1]
                options = command[command.index('-m')+2:]
                if module == 'gaussian_jscc.multiscene':
                    train(training_parser().parse_args(options))
                elif module == 'gaussian_jscc.multiscene_experiments':
                    evaluate(evaluation_parser().parse_args(options))
                else:
                    raise AssertionError(f'unexpected subprocess: {module}')

            with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
                 patch('gaussian_jscc.rendering.load_cameras', return_value=render_fixture.RenderFirstTests().cameras()), \
                 patch('gaussian_jscc.mask_checks.check_masked_renderer', return_value={'synthetic_test': True}), \
                 patch('gaussian_jscc.allocation_controls.subprocess.run', side_effect=run_in_process):
                train(source_args)
                main(args)
            complete = json.loads((out/'complete.json').read_text())
            self.assertFalse(complete['codec_retrained'])
            self.assertFalse(complete['joint_training_run'])
            self.assertEqual(len(complete['completed_jobs']), 9)
            self.assertEqual(complete['comparison_rows'], 7)
            rows = [json.loads(line) for line in (out/'comparison.jsonl').read_text().splitlines()]
            controls = [r for r in rows if r['control_sampling']]
            self.assertEqual(len({r['codec_id'] for r in controls}), 1)
            self.assertTrue(all(r['frozen_codec_verified'] for r in controls))
            self.assertTrue(all(r['deployment_draws'] == 10 for r in controls))
            self.assertEqual({r['control_beta'] for r in controls}, {.01, .003, .001})
            plan = json.loads((out/'plan.json').read_text())
            camera_sequences = []
            for job in plan['jobs']:
                if job['kind'] != 'fit':
                    continue
                command = job['command']
                dest = Path(command[command.index('--out')+1])
                losses = [json.loads(line) for line in (dest/'loss.jsonl').read_text().splitlines()]
                self.assertEqual(len(losses), 2)
                self.assertTrue(all(not r['codec_updated'] for r in losses))
                camera_sequences.append([r['training_view_indices'] for r in losses])
                self.assertFalse(list(dest.glob('scenes/*/validation_images/*/*.png')))
            self.assertTrue(all(views == camera_sequences[0] for views in camera_sequences))
            self.assertTrue((out/'comparison.csv').exists())
            self.assertEqual(len(list((out/'charts').glob('*.png'))), 2)


if __name__ == '__main__':
    unittest.main()
