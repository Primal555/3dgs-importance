"""Fixed keep-set shuffle controls: small CPU checks, no scene training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from gaussian_jscc.modular_allocation_shuffle import (
    main, parser, shuffle_retained, tier_batches, comparison_summary,
)


class ShuffleControlTests(unittest.TestCase):
    def test_shuffle_changes_positive_placement_not_keep_set_or_histogram(self):
        q = torch.tensor([0, 1, 2, 3] * 20)
        before = q.clone()
        shuffled = shuffle_retained(q, 101)
        torch.testing.assert_close(q, before)
        torch.testing.assert_close(shuffled == 0, q == 0)
        torch.testing.assert_close(torch.bincount(shuffled), torch.bincount(q))
        self.assertFalse(torch.equal(shuffled, q))
        torch.testing.assert_close(shuffled, shuffle_retained(q, 101))
        self.assertFalse(torch.equal(shuffled, shuffle_retained(q, 102)))
        for bad in (torch.tensor([0., 1.5]), torch.tensor([-1, 2]),
                    torch.tensor([1, 4]), torch.ones(2, 2), torch.tensor([])):
            with self.assertRaises(ValueError):
                shuffle_retained(bad, 1)
        torch.testing.assert_close(shuffle_retained(torch.zeros(4, dtype=torch.long), 1),
                                   torch.zeros(4, dtype=torch.long))

    def test_batches_follow_original_ids_not_morton_indices_and_keep_padding_off(self):
        q = torch.tensor([3, 0, 2, 1])
        ids = [torch.tensor([[2, 0], [3, -1]]), torch.tensor([[1, -1]])]
        batches, packet = tier_batches(q, ids)
        self.assertEqual([b.tolist() for b in batches], [[[2, 3], [1, 0]], [[0, 0]]])
        self.assertEqual(packet.tolist(), [2, 3, 1, 0])

    def test_summary_pairs_by_view_trial_and_reports_seed_variation(self):
        baseline = [dict(trial=0, view=0, source_psnr=20., source_lpips=.2),
                    dict(trial=1, view=0, source_psnr=22., source_lpips=.1)]
        rows = dict(learned=baseline,
            shuffle_101=[dict(trial=1, view=0, source_psnr=20., source_lpips=.3),
                         dict(trial=0, view=0, source_psnr=18., source_lpips=.4)],
            shuffle_102=[dict(trial=0, view=0, source_psnr=19., source_lpips=.3),
                         dict(trial=1, view=0, source_psnr=21., source_lpips=.2)])
        result = comparison_summary(rows)
        self.assertEqual(result['per_shuffle']['shuffle_101']['source_psnr'], 2.)
        self.assertAlmostEqual(result['across_shuffle_seeds']['source_psnr']['mean'], 1.5)
        self.assertAlmostEqual(result['across_shuffle_seeds']['source_psnr']['std'], .5)
        self.assertAlmostEqual(result['across_shuffle_seeds']['source_lpips']['mean'], -.15)
        rows['shuffle_102'].pop()
        with self.assertRaises(ValueError): comparison_summary(rows)

    def test_complete_evaluation_loads_final_map_preserves_weights_and_writes_metrics(self):
        from test_modular_allocation import small_model
        from test_modular_experiment import source_fixture
        from test_render_first import RenderFirstTests, synthetic_render
        from test_perceptual_metrics import backend
        from gaussian_jscc.modular_checkpoint import save_training_checkpoint, capture_rng
        from gaussian_jscc.modular_experiment import inputs, make_scene
        from gaussian_jscc.data import read_ply
        from gaussian_jscc.transport import model_id

        def cameras(*args):
            values = RenderFirstTests().cameras()
            if args[-1] == 'test':
                for i, camera in enumerate(values): camera.image_name = f'test_{i}'
            return values

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, \
             patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
             patch('gaussian_jscc.perceptual_metrics._load_lpips_model', side_effect=backend), \
             patch('torch.optim.Adam.step', side_effect=AssertionError('evaluation must not train')):
            root = Path(directory)
            source = source_fixture(root)
            _, _, _, model = small_model()
            opts = parser().parse_args(['--run', str(source), '--out', str(root/'setup')])
            old_source, manifest, spec, old = inputs(opts)
            raw, degree = read_ply(spec['ply'])
            scene = make_scene(spec, raw, degree, model, old_source, old, root/'setup')
            run = root/'beta005'
            (run/'deployment/final').mkdir(parents=True)
            q = np.array([0, 1, 2, 3] * 4, dtype=np.uint8)
            np.save(run/'deployment/final/tiers.npy', q)
            (run/'deployment/final/allocation.json').write_text(json.dumps(dict(
                hard_tier_counts=[4, 4, 4, 4], source_gaussians=16, step=2000,
                snr=10., rates=[0, 8, 24, 48])))
            identity = model_id(model)
            (run/'complete.json').write_text(json.dumps(dict(final_codec_id=identity)))
            config = dict(old_source, codec_config=model.cfg.to_dict(), scene=scene.record,
                          beta=.005, snr=10., seed=42)
            (run/'training.json').write_text(json.dumps(config))
            save_training_checkpoint(run/'checkpoints/final/codec.pt', model, None,
                dict(step=2000, channel_trained=True, train_snrs=[10.]), capture_rng(), scene.record)
            before = (run/'checkpoints/final/codec.pt').read_bytes()
            args = parser().parse_args(['--run', str(run), '--out', str(root/'control'),
                '--device', 'cpu', '--shuffle-seeds', '101', '102', '--trials', '2', '--image-views', '1'])
            main(args)
            out = root/'control'
            summary = json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['codec_unchanged'])
            self.assertEqual(set(summary['layouts']), {'learned', 'shuffle_101', 'shuffle_102'})
            for row in summary['layouts'].values():
                self.assertEqual(row['tier_counts'], [4, 4, 4, 4])
                self.assertEqual(row['payload_complex_symbols'], 320)
                self.assertEqual(row['xyz_bytes'], summary['layouts']['learned']['xyz_bytes'])
                self.assertIn('source_lpips', row)
                self.assertIn('photo_ssim', row)
                self.assertIn('metadata_bytes', row)
            self.assertEqual(len((out/'metrics.csv').read_text().splitlines()), 13)
            self.assertEqual(len(list((out/'images').glob('*.png'))), 2)
            self.assertEqual(before, (run/'checkpoints/final/codec.pt').read_bytes())
            np.testing.assert_array_equal(q, np.load(out/'learned_tiers.npy'))
            self.assertTrue((out/'complete.json').exists())
            with self.assertRaises(FileExistsError): main(args)
            args.out = str(root/'wrong_beta')
            args.expected_beta = .001
            with self.assertRaisesRegex(ValueError, 'beta'): main(args)
            self.assertFalse(Path(args.out).exists())
            args.expected_beta = .005
            wrong = q.copy(); wrong[0] = 3
            np.save(run/'deployment/final/tiers.npy', wrong)
            with self.assertRaisesRegex(ValueError, 'counts'): main(args)
            self.assertFalse(Path(args.out).exists())


if __name__ == '__main__': unittest.main()
