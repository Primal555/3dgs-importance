"""Small CPU controls only; real scene evaluation runs on the server GPU."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gaussian_jscc.allocation import GaussianTierMask, scene_fingerprint
from gaussian_jscc.allocation_shuffle import run, shuffled_tiers
from gaussian_jscc.data import write_ply
from gaussian_jscc.route2 import save_joint
from test_progressive_prefix import progressive
from test_render_first import synthetic_render
import test_render_first as render_fixture


class ShuffleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shuffle_preserves_q0_and_all_positive_counts(self):
        q = torch.tensor([0]*38 + [1]*32 + [2]*19 + [3]*11)
        first = shuffled_tiers(q, 2026)
        torch.testing.assert_close(torch.bincount(first), torch.bincount(q))
        torch.testing.assert_close(first, shuffled_tiers(q, 2026))
        self.assertFalse(torch.equal(first, q))
        self.assertFalse(torch.equal(first, shuffled_tiers(q, 2027)))

    def test_comparison_saves_paired_metrics_without_training(self):
        raw, _, _, model = progressive(24)
        wanted = torch.arange(len(raw)) % 4
        mask = GaussianTierMask(len(raw))
        with torch.no_grad():
            mask.keep_logits.copy_(torch.nn.functional.one_hot((wanted > 0).long(), 2)*40.)
            mask.logits.copy_(torch.nn.functional.one_hot((wanted-1).clamp_min(0), 3)*40.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root/'source.ply'
            write_ply(ply, raw, model.cfg.sh_degree)
            save_joint(root, '_best_joint', model, mask, scene_fingerprint(raw), 1000, {})
            config = {'ply': str(ply), 'source': 'mock', 'snr': 10., 'channel': 'awgn',
                      'seed': 42, 'blocks_per_batch': 2, 'validation_views': 1,
                      'position_net_bits_per_use': 2., 'resolution': 2}
            (root/'training.json').write_text(json.dumps(config))
            args = SimpleNamespace(training=str(root), out=None, ply=None, source=None,
                                   device='cuda', trials=2, shuffle_seed=2026)
            with patch('gaussian_jscc.cli.device_for', return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras', return_value=render_fixture.RenderFirstTests().cameras()*2), \
                 patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
                 patch('torch.optim.Adam.step', side_effect=AssertionError('no training allowed')):
                run(args)
            out = root/'allocation_shuffle'
            result = json.loads((out/'summary.json').read_text())
            self.assertTrue(result['codec_unchanged'])
            a, b = result['layouts']['learned'], result['layouts']['shuffled']
            self.assertEqual(a['tier_counts'], b['tier_counts'])
            self.assertEqual(a['payload_complex_symbols'], b['payload_complex_symbols'])
            for metric in ('source_mse', 'source_psnr', 'source_ssim', 'photo_psnr'):
                self.assertAlmostEqual(result['paired_learned_minus_shuffled'][metric], a[metric]-b[metric])
            for label in ('learned', 'shuffled'):
                self.assertTrue((out/'images'/f'{label}_view00.png').exists())
            a = np.load(out/'learned_tiers.npy')
            b = np.load(out/'shuffled_tiers.npy')
            np.testing.assert_array_equal(np.bincount(a, minlength=4), np.bincount(b, minlength=4))
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == '__main__':
    unittest.main()
