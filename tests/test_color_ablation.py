"""CPU attribute-swap tests; synthetic images are NOT actual scene evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.color_ablation import SWAPS, color_metrics, evaluate_decoded, run, swap_attributes
from gaussian_jscc.data import write_ply
from gaussian_jscc.learned_training import decode_batches
from gaussian_jscc.transport import save_checkpoint
from test_progressive_prefix import progressive
from test_render_first import synthetic_render


class ColorAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_columns_xyz_and_inputs_unchanged(self):
        source = torch.arange(3*59, dtype=torch.float32).reshape(3, 59)
        decoded = source + 1000
        source_copy, decoded_copy = source.clone(), decoded.clone()
        expected_columns = {
            'received': [], 'source_color': list(range(11, 59)),
            'source_dc': list(range(11, 14)), 'source_sh': list(range(14, 59)),
            'source_shape': list(range(4, 11)), 'source_opacity': [3],
            'source_shape_opacity': list(range(3, 11)), 'source_attributes': list(range(3, 59)),
        }
        for name in SWAPS:
            swapped = swap_attributes(decoded, source, name)
            expected = decoded.clone()
            expected[:, expected_columns[name]] = source[:, expected_columns[name]]
            torch.testing.assert_close(swapped, expected, atol=0, rtol=0)
            torch.testing.assert_close(swapped[:, :3], decoded[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(source, source_copy, atol=0, rtol=0)
        torch.testing.assert_close(decoded, decoded_copy, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            swap_attributes(decoded[:2], source, 'received')

    def test_degree_zero_sh_swap_is_identity(self):
        source, decoded = torch.randn(3, 14), torch.randn(3, 14)
        torch.testing.assert_close(swap_attributes(decoded, source, 'source_sh'), decoded)

    def test_bias_and_chroma_do_not_confuse_brightness_or_cancel_local_error(self):
        source = torch.ones(3, 4, 4)
        result = color_metrics(source+.25, source)
        self.assertEqual(result['r_bias'], .25)  # No clipping at 1.
        self.assertEqual(result['rg_error_rmse'], 0)
        self.assertEqual(result['gb_error_rmse'], 0)
        changed = source.clone()
        changed[1, :2] += .5
        changed[1, 2:] -= .5
        result = color_metrics(changed, source)
        self.assertEqual(result['g_bias'], 0)
        self.assertEqual(result['g_mae'], .5)
        self.assertEqual(result['rg_error_rmse'], .5)

    def cameras(self):
        return [SimpleNamespace(image_name=str(i), factor=.3+i/10,
                                original_image=torch.full((3, 8, 8), .5)) for i in range(5)]

    def test_panels_metrics_and_no_grad(self):
        raw, _, _, model = progressive(17)
        received = (raw+.05).requires_grad_()
        with tempfile.TemporaryDirectory() as temp, patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
            out = Path(temp)
            rows = evaluate_decoded(received, raw, self.cameras()[:2], model.cfg.sh_degree, False, out, 3, 0)
            self.assertEqual(len(rows), 16)
            self.assertTrue((out/'comparisons/q3_view00.png').is_file())
            self.assertEqual(len(list((out/'panels/q3').glob('*.png'))), 16)
            self.assertIsNone(received.grad)

    def test_real_codec_decode_once_per_trial_matches_original_validation(self):
        from gaussian_jscc.codec import CodecConfig, GaussianCodec
        from gaussian_jscc.data import prepare, to_features
        from gaussian_jscc.render_validation import validate_render
        from torch.nn.utils.rnn import pad_sequence
        from test_render_first import Reference
        raw, _, _, progressive_model = progressive(17)
        for mode in ('adaptive', 'progressive'):
            model = GaussianCodec(CodecConfig(**dict(progressive_model.cfg.to_dict(), prefix_mode=mode)))
            model.load_state_dict(progressive_model.state_dict())
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                ply = root/'source.ply'
                write_ply(ply, raw, model.cfg.sh_degree)
                save_checkpoint(root/'codec.pt', model, 5000)
                digest = hashlib.sha256((root/'codec.pt').read_bytes()).hexdigest()
                config = {'ply': str(ply), 'source': str(root), 'source_gaussians': len(raw),
                          'blocks_per_batch': 2, 'resolution': 2, 'validation_views': 2,
                          'validation_view_names': ['0', '4'], 'snr': 10, 'channel': 'awgn', 'seed': 42}
                (root/'training.json').write_text(json.dumps(config), encoding='utf-8')
                args = SimpleNamespace(training=str(root), checkpoint=None, ply=None, source=None,
                                       out=str(root/'diagnostic'), tiers=[3], trials=2, device='cpu')
                decoded_inputs = []
                def record_evaluate(received, *a, **kw):
                    decoded_inputs.append(received.clone())
                    return evaluate_decoded(received, *a, **kw)
                with patch('gaussian_jscc.rendering.load_cameras', return_value=self.cameras()), \
                     patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
                     patch('gaussian_jscc.color_ablation.decode_batches', wraps=decode_batches) as decode, \
                     patch('gaussian_jscc.color_ablation.evaluate_decoded', side_effect=record_evaluate):
                    summary = run(args)
                self.assertEqual(decode.call_count, 2)
                self.assertEqual(len(summary), 8)
                self.assertEqual(digest, hashlib.sha256((root/'codec.pt').read_bytes()).hexdigest())
                self.assertEqual(len((root/'diagnostic/metrics.jsonl').read_text().splitlines()), 32)
                # Compare exact decoded tensors with existing validation seed/batch schedule.
                from gaussian_jscc.data import read_ply
                disk_raw, _ = read_ply(ply)
                sorted_raw, geometry, _ = prepare(disk_raw, model.cfg.morton_bits)
                blocks, ids = [], []
                for start in range(0, len(raw), model.cfg.block_size):
                    f, _ = to_features(sorted_raw[start:start+model.cfg.block_size], geometry, model)
                    blocks.append(f)
                    ids.append(torch.arange(start, start+len(f)))
                groups = [pad_sequence(blocks[i:i+2], batch_first=True) for i in range(0, len(blocks), 2)]
                group_ids = [pad_sequence(ids[i:i+2], batch_first=True, padding_value=-1) for i in range(0, len(ids), 2)]
                baseline_decodes = []
                def capture(*a, **kw):
                    scene = decode_batches(*a, **kw)
                    baseline_decodes.append(scene.clone())
                    return scene
                with patch('gaussian_jscc.render_validation.decode_batches', side_effect=capture), \
                     patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
                    validate_render(model, groups, group_ids, sorted_raw, geometry, self.cameras()[::4],
                                    Reference(), 10, 'awgn', 2, 42, root, 5000, 'render')
                for trial in range(2):
                    torch.testing.assert_close(decoded_inputs[trial], baseline_decodes[4+trial], atol=0, rtol=0)
                with self.assertRaises(FileExistsError):
                    run(args)


if __name__ == '__main__':
    unittest.main()
