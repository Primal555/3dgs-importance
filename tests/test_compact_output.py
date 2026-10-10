import json
import tempfile
import unittest
import zipfile
from pathlib import Path


class CompactOutputTests(unittest.TestCase):
    def test_image_filter_does_not_change_metrics(self):
        import torch
        from unittest.mock import patch
        from gaussian_jscc.render_validation import validate_render
        from test_progressive_prefix import progressive
        from test_render_first import synthetic_render, RenderFirstTests, Reference
        raw, geometry, features, model = progressive(8)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
            common = (model, [features[None]], [torch.arange(8)[None]], raw, geometry,
                      RenderFirstTests().cameras(), Reference(), 10, 'awgn', 2, 42)
            full = validate_render(*common, Path(tmp)/'full', 0, 'joint')
            compact = validate_render(*common, Path(tmp)/'compact', 0, 'joint',
                                      image_views=1, image_layouts=('mixed', '1'))
            none = validate_render(*common, Path(tmp)/'compact', 4, 'joint', save_images=False)
            self.assertEqual(full['layouts'], compact['layouts'])
            self.assertEqual(compact['layouts'], none['layouts'])
            self.assertEqual(len(compact['saved_images']), 2)
            self.assertEqual({r['layout'] for r in compact['saved_images']}, {'mixed', '1'})
            self.assertFalse((Path(tmp)/'compact/validation_images/000004').exists())

    def test_archive_excludes_weights_arrays_and_outside_links(self):
        from gaussian_jscc import compact_output
        self.assertTrue(hasattr(compact_output, 'build_review'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage in ('initial', 'final'):
                folder = root/'evaluation'/stage
                folder.mkdir(parents=True)
                (folder/'evaluation.json').write_text('{}')
                (folder/'results.csv').write_text('psnr\n20\n')
            (root/'complete.json').write_text('{}')
            (root/'loss.jsonl').write_text('{}\n')
            (root/'checkpoints').mkdir()
            (root/'checkpoints'/'large.pt').write_bytes(b'weights')
            (root/'probabilities.npy').write_bytes(b'array')
            archive = compact_output.build_review(root)
            with zipfile.ZipFile(archive) as z:
                self.assertIn('loss.jsonl', z.namelist())
                self.assertIn('evaluation/final/results.csv', z.namelist())
                self.assertFalse(any(n.endswith(('.pt', '.npy')) for n in z.namelist()))
            manifest = json.loads((root/'review_manifest.json').read_text())
            self.assertGreater(manifest['archive_bytes'], 0)

    def test_incomplete_run_is_not_packaged(self):
        from gaussian_jscc.compact_output import build_review
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'must complete'):
                build_review(tmp)


if __name__ == '__main__':
    unittest.main()
