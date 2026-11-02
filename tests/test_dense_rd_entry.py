"""CPU full-codec integration with synthetic rendering and real shared checkpoints."""
import contextlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gaussian_jscc import dense_rd_allocation as entry
from test_budgeted_allocation import cameras, render
import test_successive_multiscene as shared_fixture


def equal_nested(test, a, b):
    if torch.is_tensor(a): torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for k in a: equal_nested(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x,y in zip(a,b): equal_nested(test, x,y)
    else: test.assertEqual(a, b)


class DenseEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gaussian_jscc import successive_multiscene
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            parent, manifest = shared_fixture.source_and_manifest(cls.root)
            checkpoint = parent/'checkpoints/final/codec.pt'
            state = torch.load(checkpoint, weights_only=True)
            state['training']['scene_record']['train_view_names'] += ['extra'+str(i) for i in range(4)]
            torch.save(state, checkpoint)
            metadata_path = parent/'training.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['scene'] = state['training']['scene_record']
            metadata_path.write_text(json.dumps(metadata))
            cls.source = cls.root/'shared'
            args = shared_fixture.SuccessiveMultisceneTests().options(parent, manifest, cls.source)
            args.steps_per_scene = 1
            successive_multiscene.main(args)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def options(self, out):
        return entry.parser().parse_args(['--run',str(self.source),'--scene','truck','--out',str(out),
            '--device','cpu','--allocation-steps','3','--window-views','1','--calibration-views','1',
            '--test-views','2','--trials','1','--image-views','1','--cpu-threads','1',
            '--save-every','1','--validate-every','2','--profile-every','0','--no-lpips'])

    def test_shared_source_dry_run_has_no_render_or_write(self):
        out = self.root/'dry'; args = self.options(out); args.dry_run = True
        with patch('gaussian_jscc.rendering.render', side_effect=AssertionError('dry rendered')), \
             contextlib.redirect_stdout(io.StringIO()):
            plan = entry.main(args)
        self.assertEqual(plan['version'], entry.FAMILY)
        self.assertFalse(out.exists())

    def test_full_frozen_run_resume_equivalence_and_compact_report(self):
        full, split = self.root/'full', self.root/'split'
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            report = entry.main(self.options(full))
            paused = entry.main(self.options(split), max_updates=1)
            self.assertEqual(paused['step'], 1)
            self.assertFalse((split/'complete.json').exists())
            with (split/'metrics/training.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=999, lost_work=True))+'\n')
            args = self.options(split); args.resume = True
            resumed = entry.main(args)
            # A completed resume must remain read-only to the training state.
            entry.main(self.options_with_resume(split))
        read = lambda p: torch.load(p/'checkpoints/latest/state.pt', map_location='cpu', weights_only=True)
        a, b = read(full), read(split)
        for key in ('mask','optimizer','views','controller','best','rng','reference','progress'):
            equal_nested(self, a[key], b[key])
        self.assertNotIn('state_dict', a)
        self.assertTrue(report['progress']['frozen_codec_verified'])
        self.assertEqual(report['rows']['initial']['step'], 0)
        self.assertEqual(resumed['rows']['latest']['step'], 3)
        rows = [json.loads(x) for x in (split/'metrics/training.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in rows], [1,2,3])
        calibration = [json.loads(x) for x in (split/'metrics/control.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in calibration], [0,2,3])
        self.assertEqual(len(list((split/'checkpoints').rglob('*.pt'))), 1)
        with tarfile.open(split/'review.tar.gz') as archive:
            names = archive.getnames()
        self.assertFalse(any('checkpoint' in n for n in names))
        self.assertIn('review_summary.json', names)
        image_folders = {p.parent.parent.name for p in (split/'evaluation').rglob('*.png')}
        self.assertEqual(image_folders, {'initial','best_rd'})
        self.assertFalse((split/'replay').exists())

    def options_with_resume(self, out):
        args = self.options(out); args.resume = True; return args

    def test_interrupted_archive_is_rebuilt_on_resume(self):
        out = self.root/'archive_interruption'
        original_add = tarfile.TarFile.add
        calls = []
        def fail_mid_archive(archive, *args, **kwargs):
            calls.append(1)
            if len(calls) == 2: raise OSError('simulated archive interruption')
            return original_add(archive, *args, **kwargs)
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(tarfile.TarFile, 'add', fail_mid_archive):
                with self.assertRaisesRegex(OSError, 'archive interruption'):
                    entry.main(self.options(out))
            self.assertTrue((out/'complete.json').exists())
            self.assertFalse((out/'review.tar.gz').exists())
            entry.main(self.options_with_resume(out))
        with tarfile.open(out/'review.tar.gz') as archive:
            self.assertIn('review_summary.json', archive.getnames())
            self.assertIn('deployment/latest_tiers.npy', archive.getnames())
        self.assertFalse((out/'review.tar.gz.tmp').exists())


if __name__ == '__main__': unittest.main()
