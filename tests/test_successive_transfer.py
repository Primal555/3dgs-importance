"""Expand a completed shared codec without fitting any heldout scene."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.data import read_ply, write_ply
from test_gaussian_jscc import fixture
from test_render_first import synthetic_render
from test_successive_joint import cameras
import test_successive_multiscene as legacy


class SuccessiveTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
            from gaussian_jscc import successive_multiscene
            parent, manifest = legacy.source_and_manifest(cls.root)
            cls.source = cls.root / 'shared'
            options = legacy.SuccessiveMultisceneTests().options(parent, manifest, cls.source)
            options.steps_per_scene = 1
            successive_multiscene.main(options)
        # Select the real three train-role records, not the old DrJohnson holdout.
        records = [s for s in json.loads(manifest.read_text())['scenes'] if s['role'] == 'train']
        train = {'bicycle', 'garden', 'stump', 'bonsai', 'counter', 'kitchen'}
        for index, name in enumerate(('bicycle', 'bonsai', 'counter', 'flowers', 'garden',
                                      'kitchen', 'room', 'stump', 'treehill'), 1):
            raw, _ = fixture(8 + index, 1)
            raw[:, :3] += index * 11
            write_ply(cls.root / (name + '.ply'), raw, 1)
            (cls.root / name).mkdir()
            records.append(dict(name=name, role='train' if name in train else 'heldout',
                                ply=name + '.ply', source=name, prior='none'))
        cls.manifest = cls.root / 'expanded.json'
        cls.manifest.write_text(json.dumps(dict(scenes=records)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.successive_transfer'),
                             'missing continuation from a completed shared codec')
        from gaussian_jscc import successive_transfer
        return successive_transfer

    def options(self, out, steps=1):
        return self.entry().parser().parse_args([
            '--run', str(self.source), '--manifest', str(self.manifest), '--out', str(out),
            '--device', 'cpu', '--steps-per-scene', str(steps), '--validation-views', '1',
            '--validate-every', '1', '--save-every', '1', '--test-views', '2',
            '--render-trials', '1', '--cpu-threads', '1', '--no-lpips'])

    def test_dry_run_selects_nine_train_and_three_unseen_scenes_without_writes(self):
        entry = self.entry()
        out = self.root / 'preflight'
        args = self.options(out, 1250)
        args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()):
            plan = entry.main(args)
        self.assertEqual(plan['total_updates'], 11250)
        self.assertEqual({s['name'] for s in plan['heldout_scene']}, {'flowers', 'treehill', 'room'})
        self.assertEqual(len(plan['selected_scenes']), 9)
        self.assertEqual(plan['settings']['lr'], 1e-5)
        self.assertFalse(out.exists())

    def test_rejects_train_holdout_overlap_and_changed_source_statistics(self):
        entry = self.entry()
        args = self.options(self.root / 'rejected')
        args.dry_run = True
        args.heldout_scenes = ['truck']
        with self.assertRaisesRegex(ValueError, 'heldout|overlap|history'):
            entry.main(args)
        args.heldout_scenes = ['flowers', 'treehill', 'room']
        args.scenes = ['train', 'truck', 'playroom', 'room']
        with self.assertRaisesRegex(ValueError, 'train|heldout|overlap'):
            entry.main(args)
        self.assertFalse(Path(args.out).exists())
        corrupted = self.root / 'corrupt_statistics'
        shutil.copytree(self.source, corrupted)
        path = corrupted / 'checkpoints/final/codec.pt'
        saved = torch.load(path, weights_only=True)
        saved['state_dict']['attr_mean'] = saved['state_dict']['attr_mean'] + .1
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, 'statistics'):
            entry.load_source(corrupted)

    def test_old_test_cameras_survive_new_scene_test_budget(self):
        entry = self.entry()
        saved, metadata, cfg, specs, _, _ = entry.inputs(self.options(self.root / 'split_probe'))
        from gaussian_jscc.model_factory import build_codec
        from gaussian_jscc.successive_multiscene_inputs import make_scene
        model = build_codec(cfg); model.load_state_dict(saved['state_dict'])
        args = self.options(self.root / 'split_probe'); args.test_views = 32
        entry.inputs(args)  # inherit the source geometry/batch settings
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
            raw, degree = read_ply(specs[0]['ply'])
            old = metadata['scene_records'][specs[0]['name']]
            scene = make_scene(specs[0], raw, degree, model, args, Path(args.out), old=old)
        self.assertEqual(scene.record, old)

    def test_rejects_unbalanced_completed_source(self):
        entry = self.entry()
        corrupted = self.root / 'corrupt_counts'
        shutil.copytree(self.source, corrupted)
        path = corrupted / 'checkpoints/final/codec.pt'
        saved = torch.load(path, weights_only=True)
        counts = saved['training']['progress']['scene_completed']
        counts['train'] += 1; counts['truck'] -= 1
        torch.save(saved, path)
        complete_path = corrupted / 'complete.json'
        complete = json.loads(complete_path.read_text())
        complete['scene_completed'] = counts
        complete_path.write_text(json.dumps(complete))
        with self.assertRaisesRegex(ValueError, 'balanced|budget|completed'):
            entry.load_source(corrupted)

    def test_shared_weights_stats_old_splits_and_exact_resume_are_preserved(self):
        entry = self.entry()
        with contextlib.redirect_stdout(io.StringIO()), \
             patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
            original = torch.load(self.source / 'checkpoints/final/codec.pt', weights_only=True)
            out = self.root / 'full'
            result = entry.main(self.options(out))
            self.assertEqual(result['step'], 9)
            self.assertEqual(result['heldout_optimizer_updates'], 0)
            self.assertEqual(set(result['heldout_scenes']), {'flowers', 'treehill', 'room'})
            first = torch.load(out / 'checkpoints/initial/codec.pt', weights_only=True)
            final = torch.load(out / 'checkpoints/final/codec.pt', weights_only=True)
            for name, value in original['state_dict'].items():
                torch.testing.assert_close(first['state_dict'][name], value, rtol=0, atol=0)
            for name in ('attr_mean', 'attr_std'):
                torch.testing.assert_close(final['state_dict'][name], original['state_dict'][name], rtol=0, atol=0)
            for name, record in original['training']['scene_record'].items():
                self.assertEqual(final['training']['scene_record'][name], record)
            for module in ('semantic_encoder', 'semantic_decoder', 'channel_encoder', 'channel_decoder'):
                self.assertTrue(any(not torch.equal(v, final['state_dict'][k]) for k, v in
                                    original['state_dict'].items() if k.startswith(module + '.')), module)
            rows = [json.loads(s) for s in (out / 'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual({r['scene'] for r in rows},
                             {'train', 'truck', 'playroom', 'bicycle', 'garden', 'stump', 'bonsai', 'counter', 'kitchen'})
            for name in ('flowers', 'treehill', 'room'):
                self.assertEqual({p.name for p in (out / 'scenes' / name / 'evaluation').iterdir()}, {'initial', 'final'})
            args = self.options(self.root / 'resumed')
            entry.main(args, max_updates=2)
            args.resume = True
            entry.main(args)
            resumed = torch.load(Path(args.out) / 'checkpoints/final/codec.pt', weights_only=True)
            for k, v in final['state_dict'].items():
                torch.testing.assert_close(v, resumed['state_dict'][k], rtol=0, atol=0)
            for key, state in final['training']['optimizer']['state'].items():
                for k, v in state.items():
                    torch.testing.assert_close(v, resumed['training']['optimizer']['state'][key][k], rtol=0, atol=0)
            summary = json.loads((out / 'comparison/summary.json').read_text())
            self.assertEqual(len(summary['rows']), 12 * 18)
            self.assertTrue((out / 'charts/heldout_quality.png').exists())
            # Heldout targets may change evaluation, never the shared optimizer.
            raw, degree = read_ply(self.root / 'room.ply')
            try:
                changed = raw.clone(); changed[:, 11:14] += 3.
                write_ply(self.root / 'room.ply', changed, degree)
                other_out = self.root / 'changed_heldout'
                entry.main(self.options(other_out))
                other = torch.load(other_out / 'checkpoints/final/codec.pt', weights_only=True)
                for k, v in final['state_dict'].items():
                    torch.testing.assert_close(v, other['state_dict'][k], rtol=0, atol=0)
            finally:
                write_ply(self.root / 'room.ply', raw, degree)


if __name__ == '__main__':
    unittest.main()
