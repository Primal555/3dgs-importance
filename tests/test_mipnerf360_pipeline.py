import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from plyfile import PlyData
from test_prune_initialization import fixture_ply


class MipPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.interrupt_pruning = False
        from gaussian_jscc.successive_transfer import NEW_TRAIN, NEW_HELDOUT, OLD_TRAIN
        self.names = NEW_TRAIN + NEW_HELDOUT
        self.old = []
        for name in OLD_TRAIN + self.names:
            source = self.root / name
            sparse = source / 'sparse/0'
            sparse.mkdir(parents=True)
            (source / 'images').mkdir()
            (sparse / 'cameras.txt').write_text('1 PINHOLE 16 16 12 12 8 8\n')
            (sparse / 'images.txt').write_text('1 1 0 0 0 0 0 0 1 a.png\n\n2 1 0 0 0 1 0 0 1 b.png\n\n')
            (sparse / 'points3D.txt').write_text('1 0 0 0 255 255 255 0\n')
            for photo in ('a.png', 'b.png'):
                Image.new('RGB', (16, 16)).save(source / 'images' / photo)
            fixture_ply(source / 'original.ply', degree=1)
            self.old.append(dict(name=name, role='train', source=str(source), ply=str(source / 'original.ply')))
        self.manifest = self.root / 'raw.json'
        self.manifest.write_text(json.dumps(dict(scenes=self.old[3:])))
        self.source = self.root / 'source'
        self.source.mkdir()
        self.out = self.root / 'pipeline'

    def entry(self):
        from utils import mipnerf360_pipeline
        return mipnerf360_pipeline

    def options(self):
        return self.entry().parser().parse_args(['--run', str(self.source), '--out', str(self.out),
                                                '--raw-manifest', str(self.manifest)])

    def source_result(self, run):
        return {}, dict(selected_scenes=self.old[:3]), SimpleNamespace(sh_degree=1), dict(source_sha256='fixture')

    def fake_prune(self, plan, gpu):
        from utils.maskgaussian_batch import run_plan
        def runner(command, **kwargs):
            if self.interrupt_pruning and 'garden' in command[command.index('-m') + 1]:
                raise RuntimeError('simulated pruning interruption')
            directory = Path(command[command.index('-m') + 1])
            if command[1].endswith('prune_finetune.py'):
                ply = directory / 'point_cloud/iteration_35000/point_cloud.ply'
                ply.parent.mkdir(parents=True)
                fixture_ply(ply, degree=1, count=1)
                (directory / 'baseline_metrics.json').write_text(json.dumps(dict(views=1, PSNR=26., SSIM=.9)))
            if command[1].endswith('metrics.py'):
                (directory / 'results.json').write_text(json.dumps({'ours_35000':dict(PSNR=25., SSIM=.9, LPIPS=.1)}))
        with patch('utils.maskgaussian_batch.subprocess.run', side_effect=runner):
            return run_plan(plan, gpu)

    def fake_train(self, command, **kwargs):
        self.command = command
        training = self.out / 'training'
        training.mkdir(exist_ok=True)
        (training / 'complete.json').write_text(json.dumps(dict(status='complete', step=11250)))

    def test_dry_run_checks_every_scene_without_creating_output(self):
        entry = self.entry()
        args = self.options(); args.dry_run = True
        with patch.object(entry, 'load_source', side_effect=self.source_result), contextlib.redirect_stdout(io.StringIO()):
            plan = entry.main(args)
        self.assertEqual(plan['total_updates'], 11250)
        self.assertEqual(len(plan['preprocessing']['scenes']), 9)
        self.assertFalse(self.out.exists())
        (self.root / 'room/images/b.png').unlink()
        with patch.object(entry, 'load_source', side_effect=self.source_result), self.assertRaises(FileNotFoundError):
            entry.main(args)
        self.assertFalse(self.out.exists())

    def test_pipeline_roles_balanced_command_and_reuse_completed_pruning(self):
        entry = self.entry(); args = self.options(); self.interrupt_pruning = True
        with patch.object(entry, 'load_source', side_effect=self.source_result), \
             patch.dict('os.environ', CUDA_VISIBLE_DEVICES='2'), \
             patch.object(entry, 'run_plan', side_effect=self.fake_prune), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                entry.main(args)
            self.assertFalse((self.out / 'dataset.json').exists())
            self.assertEqual(json.loads((self.out / 'progress.json').read_text())['status'], 'failed')
            self.interrupt_pruning = False; args.resume = True
            with patch.object(entry, 'run_training', side_effect=self.fake_train):
                result = entry.main(args)
        self.assertEqual(result['status'], 'complete')
        first = json.loads((self.out / 'preprocessing/pass_000/plan.json').read_text())
        second = json.loads((self.out / 'preprocessing/pass_001/plan.json').read_text())
        self.assertEqual(first['scenes'][0]['name'], 'bicycle')
        self.assertNotIn('bicycle', [s['name'] for s in second['scenes']])
        dataset = json.loads((self.out / 'dataset.json').read_text())['scenes']
        self.assertEqual(len(dataset), 12)
        self.assertEqual([s['name'] for s in dataset if s['role'] == 'heldout'], ['flowers', 'treehill', 'room'])
        for s in dataset[:3]:
            self.assertEqual(s, self.old[['train', 'truck', 'playroom'].index(s['name'])])
        self.assertIn('gaussian_jscc.successive_transfer', self.command)
        self.assertEqual(self.command[self.command.index('--steps-per-scene') + 1], '1250')
        self.assertEqual(self.command[self.command.index('--lr') + 1], '1e-05')
        self.assertNotIn('--resume', self.command)

    def test_incompatible_degree_fails_before_gpu_or_output(self):
        entry = self.entry(); args = self.options()
        fixture_ply(self.root / 'room/original.ply', degree=2)
        with patch.object(entry, 'load_source', side_effect=self.source_result), \
             patch.object(entry, 'run_plan') as runner, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'SH degree'):
                entry.main(args)
        runner.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_resume_rejects_same_size_input_and_export_content_changes(self):
        entry = self.entry(); args = self.options(); self.interrupt_pruning = True
        with patch.object(entry, 'load_source', side_effect=self.source_result), \
             patch.dict('os.environ', CUDA_VISIBLE_DEVICES='2'), \
             patch.object(entry, 'run_plan', side_effect=self.fake_prune), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                entry.main(args)
            args.resume = True
            # Same count, properties and bytes: content hash must still catch it.
            original = self.root / 'bicycle/original.ply'
            data = original.read_bytes()
            ply = PlyData.read(str(original), mmap=False); ply['vertex'].data['f_dc_0'] += .25
            ply.write(str(original))
            self.assertEqual(len(data), original.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'changed'):
                entry.main(args)
            original.write_bytes(data)
            exported = self.out / 'preprocessing/pass_000/bicycle/point_cloud/iteration_35000/point_cloud.ply'
            ply = PlyData.read(str(exported), mmap=False); ply['vertex'].data['f_dc_1'] += .25
            ply.write(str(exported))
            with self.assertRaisesRegex(ValueError, 'export changed'):
                entry.main(args)

    def test_training_resume_passes_exact_resume_flag(self):
        entry = self.entry(); args = self.options()
        def interrupted(command, **kwargs):
            latest = self.out / 'training/checkpoints/latest/codec.pt'
            latest.parent.mkdir(parents=True)
            latest.write_bytes(b'fake checkpoint: restoration itself is tested on a real codec')
            raise RuntimeError('simulated codec interruption')
        with patch.object(entry, 'load_source', side_effect=self.source_result), \
             patch.dict('os.environ', CUDA_VISIBLE_DEVICES='2'), \
             patch.object(entry, 'run_plan', side_effect=self.fake_prune), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(entry, 'run_training', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'codec interruption'):
                    entry.main(args)
            args.resume = True
            with patch.object(entry, 'run_training', side_effect=self.fake_train):
                entry.main(args)
        self.assertIn('--resume', self.command)


if __name__ == '__main__':
    unittest.main()
