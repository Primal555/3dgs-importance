import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from utils.maskgaussian_batch import build_plan, run_plan
from test_prune_initialization import fixture_ply


class MaskGaussianBatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scenes = []
        for name in ('truck','train','playroom','drjohnson'):
            source = self.root/name
            (source/'sparse'/'0').mkdir(parents=True)
            (source/'images').mkdir()
            (source/'sparse'/'0'/'cameras.txt').write_text('1 PINHOLE 16 16 12 12 8 8\n')
            (source/'sparse'/'0'/'images.txt').write_text('1 1 0 0 0 0 0 0 1 a.png\n\n2 1 0 0 0 1 0 0 1 b.png\n\n')
            (source/'sparse'/'0'/'points3D.txt').write_text('1 0 0 0 255 255 255 0\n')
            Image.new('RGB', (16,16)).save(source/'images'/'a.png')
            Image.new('RGB', (16,16)).save(source/'images'/'b.png')
            fixture_ply(source/'original.ply', degree=1)
            scenes.append({'name':name, 'role':'heldout' if name=='drjohnson' else 'train',
                           'source':name, 'ply':f'{name}/original.ply'})
        self.manifest = self.root/'scenes.json'
        self.manifest.write_text(json.dumps({'scenes':scenes}))
        self.out = self.root/'pruned'

    def plan(self, **kwargs):
        return build_plan(self.manifest, self.out, **kwargs)

    def test_plan_preserves_inputs_and_runs_three_scenes_sequentially(self):
        plan = self.plan()
        self.assertEqual([s['name'] for s in plan['scenes']], ['train','truck','playroom'])
        self.assertFalse(self.out.exists())
        for s in plan['scenes']:
            command = s['commands'][0]
            self.assertIn('--start_pointcloud', command)
            self.assertNotIn('--start_checkpoint', command)
            self.assertEqual(command[command.index('--iterations')+1], '35000')
            self.assertEqual(command[command.index('--data_device')+1], 'cpu')
            self.assertEqual(command[command.index('--sh_degree')+1], '1')
            self.assertNotEqual(s['input_ply'], s['output_ply'])
            self.assertEqual(len(s['commands']), 3)

    def test_missing_scene_fails_before_output_is_created(self):
        (self.root/'playroom'/'original.ply').unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()
        self.assertFalse(self.out.exists())

    def test_invalid_later_scene_values_and_camera_files_fail_preflight(self):
        fixture_ply(self.root/'playroom'/'original.ply', nonfinite=True)
        with self.assertRaises(ValueError):
            self.plan()
        fixture_ply(self.root/'playroom'/'original.ply')
        (self.root/'playroom'/'images'/'b.png').unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()
        Image.new('RGB', (16,16)).save(self.root/'playroom'/'images'/'b.png')
        (self.root/'playroom'/'sparse'/'0'/'cameras.txt').unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()
        self.assertFalse(self.out.exists())

    def test_existing_output_and_invalid_options_are_rejected(self):
        for options in ({'steps':0}, {'steps':-1}, {'lambda_mask':-1}, {'scenes':['train','train']}, {'scenes':['unknown']}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.plan(**options)
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_pipeline_publishes_manifest_only_after_metrics_and_exports_exist(self):
        plan = self.plan()
        calls = []
        def runner(command, **kwargs):
            calls.append(command[1])
            scene_dir = Path(command[command.index('-m')+1])
            if command[1].endswith('prune_finetune.py'):
                output = scene_dir/'point_cloud'/'iteration_35000'/'point_cloud.ply'
                output.parent.mkdir(parents=True)
                fixture_ply(output, degree=1, count=1)
                (scene_dir/'baseline_metrics.json').write_text(json.dumps({'PSNR':26.0,'SSIM':.91,'views':1}))
            if command[1].endswith('metrics.py'):
                (scene_dir/'results.json').write_text(json.dumps({'ours_35000':{'PSNR':25.0,'SSIM':.9,'LPIPS':.1}}))
        with patch('utils.maskgaussian_batch.subprocess.run', side_effect=runner):
            result = run_plan(plan, gpu='2')
        self.assertEqual(calls, [str(Path(plan['repo'])/f) for f in ('prune_finetune.py','render.py','metrics.py')]*3)
        manifest = json.loads(Path(result['manifest']).read_text())
        self.assertEqual(len(manifest['scenes']), 3)
        self.assertTrue(all(Path(s['ply']).is_file() for s in manifest['scenes']))
        self.assertEqual(result['scenes'][0]['retained_fraction'], .5)
        self.assertEqual(result['scenes'][0]['metrics']['PSNR'], 25.0)
        self.assertEqual(result['scenes'][0]['psnr_change_db'], -1.0)

    def test_missing_baseline_does_not_publish_complete_manifest(self):
        plan = self.plan(scenes=['train'])
        def runner(command, **kwargs):
            scene_dir = Path(plan['scenes'][0]['output_dir'])
            if command[1].endswith('prune_finetune.py'):
                output = Path(plan['scenes'][0]['output_ply'])
                output.parent.mkdir(parents=True)
                fixture_ply(output, degree=1)
            if command[1].endswith('metrics.py'):
                (scene_dir/'results.json').write_text(json.dumps({'ours_35000':{'PSNR':25.,'SSIM':.9,'LPIPS':.1}}))
        with patch('utils.maskgaussian_batch.subprocess.run', side_effect=runner):
            with self.assertRaises(FileNotFoundError):
                run_plan(plan, gpu='2')
        self.assertFalse((self.out/'multiscene_pruned.json').exists())

    def test_silent_metrics_failure_does_not_publish_complete_manifest(self):
        plan = self.plan(scenes=['train'])
        def runner(command, **kwargs):
            if command[1].endswith('prune_finetune.py'):
                output = Path(plan['scenes'][0]['output_ply'])
                output.parent.mkdir(parents=True)
                fixture_ply(output, degree=1)
        with patch('utils.maskgaussian_batch.subprocess.run', side_effect=runner):
            with self.assertRaises((FileNotFoundError, ValueError)):
                run_plan(plan, gpu='1')
        self.assertFalse((self.out/'multiscene_pruned.json').exists())
        self.assertEqual(json.loads((self.out/'summary.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
