"""CPU checks for the common digital-baseline evaluation contract."""
import hashlib
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from gaussian_jscc.data import write_ply


class PairDistance(torch.nn.Module):
    def forward(self, image, target):
        if self.training or torch.is_grad_enabled():
            raise AssertionError('LPIPS must run in frozen evaluation mode')
        return (image - target).square().mean()


def cameras():
    return [SimpleNamespace(image_name=f'photo_{i}', uid=i,
                            image_width=16, image_height=16,
                            original_image=torch.full((3, 16, 16), .03 * i),
                            world_view_transform=torch.eye(4),
                            full_proj_transform=torch.eye(4), camera_center=torch.zeros(3),
                            FoVx=1., FoVy=1.) for i in range(5)]


def synthetic_renderer(raw, camera, degree, white_background=False):
    if torch.is_grad_enabled():
        raise AssertionError('rendering must run without autograd')
    if degree != (1 if len(raw) == 3 else 0):
        raise AssertionError('each PLY must retain its own SH degree')
    step = .05 if len(raw) == 3 else .1
    return (raw[0, 11] + step * camera.uid).expand(3, 16, 16)


class ComparisonEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, directory):
        root = Path(directory)
        source = root / 'photos'
        source.mkdir()
        paths = []
        for name, count, degree, value in (('source', 3, 1, .2), ('received', 2, 0, .4)):
            raw = torch.zeros(count, 14 + 3 * ((degree + 1) ** 2 - 1))
            raw[:, 7] = 1
            raw[:, 11] = value
            path = root / (name + '.ply')
            write_ply(path, raw, degree)
            paths.append(path)
        return source, *paths, root / 'evaluation'

    def test_mean_is_per_view_and_nonfinite_or_incomplete_metrics_fail(self):
        from comparison_baselines.evaluate import aggregate_view_metrics
        rows = [{'view': 'a', 'source_psnr': 10., 'photo_psnr': 20.},
                {'view': 'b', 'source_psnr': 30., 'photo_psnr': 24.}]
        self.assertEqual(aggregate_view_metrics(rows), {'source_psnr': 20., 'photo_psnr': 22.})
        for bad in ([], [{'view': 'a'}], [rows[0], {'view': 'b', 'source_psnr': 30.}],
                    [{'view': 'a', 'source_psnr': math.nan}],
                    [{'view': 'a', 'source_psnr': math.inf}]):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                aggregate_view_metrics(bad)

    def test_same_spaced_views_score_both_independent_plys_and_save_two_named_panels(self):
        from comparison_baselines.evaluate import evaluate_plys
        with tempfile.TemporaryDirectory() as directory:
            source, teacher, received, out = self.fixture(directory)
            with patch('gaussian_jscc.rendering.load_cameras', return_value=cameras()):
                result = evaluate_plys(source, teacher, received, out, views=3, image_views=9,
                                       lpips=False, device='cpu', renderer=synthetic_renderer)
            self.assertEqual([row['view'] for row in result['views']], ['photo_0', 'photo_2', 'photo_4'])
            self.assertEqual(result['identity']['camera_indices'], [0, 2, 4])
            self.assertEqual(result['identity']['source_ply']['count'], 3)
            self.assertEqual(result['identity']['received_ply']['count'], 2)
            self.assertEqual(result['identity']['source_ply']['sh_degree'], 1)
            self.assertEqual(result['identity']['received_ply']['sh_degree'], 0)
            self.assertEqual(result['identity']['source_ply']['sha256'], hashlib.sha256(teacher.read_bytes()).hexdigest())
            self.assertEqual(result['identity']['resolution'], 2)
            self.assertEqual(result['identity']['images'], 'images')
            self.assertFalse(result['identity']['image_metric_protocol']['lpips']['enabled'])
            self.assertAlmostEqual(result['metrics']['source_mse'], (.04 + .09 + .16) / 3, places=6)
            self.assertAlmostEqual(result['metrics']['source_psnr'], sum(-10 * math.log10(v) for v in (.04, .09, .16)) / 3, places=5)
            self.assertAlmostEqual(result['metrics']['photo_mse'], (.16 + .2916 + .4624) / 3, places=6)
            self.assertAlmostEqual(result['metrics']['source_teacher_photo_mse'], (.04 + .0576 + .0784) / 3, places=6)
            self.assertNotIn('source_lpips', result['metrics'])
            saved = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, result)
            panels = sorted(out.glob('*.png'))
            self.assertEqual(len(panels), 2)
            names = []
            for panel in panels:
                with Image.open(panel) as image:
                    names.append(image.info['view'])
            self.assertEqual(names, ['photo_0', 'photo_4'])

    def test_lpips_uses_existing_protocol_and_three_reference_pairs(self):
        from comparison_baselines.evaluate import evaluate_plys
        with tempfile.TemporaryDirectory() as directory:
            source, teacher, received, out = self.fixture(directory)
            with patch('gaussian_jscc.rendering.load_cameras', return_value=cameras()), \
                 patch('gaussian_jscc.perceptual_metrics._load_lpips_model', return_value=PairDistance()):
                result = evaluate_plys(source, teacher, received, out, views=3, image_views=0,
                                       device='cpu', renderer=synthetic_renderer)
            self.assertEqual(result['identity']['image_metric_protocol']['lpips']['network'], 'vgg')
            self.assertEqual(result['identity']['image_metric_protocol']['lpips']['version'], '0.1')
            self.assertAlmostEqual(result['metrics']['source_lpips'], 4 * (.04 + .09 + .16) / 3, places=6)
            self.assertAlmostEqual(result['metrics']['photo_lpips'], 4 * (.16 + .2916 + .4624) / 3, places=6)
            self.assertAlmostEqual(result['metrics']['source_teacher_photo_lpips'], 4 * (.04 + .0576 + .0784) / 3, places=6)
            self.assertFalse(list(out.glob('*.png')))

    def test_saved_camera_names_select_exact_views_in_requested_order(self):
        from comparison_baselines.evaluate import evaluate_plys
        for names, indices in ((['photo_1', 'photo_2'], [1, 2]),
                               (['photo_2', 'photo_1'], [2, 1])):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                source, teacher, received, out = self.fixture(directory)
                with patch('gaussian_jscc.rendering.load_cameras', return_value=cameras()):
                    result = evaluate_plys(source, teacher, received, out, views=1, image_views=0,
                                           camera_names=names, lpips=False, device='cpu',
                                           renderer=synthetic_renderer)
                self.assertEqual([row['view'] for row in result['views']], names)
                self.assertEqual(result['identity']['camera_names'], names)
                self.assertEqual(result['identity']['camera_indices'], indices)
                self.assertEqual(result['identity']['camera_selection'], 'explicit_camera_names')

    def test_invalid_missing_or_duplicate_camera_names_fail_without_metrics(self):
        from comparison_baselines.evaluate import evaluate_plys
        for names in ([], ['photo_1', 'absent'], ['photo_1', 'photo_1'], 'photo_1', [1]):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                source, teacher, received, out = self.fixture(directory)
                with patch('gaussian_jscc.rendering.load_cameras', return_value=cameras()), \
                     self.assertRaises(ValueError):
                    evaluate_plys(source, teacher, received, out, camera_names=names,
                                  image_views=0, lpips=False, device='cpu', renderer=synthetic_renderer)
                self.assertFalse((out / 'metrics.json').exists())

    def test_ambiguous_loaded_camera_names_are_rejected(self):
        from comparison_baselines.evaluate import evaluate_plys
        for names in (None, ['photo_1', 'photo_2']):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                source, teacher, received, out = self.fixture(directory)
                loaded = cameras()
                loaded[4].image_name = 'photo_1'
                with patch('gaussian_jscc.rendering.load_cameras', return_value=loaded), \
                     self.assertRaisesRegex(ValueError, 'duplicate'):
                    evaluate_plys(source, teacher, received, out, camera_names=names,
                                  image_views=0, lpips=False, device='cpu', renderer=synthetic_renderer)
                self.assertFalse((out / 'metrics.json').exists())

    def test_cli_reads_camera_name_json_and_uses_the_saved_views(self):
        from comparison_baselines.evaluate import evaluate_plys, main
        with tempfile.TemporaryDirectory() as directory:
            source, teacher, received, out = self.fixture(directory)
            names_file = Path(directory) / 'camera_names.json'
            names_file.write_text(json.dumps(['photo_1', 'photo_2']), encoding='utf-8')
            def evaluate_with_cpu_renderer(**kwargs):
                return evaluate_plys(**kwargs, renderer=synthetic_renderer)
            with patch('gaussian_jscc.rendering.load_cameras', return_value=cameras()), \
                 patch('comparison_baselines.evaluate.evaluate_plys', side_effect=evaluate_with_cpu_renderer), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = main(['--source', str(source), '--source-ply', str(teacher),
                               '--received-ply', str(received), '--out', str(out), '--views', '1',
                               '--image-views', '0', '--no-lpips', '--device', 'cpu',
                               '--camera-names-json', str(names_file)])
            self.assertEqual(result['identity']['camera_names'], ['photo_1', 'photo_2'])

    def test_explicit_cuda_device_is_selected_before_camera_loading(self):
        from comparison_baselines.evaluate import evaluate_plys
        with tempfile.TemporaryDirectory() as directory:
            source, teacher, received, out = self.fixture(directory)
            events = []
            def load_on_current_device(*args, **kwargs):
                self.assertEqual(events, ['cuda:2'])
                raise RuntimeError('stop before unavailable CUDA rendering')
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.set_device', side_effect=lambda device: events.append(str(device))), \
                 patch('gaussian_jscc.rendering.load_cameras', side_effect=load_on_current_device), \
                 self.assertRaisesRegex(RuntimeError, 'stop before unavailable CUDA'):
                evaluate_plys(source, teacher, received, out, image_views=0, lpips=False, device='cuda:2')

    def test_nonfinite_or_mismatched_render_never_writes_success_metrics(self):
        from comparison_baselines.evaluate import evaluate_plys
        for image in (torch.full((3, 16, 16), math.nan), torch.zeros(3, 8, 8)):
            with self.subTest(shape=image.shape), tempfile.TemporaryDirectory() as directory:
                source, teacher, received, out = self.fixture(directory)
                with patch('gaussian_jscc.rendering.load_cameras', return_value=cameras()), \
                     self.assertRaises(ValueError):
                    evaluate_plys(source, teacher, received, out, views=3, image_views=0,
                                  lpips=False, device='cpu', renderer=lambda *args: image)
                self.assertFalse((out / 'metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
