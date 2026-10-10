import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from utils.prune_initialization import initialize_from_ply, read_prune_parameters, quantized_photo_metrics


def fixture_ply(path, degree=1, count=2, drop=None, nonfinite=False):
    names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    # Deliberately not numeric order in the file header.
    names += list(reversed([f'f_rest_{i}' for i in range(3*((degree+1)**2-1))]))
    names += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    if drop:
        names.remove(drop)
    vertex = np.zeros(count, dtype=[(n, 'f4') for n in names])
    for j, name in enumerate(names):
        vertex[name] = j+1
    for name in names:
        if name.startswith('f_rest_'):
            vertex[name] = int(name.rsplit('_', 1)[1])+100
    if nonfinite and count:
        vertex['x'][0] = np.nan
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(str(path))
    return vertex


class PruneInitializationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'input.ply'

    def test_parameters_are_exact_and_sh_channels_are_not_scrambled(self):
        source = fixture_ply(self.path)
        original = self.path.read_bytes()
        tensors = read_prune_parameters(self.path, 1, 'cpu')
        np.testing.assert_array_equal(tensors['_xyz'].numpy(), np.stack([source[n] for n in ('x','y','z')], 1))
        np.testing.assert_array_equal(tensors['_features_dc'][0].numpy(), [[4,5,6]])
        np.testing.assert_array_equal(tensors['_features_rest'][0].numpy(), [[100,103,106], [101,104,107], [102,105,108]])
        np.testing.assert_array_equal(tensors['_opacity'].numpy()[:,0], source['opacity'])
        np.testing.assert_array_equal(tensors['_scaling'].numpy()[:,0], source['scale_0'])
        np.testing.assert_array_equal(tensors['_rotation'].numpy()[:,0], source['rot_0'])
        torch.testing.assert_close(tensors['_mask_score'], torch.tensor([[10.,1.], [10.,1.]]))
        self.assertEqual(self.path.read_bytes(), original)

    def test_sh_degree_zero_has_empty_rest(self):
        fixture_ply(self.path, degree=0)
        self.assertEqual(read_prune_parameters(self.path, 0)['_features_rest'].shape, (2,0,3))

    def test_invalid_inputs_are_rejected(self):
        for kwargs in ({'drop':'rot_3'}, {'nonfinite':True}, {'count':0}, {'degree':0}):
            with self.subTest(kwargs=kwargs):
                fixture_ply(self.path, **kwargs)
                with self.assertRaises(ValueError):
                    read_prune_parameters(self.path, 1)

    def test_initializer_uses_fresh_trainable_parameters_and_scene_extent(self):
        fixture_ply(self.path)
        # CPU facade at the CUDA model boundary; optimizer itself is real Adam.
        model = SimpleNamespace(max_sh_degree=1, optimizer=None)
        def setup(_):
            model.optimizer = torch.optim.Adam([getattr(model, key) for key in
                ('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_mask_score')])
        model.training_setup = setup
        initialize_from_ply(model, self.path, object(), 2.5, device='cpu')
        self.assertEqual(model.spatial_lr_scale, 2.5)
        self.assertEqual(model.active_sh_degree, 1)
        self.assertEqual(len(model.optimizer.state), 0)
        self.assertTrue(model._xyz.requires_grad)
        self.assertEqual(model.max_radii2D.shape, (2,))
        model._xyz.square().mean().backward()
        model.optimizer.step()
        self.assertEqual(len(model.optimizer.state), 1)

    def test_photo_psnr_uses_all_rgb_channels_like_metrics_py(self):
        source = torch.zeros(3, 16, 16)
        received = torch.tensor([.1,.2,.8]).view(3,1,1).expand_as(source)
        metrics = quantized_photo_metrics(received, source)
        quantized = (received*255+.5).to(torch.uint8).float()/255
        expected = -10*torch.log10(quantized.square().mean()).item()
        self.assertAlmostEqual(metrics['PSNR'], expected, places=5)

    @unittest.skipUnless(torch.cuda.is_available(), 'actual GaussianModel initializer needs CUDA extensions')
    def test_real_gaussian_model_has_fresh_optimizer_and_differentiable_params(self):
        from argparse import ArgumentParser
        from arguments import OptimizationParams
        from scene.gaussian_model import GaussianModel
        fixture_ply(self.path)
        parser = ArgumentParser()
        params = OptimizationParams(parser)
        model = GaussianModel(1)
        initialize_from_ply(model, self.path, params.extract(parser.parse_args([])), 2.5)
        self.assertFalse(model.optimizer.state)
        self.assertEqual(len(model.optimizer.param_groups), 7)
        model._xyz.square().mean().backward()
        model.optimizer.step()
        self.assertTrue(model.optimizer.state)


if __name__ == '__main__':
    unittest.main()
