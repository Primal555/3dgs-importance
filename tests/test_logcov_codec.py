"""SPD representation, no-eigenvector gradients, codec boundaries and replay."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
import test_learned_joint as baseline_tests
from test_learned_joint import setup as old_setup
from test_gaussian_jscc import fixture
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.covariance import (source_logcov, pack_symmetric, unpack_symmetric,
    covariance_exp, scene_covariance, export_raw, is_covariance_scene, matrix_quaternion)
from gaussian_jscc.data import (to_features, to_raw, to_scene, fit_feature_statistics,
    prepare, write_ply)
from gaussian_jscc.spatial_response import spatial_response_loss
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.optimization import clip_codec_gradients


def setup(n=17):
    raw, g, _, old = old_setup(n)
    raw[:,4:7] = torch.tensor([-1.,-2.,-4.])+torch.randn(n,3)*.1
    raw[:,7:11] = F.normalize(torch.randn(n,4),dim=-1)
    cfg = old.cfg.to_dict()
    cfg['architecture'] = 'learned_split_logcov'
    m = GaussianCodec(CodecConfig.from_dict(cfg))
    fit_feature_statistics(raw, m)
    f, _ = to_features(raw, g, m)
    return raw, g, f, m


class LogcovTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_representation_roundtrip_all_sh_degrees(self):
        for degree in range(4):
            raw, _ = fixture(16, degree)
            raw[:,4:7] = torch.tensor([-1.,-2.,-4.])
            raw[:,7:11] = F.normalize(torch.randn(16,4),dim=-1)
            m = GaussianCodec(CodecConfig(architecture='learned_split_logcov',sh_degree=degree,
                                          hidden=16,depth=1,decoder_window=4))
            fit_feature_statistics(raw,m)
            raw,g,_ = prepare(raw,16)
            f,_ = to_features(raw,g,m)
            scene = to_scene(f,g,m)
            self.assertEqual(scene.shape[-1],raw.shape[-1]-1)
            self.assertTrue(is_covariance_scene(scene))
            torch.testing.assert_close(scene_covariance(scene),scene_covariance(raw),rtol=1e-4,atol=2e-7)
            restored = export_raw(scene)
            torch.testing.assert_close(scene_covariance(restored),scene_covariance(scene),rtol=2e-4,atol=3e-7)
            torch.testing.assert_close(restored[:,:4],raw[:,:4])
            torch.testing.assert_close(restored[:,11:],raw[:,11:])
            torch.testing.assert_close(covariance_exp(pack_symmetric(source_logcov(raw))),
                                       scene_covariance(raw),rtol=1e-4,atol=2e-7)

    def test_spherical_prediction_has_directional_gradient(self):
        raw,_,_,_ = setup(1)
        raw[:,4:7] = torch.tensor([-1.,-2.,-4.])
        raw[:,7:11] = F.normalize(torch.tensor([[.9,.2,.3,.1]]),dim=-1)
        target = source_logcov(raw).double()
        packed = pack_symmetric(-4*torch.eye(3,dtype=torch.double)[None]).requires_grad_()
        loss = (unpack_symmetric(packed)-target).square().mean()
        loss.backward()
        self.assertGreater(float(packed.grad[:,[1,2,4]].norm()),.1)
        # The rendering map itself must differentiate at repeated eigenvalues.
        sphere = pack_symmetric(-4*torch.eye(3,dtype=torch.double)[None]).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(covariance_exp,(sphere,),eps=1e-6,atol=1e-5))
        self.assertTrue(torch.isfinite(sphere).all())

    def test_export_is_never_an_autograd_path(self):
        _,g,f,m = setup(8)
        pred = f.clone().requires_grad_()
        with self.assertRaisesRegex(RuntimeError,'inference-only'):
            to_raw(pred,g,m)
        to_scene(pred,g,m)[:,:3].sum().backward()
        self.assertGreater(float(pred.grad[:,:3].norm()),0)
        with torch.no_grad():
            self.assertEqual(to_raw(pred,g,m).shape[-1],14)
        # 180-degree rotations and identity must survive export's branch choice.
        rotations = torch.stack((torch.eye(3),torch.diag(torch.tensor([1.,-1.,-1.]))))
        from gaussian_jscc.covariance import quaternion_matrix
        torch.testing.assert_close(quaternion_matrix(matrix_quaternion(rotations)),rotations)

    def test_transport_q0_and_packet_contracts(self):
        for name in ('test_no_hand_geometry_or_split_budget',
                     'test_power_budget_empty_windows_and_singletons',
                     'test_new_receiver_has_only_packet_and_weights',
                     'test_drop_inputs_do_not_leak_into_other_outputs',
                     'test_discrete_joint_omits_zero_rows_and_backpropagates'):
            with self.subTest(name=name),patch('test_learned_joint.setup',setup):
                getattr(baseline_tests.LearnedTests(name),name)()
        from gaussian_jscc.transport import transmit,receive
        from gaussian_jscc.rendering import hybrid_parameter_scenes
        raw,g,f,m=setup(8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'packet'
            transmit(m,raw,torch.ones(8,dtype=torch.long),10,'none',42,path)
            native=receive(m,path,native_scene=True)
            ply=receive(m,path)
            torch.testing.assert_close(scene_covariance(native),scene_covariance(ply),rtol=1e-4,atol=1e-6)
            hybrids=hybrid_parameter_scenes(native,raw)
            self.assertTrue(is_covariance_scene(hybrids['attribute_error_only']))
            self.assertFalse(is_covariance_scene(hybrids['position_error_only']))
            with self.assertRaisesRegex(ValueError,'same SH degree'):
                hybrid_parameter_scenes(native,torch.zeros(8,59))
            empty=Path(tmp)/'empty'
            transmit(m,raw,torch.zeros(8,dtype=torch.long),10,'none',42,empty)
            self.assertEqual(receive(m,empty).shape,(0,14))
            self.assertEqual(receive(m,empty,native_scene=True).shape,(0,13))

    def test_noise_batched_paths_and_all_heads_receive_gradients(self):
        _,g,f,m = setup(8)
        q = torch.tensor([1,0,2,3,1,0,2,3])
        torch.manual_seed(8)
        pred = m(f,f[:,:3],q,10,'awgn')
        torch.manual_seed(8)
        batched = m.forward_tier_batches(f[None],f[None,:,:3],F.one_hot(q,4).float()[None],10,'awgn')[0][0]
        torch.testing.assert_close(pred,batched,rtol=2e-5,atol=1e-6)
        # Forbid eigendecomposition of eigenvectors anywhere in training.
        with patch('torch.linalg.eigh',side_effect=AssertionError('training eigh')):
            loss,stats = spatial_response_loss(pred[q>0],f[q>0],g,m,directions=torch.eye(3))
            loss.backward()
        self.assertIn('spatial_logcov_shape_mse',stats)
        self.assertNotIn('spatial_native_shape_response',stats)
        self.assertIn('decoded_anisotropy_p50',stats)
        for head in ('xyz','opacity','logcov','dc'):
            gradient = m.learned.heads[head].weight.grad
            self.assertTrue(torch.isfinite(gradient).all(),head)
            self.assertGreater(float(gradient.norm()),0,head)
        _,groups = clip_codec_gradients(m,mode='none')
        self.assertIn('covariance_head',groups['gradient_groups'])

    def test_direct_replay_checkpoint_covariance_gradients(self):
        _,g,f,m = setup(8)
        batches = [(f[None],torch.tensor([[1,0,2,3,1,0,2,3]]))]
        results=[]
        for mode in ('direct','replay','checkpoint'):
            model=copy.deepcopy(m)
            torch.manual_seed(123)
            def task(scene):
                self.assertTrue(is_covariance_scene(scene))
                return scene.square().mean()
            with patch('torch.linalg.eigh',side_effect=AssertionError('training eigh')):
                loss,_=full_scene_step(model,batches,g,10,'awgn',task,attr_weight=.1,mode=mode)
            results.append((loss,[p.grad for p in model.parameters()]))
        for result in results[1:]:
            torch.testing.assert_close(results[0][0],result[0])
            for a,b in zip(results[0][1],result[1]):
                if a is not None:
                    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)

    def test_renderer_consumes_covariance_directly(self):
        from gaussian_jscc.rendering import render
        _,g,f,m=setup(8)
        f.requires_grad_()
        scene=to_scene(f,g,m)
        camera=SimpleNamespace(image_height=2,image_width=2,FoVx=1.,FoVy=1.,
                               world_view_transform=torch.eye(4),full_proj_transform=torch.eye(4),
                               camera_center=torch.zeros(3))
        called=[]
        class Rasterizer:
            def __init__(self,**kwargs): pass
            def __call__(self,**kwargs):
                called.append(kwargs)
                return kwargs['cov3D_precomp'].sum().expand(3,2,2),None
        module=SimpleNamespace(GaussianRasterizer=Rasterizer,GaussianRasterizationSettings=lambda **kw:kw)
        with patch.dict('sys.modules',{'diff_gaussian_rasterization':module}), \
             patch('torch.linalg.eigh',side_effect=AssertionError('training eigh')):
            render(scene,camera,0).mean().backward()
        self.assertIsNone(called[0]['scales'])
        self.assertIsNone(called[0]['rotations'])
        torch.testing.assert_close(called[0]['cov3D_precomp'],scene[:,4:10])
        self.assertGreater(float(f.grad[:,4:10].norm()),0)

    @unittest.skipUnless(torch.cuda.is_available(),'requires CUDA rasterizer for image/gradient equivalence')
    def test_cuda_actual_rasterizer_matches_source_and_backpropagates(self):
        try:
            import diff_gaussian_rasterization
        except ImportError:
            self.skipTest('CUDA rasterizer not installed')
        from gaussian_jscc.rendering import render
        from utils.graphics_utils import getProjectionMatrix
        raw,_,_,m=setup(8)
        raw=raw.cuda()
        raw[:,:2]-=.5
        raw[:,2]+=2
        from gaussian_jscc.data import Geometry
        g=Geometry.fit(raw[:,:3],16)
        m=m.cuda()
        f,_=to_features(raw,g,m)
        f=f.detach().requires_grad_()
        scene=to_scene(f,g,m)
        camera=SimpleNamespace(image_height=32,image_width=32,FoVx=1.,FoVy=1.,
                               world_view_transform=torch.eye(4,device='cuda'),
                               full_proj_transform=getProjectionMatrix(.01,100.,1.,1.).T.cuda(),
                               camera_center=torch.zeros(3,device='cuda'))
        reference=render(raw,camera,0).detach()
        with patch('torch.linalg.eigh',side_effect=AssertionError('training eigh')):
            image=render(scene,camera,0)
            torch.testing.assert_close(image,reference,rtol=2e-3,atol=2e-4)
            image.square().mean().backward()
        self.assertTrue(torch.isfinite(f.grad).all())
        self.assertGreater(float(f.grad[:,4:10].norm()),0)

    def test_short_random_fit_and_cli_checkpoint(self):
        torch.manual_seed(42)
        raw,g,f,m=setup(8)
        q=torch.arange(8)%3+1
        optimizer=torch.optim.Adam(m.parameters(),lr=2e-4)
        def objective():
            pred=m(f,f[:,:3],q,10,'none')
            return spatial_response_loss(pred,f,g,m,directions=torch.eye(3))[0]
        initial=float(objective().detach())
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss=objective()
            loss.backward()
            clip_codec_gradients(m,mode='none')
            optimizer.step()
        self.assertLess(float(objective().detach()),initial*.9)
        from gaussian_jscc.cli import main
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train','--architecture','learned_split_logcov',
                  '--ply',str(root/'input.ply'),'--out',str(root/'run'),'--device','cpu',
                  '--bootstrap-objective','spatial-response','--bootstrap-steps','2',
                  '--render-steps','0','--hidden','16','--depth','1','--block-size','4',
                  '--decoder-window','4','--blocks-per-batch','1','--validation-blocks','1',
                  '--validation-trials','1','--validate-every','1','--save-every','1']
            with patch('sys.argv',argv),patch('gaussian_jscc.plots.safe_plot'):
                main()
            model=load_checkpoint(root/'run/codec.pt','cpu')
            self.assertEqual(model.cfg.architecture,'learned_split_logcov')
            config=json.loads((root/'run/training.json').read_text())
            self.assertEqual(config['objective'],'spatial_logcov_v1')
            logs=[json.loads(s) for s in (root/'run/loss.jsonl').read_text().splitlines()]
            self.assertIn('covariance_head',logs[-1]['gradient_groups'])
            from gaussian_jscc.plots import plot_training
            plot_training(root/'run')
            self.assertTrue((root/'run/charts/bootstrap_logcov_shape.png').exists())

    def test_checkpoint_history_keeps_native_covariance_and_nested_output(self):
        from test_checkpoint_history import HistoryTests
        from test_render_first import synthetic_render
        def checked_render(scene,*args):
            # Source reference is ordinary PLY, received scene is covariance.
            if scene.shape[-1] == 13:
                self.assertTrue(is_covariance_scene(scene))
            return synthetic_render(scene,*args)
        with patch('test_checkpoint_history.setup',setup), \
             patch('test_checkpoint_history.synthetic_render',side_effect=checked_render):
            case=HistoryTests('test_fixed_noise_images_metrics_and_readonly_weights')
            case.test_fixed_noise_images_metrics_and_readonly_weights()


if __name__=='__main__':
    unittest.main()
