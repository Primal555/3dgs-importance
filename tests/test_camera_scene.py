"""CPU: real codec/noise/VJP; synthetic renderer, not GPU quality evidence."""
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from test_logcov_codec import setup
from test_render_first import synthetic_render, Reference
from gaussian_jscc.camera_objective import camera_position_loss, CameraInitializationTask
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.render_validation import validate_render


def cameras():
    result=[]
    for factor in (.4,1.3,.8,1.1):
        matrix=torch.eye(4)
        matrix[3,2]=5.
        result.append(SimpleNamespace(factor=factor,world_view_transform=matrix,
                      FoVx=math.pi/2,FoVy=math.pi/2,image_width=64,image_height=64,
                      original_image=torch.full((3,8,8),.5),image_name=str(factor),znear=.01,zfar=100.))
    return result


class CameraSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pixel_units_depth_and_outside_prediction(self):
        c=cameras()[0]
        source=torch.zeros(1,3,requires_grad=True)
        xyz=torch.tensor([[1.,0.,0.]],requires_grad=True)
        loss,stats=camera_position_loss(xyz,source,c)
        self.assertAlmostEqual(float(stats['pixel_error_mean']),6.4,places=5)
        loss.backward()
        self.assertIsNone(source.grad)
        self.assertGreater(float(xyz.grad.norm()),0)
        # Equal ray but incorrect depth still has a gradient.
        along=torch.tensor([[0.,0.,1.]],requires_grad=True)
        loss,stats=camera_position_loss(along,source,c)
        self.assertEqual(float(stats['pixel_loss']),0.)
        self.assertGreater(float(stats['depth_loss']),0)
        loss.backward()
        self.assertGreater(float(along.grad.norm()),0)
        behind=torch.tensor([[100.,0.,-6.]],requires_grad=True)
        loss,stats=camera_position_loss(behind,source,c)
        loss.backward()
        self.assertTrue(torch.isfinite(behind.grad).all())
        self.assertEqual(stats['source_frustum_count'],1)
        empty=torch.tensor([[0.,0.,0.]],requires_grad=True)
        loss,stats=camera_position_loss(empty,torch.tensor([[0.,0.,-10.]]),c)
        loss.backward()
        self.assertEqual(stats['source_frustum_count'],0)
        self.assertEqual(float(empty.grad.norm()),0)

    def test_training_render_teacher_xyz_only_attributes_get_rgb_gradient(self):
        scene=torch.randn(8,13,requires_grad=True)
        source=torch.zeros(8,3,requires_grad=True)
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            task=CameraInitializationTask(cameras()[:2],Reference(),0,source,geometry_weight=0.)
            task.backward_scene(scene)
        self.assertEqual(float(scene.grad[:,:3].norm()),0.)
        self.assertGreater(float(scene.grad[:,3:].norm()),0.)
        self.assertIsNone(source.grad)
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            CameraInitializationTask(cameras()[:2],Reference(),0,source).backward_scene(scene)
        self.assertGreater(float(scene.grad[:,:3].norm()),0.)

    def test_small_random_codec_projection_fit(self):
        raw,g,f,model=setup(16)
        q=torch.tensor([[1,2,3,1,2,3,1,2]*2])
        optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
        batches=[(f[None],q)]
        losses=[]
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            for step in range(30):
                # Matched noisy observations isolate ability to optimize;
                # this deliberately is NOT a generalization/Truck quality test.
                torch.manual_seed(123)
                optimizer.zero_grad(set_to_none=True)
                task=CameraInitializationTask(cameras()[:2],Reference(),0,raw[:,:3])
                loss,_=full_scene_step(model,batches,g,10,'awgn',task,attr_weight=0)
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(v).all() for v in grads))
                losses.append(float(loss))
                optimizer.step()
        self.assertLess(losses[-1],losses[0])

    def test_noisy_mixed_tiers_all_backward_modes_match(self):
        raw,g,f,model=setup(16)
        qs=[torch.tensor([[1,2,3,0,2,3,1,2]]),torch.tensor([[3,2,1,0,2,1,3,1]])]
        batches=[(f[:8][None],qs[0]),(f[8:][None],qs[1])]
        source=raw[torch.cat([q.flatten() for q in qs])>0,:3]
        values=[]
        gradients=[]
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.training.reconstruction_loss',side_effect=AssertionError('no old auxiliary')):
            for mode in ('direct','replay','checkpoint'):
                m=copy.deepcopy(model)
                torch.manual_seed(142)
                task=CameraInitializationTask(cameras()[:2],Reference(),0,source)
                value,stats=full_scene_step(m,batches,g,10,'awgn',task,attr_weight=0,mode=mode)
                values.append(value)
                gradients.append([p.grad.clone() if p.grad is not None else None for p in m.parameters()])
                self.assertGreater(task.stats['pixel_error_mean'],0)
                self.assertEqual(stats['aux_loss'],0)
        for i in (1,2):
            torch.testing.assert_close(values[0],values[i])
            for p,q in zip(gradients[0],gradients[i]):
                self.assertEqual(p is None,q is None)
                if p is not None:
                    torch.testing.assert_close(p,q,atol=2e-6,rtol=3e-4)

    def test_teacher_validation_does_not_change_actual_score_or_decode(self):
        raw,g,f,model=setup(16)
        groups=[f.reshape(2,8,-1)]
        ids=[torch.arange(16).reshape(2,8)]
        from gaussian_jscc.learned_training import decode_batches
        with tempfile.TemporaryDirectory() as temp,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            root=Path(temp)
            first=validate_render(model,groups,ids,raw,g,cameras()[:2],Reference(),10,'awgn',2,42,root,0,'initial')
            with patch('gaussian_jscc.render_validation.decode_batches',wraps=decode_batches) as decode:
                second=validate_render(model,groups,ids,raw,g,cameras()[:2],Reference(),10,'awgn',2,42,root,1,
                                       'camera_init',teacher_xyz_diagnostic=True)
                self.assertEqual(decode.call_count,8)
            self.assertEqual(first['score'],second['score'])
            self.assertEqual(len(list((root/'teacher_xyz_images'/'000001').glob('*.png'))),8)
            for a,b in zip(first['layouts'],second['layouts']):
                self.assertEqual(a['source_psnr'],b['source_psnr'])
                self.assertIn('teacher_xyz_source_psnr',b)

    def test_random_two_phase_full_flow_and_plots(self):
        from gaussian_jscc.cli import main
        from gaussian_jscc.data import write_ply
        from gaussian_jscc.transport import load_checkpoint
        from gaussian_jscc.plots import plot_training
        raw,_,_,_=setup(17)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--source','mock','--device','cuda','--camera-init-steps','4','--render-steps','2',
                  '--architecture','learned_split_logcov','--validation-views','1','--validation-trials','1',
                  '--validate-every','2','--save-every','2','--hidden','16','--depth','1','--block-size','8',
                  '--decoder-window','4','--blocks-per-batch','2']
            with patch('sys.argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=cameras()), \
                 patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),patch('gaussian_jscc.plots.safe_plot'), \
                 patch('gaussian_jscc.learned_train.load_checkpoint',side_effect=AssertionError('random must not load')), \
                 patch('gaussian_jscc.training.reconstruction_loss',side_effect=AssertionError('no old auxiliary')):
                main()
            run=root/'run'
            rows=[json.loads(line) for line in (run/'loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['camera_init']*4+['render']*2)
            self.assertTrue(all(r['lr']==2e-4 and r['aux_loss']==0 for r in rows))
            self.assertTrue(all(r['grad_norm']>0 and r['update_norm']>0 for r in rows))
            for r in rows[:4]:
                self.assertAlmostEqual(r['loss'],r['teacher_xyz_image_mse']+r['geometry_contribution'],places=5)
                self.assertNotIn('render_loss',r)
            for r in rows[4:]:
                self.assertEqual(r['loss'],r['image_mse'])
                self.assertNotIn('geometry_contribution',r)
            self.assertTrue((run/'codec_best_camera_init.pt').exists())
            self.assertTrue((run/'codec_best_render.pt').exists())
            self.assertEqual(load_checkpoint(run/'codec.pt','cpu').cfg.position_delivery,'learned')
            plot_training(run)
            self.assertTrue((run/'charts'/'teacher_xyz_gap.png').exists())
            self.assertTrue((run/'charts'/'camera_geometry.png').exists())


if __name__=='__main__':
    unittest.main()
