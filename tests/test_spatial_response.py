"""CPU tests of the experimental learned-XYZ bootstrap; no mocked gradients."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from gaussian_jscc.spatial_response import spatial_response_loss, position_metrics, native_shape_response, fine_position_response
from gaussian_jscc.local_response import view_frames
from gaussian_jscc.data import write_ply
from test_learned_joint import setup


class SpatialResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        torch.manual_seed(12)
        return setup(16)

    def test_identity_translation_and_detached_teacher(self):
        _, g, f, m = self.fixture()
        target = f.clone().requires_grad_()
        pred = f.clone().requires_grad_()
        loss, _ = spatial_response_loss(pred,target,g,m)
        self.assertLess(abs(float(loss.detach())),1e-10)
        loss.backward()
        self.assertIsNone(target.grad)
        shifted = f.clone(); shifted[:, :3] += .2; shifted.requires_grad_()
        loss, _ = spatial_response_loss(shifted,target,g,m)
        loss.backward()
        self.assertGreater(float(loss.detach()),0)
        self.assertGreater(float(shifted.grad[:, :3].sum()),0)
        self.assertTrue(torch.isfinite(shifted.grad).all())

    def test_far_centers_tiny_opacity_have_attractive_gradient(self):
        _, g, f, m = self.fixture()
        target=f.clone(); target[:,3]=-100
        pred=target.clone(); pred[:,:3]+=.8; pred.requires_grad_()
        loss, _=spatial_response_loss(pred,target,g,m)
        loss.backward()
        self.assertGreater(float(pred.grad[:,:3].sum()),1e-6)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_fine_gradient_bound_tiny_scales_and_zero(self):
        raw,g,_,_=self.fixture()
        for logscale in (-20.,-10.,0.):
            source=raw.double().clone();source[:,4:7]=logscale
            for shift in (0.,1e-8,.01,1000.):
                pred=source.clone();pred[:,0]+=shift;pred.requires_grad_()
                loss=fine_position_response(pred,source,g,.0078125)
                loss.backward()
                self.assertTrue(torch.isfinite(pred.grad).all())
                # derivative wrt world XYZ includes 1 / bbox diagonal and mean N
                bound=1/(.0078125*float(g.span.double().norm())*len(raw))
                self.assertLessEqual(float(pred.grad[:,:3].norm(dim=-1).max()),bound*(1+1e-8))
                self.assertEqual(float(pred.grad[:,3:].abs().sum()),0.)
                if shift==0:
                    self.assertEqual(float(loss.detach()),0.)
                    self.assertEqual(float(pred.grad.abs().sum()),0.)
                else:self.assertGreater(float(pred.grad[:,0].sum()),0.)

    def test_fine_target_detached_and_prediction_shape_independent(self):
        raw,g,_,_=self.fixture()
        target=raw.clone().requires_grad_()
        pred=raw.clone();pred[:,:3]+=.1;pred.requires_grad_()
        first=fine_position_response(pred,target,g,.0078125)
        first.backward()
        self.assertIsNone(target.grad)
        changed=pred.detach().clone();changed[:,3:]+=5
        torch.testing.assert_close(first.detach(),fine_position_response(changed,target,g,.0078125))

    def test_zero_weight_reproduces_v2_composition(self):
        _,g,f,m=self.fixture()
        pred=f.clone();pred[:,:3]+=.05
        old,s=spatial_response_loss(pred,f,g,m,directions=torch.eye(3),fine_weight=0.)
        new,t=spatial_response_loss(pred,f,g,m,directions=torch.eye(3),fine_weight=1.)
        self.assertAlmostEqual(float(old),(s['spatial_coarse_position_response']+s['spatial_native_shape_response']+s['spatial_appearance_response'])/3,places=6)
        self.assertAlmostEqual(float(new-old),t['spatial_fine_position_response']/3,places=6)

    def test_displaced_centers_cannot_reward_inflation(self):
        _,g,f,m=self.fixture()
        directions=torch.eye(3)
        shifted=f.clone();shifted[:,:3]+=.2
        base,bs=spatial_response_loss(shifted,f,g,m,directions=directions)
        for factor in (.01,.1,.5,2.,10.,34.,266.):
            inflated=shifted.clone()
            inflated[:,4:7]+=torch.log(torch.tensor(factor))/m.attr_std[1:4]
            inflated.requires_grad_()
            loss,stats=spatial_response_loss(inflated,f,g,m,directions=directions)
            self.assertAlmostEqual(stats['spatial_position_response'],bs['spatial_position_response'],places=10)
            self.assertGreater(float(loss.detach()),float(base.detach()))
            loss.backward()
            # Correct both inflation and contraction, not just make all small.
            derivative=float((inflated.grad[:,4:7]/m.attr_std[1:4]).sum())
            self.assertGreater(derivative*(1 if factor>1 else -1),0)
            self.assertTrue(torch.isfinite(inflated.grad).all())

    def test_shape_cannot_hide_in_low_opacity_or_shift(self):
        _,g,f,m=self.fixture()
        inflated=f.clone();inflated[:,4:7]+=2/m.attr_std[1:4]
        _,a=spatial_response_loss(inflated,f,g,m,directions=torch.eye(3))
        inflated[:,:3]+=10;inflated[:,3]=-100
        _,b=spatial_response_loss(inflated,f,g,m,directions=torch.eye(3))
        self.assertAlmostEqual(a['spatial_native_shape_response'],b['spatial_native_shape_response'],places=10)
        self.assertGreater(b['spatial_native_shape_response'],0)

    def test_native_shape_scale_invariance_and_nonsaturating_gradient(self):
        raw,_,_,_=self.fixture()
        _,frames=view_frames(torch.eye(3))
        target=raw.clone(); target[:,4:7]=-8
        for factor in (34.,266.,10000.):
            pred=target.clone();pred[:,4:7]+=torch.log(torch.tensor(factor));pred.requires_grad_()
            loss=native_shape_response(pred,target,frames)
            loss.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertGreater(float(pred.grad[:,4:7].sum()),.5)
            scaled_p=pred.detach().clone();scaled_t=target.clone()
            scaled_p[:,4:7]+=3;scaled_t[:,4:7]+=3
            self.assertAlmostEqual(float(loss.detach()),float(native_shape_response(scaled_p,scaled_t,frames)),places=5)

    def test_extreme_anisotropy_finite_gradients(self):
        _,g,f,m=self.fixture()
        target=f.clone();pred=f.clone()
        target[:,4:7]=(torch.tensor([-12.,-5.,0.])-m.attr_mean[1:4])/m.attr_std[1:4]
        pred[:,4:7]=(torch.tensor([8.,-10.,-3.])-m.attr_mean[1:4])/m.attr_std[1:4]
        pred.requires_grad_()
        loss,_=spatial_response_loss(pred,target,g,m,directions=torch.eye(3))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_all_heads_receive_gradients_and_loss_can_decrease(self):
        _,g,f,m=self.fixture()
        q=torch.arange(len(f))%3+1
        optimizer=torch.optim.Adam(m.parameters(),lr=2e-4)
        directions=torch.eye(3)
        def evaluate():
            torch.manual_seed(71)
            pred=m(f,f[:,:3],q,10,'awgn')
            loss,stats=spatial_response_loss(pred,f,g,m,directions=directions)
            return loss,stats
        initial, before=evaluate()
        initial=float(initial.detach())
        for step in range(120):
            optimizer.zero_grad()
            loss,_=evaluate(); loss.backward()
            if step == 0:
                for name,head in m.learned.heads.items():
                    self.assertIsNotNone(head.weight.grad,name)
                    self.assertGreater(float(head.weight.grad.norm()),0,name)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            optimizer.step()
        final, after=evaluate()
        self.assertLess(float(final.detach()),initial)
        self.assertLess(after['xyz_nrmse_bbox'],before['xyz_nrmse_bbox'])

    def test_cli_bootstrap_only_never_loads_cameras_or_old_weights(self):
        from gaussian_jscc.cli import main
        raw,_,_,_=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--device','cpu','--bootstrap-steps','3','--render-steps','0','--joint-steps','0',
                  '--bootstrap-objective','spatial-response','--position-delivery','learned',
                  '--hidden','16','--depth','1','--grid-dim','4','--levels','2',
                  '--block-size','4','--blocks-per-batch','2','--decoder-window','4',
                  '--validation-blocks','1','--validation-trials','2','--validate-every','1']
            with patch('sys.argv',argv),patch('gaussian_jscc.rendering.load_cameras',side_effect=AssertionError('no cameras')), \
                 patch('gaussian_jscc.learned_train.load_checkpoint',side_effect=AssertionError('random start')), \
                 patch('gaussian_jscc.plots.safe_plot'):
                main()
            rows=[json.loads(s) for s in (root/'run/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),3)
            for r in rows:
                self.assertEqual(r['phase'],'bootstrap')
                self.assertEqual(r['objective'],'spatial_response_v3')
                self.assertGreater(r['gradient_groups']['xyz_head']['before'],0)
                for key in ('spatial_position_response','spatial_native_shape_response',
                            'max_axis_ratio_p50','xyz_distance_over_source_radius_p50'):
                    self.assertIn(key,r)
            vals=[json.loads(s) for s in (root/'run/bootstrap_validation.jsonl').read_text().splitlines()]
            self.assertEqual(len(vals),4)
            self.assertEqual([r['layout'] for r in vals[-1]['layouts']],['1','2','3','mixed'])
            self.assertTrue(all(r['position_side_stream_bits']==0 for r in vals[-1]['layouts']))
            self.assertEqual(vals[-1]['loss_version'],'spatial_response_v3')
            self.assertTrue((root/'run/codec_best_bootstrap.pt').exists())
            self.assertTrue((root/'run/codec_end_bootstrap.pt').exists())
            self.assertFalse((root/'run/validation.jsonl').exists())
            from gaussian_jscc.plots import plot_training
            plot_training(root/'run')
            self.assertTrue((root/'run/charts/bootstrap_position_validation.png').exists())
            self.assertTrue((root/'run/charts/bootstrap_position_validation.csv').exists())

    def test_invalid_bandwidth_and_metric_definition(self):
        _,g,f,m=self.fixture()
        with self.assertRaises(ValueError):
            spatial_response_loss(f,f,g,m,bandwidths=[0])
        x=f.clone();x[:,:3]+=.1
        stats=position_metrics(x,f,g)
        self.assertAlmostEqual(stats['xyz_nrmse_bbox'],.1/(3**.5),places=6)


if __name__=='__main__':
    unittest.main()
