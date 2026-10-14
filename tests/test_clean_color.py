"""Color supervision must not disappear with alpha/shape or invoke a channel."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_modular_training import setup
from test_modular_experiment import source_fixture
from test_render_first import synthetic_render
import test_render_first as render_fixture


class CleanColorTests(unittest.TestCase):
    def color_loss(self):
        from gaussian_jscc import local_response
        self.assertTrue(hasattr(local_response, 'directional_color_loss'), 'independent color supervision missing')
        return local_response.directional_color_loss

    def test_dc_error_has_physical_rgb_scale_and_no_alpha_shape_gradient(self):
        loss_fn=self.color_loss(); m,g,f,q=setup()
        target=torch.zeros(2,f.shape[-1],requires_grad=True)
        pred=target.detach().clone(); pred[:,11]=1.; pred[:,:11]=50.; pred.requires_grad_()
        directions=torch.tensor([[1.,0,0],[0,0,1.]])
        loss,stats=loss_fn(pred,target,m,directions)
        self.assertAlmostEqual(float(loss.detach()),0.02652582384864923,places=7)  # C0**2 / RGB(3)
        loss.backward()
        self.assertIsNone(target.grad)
        torch.testing.assert_close(pred.grad[:,:11],torch.zeros_like(pred.grad[:,:11]))
        self.assertGreater(float(pred.grad[:,11:].abs().sum()),0)
        self.assertEqual(stats['color_directions'],2)

    def test_negative_predicted_color_still_has_gradient(self):
        loss_fn=self.color_loss(); m,g,f,q=setup()
        target=torch.zeros(1,f.shape[-1]); pred=target.clone(); pred[:,11:14]=-5.; pred.requires_grad_()
        loss,_=loss_fn(pred,target,m,torch.tensor([[1.,0,0]])); loss.backward()
        self.assertLess(float(pred.grad[:,11:14].max()),0.)

    def test_sh_sign_channel_order_and_per_point_directions(self):
        loss_fn=self.color_loss(); m,g,f,q=setup()
        target=torch.zeros(1,f.shape[-1]); pred=target.clone()
        pred[:,14+2]=1.  # red x SH: -C1*x; no green/blue coefficient
        loss,_=loss_fn(pred,target,m,torch.tensor([[[1.,0,0],[-1.,0,0]]]))
        self.assertAlmostEqual(float(loss),0.07957747154594767,places=7)  # C1**2 / RGB(3)
        with self.assertRaises(ValueError): loss_fn(pred,target,m,torch.zeros(2,3))

    def test_color_step_keeps_shape_heads_and_channel_weights_unchanged(self):
        from gaussian_jscc import modular_training
        self.assertTrue(hasattr(modular_training,'color_bootstrap_step'), 'color-only update missing')
        m,g,f,q=setup(); opt=modular_training.configure_phase(m,'semantic_bootstrap',{})
        # Seed Adam momentum through a real ordinary update first.
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        modular_training.bootstrap_step(m,f,q,g,10,'none','semantic_bootstrap')
        modular_training.finish_update(m,opt,before); opt.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        centers=torch.tensor([[3.,2.,4.],[4.,3.,2.]])
        loss,stats=modular_training.color_bootstrap_step(m,f,q,g,centers,8)
        modular_training.finish_update(m,opt,before)
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(stats['color_directions'],8)
        self.assertEqual(stats['color_camera_directions'],4)
        for n,p in m.named_parameters():
            if n.startswith(('channel_', 'semantic_decoder.heads.opacity.', 'semantic_decoder.heads.scale.', 'semantic_decoder.heads.rotation.')):
                torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if 'heads.dc.' in n))

    def args(self,source,out):
        from gaussian_jscc.modular_experiment import parser
        try:
            return parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu',
                '--context-variant','both','--context-centers','4','2','--context-neighbors','4','8',
                '--clean-only','--semantic-color-every','4','--color-metrics','--save-semantic-end-images',
                '--semantic-bootstrap-steps','8','--semantic-render-steps','2',
                '--channel-bootstrap-steps','0','--channel-render-steps','0','--validate-every','4','--save-every','4'])
        except SystemExit:
            self.fail('clean-only color experiment CLI missing')

    def test_clean_pipeline_never_calls_channel_and_resume_is_exact(self):
        from gaussian_jscc.modular_experiment import main
        from gaussian_jscc.transport import load_checkpoint
        cams=render_fixture.RenderFirstTests().cameras()
        for i,c in enumerate(cams): c.camera_center=torch.tensor([4.,2.+i,3.])
        with tempfile.TemporaryDirectory() as t, \
             patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras',return_value=cams), \
             patch('gaussian_jscc.modular_core.ChannelEncoder.forward',side_effect=AssertionError('channel must not run')):
            root=Path(t); source=source_fixture(root); a=root/'full'; b=root/'resume'
            main(self.args(source,a)); main(self.args(source,b),max_updates=5)
            opts=self.args(source,b); opts.resume=str(b/'checkpoints/latest/codec.pt'); main(opts)
            x=load_checkpoint(a/'checkpoints/final/codec.pt','cpu'); y=load_checkpoint(b/'checkpoints/final/codec.pt','cpu')
            for pa,pb in zip(x.parameters(),y.parameters()): torch.testing.assert_close(pa,pb,rtol=0,atol=0)
            done=json.loads((a/'complete.json').read_text())
            self.assertEqual(done['completed'],dict(semantic_bootstrap=8,semantic_render=2,channel_bootstrap=0,channel_render=0))
            self.assertFalse(done['channel_trained'])
            logs=[json.loads(line) for line in (a/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual(sum(row['objective']=='directional_color_mse_unclipped_v1' for row in logs),2)
            for name in ('validation','test'):
                records=[json.loads(line) for line in (a/'metrics'/f'{name}.jsonl').read_text().splitlines()]
                self.assertTrue(all(row['path']=='clean' and row['actual_snr'] is None for row in records))
                self.assertTrue(all('source_color_rg_error_rmse' in row for row in records))
            self.assertTrue((a/'images/end_semantic_bootstrap/clean_clean_view0000.png').exists())
            self.assertTrue((a/'images/final/clean_clean_view0000.png').exists())
            self.assertTrue((a/'charts/color_errors.png').exists(), 'saved color metrics need a comparison curve')
            self.assertFalse((a/'charts/psnr_snr.png').exists(), 'clean-only run must not imply an SNR sweep')
            boundary=root/'boundary'
            main(self.args(source,boundary),max_updates=8)
            self.assertFalse((boundary/'images/end_semantic_bootstrap').exists())
            # A interrupted PNG write can create a directory before endpoint completion.
            (boundary/'images/end_semantic_bootstrap').mkdir(parents=True)
            opts=self.args(source,boundary); opts.resume=str(boundary/'checkpoints/latest/codec.pt'); main(opts)
            self.assertTrue((boundary/'images/end_semantic_bootstrap/clean_clean_view0000.png').exists(),
                            'exact phase-boundary resume must recover endpoint images before rendering updates')
            self.assertTrue((boundary/'images/end_semantic_bootstrap/complete.json').exists())
            self.assertTrue((a/'charts/clean_quality.png').exists())
            opts=self.args(source,root/'invalid'); opts.channel_render_steps=1
            with self.assertRaisesRegex(ValueError,'clean-only'): main(opts)
            opts=self.args(source,b); opts.resume=str(b/'checkpoints/latest/codec.pt'); opts.semantic_color_every=2
            with self.assertRaisesRegex(ValueError,'objective'): main(opts)


if __name__=='__main__': unittest.main()
