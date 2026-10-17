"""Modular learned allocation: hard-forward parity, feedback and deployment maps."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from gaussian_jscc.allocation import GaussianTierMask
from gaussian_jscc.allocation_diagnostics import AllocationCostMeter
from gaussian_jscc.data import Geometry, to_features
from gaussian_jscc.mask_gradient import local_mask_step
from gaussian_jscc.modular_codec import ModularGaussianCodec
from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.position_delivery import PositionCostMeter
from gaussian_jscc.render_objective import MaskedMultiViewRenderTask
from test_gaussian_jscc import fixture
from test_mask_gradient import Reference, toy_mask_render


def small_model():
    raw,_=fixture(16,1)
    cfg=ModularCodecConfig(sh_degree=1,hidden=16,grid_dim=4,levels=(2,),block_size=8,
        decoder_window=4,latent_dim=8,rates=(0,8,24,48),channel_variant='progressive',
        channel_hidden=16,channel_prefix_mode='full_prefix',channel_denoiser_mode='pointwise',
        channel_denoiser_hidden=8,channel_neighbor_mode='plain',channel_neighbor_hidden=8,
        channel_neighbor_neighbors=3)
    model=ModularGaussianCodec(cfg)
    geometry=Geometry.fit(raw[:,:3],16); model.set_scene_geometry(geometry)
    f,_=to_features(raw,geometry,model)
    return raw,geometry,f.reshape(2,8,-1),model


class ModularAllocationGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_st_forward_keeps_exact_hard_prefix_and_has_positive_tier_feedback(self):
        _,_,f,model=small_model()
        self.assertTrue(hasattr(model,'forward_st_prefix_batches'),
                        'modular codec must support the local allocator training boundary')
        q=torch.tensor([[0,1,2,3,1,2,3,0],[1,2,3,1,2,3,0,0]])
        logits=torch.zeros(2,8,4,requires_grad=True)
        torch.manual_seed(19)
        expected=model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).to(f),10,'awgn',paired_noise=True)[0]
        torch.manual_seed(19)
        observed=model.forward_st_prefix_batches(f,f[...,:3],q,logits.softmax(-1),10,'awgn')
        torch.testing.assert_close(expected,observed,atol=1e-6,rtol=1e-5)
        observed[...,3:].square().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[q>0].norm()),0)
        self.assertEqual(int(observed[q==0].count_nonzero()),0)

    def test_every_q0_gets_feedback_and_frozen_codec_stays_frozen(self):
        raw,geometry,f,model=small_model()
        self.assertTrue(hasattr(model,'forward_st_prefix_batches'),'missing modular allocator feedback')
        mask=GaussianTierMask(16,existence_prior=torch.full((16,),.5))
        meter=AllocationCostMeter(model.cfg,PositionCostMeter(model.cfg,geometry.normalize(raw[:,:3])),16,2.)
        task=MaskedMultiViewRenderTask([SimpleNamespace(factor=1.,source_count=16)],Reference(),1)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with patch('gaussian_jscc.rendering.render',side_effect=toy_mask_render):
            torch.manual_seed(11)
            _,stats=local_mask_step(model,mask,[f],[torch.arange(16).reshape(2,8)],geometry,
                10,'awgn',task,.01,meter,train_codec=False)
        self.assertGreater(stats['sampled_tier_counts'][0],0)
        self.assertEqual(stats['sampled_tier_counts'][0],stats['sampled_q0_shadow_candidates'])
        self.assertGreater(stats['mask_keep_image_grad_norm'],0)
        self.assertGreater(stats['mask_tier_image_grad_norm'],0)
        self.assertFalse(any(p.grad is not None for p in model.parameters()))
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_joint_feedback_reaches_all_four_modules(self):
        raw,geometry,f,model=small_model()
        self.assertTrue(hasattr(model,'forward_st_prefix_batches'),'missing joint allocator feedback')
        mask=GaussianTierMask(16,existence_prior=torch.full((16,),.8))
        meter=AllocationCostMeter(model.cfg,PositionCostMeter(model.cfg,geometry.normalize(raw[:,:3])),16,2.)
        task=MaskedMultiViewRenderTask([SimpleNamespace(factor=1.,source_count=16)],Reference(),1)
        with patch('gaussian_jscc.rendering.render',side_effect=toy_mask_render):
            torch.manual_seed(11)
            local_mask_step(model,mask,[f],[torch.arange(16).reshape(2,8)],geometry,
                10,'awgn',task,.01,meter,train_codec=True)
        for name in ('semantic_encoder','semantic_decoder','channel_encoder','channel_decoder'):
            self.assertTrue(any(p.grad is not None and p.grad.norm()>0 for p in getattr(model,name).parameters()),name)

    def test_unreceived_enhancement_parameters_do_not_get_adam_momentum_updates(self):
        _,_,f,model=small_model()
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        # Establish nonzero Adam moments, as after a q3 prefix anchor.
        q=torch.full(f.shape[:2],3,dtype=torch.long)
        pred=model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).to(f),10,'none')[0]
        pred[...,3:].square().sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in model.channel_decoder.stages[2].named_parameters()}
        q.fill_(1);logits=torch.zeros(*q.shape,4,requires_grad=True)
        pred=model.forward_st_prefix_batches(f,f[...,:3],q,logits.softmax(-1),10,'none')
        pred[...,3:].square().sum().backward()
        self.assertTrue(all(p.grad is None for p in model.channel_decoder.stages[2].parameters()),
                        'unreceived stage must not receive zero tensors that trigger stale Adam momentum')
        self.assertGreater(float(logits.grad.norm()),0)
        optimizer.step()
        for n,p in model.channel_decoder.stages[2].named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)


class TierMapTests(unittest.TestCase):
    def test_dominant_contribution_label_has_no_blended_or_q0_colors(self):
        try:
            from gaussian_jscc.tier_visualization import contribution_labels,colorize_labels,TIER_COLORS
        except ImportError as exc:
            self.fail(str(exc))
        weights=torch.tensor([[[.8,0.,.1,0.]],[[.1,.7,.2,0.]],[[.05,.2,.6,0.]]])
        labels=contribution_labels(weights)
        self.assertEqual(labels.tolist(),[[1,2,3,0]])
        rgb=colorize_labels(labels)
        self.assertEqual(tuple(rgb.shape),(3,1,4))
        for x,tier in enumerate((1,2,3,0)):
            torch.testing.assert_close(rgb[:,0,x],torch.tensor(TIER_COLORS[tier])/255)
        with self.assertRaises(ValueError): contribution_labels(torch.full((3,1,1),float('nan')))


class AllocationExperimentTests(unittest.TestCase):
    def test_two_phase_run_metrics_images_and_exact_resume(self):
        try:
            from gaussian_jscc.modular_allocation import main,parser
        except ImportError as exc:
            self.fail(str(exc))
        from gaussian_jscc.modular_experiment import main as train_codec,parser as codec_parser
        from gaussian_jscc.transport import load_checkpoint
        from test_modular_experiment import source_fixture
        from test_render_first import synthetic_render,RenderFirstTests
        from test_perceptual_metrics import backend

        def fake_render(raw,camera,degree,white_background=False,existence=None,colors=None):
            if colors is not None:
                # Rasterizer replacement only; real map classification/saving is exercised.
                return colors.mean(0)[:,None,None].expand(3,*camera.original_image.shape[-2:]) if len(raw) else torch.zeros_like(camera.original_image)
            if existence is not None:
                image=toy_mask_render(raw,camera,degree,white_background,existence)
                return F.interpolate(image[None],size=camera.original_image.shape[-2:],mode='nearest')[0]
            return synthetic_render(raw,camera,degree,white_background)

        with tempfile.TemporaryDirectory() as t, \
             patch('gaussian_jscc.rendering.load_cameras',return_value=RenderFirstTests().cameras()), \
             patch('gaussian_jscc.rendering.render',side_effect=fake_render), \
             patch('gaussian_jscc.perceptual_metrics._load_lpips_model',side_effect=backend):
            root=Path(t); source=source_fixture(root); trained=root/'trained'
            opts=codec_parser().parse_args(['--run',str(source),'--out',str(trained),'--device','cpu',
                '--channel-variant','progressive','--channel-hidden','16','--rates','8','24','48',
                '--semantic-bootstrap-steps','1','--semantic-render-steps','1',
                '--channel-bootstrap-steps','1','--channel-render-steps','1','--train-snrs','10'])
            train_codec(opts)
            def args(dest):
                return parser().parse_args(['--run',str(trained),'--out',str(dest),'--device','cpu',
                    '--allocation-steps','2','--joint-steps','2','--validate-every','2',
                    '--save-every','2','--prefix-anchor-every','1','--test-views','1','--trials','1'])
            main(args(root/'full'))
            main(args(root/'resumed'),max_updates=3)
            resume=args(root/'resumed');resume.resume=True;main(resume)
            done=json.loads((root/'full/complete.json').read_text())
            self.assertEqual(done['completed'],{'allocation':2,'joint':2})
            self.assertTrue(done['frozen_codec_verified'])
            a=load_checkpoint(root/'full/checkpoints/final/codec.pt','cpu')
            b=load_checkpoint(root/'resumed/checkpoints/final/codec.pt','cpu')
            for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
            ma=torch.load(root/'full/checkpoints/final/allocation.pt',weights_only=True)
            mb=torch.load(root/'resumed/checkpoints/final/allocation.pt',weights_only=True)
            for k,v in ma['state_dict'].items(): torch.testing.assert_close(v,mb['state_dict'][k],rtol=0,atol=0)
            rows=[json.loads(x) for x in (root/'full/metrics/test.jsonl').read_text().splitlines()]
            self.assertEqual({r['layout'] for r in rows},{'1','2','3','deployment'})
            self.assertTrue(all('source_lpips' in r and 'photo_lpips' in r and 'source_ssim' in r for r in rows))
            deployed=next(r for r in rows if r['layout']=='deployment')
            self.assertEqual(sum(deployed['tier_counts']),16)
            self.assertTrue((root/'full/images/final/deployment_view0000_tiers.png').exists())
            self.assertTrue((root/'full/deployment/final/tiers.npy').exists())
            loss=[json.loads(x) for x in (root/'full/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in loss],[1,2,3,4])
            self.assertTrue(all(not r['codec_updated'] for r in loss[:2]))
            self.assertTrue(all(r['codec_updated'] for r in loss[2:]))
            for name in ('semantic_encoder','semantic_decoder','channel_encoder','channel_decoder'):
                self.assertGreater(loss[-1]['updates'][name]['update_norm'],0,name)


if __name__=='__main__': unittest.main()
