"""Conditional-channel allocation preserves hard deployment and Joint A scope."""
import importlib.util
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
from gaussian_jscc.mask_gradient import local_mask_step
from gaussian_jscc.position_delivery import PositionCostMeter
from gaussian_jscc.render_objective import MaskedMultiViewRenderTask
from test_mask_gradient import Reference,toy_mask_render


def small_successive():
    from test_modular_allocation import small_model
    from gaussian_jscc.model_factory import build_codec
    raw,geometry,f,old=small_model()
    old.cfg.channel_variant='successive';old.cfg.channel_power_mode='block'
    model=build_codec(old.cfg);model.set_scene_geometry(geometry)
    model.channel_decoder.neighbor_enabled.fill_(True)
    return raw,geometry,f,model


class SuccessiveAllocationGradientTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(71)

    def test_actual_forward_and_rng_equal_hard_deployment_and_only_logits_get_proxy_gradients(self):
        _,_,f,model=small_successive();model.train();model.requires_grad_(False)
        q=torch.tensor([[0,1,2,3,1,2,3,0],[1,2,3,1,2,3,0,0]])
        logits=torch.zeros(2,8,4,requires_grad=True)
        torch.manual_seed(91)
        expected=model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).to(f),10.,'awgn',paired_noise=True)[0]
        expected_rng=torch.get_rng_state()
        torch.manual_seed(91)
        try:actual=model.forward_st_prefix_batches(f,f[...,:3],q,logits.softmax(-1),10.,'awgn')
        except ValueError as exc:self.fail('missing conditional allocation forward: '+str(exc))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
        actual[...,3:].square().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[q>0].norm()),0.)
        self.assertEqual(float(logits.grad[q==0].norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_unreceived_stages_keep_grad_none_despite_counterfactual_candidates(self):
        _,_,f,model=small_successive()
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        q=torch.full(f.shape[:2],3,dtype=torch.long)
        model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).to(f),10.,'none')[0].square().sum().backward()
        opt.step();opt.zero_grad(set_to_none=True)
        before={n:p.clone() for n,p in model.named_parameters()}
        q.fill_(1);logits=torch.zeros(*q.shape,4,requires_grad=True)
        try:pred=model.forward_st_prefix_batches(f,f[...,:3],q,logits.softmax(-1),10.,'none')
        except ValueError as exc:self.fail(str(exc))
        pred[...,3:].square().sum().backward()
        for module in (model.channel_encoder,model.channel_decoder):
            for stage in module.stages[1:]:self.assertTrue(all(p.grad is None for p in stage.parameters()))
        opt.step()
        for n,p in model.named_parameters():
            if n.startswith(('channel_encoder.stages.1.','channel_encoder.stages.2.',
                             'channel_decoder.stages.1.','channel_decoder.stages.2.')):
                torch.testing.assert_close(p,before[n],rtol=0,atol=0)

    def test_q0_feedback_reaches_every_closed_point_and_latent_anchor_stays_off_allocator(self):
        raw,geometry,f,model=small_successive()
        mask=GaussianTierMask(16,existence_prior=torch.full((16,),.5))
        model.semantic_encoder.requires_grad_(False)
        meter=AllocationCostMeter(model.cfg,PositionCostMeter(model.cfg,geometry.normalize(raw[:,:3])),16,2.)
        initial={n:v.clone() for n,v in model.semantic_encoder.state_dict().items()}
        task=MaskedMultiViewRenderTask([SimpleNamespace(factor=1.,source_count=16)],Reference(),1)
        with patch('gaussian_jscc.rendering.render',side_effect=toy_mask_render):
            try:
                _,stats=local_mask_step(model,mask,[f],[torch.arange(16).reshape(2,8)],geometry,10.,'awgn',
                    task,.005,meter,train_codec=True,channel_latent_weight=1.)
            except (ValueError,TypeError) as exc:self.fail(str(exc))
        self.assertGreater(stats['sampled_tier_counts'][0],0)
        self.assertEqual(stats['sampled_tier_counts'][0],stats['sampled_q0_shadow_candidates'])
        self.assertGreater(stats['mask_keep_image_grad_norm'],0.)
        self.assertGreater(stats['mask_tier_image_grad_norm'],0.)
        self.assertGreater(stats['weighted_latent_loss'],0.)
        for name in ('channel_encoder','channel_decoder','semantic_decoder'):
            self.assertTrue(any(p.grad is not None and p.grad.norm()>0 for p in getattr(model,name).parameters()),name)
        for n,v in model.semantic_encoder.state_dict().items():torch.testing.assert_close(v,initial[n],rtol=0,atol=0)

    def test_measured_cost_and_training_meter_include_real_block_scale_frames(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.successive_allocation'))
        from gaussian_jscc.successive_allocation import SuccessiveAllocationCostMeter
        from gaussian_jscc.modular_allocation import measured_cost
        from gaussian_jscc.transport import model_id
        raw,geometry,_,model=small_successive()
        scene=SimpleNamespace(raw=raw,geometry=geometry,group_ids=[torch.arange(16).reshape(2,8)],net_bits_per_use=2.)
        q=torch.tensor([3,1,0,2,1,3,0,0,1,1,0,0,0,0,0,0])
        base=AllocationCostMeter(model.cfg,PositionCostMeter(model.cfg,geometry.normalize(raw[:,:3])),16,2.)
        meter=SuccessiveAllocationCostMeter(model,scene,PositionCostMeter(model.cfg,geometry.normalize(raw[:,:3])))
        # Four nonempty block/segments: 8-byte header plus 4 binary16 values.
        cost=measured_cost(model,scene,q,10.,model_id(model))
        self.assertEqual(cost['scale_stream_bytes'],16)
        self.assertEqual(cost['scale_channel_uses'],64)
        self.assertEqual(cost['total_channel_uses'],cost['payload_complex_symbols']+
            cost['coordinate_channel_uses']+cost['metadata_channel_uses']+64)
        self.assertAlmostEqual(meter.details(q)['allocation_uses_per_source_gaussian'],
                               base.details(q)['allocation_uses_per_source_gaussian']+4.)

    def test_latent_anchor_changes_codec_gradient_without_biasing_table_gradient(self):
        raw,geometry,f,model=small_successive();model.semantic_encoder.requires_grad_(False)
        mask=GaussianTierMask(16,existence_prior=torch.full((16,),.7))
        meter=AllocationCostMeter(model.cfg,PositionCostMeter(model.cfg,geometry.normalize(raw[:,:3])),16,2.)
        gradients=[]
        with patch('gaussian_jscc.rendering.render',side_effect=toy_mask_render):
            for weight in (0.,1.):
                mask.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);torch.manual_seed(6)
                task=MaskedMultiViewRenderTask([SimpleNamespace(factor=1.,source_count=16)],Reference(),1)
                local_mask_step(model,mask,[f],[torch.arange(16).reshape(2,8)],geometry,10.,'awgn',
                    task,.005,meter,train_codec=True,channel_latent_weight=weight)
                gradients.append(([p.grad.clone() for p in mask.parameters()],
                    torch.cat([p.grad.flatten() for p in model.channel_decoder.parameters() if p.grad is not None])))
        for a,c in zip(gradients[0][0],gradients[1][0]):torch.testing.assert_close(a,c,rtol=0,atol=0)
        self.assertGreater(float((gradients[0][1]-gradients[1][1]).norm()),0.)


class SuccessiveAllocationEntryTests(unittest.TestCase):
    def test_zero_test_views_selects_all_real_held_out_views(self):
        from gaussian_jscc.power_followup_evaluation import bind_test_cameras
        cameras=[SimpleNamespace(image_name='test_'+str(i)) for i in range(3)]
        scene=SimpleNamespace(record=dict(resolution=4,images='images',train_view_names=['train']),white_background=False)
        with patch('gaussian_jscc.rendering.load_cameras',return_value=cameras):
            self.assertEqual(bind_test_cameras(scene,dict(source='unused'),0),[c.image_name for c in cameras])

    def test_two_stages_freeze_encoder_save_deployment_and_resume_exactly(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.successive_allocation'))
        from gaussian_jscc import successive_allocation as entry
        from gaussian_jscc.successive_joint import main as joint,parser as joint_parser
        from test_successive_joint import source_run,cameras
        from test_render_first import synthetic_render
        from test_perceptual_metrics import backend

        def render(raw,camera,degree,white_background=False,existence=None,colors=None):
            if colors is not None:
                return colors.mean(0)[:,None,None].expand_as(camera.original_image) if len(raw) else torch.zeros_like(camera.original_image)
            if existence is not None:
                image=toy_mask_render(raw,camera,degree,white_background,existence)
                return F.interpolate(image[None],size=camera.original_image.shape[-2:],mode='nearest')[0]
            return synthetic_render(raw,camera,degree,white_background)

        with tempfile.TemporaryDirectory() as folder, \
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=render), \
             patch('gaussian_jscc.perceptual_metrics._load_lpips_model',side_effect=backend):
            root=Path(folder);parent=source_run(root,context_variant='both');source=root/'jointA'
            joint(joint_parser().parse_args(['--run',str(parent),'--out',str(source),'--device','cpu',
                '--steps','1','--test-views','2','--render-trials','1','--no-lpips']))
            source_bytes=(source/'checkpoints/final/codec.pt').read_bytes()
            def options(out):
                return entry.parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu',
                    '--allocation-steps','2','--joint-steps','2','--validate-every','2','--save-every','1',
                    '--test-views','2','--trials','1','--prefix-anchor-every','1'])
            dry=options(root/'dry');dry.dry_run=True
            self.assertFalse(entry.main(dry)['writes_output'])
            self.assertFalse((root/'dry').exists())
            saved=torch.load(source/'checkpoints/final/codec.pt',weights_only=True)
            broken=dict(saved,step=saved['step']+1)
            torch.save(broken,source/'checkpoints/final/codec.pt')
            try:
                with self.assertRaisesRegex(ValueError,'completed Joint A'):entry.source_plan(options(root/'invalid'))
            finally:(source/'checkpoints/final/codec.pt').write_bytes(source_bytes)
            result=entry.main(options(root/'full'))
            self.assertEqual(result['completed'],dict(allocation=2,joint=2))
            self.assertTrue(result['frozen_encoder_verified'])
            self.assertEqual((source/'checkpoints/final/codec.pt').read_bytes(),source_bytes)
            source_state=torch.load(source/'checkpoints/final/codec.pt',weights_only=True)['state_dict']
            final=torch.load(root/'full/checkpoints/final/state.pt',weights_only=True)
            frozen=torch.load(root/'full/checkpoints/end_allocation/state.pt',weights_only=True)
            for n,v in source_state.items():
                torch.testing.assert_close(frozen['state_dict'][n],v,rtol=0,atol=0)
                if n.startswith('semantic_encoder.') or n in ('attr_mean','attr_std'):
                    torch.testing.assert_close(final['state_dict'][n],v,rtol=0,atol=0)
            logs=[json.loads(s) for s in (root/'full/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in logs],[1,2,3,4])
            for row in logs:
                self.assertEqual(row['updates']['semantic_encoder']['update_norm'],0.)
            for name in ('channel_encoder','channel_decoder','semantic_decoder'):
                self.assertGreater(logs[-1]['updates'][name]['update_norm'],0.)
            self.assertGreater(logs[-1]['weighted_latent_loss'],0.)
            self.assertTrue((root/'full/deployment/end_allocation/tiers.npy').exists())
            self.assertTrue((root/'full/images/final/deployment_view0000_tiers.png').exists())
            summary=json.loads((root/'full/comparison/summary.json').read_text())
            self.assertEqual({r['phase'] for r in summary['rows']},{'initial','end_allocation','final'})
            self.assertEqual(len(summary['rows']),12)
            self.assertTrue((root/'full/comparison/endpoints.csv').exists())
            self.assertTrue((root/'full/charts/tier_shares.png').exists())
            test=[json.loads(s) for s in (root/'full/metrics/test.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['scale_stream_bytes']>0 and 'source_lpips' in r for r in test))
            # Include an endpoint interrupted after its first metric row and
            # allocation-history write; exact resume must repair both.
            for stop in (2,3,'evaluation'):
                dest=root/f'resume{stop}';args=options(dest)
                if stop=='evaluation':
                    real=entry.evaluate
                    def interrupt(*a,**kw):
                        real(*a,**kw)
                        raise RuntimeError('simulated evaluation interruption')
                    with patch.object(entry,'evaluate',side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError,'simulated evaluation'):entry.main(args)
                else:entry.main(args,max_updates=stop)
                args.resume=True;entry.main(args)
                resumed=torch.load(dest/'checkpoints/final/state.pt',weights_only=True)
                for n,v in final['state_dict'].items():torch.testing.assert_close(v,resumed['state_dict'][n],rtol=0,atol=0)
                for n,v in final['mask'].items():torch.testing.assert_close(v,resumed['mask'][n],rtol=0,atol=0)
                history=lambda p:[json.loads(s) for s in (p/'deployment/allocation_history.jsonl').read_text().splitlines()]
                self.assertEqual(history(root/'full'),history(dest))


if __name__=='__main__':unittest.main()
