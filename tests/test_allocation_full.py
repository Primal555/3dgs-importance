"""CPU integration checks; synthetic rendering is NOT quality evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from gaussian_jscc.allocation import GaussianTierMask
from gaussian_jscc.allocation_diagnostics import (load_existence_prior, AllocationCostMeter,
                                                prior_ranked_tiers, record_allocation)
from gaussian_jscc.position_delivery import PositionCostMeter, encode_positions
from gaussian_jscc.learned_training import discrete_joint_step
from gaussian_jscc.render_validation import validate_render
from gaussian_jscc.render_plots import plot_render_training
from gaussian_jscc.render_objective import MultiViewRenderTask
from gaussian_jscc.data import write_ply, read_ply
from gaussian_jscc.transport import load_checkpoint, model_id
from gaussian_jscc.route2 import load_mask
from test_dense_prefix import dense
from test_render_first import synthetic_render, Reference
import test_render_first as rf
import test_training_launcher as launcher


class AllocationFullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prior_column_and_missing_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'source.ply'
            rows=np.array([(2.,-1.),(-1.,2.)],dtype=[('masks_0','f4'),('masks_1','f4')])
            PlyData([PlyElement.describe(rows,'vertex')]).write(str(path))
            prior,info=load_existence_prior('ply',path,2)
            torch.testing.assert_close(prior,torch.tensor([[2.,-1.],[-1.,2.]]).softmax(-1)[:,0])
            raw,_,_,_=dense(8)
            write_ply(path,raw,0)
            prior,info=load_existence_prior('auto',path,8)
            self.assertIsNone(prior)
            self.assertIn('no mask logits',info['reason'])
            with self.assertRaises(ValueError):
                load_existence_prior('ply',path,8)
            np.save(Path(temp)/'bad.npy',np.ones(7))
            with self.assertRaises(ValueError):
                load_existence_prior(Path(temp)/'bad.npy',path,8)

    def test_rate_charges_actual_xyz_and_tier_stream(self):
        raw,g,_,model=dense(16)
        unit=g.normalize(raw[:,:3])
        meter=AllocationCostMeter(model.cfg,PositionCostMeter(model.cfg,unit),len(raw),2.)
        q=torch.arange(16)%9
        stats=meter.details(q)
        self.assertEqual(stats['position_stream_bytes'],len(encode_positions(unit,q,model.cfg)))
        payload=int(torch.tensor(model.cfg.rates)[q].sum())/len(q)
        self.assertAlmostEqual(stats['allocation_uses_per_source_gaussian'],payload+stats['allocation_side_uses_per_source_gaussian'])
        self.assertGreater(stats['tier_map_proxy_bytes'],0)
        self.assertGreater(meter.details(torch.zeros_like(q))['allocation_uses_per_source_gaussian'],0)

    def test_ranking_same_histogram_and_snapshot_order(self):
        mask=GaussianTierMask(5)
        wanted=torch.tensor([3,0,2,1,3])
        with torch.no_grad():
            mask.logits.copy_(torch.nn.functional.one_hot(wanted,4)*20.)
        prior=torch.tensor([.1,.9,.5,.2,.6])
        ranked=prior_ranked_tiers(prior,wanted)
        torch.testing.assert_close(torch.bincount(wanted),torch.bincount(ranked))
        self.assertEqual(int(ranked[1]),3)
        self.assertEqual(int(ranked[0]),0)
        with tempfile.TemporaryDirectory() as temp:
            q,info=record_allocation(temp,0,mask,10,(0,8,16,32),prior)
            torch.testing.assert_close(q,wanted)
            np.testing.assert_array_equal(np.load(Path(temp)/'allocation_latest'/'tiers.npy'),wanted.numpy())
            _,info=record_allocation(temp,1,mask,10,(0,8,16,32),prior)
            self.assertEqual(info['changed_since_previous'],0)
            self.assertEqual(sum(map(sum,info['existence_decile_tier_counts'])),5)

    def test_mask_only_and_joint_have_separate_gradients(self):
        raw,g,f,model=dense(8)
        mask=GaussianTierMask(8,tier_count=9)
        meter=AllocationCostMeter(model.cfg,PositionCostMeter(model.cfg,g.normalize(raw[:,:3])),8,2.)
        task=MultiViewRenderTask(rf.RenderFirstTests().cameras(),Reference(),0)
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            for train_codec in (False,True):
                model.zero_grad(set_to_none=True)
                mask.zero_grad(set_to_none=True)
                torch.manual_seed(29)
                before=model_id(model)
                loss,stats=discrete_joint_step(model,mask,[f[None]],[torch.arange(8)[None]],g,10,'awgn',task,
                                               beta=.01,rate_meter=meter,train_codec=train_codec,paired_noise=True)
                self.assertEqual(before,model_id(model))
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(float(mask.logits.grad.norm()),0)
                self.assertEqual(any(p.grad is not None for p in model.parameters()),train_codec)
                self.assertEqual(stats['codec_updated'],train_codec)
                self.assertTrue(stats['paired_mask_channel_noise'])
                self.assertTrue(all(v>0 for v in stats['sample_side_uses_per_gaussian']))

    def test_cost_only_training_reduces_expected_payload(self):
        raw,g,f,model=dense(8)
        mask=GaussianTierMask(8,tier_count=9)
        optimizer=torch.optim.Adam(mask.parameters(),lr=.1)
        rates=torch.tensor(model.cfg.rates)
        def expected():
            return float((mask.logits.softmax(-1)*rates).sum(-1).mean().detach())
        initial=expected()
        meter=AllocationCostMeter(model.cfg,PositionCostMeter(model.cfg,g.normalize(raw[:,:3])),8,2.)
        for step in range(30):
            torch.manual_seed(step)
            optimizer.zero_grad(set_to_none=True)
            discrete_joint_step(model,mask,[f[None]],[torch.arange(8)[None]],g,10,'none',
                                lambda scene,ids:scene.new_tensor(.1),beta=1.,rate_meter=meter,train_codec=False)
            optimizer.step()
        self.assertLess(expected(),initial*.7)

    def test_discrete_side_cost_changes_policy_gradient(self):
        _,g,f,model=dense(8)
        mask=GaussianTierMask(8,existence_prior=torch.full((8,),.5),tier_count=9)
        class Meter:
            normalizer=32.
            def details(self,q):
                return {'allocation_side_uses_per_source_gaussian':float((q>0).sum())**2}
        gradients=[]
        for meter in (None,Meter()):
            mask.zero_grad(set_to_none=True)
            torch.manual_seed(37)
            discrete_joint_step(model,mask,[f[None]],[torch.arange(8)[None]],g,10,'none',
                                lambda scene,ids:scene.new_tensor(.1),beta=1.,rate_meter=meter,
                                train_codec=False,samples=4)
            gradients.append(mask.logits.grad.clone())
        self.assertGreater(float((gradients[1]-gradients[0]).norm()),1e-5)

    def test_full_three_stage_fixed_coordinates(self):
        from gaussian_jscc.cli import main
        raw,_,_,_=dense(24)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'source.ply',raw,0)
            np.save(root/'prior.npy',np.linspace(.2,.9,24,dtype=np.float32))
            run=root/'run'
            argv=['jscc','train-learned','--ply',str(root/'source.ply'),'--out',str(run),'--source','mock',
                  '--device','cuda','--rates','0','8','16','32','--prefix-mode','progressive',
                  '--position-delivery','quantized','--position-bits','16','--position-compression','delta_zlib',
                  '--bootstrap-steps','1','--bootstrap-objective','local-response','--render-steps','1',
                  '--joint-steps','2','--mask-only-steps','1','--existence-prior',str(root/'prior.npy'),
                  '--validate-every','1','--validation-views','1','--validation-trials','1','--save-every','1',
                  '--views-per-step','1','--blocks-per-batch','2','--block-size','8','--decoder-window','4',
                  '--hidden','16','--depth','1','--grid-dim','4','--levels','2','--patience','0']
            with patch('sys.argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=rf.RenderFirstTests().cameras()*2), \
                 patch('gaussian_jscc.plots.safe_plot'):
                main()
            loss=[json.loads(line) for line in (run/'loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in loss],['bootstrap','render','joint','joint'])
            self.assertEqual(loss[-2]['update_norm'],0.)
            self.assertIsNone(loss[1]['position_stream_bytes'])
            for r in loss[-2:]:
                self.assertGreater(r['rate_accounting_seconds'],0)
                self.assertGreater(r['position_compression_seconds'],0)
                self.assertGreater(r['tier_map_compression_seconds'],0)
                self.assertGreater(r['rate_loss'],0)
            self.assertGreater(loss[-1]['update_norm'],0.)
            self.assertEqual(model_id(load_checkpoint(run/'codec_end_render.pt','cpu')),
                             model_id(load_checkpoint(run/'codec_3.pt','cpu')))
            model=load_checkpoint(run/'codec_best_joint.pt','cpu')
            self.assertEqual(model.cfg.position_compression_level,6)
            source,_=read_ply(root/'source.ply')
            loaded=load_mask(run/'route2_best_joint.pt',source,model,'cpu')
            self.assertEqual(len(loaded.logits),24)
            checks=[json.loads(line) for line in (run/'validation.jsonl').read_text().splitlines()]
            last=checks[-1]
            mask=next(e for e in last['layouts'] if e['layout']=='mask')
            ranked=next(e for e in last['layouts'] if e['layout']=='prior_ranked')
            self.assertEqual(mask['tier_counts'],ranked['tier_counts'])
            record=json.loads((run/'training.json').read_text())
            self.assertAlmostEqual(last['score'],mask['source_mse']+record['beta']*mask['allocation_uses_per_source_gaussian']/record['allocation_rate_normalizer'])
            plot_render_training(run)
            for name in ('allocation_history.png','allocation_history.csv','existence_vs_allocation.png'):
                self.assertTrue((run/'charts'/name).exists())

    def test_full_launcher(self):
        result=launcher.LauncherTests().launch({'CUDA_VISIBLE_DEVICES':'2'},script='scripts/train_progressive16_mask_full.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        for text in ('--rates 0 8 16 32','--joint-steps 3000','--mask-only-steps 500','--existence-prior auto',
                     '--beta 0.01','codec_best_joint.pt','route2_best_joint.pt','--save-images'):
            self.assertIn(text,result.stdout)


if __name__=='__main__':
    unittest.main()
