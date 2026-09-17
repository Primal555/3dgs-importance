"""V6: wire equivalence, common-reference gradients, phase-continuous objectives."""
import copy
from pathlib import Path
import tempfile
import unittest
import json
import sys
from unittest.mock import patch
import torch
from torch.nn import functional as F
from gaussian_jscc.reference_geometry import ReferenceGeometry, reference_position_rows
from gaussian_jscc.codec import CodecConfig,GaussianCodec,channel
from gaussian_jscc.data import prepare,to_features,write_ply
from gaussian_jscc.losses import reconstruction_loss,position_training_inputs
from gaussian_jscc.training import full_scene_step,joint_scene_step,codec_batch
from gaussian_jscc.optimization import all_tier_geometry_step
from gaussian_jscc.transport import save_checkpoint,load_checkpoint,transmit,receive,model_id
from gaussian_jscc.allocation import GaussianTierMask
from test_gaussian_jscc import fixture


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def scene(self,n=96):
        torch.manual_seed(123)
        raw,_=fixture(n,degree=0);raw,geometry,_=prepare(raw,16)
        model=GaussianCodec(CodecConfig(sh_degree=0,hidden=16,grid_dim=4,depth=1,levels=(2,3),
                            block_size=64,geometry_group_size=32,position_head='reference_v6',loss_profile='position_v3'))
        f,_=to_features(raw,geometry,model)
        return raw,model,geometry,f

    def test_noiseless_identity_power_mixed_drop_and_tiny_groups(self):
        torch.manual_seed(42);g=ReferenceGeometry((0,4,8,16),16,64)
        for n in (1,23,24,65,257):
            x=torch.rand(n,3);q=torch.arange(n)%4
            q[0]=1
            z=g.encode(x,q,10);y=g.decode(z,q,10)
            torch.testing.assert_close(y[q>0],x[q>0],atol=3e-6,rtol=3e-6)
            self.assertAlmostEqual(float(z.detach().square().sum()/g.rates[q].sum()),1.,places=5)
            c=F.one_hot(q,4).float()
            torch.testing.assert_close(z,g.encode_choices(x,c,10),atol=3e-6,rtol=3e-6)
        for p in (g.encoder[-1],g.reference_encoder[-1]):
            torch.nn.init.normal_(p.weight,std=.1)
        z=g.encode(x,q,10)
        self.assertAlmostEqual(float(z.detach().square().sum()/g.rates[q].sum()),1.,places=5)

    def test_source_free_fresh_receiver_and_old_hash(self):
        raw,model,_,_=self.scene()
        q=torch.arange(len(raw))%4
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save_checkpoint(root/'codec.pt',model,0)
            fresh=load_checkpoint(root/'codec.pt','cpu')
            self.assertEqual(model_id(model),model_id(fresh))
            for kind in ('none','awgn'):
                stats=transmit(model,raw,q,10,kind,42,root/kind)
                a=receive(fresh,root/kind);b=receive(model,root/kind)
                torch.testing.assert_close(a,b)
                self.assertFalse(stats['block_reference_in_metadata'])
                self.assertEqual(stats['payload_complex_symbols'],sum(model.cfg.rates[int(t)] for t in q))
                if kind=='none': torch.testing.assert_close(a[:,:3],raw[q>0,:3],atol=3e-6,rtol=3e-6)

    def test_st_forward_matches_packed_and_receives_quality_gradient(self):
        _,model,geometry,f=self.scene(64)
        logits=torch.randn(64,4,requires_grad=True)
        choices=F.gumbel_softmax(logits,hard=True)
        q=choices.detach().argmax(-1);keep=q>0
        pred,_,active=model.forward_tiers(f,f[:,:3],choices,10,'none')
        expected=model(f[keep],f[keep,:3],q[keep],10,'none')
        torch.testing.assert_close(pred[keep],expected,atol=3e-6,rtol=5e-5)
        # Nonzero gradients without any rate penalty, including positive tiers.
        objective=(pred*active[:,None]-f).square().mean()
        objective.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[:,1:].abs().sum()),1e-7)

    def test_actual_group_supervision_not_parent_block(self):
        _,model,geometry,f=self.scene(64)
        f[:32,:3]=torch.rand(32,3)*.01
        f[32:,:3]=.9+torch.rand(32,3)*.01
        q=torch.ones(64,dtype=torch.long)
        scale=model.block_geometry.supervision_scale(f[:,:3],q)
        self.assertLess(float(scale.max()),.011)
        pred=(f.clone()+.001).requires_grad_()
        loss=reference_position_rows(pred,f,geometry,model,scale).mean();loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        for i in (0,32):
            _,r=model.block_geometry.reference(f[i:i+32,:3],torch.ones(32))
            torch.testing.assert_close(scale[i:i+32],r.reshape(()).expand(32).clamp_min(.001))

    def test_reference_branches_receive_finite_gradients(self):
        _,model,geometry,f=self.scene(64)
        for n,p in model.named_parameters(): p.requires_grad_(n.startswith('block_geometry.'))
        loss,stats=all_tier_geometry_step(model,f,geometry,10,'awgn')
        self.assertIn('mixed',stats['per_tier'])
        for sub in (model.block_geometry.reference_encoder,model.block_geometry.reference_decoder):
            grads=[p.grad for p in sub.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(t).all() for t in grads))
            self.assertGreater(sum(float(t.abs().sum()) for t in grads),0)

    def test_geometry_objective_unchanged_by_render_attribute_weight(self):
        _,model,geometry,f=self.scene(64)
        batches=[(f[None],torch.full((1,64),2))]
        norms=[]
        for weight in (0.,.1,1.):
            model.zero_grad(set_to_none=True);torch.manual_seed(42)
            loss,stats=full_scene_step(model,batches,geometry,10,'awgn',lambda raw:raw.sum()*0,
                                     attr_weight=weight,mode='replay')
            # Turn off attribute objectives for exact geometry-only comparison.
            norms.append((loss,stats))
            self.assertGreater(stats['geometry_loss'],0.)
            expected=weight*stats['aux_loss']+(1-weight)*model.cfg.geometry_weight*stats['geometry_loss']
            self.assertAlmostEqual(float(loss),expected,places=4)

    def test_replay_checkpoint_joint_gradients(self):
        _,model,geometry,f=self.scene(64)
        for joint in (False,True):
            mask=GaussianTierMask(64)
            results=[]
            for mode in ('replay','checkpoint'):
                model.zero_grad(set_to_none=True);mask.zero_grad(set_to_none=True);torch.manual_seed(42)
                if joint:
                    loss,_=joint_scene_step(model,mask,[(f[None],torch.arange(64)[None])],geometry,10,'awgn',
                         lambda raw,a:(raw*a[:,None]).square().mean(),beta=0.,attr_weight=.1,mode=mode)
                else:
                    loss,_=full_scene_step(model,[(f[None],torch.full((1,64),2))],geometry,10,'awgn',
                                           lambda raw:raw.square().mean(),attr_weight=.1,mode=mode)
                grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
                if joint:
                    self.assertGreater(float(mask.logits.grad.abs().sum()),0.)
                    grads['mask']=mask.logits.grad.clone()
                results.append((loss,grads))
            torch.testing.assert_close(results[0][0],results[1][0])
            for n in results[0][1]:
                torch.testing.assert_close(results[0][1][n],results[1][1][n],atol=1e-4,rtol=1e-3)

    def test_cli_geometry_attribute_joint_checkpoint_handoff(self):
        from gaussian_jscc.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,model,_,_=self.scene(64)
            write_ply(root/'source.ply',raw,0);save_checkpoint(root/'init.pt',model,0)
            common=['--ply',str(root/'source.ply'),'--device','cpu','--position-head','reference_v6',
                    '--loss-profile','position_v3','--snr-range','10','10','--clip-mode','branch']
            commands=[['train',*common,'--init',str(root/'init.pt'),'--out',str(root/'geometry'),
                       '--geometry-only','--fixed-snr','10','--tier-training','all','--attribute-drop','0','--steps','2'],
                      ['train',*common,'--init',str(root/'geometry/codec.pt'),'--out',str(root/'attribute'),
                       '--tier-training','all','--attribute-drop','0','--steps','2'],
                      ['train-route2',*common,'--codec-init',str(root/'attribute/codec.pt'),
                       '--out',str(root/'joint'),'--attribute-only','--warmup-steps','1','--joint-steps','2']]
            for command in commands:
                with patch.object(sys,'argv',['codec',*command]),patch('gaussian_jscc.plots.safe_plot'):
                    main()
            trained=load_checkpoint(root/'joint/codec.pt','cpu')
            self.assertTrue(trained.cfg.reference_bounded)
            self.assertEqual(trained.cfg.geometry_clean_weight,1.)
            rows=[json.loads(line) for line in (root/'joint/loss.jsonl').read_text().splitlines()]
            self.assertGreater(rows[-1]['mask_grad_norm'],0.)


if __name__=='__main__': unittest.main()
