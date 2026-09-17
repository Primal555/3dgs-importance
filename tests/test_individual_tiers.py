"""Stable source groups, per-row budgets and bounded mixed-tier optimization."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from gaussian_jscc.codec import CodecConfig,GaussianCodec,prefix_mask
from gaussian_jscc.reference_geometry import ReferenceGeometry
from gaussian_jscc.data import prepare,to_features,write_ply
from gaussian_jscc.transport import transmit,receive,save_checkpoint,load_checkpoint,model_id
from gaussian_jscc.losses import reconstruction_loss,position_training_inputs
from gaussian_jscc.optimization import all_tier_attribute_step,training_layouts
from gaussian_jscc.training import full_scene_step
from test_gaussian_jscc import fixture


class IndividualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def scene(self,n=100):
        torch.manual_seed(12)
        raw,_=fixture(n,degree=0);raw,g,_=prepare(raw,16)
        m=GaussianCodec(CodecConfig(sh_degree=0,hidden=16,grid_dim=4,levels=(2,),depth=1,
            block_size=64,geometry_group_size=32,position_head='reference_v6',
            individual_tiers=True,loss_profile='robust_v4'))
        f,_=to_features(raw,g,m)
        return raw,g,m,f

    def test_identity_row_energy_sparse_and_drop(self):
        g=ReferenceGeometry((0,4,8,16),16,64,individual=True)
        for n in (1,23,24,64,65,130):
            x=torch.rand(n,3);q=torch.arange(n)%4;q[0]=1
            z=g.encode(x,q,10);y=g.decode(z,q,10)
            torch.testing.assert_close(y[q>0],x[q>0],atol=3e-6,rtol=3e-6)
            torch.testing.assert_close(z.square().sum(),g.rates[q].float().sum(),atol=5e-5,rtol=1e-6)
            torch.testing.assert_close(z,g.encode_choices(x,F.one_hot(q,4).float(),10))

    def test_drop_does_not_repartition_other_groups(self):
        g=ReferenceGeometry((0,4,8,16),16,32,individual=True)
        x=torch.rand(96,3);q=torch.ones(96,dtype=torch.long)
        a=g.encode(x,q,10);q[0]=0;b=g.encode(x,q,10)
        torch.testing.assert_close(a[32:],b[32:],rtol=0,atol=0)

    def test_positive_upgrade_only_changes_own_geometry_payload(self):
        g=ReferenceGeometry((0,4,8,16),16,64,individual=True)
        for layer in (g.encoder[-1],g.decoder[-1],g.reference_encoder[-1],g.reference_decoder[-1]):
            torch.nn.init.normal_(layer.weight,std=.02)
        x=torch.rand(64,3);q=torch.ones(64,dtype=torch.long)
        a=g.encode(x,q,10);q[4]=3;b=g.encode(x,q,10)
        other=torch.arange(64)!=4
        torch.testing.assert_close(a[other],b[other],rtol=0,atol=0)
        torch.testing.assert_close(a[:,:4],b[:,:4],rtol=0,atol=0)
        torch.testing.assert_close(g.decode(a,torch.ones_like(q),10)[other],g.decode(b,q,10)[other])

    def test_wire_and_dense_mixed_equivalence(self):
        raw,g,m,f=self.scene(64)
        q=torch.arange(64)%4
        logits=torch.randn(64,4,requires_grad=True)
        soft=logits.softmax(-1);choices=F.one_hot(q,4).float()-soft.detach()+soft
        dense,_,active=m.forward_tiers(f,f[:,:3],choices,10,'none')
        packed=m(f,f[:,:3],q,10,'none')
        torch.testing.assert_close(dense[q>0],packed[q>0],atol=3e-6,rtol=1e-5)
        loss=reconstruction_loss(dense,f,g,m,active=active,**position_training_inputs(m,f,choices,10))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);save_checkpoint(p/'codec.pt',m,0)
            fresh=load_checkpoint(p/'codec.pt','cpu')
            self.assertEqual(model_id(fresh),model_id(m))
            stats=transmit(m,raw,q,10,'none',42,p/'packet')
            recovered=receive(fresh,p/'packet')
            torch.testing.assert_close(recovered[:,:3],raw[q>0,:3],atol=3e-6,rtol=1e-5)
            self.assertEqual(stats['payload_complex_symbols'],sum(m.cfg.rates[int(t)] for t in q))

    def test_all_drop_finite_mask_gradient(self):
        _,g,m,f=self.scene(32)
        logits=torch.zeros(32,4,requires_grad=True);soft=logits.softmax(-1)
        c=F.one_hot(torch.zeros(32,dtype=torch.long),4).float()-soft.detach()+soft
        pred,_,active=m.forward_tiers(f,f[:,:3],c,10,'awgn')
        loss=reconstruction_loss(pred,f,g,m,active=active,**position_training_inputs(m,f,c,10))
        loss=loss+(pred*active[:,None]).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad.norm()),100)

    def test_mixed_training_and_finite_gradients(self):
        _,g,m,f=self.scene(64)
        layouts=training_layouts(m,f)
        self.assertEqual(sum(name.startswith('mixed') for name,q in layouts),3)
        loss,stats=all_tier_attribute_step(m,f,g,10,'awgn')
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_old_hash_unchanged_and_new_mode_explicit(self):
        cfg=CodecConfig(position_head='reference_v6')
        self.assertNotIn('individual_tiers',cfg.to_dict())
        cfg.individual_tiers=True
        self.assertTrue(cfg.to_dict()['individual_tiers'])

    def test_attribute_power_is_row_local_and_bounded(self):
        _,_,m,f=self.scene(64)
        h=torch.randn(64,m.cfg.hidden);q=torch.ones(64,dtype=torch.long)
        a=m.block_payload(h,f[:,:3],prefix_mask(q,m.cfg.rates).float(),10)
        q[4]=3;b=m.block_payload(h,f[:,:3],prefix_mask(q,m.cfg.rates).float(),10)
        other=torch.arange(64)!=4
        torch.testing.assert_close(a[other][:,m.attribute_slots],b[other][:,m.attribute_slots],rtol=0,atol=0)
        budget=(torch.tensor(m.cfg.rates)-torch.tensor(m.cfg.geometry_rates))[q]
        self.assertTrue((b[:,m.attribute_slots].square().sum(-1)<=budget+1e-5).all())

    def test_replay_matches_checkpoint_with_empty_packet(self):
        _,g,m,f=self.scene(128)
        q=torch.arange(128)%4;q[:64]=0
        batches=[(f[:64][None],q[:64][None]),(f[64:][None],q[64:][None])]
        models=[copy.deepcopy(m),copy.deepcopy(m)]
        values=[]
        for model,mode in zip(models,('replay','checkpoint')):
            torch.manual_seed(55)
            loss,_=full_scene_step(model,batches,g,10,'awgn',lambda raw:raw[:,:3].square().mean(),mode=mode)
            values.append(loss)
        torch.testing.assert_close(values[0],values[1])
        for a,b in zip(models[0].parameters(),models[1].parameters()):
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=3e-5,rtol=3e-4)

    def test_cli_explicit_upgrade_and_save(self):
        from gaussian_jscc.cli import main
        raw,_,m,_=self.scene(64)
        m.cfg.individual_tiers=False;m.block_geometry.individual=False
        m.cfg.loss_profile='position_v3'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_ply(root/'input.ply',raw,0);save_checkpoint(root/'old.pt',m,0)
            args=['train','--ply',str(root/'input.ply'),'--init',str(root/'old.pt'),
                  '--out',str(root/'train'),'--position-head','reference_v6','--individual-tiers',
                  '--loss-profile','robust_v4','--tier-training','all','--attribute-drop','0','--fixed-snr','10',
                  '--steps','1','--render-steps','0','--device','cpu','--training-data-device','cpu']
            with patch('gaussian_jscc.plots.safe_plot'),patch('sys.argv',['gaussian_jscc',*args]): main()
            trained=load_checkpoint(root/'train/codec.pt','cpu')
            self.assertTrue(trained.cfg.individual_tiers)
            self.assertEqual(trained.cfg.loss_profile,'robust_v4')


if __name__=='__main__': unittest.main()
