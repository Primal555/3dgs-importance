"""Progressive XYZ: protocol, masking, geometry feedback, gradients and replay."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import test_transformer_trunk as trunk
import test_multiscale_codec as contracts
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.learned_train import bootstrap_split
from gaussian_jscc.transport import save_checkpoint, load_checkpoint


def setup(n=17):
    raw,g,f,base=trunk.setup(n)
    cfg=base.cfg.to_dict();cfg['decoder_refinement']='progressive'
    model=GaussianCodec(CodecConfig.from_dict(cfg))
    model.attr_mean.copy_(base.attr_mean);model.attr_std.copy_(base.attr_std)
    return raw,g,f,model


class ProgressiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mask_permutation_empty_and_distant_dependency(self):
        _,_,_,m=setup()
        q=torch.full((2,65),3,dtype=torch.long);q[0,1]=0;q[1]=0
        z=torch.randn(2,65,2*m.cfg.rates[-1],requires_grad=True)
        y=m.learned.decode(z,q,10)
        perm=torch.randperm(65)
        torch.testing.assert_close(m.learned.decode(z[:,perm],q[:,perm],10),y[:,perm],atol=3e-6,rtol=3e-5)
        bad=z.detach().clone();bad[q==0]=float('nan')
        torch.testing.assert_close(m.learned.decode(bad,q,10),y)
        self.assertEqual(float(y[q==0].detach().abs().sum()),0)
        y[0,0,:3].square().sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(float(z.grad[q==0].abs().sum()),0)
        self.assertGreater(float(z.grad[0,-1].norm()),0)
        self.assertEqual(m.learned.decode(z[:,:0],q[:,:0],10).shape,(2,0,m.cfg.attr_dim+3))

    def test_predicted_geometry_reenters_and_receives_gradient(self):
        _,_,_,m=setup()
        d=m.learned.dec_trunk
        packet=torch.randn(1,9,4*m.cfg.rates[-1],requires_grad=True)
        active=torch.ones(1,9,dtype=torch.bool)
        out,stages=d.forward_stages(packet,torch.zeros(1,9,m.cfg.hidden),active)
        self.assertEqual(len(stages),4)
        for xyz in stages: xyz.retain_grad()
        out.square().mean().backward()
        for xyz in stages:
            self.assertTrue(torch.isfinite(xyz.grad).all())
            self.assertGreater(float(xyz.grad.norm()),0)
        for name,p in d.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        for r in d.refiners:
            self.assertGreater(float(r.position[0].weight.grad.norm()),0)
            self.assertGreater(float(r.received.weight.grad.norm()),0)
            self.assertGreater(float(r.log_precision.grad.norm()),0)
        # Change only predicted geometry, keep features and received memory fixed.
        x=torch.randn(1,9,m.cfg.hidden);xyz=torch.randn(1,9,3)*.1
        a=d.refiners[0](x,x,xyz,active)[0]
        xyz[:,0]+=1
        b=d.refiners[0](x,x,xyz,active)[0]
        self.assertGreater(float((a-b).detach().abs().max()),1e-5)

    def test_fixed_holdout_point_sets(self):
        sets=[]
        for size in (256,512):
            train,val=bootstrap_split(16001,size,8,512)
            held={j for i in val for j in range(i*size,(i+1)*size)}
            fit={j for i in train for j in range(i*size,min((i+1)*size,16001))}
            self.assertFalse(held & fit)
            self.assertEqual(len(held|fit),16001)
            sets.append(held)
        self.assertEqual(sets[0],sets[1])
        for args in ((1024,256,2,512),(4096,512,2,256),(4096,256,2,300)):
            with self.assertRaises(ValueError):bootstrap_split(*args)

    def test_contracts_fit_replay_and_checkpoint(self):
        for name in ('test_transport_power_drop_and_replay_contracts','test_short_fit_and_all_backward_modes'):
            with self.subTest(name=name),patch('test_multiscale_codec.setup',setup):
                getattr(contracts.MultiScaleTests(name),name)()
        _,_,f,m=setup()
        q=torch.full((len(f),),3,dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'codec.pt';save_checkpoint(path,m,0,{})
            other=load_checkpoint(path,'cpu')
            torch.testing.assert_close(m(f,f[:,:3],q,10,'none'),other(f,f[:,:3],q,10,'none'))
            self.assertEqual(other.cfg.decoder_refinement,'progressive')
        self.assertNotIn('decoder_refinement',CodecConfig().to_dict())
        with self.assertRaises(ValueError):CodecConfig(decoder_refinement='progressive')

    def test_real_512_sh3_block(self):
        m=GaussianCodec(CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',
            encoder_attention='geometric_point',decoder_attention='transformer_trunk',decoder_refinement='progressive',block_size=512))
        f=torch.randn(2,512,m.cfg.attr_dim+3);q=torch.full((2,512),3,dtype=torch.long);q[1]=0
        y=m.forward_tier_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4).float(),10,'none')[0]
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(float(y[1].detach().abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


class ProgressiveCLI(trunk.TransformerTrunkCLI):
    decoder_refinement='progressive'


if __name__=='__main__':unittest.main()
