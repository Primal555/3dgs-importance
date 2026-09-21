"""Received-only centroid path: constraints, compatibility and real CPU autograd."""
import unittest
from unittest.mock import patch
import torch
import test_multiscale_codec as contracts
import test_q3_noiseless as q3_contracts
import test_position_path_diagnosis as diagnosis_contracts
from test_point_attention import setup as old_setup
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.multiscale_codec import masked_mean, zero_mean


def setup(n=17):
    raw,g,f,old=old_setup(n)
    cfg=old.cfg.to_dict();cfg['xyz_decoder']='block_center'
    model=GaussianCodec(CodecConfig.from_dict(cfg))
    model.attr_mean.copy_(old.attr_mean);model.attr_std.copy_(old.attr_std)
    return raw,g,f,model


class BlockCenterCLI(q3_contracts.Q3NoiselessTests):
    xyz_decoder='block_center'


class BlockCenterDiagnosis(diagnosis_contracts.PositionPathTests):
    xyz_decoder='block_center'


class BlockCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masked_centering_and_empty_block(self):
        x=torch.randn(2,8,3)
        active=torch.tensor([[True,False,True,True,False,True,False,False],[False]*8])
        centered=zero_mean(x,active)
        torch.testing.assert_close(masked_mean(centered,active),torch.zeros(2,1,3),atol=1e-7,rtol=0)
        self.assertEqual(float(centered[~active].abs().sum()),0.)
        self.assertTrue(torch.isfinite(centered).all())

    def test_center_is_received_only_and_context_cannot_translate(self):
        _,_,f,m=setup(17)
        q=torch.full((1,17),3,dtype=torch.long);q[:,2]=0
        z=m.learned.encode(f[None],f[None,:,:3],q,10)
        capture={}
        hook=m.learned.dec_xyz_center.register_forward_hook(lambda module,args,value:capture.update(center=value))
        out=m.learned.decode(z,q,10)
        torch.testing.assert_close(masked_mean(out[...,:3],q>0),capture['center'])
        out[...,:3].sum().backward()
        self.assertGreater(float(m.learned.dec_xyz_center[-1].weight.grad.norm()),0)
        self.assertLess(float(m.learned.context_heads['xyz'].weight.grad.norm()),1e-4)
        with torch.no_grad():
            m.learned.dec_geometry_gate.fill_(2)
            m.learned.context_heads['xyz'].weight.mul_(3)
        changed=m.learned.decode(z.detach(),q,10)
        torch.testing.assert_close(masked_mean(changed[...,:3],q>0),masked_mean(out[...,:3],q>0))
        hook.remove()

    def test_unchanged_encoder_and_other_outputs_at_same_seed(self):
        _,_,f,old=old_setup(17)
        cfg=old.cfg.to_dict()
        torch.manual_seed(91);baseline=GaussianCodec(CodecConfig.from_dict(cfg))
        cfg['xyz_decoder']='block_center'
        torch.manual_seed(91);new=GaussianCodec(CodecConfig.from_dict(cfg))
        q=torch.full((1,17),3,dtype=torch.long)
        z=baseline.learned.encode(f[None],f[None,:,:3],q,10)
        torch.testing.assert_close(z,new.learned.encode(f[None],f[None,:,:3],q,10),rtol=0,atol=0)
        torch.testing.assert_close(baseline.learned.decode(z,q,10)[...,3:],new.learned.decode(z,q,10)[...,3:],rtol=0,atol=0)
        self.assertNotIn('xyz_decoder',baseline.cfg.to_dict())
        self.assertEqual(new.cfg.to_dict()['xyz_decoder'],'block_center')
        loaded=GaussianCodec(CodecConfig.from_dict(new.cfg.to_dict()))
        loaded.load_state_dict(new.state_dict(),strict=True)

    def test_transport_gradients_fit_and_all_backward_modes(self):
        for name in ('test_transport_power_drop_and_replay_contracts',
                     'test_mixed_tiers_batching_and_all_heads_receive_gradients',
                     'test_short_fit_and_all_backward_modes'):
            with self.subTest(name=name),patch('test_multiscale_codec.setup',setup):
                getattr(contracts.MultiScaleTests(name),name)()


if __name__=='__main__':
    unittest.main()
