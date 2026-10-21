"""Wire-scale, causal-prefix and real PyTorch channel boundary checks."""
import tempfile
import unittest
from pathlib import Path

import torch


class SuccessiveChannelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(21)

    def model(self):
        from gaussian_jscc.modular_config import ModularCodecConfig
        from gaussian_jscc.model_factory import build_codec
        return build_codec(ModularCodecConfig(channel_variant='successive',channel_prefix_mode='full_prefix',
            channel_power_mode='block',channel_neighbor_mode='plain',latent_dim=8,channel_hidden=16,
            hidden=16,block_size=8,depth=2,grid_dim=4,levels=(2,),attention_heads=4,decoder_window=4))

    def test_scale_wire_restores_only_present_segments_and_rejects_corruption(self):
        from gaussian_jscc.successive_channel import encode_scales,decode_scales
        q=torch.tensor([[1,0,0],[2,1,0]])
        scales=torch.tensor([[2.,9.,8.],[4.,.5,7.]])
        data=encode_scales(scales,q)
        self.assertEqual(len(data),14)  # 8 framing + three binary16 values.
        got=decode_scales(data,q)
        torch.testing.assert_close(got,torch.tensor([[2.,1.,1.],[4.,.5,1.]]))
        for broken in (data[:-1],data+b'0',b'BAD!'+data[4:]):
            with self.assertRaises(ValueError):decode_scales(broken,q)

    def test_block_power_and_padding_preserve_common_amplitude_in_scale_frame(self):
        from gaussian_jscc.successive_channel import normalize_segment
        raw=torch.tensor([[[1.,0.],[3.,0.],[123.,99.]]],requires_grad=True)
        donors=torch.tensor([[True,True,False]])
        signal,scale=normalize_segment(raw,donors,.01)
        self.assertLessEqual(float(signal.detach().square().sum()/2),1.)
        self.assertTrue(torch.equal(signal[:,2],torch.zeros(1,2)))
        recovered=signal*scale[:,None,None]
        torch.testing.assert_close(recovered[:,:2],raw[:,:2])
        _,double_scale=normalize_segment(raw*2,donors,.01)
        self.assertGreater(float(double_scale.detach()),float(scale.detach())*1.9)
        recovered.square().sum().backward()
        self.assertGreater(float(raw.grad[:,:2].abs().sum()),0.)
        self.assertEqual(float(raw.grad[:,2].abs().sum()),0.)

    def test_receiver_packet_matches_training_and_lower_tiers_ignore_future_slots(self):
        model=self.model().eval()
        y=torch.randn(2,8,8);q=torch.tensor([[1,2,3,1,0,0,0,0],[3,2,1,3,2,1,3,0]])
        xyz=torch.rand(2,8,3)
        packet=model.encode_channel_packet(y,q,10.)
        got=model.decode_channel_packet(packet['payload'],q,10.,packet['scales'],xyz)
        signal,scales=model.channel_encoder(y,q,10.,model.channel_decoder)
        mask=torch.arange(signal.shape[-1])[None,None,:]<2*torch.tensor(model.cfg.rates)[q][...,None]
        expected=model.channel_decoder(signal,mask,q,10.,scales=scales,delivered_xyz=xyz)
        torch.testing.assert_close(got,expected,rtol=0,atol=0)
        changed=signal.clone();changed[q==1,8:]=999.
        altered=scales.clone();altered[:,1:]=777.
        # Isolate stage1 with all-positive q1; no future scale may enter it.
        ones=(q>0).long(); base_mask=torch.arange(signal.shape[-1])[None,None,:]<2*torch.tensor(model.cfg.rates)[ones][...,None]
        a=model.channel_decoder(signal,base_mask,ones,10.,scales=scales)
        b=model.channel_decoder(changed,base_mask,ones,10.,scales=altered)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(got[q==0],torch.zeros_like(got[q==0])))
        with self.assertRaises(ValueError):model.decode_channel_packet(packet['payload'],q,10.,b'',xyz)

    def test_local_proxy_is_detached_but_receiver_trains_every_channel_stage(self):
        model=self.model()
        y=torch.randn(1,8,8);q=torch.full((1,8),3,dtype=torch.long)
        signal,scales=model.channel_encoder(y,q,10.,model.channel_decoder)
        signal.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.channel_decoder.parameters()))
        model.zero_grad(set_to_none=True)
        signal,scales=model.channel_encoder(y,q,10.,model.channel_decoder)
        mask=torch.ones_like(signal,dtype=torch.bool)
        stages,_=model.channel_decoder(signal,mask,q,10.,scales=scales,return_stages=True)
        (stages-y[...,None,:]).square().mean().backward()
        for module in (*model.channel_encoder.stages,*model.channel_decoder.stages):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0.)

    def test_checkpoint_factory_and_legacy_symbol_only_boundary(self):
        from gaussian_jscc.transport import save_checkpoint,load_checkpoint
        model=self.model()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'codec.pt';save_checkpoint(path,model,0)
            other=load_checkpoint(path,'cpu')
            for name,value in model.state_dict().items():
                torch.testing.assert_close(value,other.state_dict()[name],rtol=0,atol=0)
        with self.assertRaises(ValueError):model.encode(torch.randn(8,59),torch.rand(8,3),torch.ones(8,dtype=torch.long),10.)

    def test_empty_donors_padding_and_missing_scale_are_receiver_safe(self):
        model=self.model().eval()
        y=torch.randn(1,4,8);q=torch.full((1,4),3,dtype=torch.long)
        signal,scales=model.channel_encoder(y,q,10.,model.channel_decoder)
        padded=torch.cat((y,torch.full_like(y,123.)),1)
        padded_q=torch.cat((q,torch.zeros_like(q)),1)
        actual,actual_scales=model.channel_encoder(padded,padded_q,10.,model.channel_decoder)
        torch.testing.assert_close(actual[:,:4],signal,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(actual_scales,scales,rtol=0,atol=0)
        zero_q=torch.zeros_like(q);packet=model.encode_channel_packet(y,zero_q,10.)
        self.assertEqual(packet['payload'].shape,(0,2));self.assertEqual(len(packet['scales']),8)
        result=model.decode_channel_packet(packet['payload'],zero_q,10.,packet['scales'])
        self.assertTrue(torch.equal(result,torch.zeros_like(result)))
        with self.assertRaises(ValueError):
            model.channel_decoder(signal,torch.ones_like(signal,dtype=torch.bool),q,10.)
        damaged=signal.clone();damaged[0,0,0]=float('nan')
        with self.assertRaises(ValueError):
            model.channel_decoder(damaged,torch.ones_like(signal,dtype=torch.bool),q,10.,scales=scales)


if __name__=='__main__':unittest.main()
