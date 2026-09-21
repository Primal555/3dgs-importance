"""Received-only relation Transformer, protocol and gradient checks."""
import unittest
from unittest.mock import patch
import torch
import test_multiscale_codec as contracts
import test_q3_noiseless as q3_contracts
import test_position_path_diagnosis as diagnosis
from test_point_attention import setup as old_setup
from gaussian_jscc.codec import CodecConfig,GaussianCodec
from gaussian_jscc.decoder_attention import ReceivedPointBlock,feature_neighbors


def setup(n=17):
    raw,g,f,old=old_setup(n)
    cfg=old.cfg.to_dict();cfg['decoder_attention']='feature_point'
    m=GaussianCodec(CodecConfig.from_dict(cfg))
    m.attr_mean.copy_(old.attr_mean);m.attr_std.copy_(old.attr_std)
    return raw,g,f,m


class DecoderPointCLI(q3_contracts.Q3NoiselessTests):
    decoder_attention='feature_point'


class DecoderPointDiagnosis(diagnosis.PositionPathTests):
    decoder_attention='feature_point'


class DecoderPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_neighbors_use_features_not_slot_distance(self):
        x=torch.tensor([[[1.,0],[0.,1],[-1.,0],[1.,.01]]])
        active=torch.tensor([[True,False,True,True]])
        ix,valid=feature_neighbors(x,active,2)
        self.assertEqual(set(ix[0,0].tolist()),{0,3})
        self.assertFalse(valid[0,1].any())
        self.assertTrue(valid[0,0].all())
        _,valid=feature_neighbors(x,torch.zeros_like(active),16)
        self.assertFalse(valid.any())

    def test_single_attention_block_equivariance_and_masked_gradients(self):
        _,_,_,m=setup()
        layer=ReceivedPointBlock(m.cfg)
        h=torch.randn(2,17,m.cfg.hidden,requires_grad=True)
        active=torch.ones(2,17,dtype=torch.bool);active[0,3]=False;active[1]=False
        y=layer(h,active)
        perm=torch.randperm(17)
        torch.testing.assert_close(layer(h[:,perm],active[:,perm]),y[:,perm],atol=2e-6,rtol=2e-5)
        self.assertEqual(float(y[~active].detach().abs().sum()),0)
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertEqual(float(h.grad[~active].abs().sum()),0)
        self.assertGreater(float(layer.relative[0].weight.grad.norm()),0)
        self.assertGreater(float(layer.relation[-1][-1].weight.grad.norm()),0)

    def test_encoder_and_output_head_initialization_unchanged(self):
        _,_,f,old=old_setup()
        cfg=old.cfg.to_dict()
        torch.manual_seed(42);a=GaussianCodec(CodecConfig.from_dict(cfg))
        cfg['decoder_attention']='feature_point'
        torch.manual_seed(42);b=GaussianCodec(CodecConfig.from_dict(cfg))
        q=torch.full((1,17),3,dtype=torch.long)
        torch.testing.assert_close(a.learned.encode(f[None],f[None,:,:3],q,10),
                                   b.learned.encode(f[None],f[None,:,:3],q,10),atol=0,rtol=0)
        for key,value in a.state_dict().items():
            if not key.startswith(('learned.dec_geometry_context.fine','learned.dec_geometry_context.coarse',
                                   'learned.dec_appearance_context.fine','learned.dec_appearance_context.coarse')):
                torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
        self.assertNotIn('decoder_attention',a.cfg.to_dict())
        with self.assertRaisesRegex(ValueError,'requires'):
            CodecConfig(decoder_attention='feature_point')

    def test_receiver_transport_replay_and_short_fit(self):
        for name in ('test_transport_power_drop_and_replay_contracts',
                     'test_zero_context_gates_leave_pointwise_information_path',
                     'test_mixed_tiers_batching_and_all_heads_receive_gradients',
                     'test_short_fit_and_all_backward_modes'):
            with self.subTest(name=name),patch('test_multiscale_codec.setup',setup):
                getattr(contracts.MultiScaleTests(name),name)()

    def test_full_size_block_and_centroid_output(self):
        for mode in ('additive','block_center'):
            cfg=CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',
                            encoder_attention='geometric_point',decoder_attention='feature_point',xyz_decoder=mode)
            m=GaussianCodec(cfg)
            features=torch.randn(2,256,cfg.attr_dim+3)
            q=torch.randint(0,4,(2,256));q[1]=0
            out=m.forward_tier_batches(features,features[...,:3],torch.nn.functional.one_hot(q,4).float(),10,'awgn')[0]
            out.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            self.assertEqual(float(out[1].detach().abs().sum()),0)

    def test_local_comparison_attention_selection(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from scripts.compare_xyz_decoders_local import run,summarize
        from gaussian_jscc.data import write_ply
        from gaussian_jscc.transport import load_checkpoint
        raw,_,_,_=old_setup(1024)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_ply(root/'input.ply',raw,0)
            args=SimpleNamespace(ply=str(root/'input.ply'),out=str(root/'run'),steps=1,
                                 every=1,threads=1,seeds=[42],modes=['additive'],blocks=[0,1,2,3],
                                 decoder_attentions=['window','feature_point'])
            result=run(args)
            self.assertEqual(set(result),{'additive_seed42','feature_point__additive_seed42'})
            m=load_checkpoint(root/'run/feature_point__additive_seed42/codec.pt','cpu')
            self.assertEqual(m.cfg.decoder_attention,'feature_point')
            report=summarize([root/'run'],root/'report',tail_checks=1)
            self.assertEqual(report['paired_decoder_attention'][0]['reference'],'additive')
            self.assertTrue((root/'report/comparison.png').is_file())


if __name__=='__main__':
    unittest.main()
