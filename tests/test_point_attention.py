"""Encoder-only geometric attention: contracts, gradients and CPU fitting."""
import unittest
from unittest.mock import patch
import torch
import test_multiscale_codec as baseline
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.point_attention import GeometricPointBlock, geometric_neighbors


def setup(n=17):
    raw,g,f,old=baseline_setup(n)
    cfg=old.cfg.to_dict();cfg['encoder_attention']='geometric_point'
    model=GaussianCodec(CodecConfig.from_dict(cfg))
    model.attr_mean.copy_(old.attr_mean);model.attr_std.copy_(old.attr_std)
    return raw,g,f,model


baseline_setup=baseline.setup


class PointAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_existing_transport_gradients_fit_and_backward_contracts(self):
        for name in ('test_transport_power_drop_and_replay_contracts',
                     'test_zero_context_gates_leave_pointwise_information_path',
                     'test_mixed_tiers_batching_and_all_heads_receive_gradients',
                     'test_short_fit_and_all_backward_modes'):
            with self.subTest(name=name),patch('test_multiscale_codec.setup',setup):
                getattr(baseline.MultiScaleTests(name),name)()

    def test_true_neighbors_ignore_slot_distance_and_dropped_points(self):
        xyz=torch.tensor([[[0.,0,0],[10.,0,0],[.001,0,0],[.01,0,0]]])
        active=torch.tensor([[True,True,False,True]])
        idx,valid,relative=geometric_neighbors(xyz,active,2)
        self.assertEqual(set(idx[0,0].tolist()),{0,3})
        self.assertTrue(valid[0,0].all())
        self.assertFalse(valid[0,2].any())
        self.assertTrue(torch.isfinite(relative).all())
        duplicate_idx,_,_=geometric_neighbors(torch.zeros(1,5,3),torch.ones(1,5,dtype=torch.bool),1)
        self.assertEqual(duplicate_idx[0,:,0].tolist(),list(range(5)))

    def test_single_scale_permutation_translation_and_q0_isolation(self):
        torch.manual_seed(9)
        cfg=CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',
                        encoder_attention='geometric_point',encoder_neighbors=4,hidden=16)
        block=GeometricPointBlock(cfg)
        h=torch.randn(2,17,16,requires_grad=True);xyz=torch.randn(2,17,3)
        active=torch.ones(2,17,dtype=torch.bool);active[:,4]=False;active[1]=False
        out=block(h,xyz,active)
        order=torch.randperm(17)
        torch.testing.assert_close(block(h[:,order],xyz[:,order],active[:,order]),out[:,order],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(block(h,xyz+3,active),out,atol=2e-6,rtol=2e-5)
        changed=xyz.clone();changed[~active]=1e8
        altered=h.detach().clone();altered[~active]=1e8
        torch.testing.assert_close(block(altered,changed,active),out)
        self.assertEqual(float(out[~active].detach().abs().sum()),0)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertGreater(float(block.position[0].weight.grad.norm()),0)
        self.assertGreater(float(block.relation[1][0].weight.grad.norm()),0)
        self.assertEqual(block(h[:,:0],xyz[:,:0],active[:,:0]).shape,(2,0,16))
        self.assertTrue(torch.isfinite(block(h[:,:1],xyz[:,:1],active[:,:1])).all())

    def test_decoder_is_exactly_unchanged_with_same_weights(self):
        _,_,_,old=baseline_setup()
        _,_,_,new=setup()
        decoder=lambda k:k.startswith(('learned.dec','learned.heads','learned.context_heads','learned.tier','learned.snr'))
        old_state={k:v for k,v in old.state_dict().items() if decoder(k)}
        new_state={k:v for k,v in new.state_dict().items() if decoder(k)}
        self.assertEqual(old_state.keys(),new_state.keys())
        new.load_state_dict(old_state,strict=False)
        q=(torch.arange(17)%4)[None];z=torch.randn(1,17,2*old.cfg.rates[-1])
        torch.testing.assert_close(old.learned.decode(z,q,10),new.learned.decode(z,q,10),rtol=0,atol=0)

    def test_objective_and_full_size_sh3_forward_backward(self):
        from gaussian_jscc.spatial_response import spatial_response_loss
        _,g,f,new=setup()
        _,_,_,old=baseline_setup()
        old.attr_mean.copy_(new.attr_mean);old.attr_std.copy_(new.attr_std)
        pred=f+torch.randn_like(f)*.01
        torch.testing.assert_close(spatial_response_loss(pred,f,g,new,directions=torch.eye(3))[0],
                                   spatial_response_loss(pred,f,g,old,directions=torch.eye(3))[0],rtol=0,atol=0)
        cfg=CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',
                        encoder_attention='geometric_point')
        m=GaussianCodec(cfg)
        features=torch.randn(2,256,cfg.attr_dim+3)
        q=torch.randint(0,4,(2,256));q[1]=0
        received=m.forward_tier_batches(features,features[...,:3],torch.nn.functional.one_hot(q,4).float(),10,'awgn')[0]
        received.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        self.assertEqual(float(received[1].detach().abs().sum()),0)

    def test_cli_checkpoint_and_initializer_validation(self):
        import json
        import tempfile
        from pathlib import Path
        from gaussian_jscc.cli import main
        from gaussian_jscc.data import write_ply
        from gaussian_jscc.transport import load_checkpoint
        raw,_,_,_=setup()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train-learned','--architecture','learned_split_logcov',
                  '--context-mode','multiscale_self','--encoder-attention','geometric_point',
                  '--encoder-neighbors','4','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--device','cpu','--bootstrap-objective','spatial-response','--bootstrap-steps','2',
                  '--render-steps','0','--joint-steps','0','--hidden','16','--depth','1',
                  '--block-size','8','--decoder-window','4','--blocks-per-batch','2',
                  '--validation-blocks','1','--validation-trials','1','--validate-every','1']
            with patch('sys.argv',argv),patch('gaussian_jscc.rendering.load_cameras',side_effect=AssertionError),patch('gaussian_jscc.plots.safe_plot'):
                main()
            m=load_checkpoint(root/'run/codec.pt','cpu')
            self.assertEqual(m.cfg.encoder_attention,'geometric_point')
            self.assertEqual(m.cfg.encoder_neighbors,4)
            record=json.loads((root/'run/training.json').read_text())
            self.assertEqual(record['objective'],'spatial_logcov_v1')
            bad=argv.copy();bad[bad.index('geometric_point')]='window'
            bad[bad.index(str(root/'run'))]=str(root/'bad');bad+=['--init',str(root/'run/codec.pt')]
            with patch('sys.argv',bad),self.assertRaisesRegex(ValueError,'encoder attention mismatch'):
                main()

    def test_legacy_config_and_invalid_modes(self):
        cfg=CodecConfig()
        self.assertNotIn('encoder_attention',cfg.to_dict())
        self.assertNotIn('encoder_neighbors',cfg.to_dict())
        with self.assertRaisesRegex(ValueError,'requires'):
            CodecConfig(encoder_attention='geometric_point')
        with self.assertRaisesRegex(ValueError,'positive'):
            CodecConfig(encoder_neighbors=0)


if __name__=='__main__':
    unittest.main()
