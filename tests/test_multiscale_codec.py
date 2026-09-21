"""Real CPU codec/channel/autograd checks, not CUDA render quality evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
import test_learned_joint as contracts
from test_logcov_codec import setup as baseline_setup
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.multiscale_codec import pool_slots
from gaussian_jscc.spatial_response import spatial_response_loss
from gaussian_jscc.training import full_scene_step


def setup(n=17):
    raw,g,f,old = baseline_setup(n)
    cfg = old.cfg.to_dict()
    cfg['context_mode'] = 'multiscale_self'
    m = GaussianCodec(CodecConfig.from_dict(cfg))
    m.attr_mean.copy_(old.attr_mean)
    m.attr_std.copy_(old.attr_std)
    return raw,g,f,m


class MultiScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_transport_power_drop_and_replay_contracts(self):
        for name in ('test_no_hand_geometry_or_split_budget','test_power_budget_empty_windows_and_singletons',
                     'test_new_receiver_has_only_packet_and_weights','test_drop_inputs_do_not_leak_into_other_outputs',
                     'test_replay_equals_checkpoint','test_reject_fake_st_gradients'):
            with self.subTest(name=name),patch('test_learned_joint.setup',setup):
                getattr(contracts.LearnedTests(name),name)()

    def test_pooling_ignores_dropped_and_handles_partial_groups(self):
        x=torch.arange(10.).reshape(1,5,2)
        mask=torch.tensor([[True,False,True,False,True]])
        pooled,active=pool_slots(x,mask,4)
        torch.testing.assert_close(pooled,torch.tensor([[[2.,3.],[8.,9.]]]))
        self.assertTrue(active.all())
        pooled,active=pool_slots(x,torch.zeros_like(mask),16)
        self.assertEqual(float(pooled.abs().sum()),0)
        self.assertFalse(active.any())

    def test_zero_context_gates_leave_pointwise_information_path(self):
        _,_,f,m=setup(32)
        q=torch.ones(1,32,dtype=torch.long)
        with torch.no_grad():
            m.learned.enc_context_gate.zero_()
            m.learned.dec_geometry_gate.zero_()
            m.learned.dec_appearance_gate.zero_()
        z=m.learned.encode(f[None],f[None,:,:3],q,10)
        changed=f.clone();changed[1:]+=1.
        other=m.learned.encode(changed[None],changed[None,:,:3],q,10)
        torch.testing.assert_close(z[:,0],other[:,0])
        decoded=m.learned.decode(z,q,10)
        perturb=z.detach().clone();perturb[:,1:]+=3
        torch.testing.assert_close(decoded[:,0],m.learned.decode(perturb,q,10)[:,0])
        z=z.detach().requires_grad_()
        m.learned.decode(z,q,10)[0,0,:3].sum().backward()
        self.assertGreater(float(z.grad[0,0].norm()),0)
        self.assertEqual(float(z.grad[0,1:].norm()),0)

    def test_coarse_context_connects_distant_slots(self):
        _,_,_,m=setup(64)
        ctx=m.learned.enc_geometry_context
        h=torch.randn(1,64,16,requires_grad=True)
        xyz=torch.rand(1,64,3)
        y=ctx(h,torch.ones(1,64,dtype=torch.bool),xyz)
        y[0,0].square().sum().backward()
        self.assertGreater(float(h.grad[0,-1].norm()),0)

    def test_mixed_tiers_batching_and_all_heads_receive_gradients(self):
        _,g,f,m=setup(17)
        q=torch.arange(17)%4
        torch.manual_seed(42)
        a=m(f,f[:,:3],q,10,'awgn')
        torch.manual_seed(42)
        b=m.forward_tier_batches(f[None],f[None,:,:3],F.one_hot(q,4).float()[None],10,'awgn')[0][0]
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        loss,_=spatial_response_loss(b[q>0],f[q>0],g,m,directions=torch.eye(3))
        loss.backward()
        for heads in (m.learned.heads,m.learned.context_heads):
            for head in heads.values():
                self.assertGreater(float(head.weight.grad.norm()),0)
        for key in ('enc_self_symbols','enc_geometry_context','enc_appearance_context','dec_geometry_context','dec_appearance_context'):
            grads=[p.grad for p in getattr(m.learned,key).parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(v).all() for v in grads))
            self.assertGreater(sum(float(v.norm()) for v in grads),0)
        for key in ('enc_context_gate','dec_geometry_gate','dec_appearance_gate'):
            self.assertGreater(float(getattr(m.learned,key).grad.abs()),0)

    def test_loss_is_unchanged_for_identical_predictions(self):
        _,g,f,m=setup(16)
        cfg=m.cfg.to_dict();cfg['context_mode']='window'
        old=GaussianCodec(CodecConfig.from_dict(cfg))
        old.attr_mean.copy_(m.attr_mean);old.attr_std.copy_(m.attr_std)
        pred=f+.01*torch.randn_like(f)
        a=spatial_response_loss(pred,f,g,m,directions=torch.eye(3))[0]
        b=spatial_response_loss(pred,f,g,old,directions=torch.eye(3))[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_short_fit_and_all_backward_modes(self):
        torch.manual_seed(42)
        _,g,f,m=setup(16)
        q=torch.arange(16)%3+1
        optimizer=torch.optim.Adam(m.parameters(),lr=2e-4)
        values=[]
        for step in range(60):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(142)
            pred=m(f,f[:,:3],q,10,'awgn')
            loss,_=spatial_response_loss(pred,f,g,m,directions=torch.eye(3))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            optimizer.step();values.append(float(loss.detach()))
        self.assertLess(values[-1],values[0])
        gradients=[]
        for mode in ('direct','replay','checkpoint'):
            model=copy.deepcopy(m);model.zero_grad(set_to_none=True);torch.manual_seed(42)
            loss,_=full_scene_step(model,[(f[None],q[None])],g,10,'awgn',lambda scene:scene.square().mean(),
                                   attr_weight=0,mode=mode)
            gradients.append((loss.detach(),[p.grad for p in model.parameters()]))
        for value,grads in gradients[1:]:
            torch.testing.assert_close(value,gradients[0][0])
            for a,b in zip(grads,gradients[0][1]):
                if a is not None:
                    torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4)

    def test_cli_logs_new_structure_and_rejects_old_initializer(self):
        from gaussian_jscc.cli import main
        from gaussian_jscc.data import write_ply
        from gaussian_jscc.transport import load_checkpoint
        raw,_,_,_=setup(17)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train-learned','--architecture','learned_split_logcov',
                  '--context-mode','multiscale_self','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--device','cpu','--bootstrap-objective','spatial-response','--bootstrap-steps','2',
                  '--render-steps','0','--joint-steps','0','--hidden','16','--depth','1',
                  '--block-size','8','--decoder-window','4','--blocks-per-batch','2',
                  '--validation-blocks','1','--validation-trials','1','--validate-every','1']
            with patch('sys.argv',argv),patch('gaussian_jscc.rendering.load_cameras',side_effect=AssertionError),patch('gaussian_jscc.plots.safe_plot'):
                main()
            m=load_checkpoint(root/'run/codec.pt','cpu')
            self.assertEqual(m.cfg.context_mode,'multiscale_self')
            rows=[json.loads(s) for s in (root/'run/loss.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['objective']=='spatial_logcov_v1' and r['phase']=='bootstrap' for r in rows))
            self.assertIn('context_gates',rows[0])
            bad=argv.copy();bad[bad.index('multiscale_self')]='window';bad[bad.index(str(root/'run'))]=str(root/'bad')
            bad+=['--init',str(root/'run/codec.pt')]
            with patch('sys.argv',bad),self.assertRaisesRegex(ValueError,'context mode mismatch'):
                main()

    def test_config_is_explicit_and_old_default_hash_schema_unchanged(self):
        self.assertNotIn('context_mode',CodecConfig().to_dict())
        with self.assertRaisesRegex(ValueError,'requires'):
            CodecConfig(context_mode='multiscale_self')
        with self.assertRaisesRegex(ValueError,'requires'):
            CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',position_delivery='quantized')

    def test_sh3_partial_and_all_dropped_batches(self):
        m=GaussianCodec(CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',
                                    hidden=16,depth=1,decoder_window=4,sh_degree=3))
        f=torch.randn(2,17,m.cfg.attr_dim+3)
        q=torch.stack((torch.arange(17)%4,torch.zeros(17,dtype=torch.long)))
        out=m.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).float(),10,'awgn')[0]
        self.assertEqual(out.shape,f.shape)
        self.assertEqual(float(out[q==0].detach().abs().sum()),0)
        out[q>0].square().mean().backward()
        for heads in (m.learned.heads,m.learned.context_heads):
            self.assertGreater(float(heads['sh'].weight.grad.norm()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


if __name__=='__main__':
    unittest.main()
