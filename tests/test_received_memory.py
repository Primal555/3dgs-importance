"""Received-only rereading: equivalence, masks, gradients and real training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import test_transformer_trunk as trunk
import test_multiscale_codec as contracts
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.transformer_decoder import ReceivedMemoryRead
from gaussian_jscc.optimization import parameter_group


def setup(n=17):
    raw,g,f,old=trunk.setup(n)
    cfg=old.cfg.to_dict();cfg['decoder_memory']='received'
    model=GaussianCodec(CodecConfig.from_dict(cfg))
    model.attr_mean.copy_(old.attr_mean);model.attr_std.copy_(old.attr_std)
    return raw,g,f,model


class MemoryCLI(trunk.TransformerTrunkCLI):
    decoder_memory='received'


class MemoryDiagnosis(trunk.TransformerTrunkDiagnosis):
    decoder_memory='received'


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_initial_outputs_weights_and_gradients_exact(self):
        _,_,f,base=trunk.setup()
        cfg=base.cfg.to_dict()
        torch.manual_seed(42);a=GaussianCodec(CodecConfig.from_dict(cfg))
        torch.manual_seed(42);b=GaussianCodec(CodecConfig.from_dict(dict(cfg,decoder_memory='received')))
        for name,p in a.state_dict().items():
            torch.testing.assert_close(p,b.state_dict()[name],atol=0,rtol=0)
        q=torch.arange(len(f))%4
        ya,yb=a(f,f[:,:3],q,10,'none'),b(f,f[:,:3],q,10,'none')
        torch.testing.assert_close(ya,yb,atol=0,rtol=0)
        ya.square().sum().backward();yb.square().sum().backward()
        bp=dict(b.named_parameters())
        for name,p in a.named_parameters():
            torch.testing.assert_close(p.grad,bp[name].grad,atol=0,rtol=0)
        for read in b.learned.dec_trunk.memory_reads:
            self.assertGreater(float(read.attention.out_proj.weight.grad.norm()),0)
            self.assertEqual(float(read.attention.in_proj_weight.grad.norm()),0)
        self.assertNotIn('decoder_memory',a.cfg.to_dict())
        self.assertEqual(CodecConfig.from_dict(b.cfg.to_dict()).decoder_memory,'received')
        with self.assertRaises(ValueError):
            CodecConfig(decoder_memory='received')

    def test_nonzero_reads_masks_permutation_and_empty(self):
        _,_,_,m=setup()
        for read in m.learned.dec_trunk.memory_reads:
            torch.nn.init.normal_(read.attention.out_proj.weight,std=.02)
        q=torch.full((2,65),3,dtype=torch.long);q[0,1]=0;q[1]=0
        z=torch.randn(2,65,2*m.cfg.rates[-1],requires_grad=True)
        y=m.learned.decode(z,q,10)
        perm=torch.randperm(65)
        torch.testing.assert_close(m.learned.decode(z[:,perm],q[:,perm],10),y[:,perm],atol=2e-6,rtol=2e-5)
        altered=z.detach().clone();altered[q==0]=float('nan')
        torch.testing.assert_close(m.learned.decode(altered,q,10),y)
        self.assertEqual(float(y[q==0].detach().abs().sum()),0)
        y[0,0].square().sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(float(z.grad[q==0].abs().sum()),0)
        self.assertGreater(float(z.grad[0,-1].norm()),0)
        for read in m.learned.dec_trunk.memory_reads:
            self.assertGreater(float(read.attention.in_proj_weight.grad.norm()),0)
        self.assertEqual(m.learned.decode(z[:,:0],q[:,:0],10).shape,(2,0,m.cfg.attr_dim+3))
        self.assertTrue(torch.isfinite(m.learned.decode(z[:,:1],q[:,:1],10)).all())

    def test_immutable_memory_and_memory_gradients(self):
        _,_,_,m=setup()
        seen=[]
        def before(module,args):
            seen.append((args[1],args[1].detach().clone()))
        hooks=[r.register_forward_pre_hook(before) for r in m.learned.dec_trunk.memory_reads]
        q=torch.full((1,7),3,dtype=torch.long)
        m.learned.decode(torch.randn(1,7,2*m.cfg.rates[-1]),q,10)
        for h in hooks:h.remove()
        self.assertEqual(len(seen),4)
        for memory,snapshot in seen:
            self.assertIs(memory,seen[0][0])
            self.assertTrue(memory.requires_grad)
            torch.testing.assert_close(memory,snapshot,atol=0,rtol=0)
        read=ReceivedMemoryRead(16,4)
        torch.nn.init.normal_(read.attention.out_proj.weight,std=.02)
        memory=torch.randn(1,7,16,requires_grad=True)
        read(torch.randn(1,7,16),memory,q>0).square().sum().backward()
        self.assertGreater(float(memory.grad.norm()),0)
        self.assertIn('decoder_memory_layer_3',{parameter_group(n) for n,_ in m.named_parameters()})

    def test_contracts_and_fit(self):
        for name in ('test_transport_power_drop_and_replay_contracts','test_short_fit_and_all_backward_modes'):
            with self.subTest(name=name),patch('test_multiscale_codec.setup',setup):
                getattr(contracts.MultiScaleTests(name),name)()

    def test_full_block_sh3(self):
        cfg=CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',
                        encoder_attention='geometric_point',decoder_attention='transformer_trunk',decoder_memory='received')
        m=GaussianCodec(cfg)
        f=torch.randn(2,256,cfg.attr_dim+3);q=torch.randint(0,4,(2,256));q[1]=0
        out=m.forward_tier_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4).float(),10,'awgn')[0]
        self.assertEqual(out.shape,f.shape)
        self.assertEqual(float(out[1].detach().abs().sum()),0)
        out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_paired_runner(self):
        from scripts.compare_received_memory import build_parser,run
        from gaussian_jscc.data import write_ply
        from gaussian_jscc.transport import load_checkpoint
        raw,_,_,_=trunk.old_setup(2048)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_ply(root/'input.ply',raw,0)
            args=build_parser().parse_args(['--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--blocks','8','--steps','2','--every','1','--save-every','2','--seeds','42',
                  '--threads','1','--blocks-per-batch','2'])
            report=run(args)
            self.assertEqual(len(report['rows']),2)
            starts=[json.loads((root/f'run/{mode}_seed42/validation.jsonl').read_text().splitlines()[0])
                    for mode in ('none','received')]
            self.assertEqual(starts[0],starts[1])
            self.assertEqual(load_checkpoint(root/'run/received_seed42/codec_2.pt','cpu').cfg.decoder_memory,'received')
            self.assertTrue((root/'run/mse.png').is_file())
            args.out=str(root/'reuse');args.reference_root=str(root/'run');args.modes=['none']
            reused=run(args)
            self.assertEqual(reused['rows'][0],report['rows'][0])
            self.assertEqual((root/'reuse/none_seed42/validation.jsonl').read_bytes(),
                             (root/'run/none_seed42/validation.jsonl').read_bytes())
            self.assertTrue((root/'reuse/none_seed42/reference.json').is_file())
            args.out=str(root/'mismatch');args.steps=3
            with self.assertRaisesRegex(ValueError,'reference protocol mismatch: steps'):
                run(args)


if __name__=='__main__':
    unittest.main()
