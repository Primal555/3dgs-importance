"""Full received-prefix reread must preserve causal tiers and warm-start Adam."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_codec import ModularGaussianCodec
from gaussian_jscc.codec import prefix_mask


def config(**extra):
    return ModularCodecConfig(hidden=16, grid_dim=4, levels=(2,), decoder_window=4,
        block_size=8, channel_variant='progressive', channel_hidden=32, rates=(0,8,24,48), **extra)


class FullPrefixTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_optional_config_roundtrips_and_rejects_nonprogressive_use(self):
        old=config()
        self.assertNotIn('channel_prefix_mode', old.to_dict())
        full=config(channel_prefix_mode='full_prefix')
        self.assertEqual(ModularCodecConfig.from_dict(full.to_dict()), full)
        with self.assertRaises(ValueError): config(channel_prefix_mode='unknown')
        with self.assertRaises(ValueError): ModularCodecConfig(channel_prefix_mode='full_prefix')

    def test_zero_upgrade_preserves_outputs_weights_and_existing_adam(self):
        from gaussian_jscc.progressive_channel import enable_full_prefix
        from gaussian_jscc.modular_training import configure_phase
        from gaussian_jscc.modular_checkpoint import save_training_checkpoint, restore_training_checkpoint, capture_rng
        model=ModularGaussianCodec(config()); model.set_training_snrs([10.])
        opt=configure_phase(model,'channel_bootstrap',{},freeze_semantic=True)
        signal=torch.randn(2,5,96); high=torch.full((2,5),3); mask=torch.ones_like(signal,dtype=torch.bool)
        model.channel_decoder(signal,mask,high,10.).square().mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
        old_weights={n:p.detach().clone() for n,p in model.named_parameters()}
        old_state={n:{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in opt.state[p].items()}
                   for n,p in model.named_parameters() if p in opt.state}
        before=model.channel_decoder.decode_stages(signal,mask,high,10.).detach()
        new_names=enable_full_prefix(model,opt)
        self.assertEqual(len(new_names),2)
        for n,p in model.named_parameters():
            if n in old_weights: torch.testing.assert_close(p,old_weights[n],rtol=0,atol=0)
            else:
                self.assertEqual(p.count_nonzero(),0); self.assertNotIn(p,opt.state)
        after=model.channel_decoder.decode_stages(signal,mask,high,10.)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        for n,p in model.named_parameters():
            if n in old_state:
                for k,v in old_state[n].items(): torch.testing.assert_close(opt.state[p][k],v,rtol=0,atol=0)
        model.channel_decoder(signal,mask,high,10.).square().mean().backward(); opt.step()
        self.assertTrue(all(float(dict(model.named_parameters())[n].grad.norm())>0 for n in new_names))
        for n in new_names: self.assertEqual(float(opt.state[dict(model.named_parameters())[n]]['step']),1.)
        progress={'phase':'channel_bootstrap','step':2,'completed':{'channel_bootstrap':2},
                  'freeze_semantic':True,'module_lrs':{},'train_snrs':[10.]}
        with tempfile.TemporaryDirectory() as t:
            checkpoint=Path(t)/'codec.pt'
            save_training_checkpoint(checkpoint,model,opt,progress,capture_rng(),{})
            resumed,restored,*_=restore_training_checkpoint(checkpoint,'cpu')
            names=dict(resumed.named_parameters())
            for n,p in model.named_parameters():
                torch.testing.assert_close(names[n],p,rtol=0,atol=0)
                if p in opt.state:
                    for k,v in opt.state[p].items(): torch.testing.assert_close(restored.state[names[n]][k],v,rtol=0,atol=0)

    def test_enhancements_can_reinterpret_earlier_symbols_without_previous_y_change(self):
        from gaussian_jscc.progressive_channel import enable_full_prefix
        base=ModularGaussianCodec(config()); base.channel_decoder.stages[0].readout[-1].weight.data.zero_()
        base.channel_decoder.stages[0].readout[-1].bias.data.zero_()
        old=ModularGaussianCodec(config()); old.load_state_dict(base.state_dict())
        enable_full_prefix(base)
        with torch.no_grad(): base.channel_decoder.stages[1].prefix_input.weight.fill_(.1)
        signal=torch.randn(1,4,96); changed=signal.clone(); changed[...,:16]+=1
        q=torch.full((1,4),2); mask=prefix_mask(q.flatten(),base.cfg.rates).reshape_as(signal)
        a=old.channel_decoder(signal,mask,q,10.); b=old.channel_decoder(changed,mask,q,10.)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a=base.channel_decoder.decode_stages(signal,mask,q,10.)
        b=base.channel_decoder.decode_stages(changed,mask,q,10.)
        torch.testing.assert_close(a[...,0,:],b[...,0,:],rtol=0,atol=0)
        self.assertGreater(float((a[...,1,:]-b[...,1,:]).detach().abs().max()),0)

    def test_full_prefix_keeps_untransmitted_slots_invisible_and_q1_updates_isolated(self):
        model=ModularGaussianCodec(config(channel_prefix_mode='full_prefix'))
        for stage in model.channel_decoder.stages[1:]: stage.prefix_input.weight.data.normal_(0,.1)
        signal=torch.randn(2,5,96); q3=torch.full((2,5),3)
        all_stages=model.channel_decoder.decode_stages(signal,torch.ones_like(signal,dtype=torch.bool),q3,10.)
        for tier in (1,2):
            q=torch.full_like(q3,tier); mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(signal)
            pred=model.channel_decoder(signal.masked_fill(~mask,float('nan')),mask,q,10.)
            torch.testing.assert_close(pred,all_stages[...,tier-1,:],rtol=0,atol=0)
        opt=torch.optim.Adam(model.channel_decoder.parameters(),lr=1e-3)
        all_stages[...,-1,:].square().mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
        saved={n:p.detach().clone() for n,p in model.channel_decoder.stages[1:].named_parameters()}
        q=torch.ones_like(q3); mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(signal)
        model.channel_decoder(signal,mask,q,10.).square().mean().backward(); opt.step()
        for n,p in model.channel_decoder.stages[1:].named_parameters():
            self.assertIsNone(p.grad); torch.testing.assert_close(p,saved[n],rtol=0,atol=0)

    def test_receiver_only_roundtrip_has_identical_payload_budget_and_xyz(self):
        from test_gaussian_jscc import fixture
        from gaussian_jscc.transport import save_checkpoint,load_checkpoint,transmit,receive
        from gaussian_jscc.data import prepare,to_features,to_raw
        from gaussian_jscc.position_delivery import delivered_positions
        model=ModularGaussianCodec(config(channel_prefix_mode='full_prefix',sh_degree=1))
        for stage in model.channel_decoder.stages[1:]: stage.prefix_input.weight.data.normal_(0,.1)
        raw,_=fixture(19,1); q=torch.arange(19)%4
        ordered,g,tiers=prepare(raw,16,q); expected=[]
        for start in range(0,19,8):
            f,x=to_features(ordered[start:start+8],g,model); tq=tiers[start:start+8]
            pred=model(f,x,tq,10.,'none')
            torch.testing.assert_close(pred[:,:3],delivered_positions(f[:,:3],tq,model.cfg),rtol=0,atol=0)
            expected.append(to_raw(pred[tq>0],g,model))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); save_checkpoint(p/'codec.pt',model,0)
            stats=transmit(model,raw,q,10.,'none',7,p/'packet'); receiver=load_checkpoint(p/'codec.pt','cpu')
            receiver.semantic_encoder.forward=lambda *a: (_ for _ in ()).throw(AssertionError('sender access'))
            torch.testing.assert_close(receive(receiver,p/'packet'),torch.cat(expected),rtol=1e-5,atol=1e-6)
            self.assertEqual(stats['payload_complex_symbols'],sum((0,8,24,48)[int(v)] for v in q))

    def test_prefix_suite_retains_source_initial_function_and_resumes_new_parameters(self):
        import test_modular_experiment as fixture
        import test_render_first as rendering
        from gaussian_jscc.modular_experiment import main as train
        from gaussian_jscc.channel_inverse_controls import parser,build_plan,run_arm,summarize
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=rendering.RenderFirstTests().cameras()):
            root=Path(t); source=fixture.source_fixture(root); helper=fixture.ModularExperimentTests()
            args=helper.args(source,root/'teacher'); args.channel_bootstrap_steps=args.channel_render_steps=0; train(args)
            args=helper.args(source,root/'channel'); args.init_semantic=str(root/'teacher/checkpoints/end_semantic_render/codec.pt')
            args.freeze_semantic=True; args.train_snrs=[10.]; args.rates=[8,24,48]
            args.channel_variant='progressive'; args.channel_hidden=32
            args.channel_bootstrap_steps=4; args.channel_render_steps=0; train(args)
            options=parser().parse_args(['--run',str(root/'channel'),'--out',str(root/'controls'),'--suite','prefix',
                '--device','cpu','--steps','4','--validate-every','2','--save-every','2',
                '--diagnostic-points','8','--diagnostic-trials','2','--diagnostic-batch','4'])
            plan=build_plan(options)
            self.assertEqual([(j['arm'],j['kind']) for j in plan['jobs']],[('suffix10','awgn'),('full_prefix10','awgn')])
            run_arm(options,'suffix10'); run_arm(options,'full_prefix10')
            records=[json.loads((root/'controls'/arm/'training.json').read_text()) for arm in ('suffix10','full_prefix10')]
            self.assertNotIn('channel_prefix_mode',records[0]['codec_config'])
            self.assertEqual(records[1]['codec_config']['channel_prefix_mode'],'full_prefix')
            rows=[]
            for arm in ('suffix10','full_prefix10'):
                rows.append([json.loads(l) for l in (root/'controls'/arm/'metrics/validation.jsonl').read_text().splitlines() if json.loads(l)['phase']=='initial'])
            for a,b in zip(*rows):
                self.assertEqual(a['latent_mse'],b['latent_mse']); self.assertEqual(a['source_psnr'],b['source_psnr'])
            options.out=str(root/'resumed'); run_arm(options,'full_prefix10',max_updates=3)
            options.resume_suite=True; run_arm(options,'full_prefix10')
            full=load_checkpoint(root/'controls/full_prefix10/checkpoints/final/codec.pt','cpu')
            resumed=load_checkpoint(root/'resumed/full_prefix10/checkpoints/final/codec.pt','cpu')
            for n,v in full.state_dict().items(): torch.testing.assert_close(v,resumed.state_dict()[n],rtol=0,atol=0)
            summary=summarize(root/'controls')
            self.assertEqual(summary['groups'],['suffix10','full_prefix10'])
            self.assertTrue(summary['matched_initial_function'])
            self.assertEqual(len(summary['comparison']),8)


if __name__=='__main__': unittest.main()
