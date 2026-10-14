import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gaussian_jscc.codec import prefix_mask
from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_codec import ModularGaussianCodec


def small_config():
    return ModularCodecConfig(hidden=16, grid_dim=4, levels=(2,), decoder_window=4,
                              block_size=8, channel_variant='progressive', channel_hidden=32)


class ProgressiveLatentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_stage_outputs_are_additive_and_respect_mixed_tiers(self):
        cfg=small_config()
        self.assertEqual(ModularCodecConfig.from_dict(cfg.to_dict()),cfg)
        model=ModularGaussianCodec(cfg)
        with torch.no_grad():
            for value,stage in zip((1.,2.,3.),model.channel_decoder.stages):
                stage.readout[-1].weight.zero_(); stage.readout[-1].bias.fill_(value)
        q=torch.tensor([[0,1,2,3]])
        received=torch.randn(1,4,48)
        mask=prefix_mask(q.flatten(),cfg.rates).reshape_as(received)
        out=model.channel_decoder(received,mask,q,10.)
        expected=torch.tensor([0.,1.,3.,6.])[None,:,None].expand(1,4,48)
        torch.testing.assert_close(out,expected,rtol=0,atol=0)

    def test_shared_prefix_is_invariant_to_tier_and_unused_suffix(self):
        model=ModularGaussianCodec(small_config())
        signal=torch.randn(2,5,48)
        high=torch.full((2,5),3)
        all_stages=model.channel_decoder.decode_stages(signal,torch.ones_like(signal,dtype=torch.bool),high,10.)
        for tier in (1,2):
            q=torch.full_like(high,tier)
            mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(signal)
            poisoned=signal.masked_fill(~mask,float('nan'))
            out=model.channel_decoder(poisoned,mask,q,10.)
            torch.testing.assert_close(out,all_stages[...,tier-1,:],rtol=0,atol=0)

    def test_higher_stage_learns_from_previous_latent_and_backpropagates(self):
        model=ModularGaussianCodec(small_config())
        y=torch.randn(2,5,48,requires_grad=True); q=torch.full((2,5),3)
        z=model.channel_encoder(y,q>0,10.)
        prediction=model.channel_decoder(z,torch.ones_like(z,dtype=torch.bool),q,10.)
        prediction.square().mean().backward()
        for stage in model.channel_decoder.stages:
            self.assertGreater(float(stage.input.weight.grad.norm()),0)
            self.assertGreater(float(stage.readout[-1].weight.grad.norm()),0)
        self.assertGreater(float(y.grad.norm()),0)
        # Decoder conditioning really uses the recovered feature, not just the suffix.
        stage=model.channel_decoder.stages[1]
        suffix=torch.randn(2,5,16)
        a=stage(suffix,torch.zeros_like(y),10.)
        b=stage(suffix,torch.ones_like(y),10.)
        self.assertGreater(float((a-b).detach().norm()),0)

    def test_encoder_is_same_as_residual_control_and_has_same_power_budget(self):
        torch.manual_seed(91); progressive=ModularGaussianCodec(small_config())
        cfg=small_config(); cfg.channel_variant='residual'
        torch.manual_seed(91); residual=ModularGaussianCodec(cfg)
        for name,value in residual.channel_encoder.state_dict().items():
            torch.testing.assert_close(value,progressive.channel_encoder.state_dict()[name],rtol=0,atol=0)
        y=torch.randn(2,5,48); active=torch.ones(2,5,dtype=torch.bool); active[0,0]=False
        signal=progressive.channel_encoder(y,active,10.)
        self.assertEqual(int(signal[~active].count_nonzero()),0)
        for start,end in ((0,4),(4,12),(12,24)):
            power=signal[...,2*start:2*end].square().sum(-1)/(end-start)
            self.assertLessEqual(float(power.detach().max()),1.00001)

    def test_q1_update_does_not_move_unused_enhancement_parameters(self):
        model=ModularGaussianCodec(small_config())
        optimizer=torch.optim.Adam(model.channel_decoder.parameters(),lr=1e-3)
        signal=torch.randn(2,5,48)
        q=torch.full((2,5),3)
        model.channel_decoder(signal,torch.ones_like(signal,dtype=torch.bool),q,10.).square().mean().backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        saved={n:p.detach().clone() for n,p in model.channel_decoder.stages[1:].named_parameters()}
        q.fill_(1); mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(signal)
        model.channel_decoder(signal,mask,q,10.).square().mean().backward()
        optimizer.step()
        for n,p in model.channel_decoder.stages[1:].named_parameters():
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p,saved[n],rtol=0,atol=0)

    def test_receiver_roundtrip_has_no_sender_access_or_extra_xyz_changes(self):
        from test_gaussian_jscc import fixture
        from gaussian_jscc.transport import save_checkpoint,load_checkpoint,transmit,receive
        from gaussian_jscc.data import prepare,to_features,to_raw
        from gaussian_jscc.position_delivery import delivered_positions
        cfg=small_config(); cfg.sh_degree=1
        model=ModularGaussianCodec(cfg); raw,_=fixture(19,1); q=torch.arange(19)%4
        ordered,geometry,tiers=prepare(raw,16,q); expected=[]
        for start in range(0,19,8):
            f,x=to_features(ordered[start:start+8],geometry,model); tq=tiers[start:start+8]
            pred=model(f,x,tq,10.,'none')
            torch.testing.assert_close(pred[:,:3],delivered_positions(f[:,:3],tq,cfg),rtol=0,atol=0)
            expected.append(to_raw(pred[tq>0],geometry,model))
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); save_checkpoint(root/'codec.pt',model,0)
            stats=transmit(model,raw,q,10.,'none',7,root/'packet')
            receiver=load_checkpoint(root/'codec.pt','cpu')
            receiver.semantic_encoder.forward=lambda *a: (_ for _ in ()).throw(AssertionError('sender access'))
            decoded=receive(receiver,root/'packet')
            torch.testing.assert_close(decoded,torch.cat(expected),rtol=1e-5,atol=1e-6)
            self.assertEqual(stats['payload_complex_symbols'],sum((0,4,12,24)[int(v)] for v in q))

    def test_training_plan_selects_only_progressive_when_requested(self):
        import test_modular_experiment as fixture
        from gaussian_jscc.channel_capacity_controls import parser,build_plan
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); source=fixture.source_fixture(root)
            teacher=root/'teacher.pt'; teacher.write_bytes(b'plan only')
            args=parser().parse_args(['--run',str(source),'--out',str(root/'out'),
                                     '--init-semantic',str(teacher),'--groups','progressive'])
            plan=build_plan(args)
            self.assertEqual(len(plan['jobs']),1)
            command=plan['jobs'][0]['command']
            self.assertEqual(command[command.index('--channel-variant')+1],'progressive')
            self.assertIn('--freeze-semantic',command)
            self.assertEqual(plan['rates'],[0,4,12,24])

    def test_suffix_shuffle_preserves_prefix_padding_and_global_rng(self):
        from gaussian_jscc.latent_prefix_probe import shuffle_suffix
        signal=torch.arange(3*4*48).reshape(3,4,48).float()
        active=torch.tensor([[True,False,True,True],[False,False,True,True],[True,True,True,False]])
        rng=torch.get_rng_state().clone()
        shuffled,changed=shuffle_suffix(signal,active,4,12,torch.Generator().manual_seed(19))
        self.assertEqual(changed,int(active.sum()))
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        torch.testing.assert_close(signal[...,:8],shuffled[...,:8],rtol=0,atol=0)
        torch.testing.assert_close(signal[...,24:],shuffled[...,24:],rtol=0,atol=0)
        torch.testing.assert_close(signal[~active],shuffled[~active],rtol=0,atol=0)
        original=signal[active][:,8:24]; altered=shuffled[active][:,8:24]
        self.assertTrue(((original-altered).abs().sum(-1)>0).all())
        self.assertEqual(sorted(original[:,0].tolist()),sorted(altered[:,0].tolist()))
        single=torch.zeros_like(active); single[0,0]=True
        unchanged,count=shuffle_suffix(signal,single,4,12,torch.Generator().manual_seed(19))
        self.assertEqual(count,0); torch.testing.assert_close(unchanged,signal,rtol=0,atol=0)

    def test_progressive_training_exact_resume_and_frozen_teacher(self):
        import test_modular_experiment as fixture
        import test_render_first as rendering
        from gaussian_jscc.modular_experiment import main
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t, patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=rendering.RenderFirstTests().cameras()):
            root=Path(t); source=fixture.source_fixture(root); helper=fixture.ModularExperimentTests()
            teacher_out=root/'teacher'; teacher_args=helper.args(source,teacher_out)
            teacher_args.channel_bootstrap_steps=teacher_args.channel_render_steps=0
            main(teacher_args)
            teacher=teacher_out/'checkpoints/end_semantic_render/codec.pt'
            reference=load_checkpoint(teacher,'cpu')
            for name in ('full','resumed'):
                opts=helper.args(source,root/name); opts.init_semantic=str(teacher)
                opts.channel_variant='progressive'; opts.channel_hidden=32; opts.freeze_semantic=True
                opts.channel_bootstrap_steps=opts.channel_render_steps=4
                main(opts,max_updates=3 if name=='resumed' else None)
                if name=='resumed':
                    opts.init_semantic=None; opts.resume=str(root/name/'checkpoints/latest/codec.pt')
                    main(opts)
                model=load_checkpoint(root/name/'checkpoints/final/codec.pt','cpu')
                for key,value in reference.state_dict().items():
                    if key.startswith(('semantic_encoder.','semantic_decoder.')) or key in ('attr_mean','attr_std'):
                        torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
            full=load_checkpoint(root/'full/checkpoints/final/codec.pt','cpu')
            resumed=load_checkpoint(root/'resumed/checkpoints/final/codec.pt','cpu')
            for key,value in full.state_dict().items():
                torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
            metadata=json.loads((root/'full/training.json').read_text())
            self.assertEqual(metadata['codec_config']['rates'],[0,4,12,24])
            self.assertEqual(metadata['codec_config']['position_bits'],16)
            from gaussian_jscc.latent_prefix_probe import parser as probe_parser,main as probe_main
            checkpoint=root/'full/checkpoints/final/codec.pt'
            before=checkpoint.read_bytes()
            summary=probe_main(probe_parser().parse_args(['--run',str(root/'full'),
                '--out',str(root/'full/diagnostics/prefix_probe'),'--device','cpu','--trials','1']))
            self.assertEqual(summary['optimizer_updates'],0)
            self.assertEqual(checkpoint.read_bytes(),before)
            self.assertEqual(len(summary['comparison']),18)
            for row in summary['comparison']:
                if row['negative_control']:
                    self.assertEqual(row['source_psnr_drop_db'],0.)
                    self.assertEqual(row['latent_mse_increase'],0.)

    def test_probe_is_rng_neutral_and_keeps_untransmitted_controls_exact(self):
        from gaussian_jscc.latent_prefix_probe import evaluate_probe
        from test_modular_evaluation import scene_fixture
        from test_modular_training import setup
        from test_render_first import synthetic_render
        old,g,f,q=setup()
        cfg=small_config(); cfg.sh_degree=1
        model=ModularGaussianCodec(cfg); model.set_scene_geometry(g)
        model.set_training_snrs([10.])
        scene=scene_fixture(model,g,f)
        rng=torch.get_rng_state().clone(); state={k:v.clone() for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            rows=evaluate_probe(model,scene,out=Path(t),trials=2)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training)
        for key,value in state.items(): torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        for row in rows:
            if row['negative_control']:
                self.assertEqual(row['source_psnr_drop_db'],0.)
                self.assertEqual(row['latent_mse_increase'],0.)
        positive=[r for r in rows if r['layout']=='3' and r['intervention']=='shuffle_all']
        self.assertTrue(any(abs(r['latent_mse_increase'])>1e-10 for r in positive))
        self.assertTrue(all(r['payload_complex_symbols']==16*(0,4,12,24)[int(r['layout'])] for r in rows))
        for row in rows:
            if row['intervention']!='baseline':
                self.assertEqual(row['shuffle_audit']['deranged_rows_per_trial'],[16]*row['trials'])
                expected=0 if row['negative_control'] else 16
                self.assertEqual(row['shuffle_audit']['affected_transmitted_rows_per_trial'],[expected]*row['trials'])

    def test_negative_control_guard_rejects_accidentally_changed_noise(self):
        from gaussian_jscc.latent_prefix_probe import check_negative_control
        baseline={'latent_mse':.1,'source_psnr':20.,'source_ssim':.9,'photo_psnr':18.}
        changed=dict(baseline,latent_mse=.100001)
        with self.assertRaisesRegex(RuntimeError,'negative control'):
            check_negative_control(changed,baseline)
        check_negative_control(dict(baseline),baseline)


if __name__=='__main__': unittest.main()
