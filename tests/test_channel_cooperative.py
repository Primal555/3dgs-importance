"""Available-neighbor refinement must never read unsent receiver slots."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gaussian_jscc.codec import prefix_mask
from gaussian_jscc.modular_codec import ModularGaussianCodec
from gaussian_jscc.modular_config import ModularCodecConfig
from test_channel_spatial_denoising import config


class CooperativeChannelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)

    def model(self, mode='spatial'):
        from gaussian_jscc.channel_spatial import enable_channel_denoising
        model=ModularGaussianCodec(config(sh_degree=1))
        enable_channel_denoising(model,mode,hidden=16,neighbors=3,scope='final')
        model.set_scene_geometry(SimpleNamespace(span=torch.ones(3)))
        return model

    def test_scope_serialization_keeps_existing_identities_and_rejects_invalid_combinations(self):
        self.assertNotIn('channel_denoiser_scope',config().to_dict())
        self.assertNotIn('channel_denoiser_scope',config(channel_denoiser_mode='spatial').to_dict())
        cfg=config(channel_denoiser_mode='spatial',channel_denoiser_scope='final')
        self.assertEqual(ModularCodecConfig.from_dict(cfg.to_dict()),cfg)
        for kwargs in ({'channel_denoiser_scope':'unknown'}, {'channel_denoiser_scope':'final'}):
            with self.assertRaises(ValueError): config(**kwargs)

    def test_final_zero_upgrade_keeps_base_stages_and_learns_only_new_weights(self):
        from gaussian_jscc.channel_spatial import enable_channel_denoising
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step,finish_update
        from test_modular_training import setup
        _,geometry,f,q=setup()
        model=ModularGaussianCodec(config(sh_degree=1));model.set_scene_geometry(geometry)
        original={n:v.clone() for n,v in model.state_dict().items()}
        z=torch.randn(*q.shape,96);mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
        before=model.channel_decoder.decode_stages(z,mask,q,10.)
        enable_channel_denoising(model,'spatial',hidden=16,neighbors=3,scope='final')
        model.set_scene_geometry(geometry)
        side=f[...,:3]
        torch.testing.assert_close(before,model.channel_decoder.decode_stages(z,mask,q,10.,delivered_xyz=side),rtol=0,atol=0)
        torch.testing.assert_close(before[...,-1,:],model.channel_decoder(z,mask,q,10.,delivered_xyz=side),rtol=0,atol=0)
        opt=configure_phase(model,'channel_bootstrap',{},freeze_semantic=True,denoiser_only=True)
        params={n:p.detach().clone() for n,p in model.named_parameters()}
        bootstrap_step(model,f,q,geometry,10.,'awgn','channel_bootstrap')
        finish_update(model,opt,params)
        self.assertTrue(any(not torch.equal(params[n],p) for n,p in model.named_parameters() if p.requires_grad))
        for n,v in original.items(): torch.testing.assert_close(v,model.state_dict()[n],rtol=0,atol=0)
        # Final refinement must not contaminate the independently usable base-stage outputs.
        torch.testing.assert_close(before,model.channel_decoder.decode_stages(z,mask,q,10.,delivered_xyz=side),rtol=0,atol=0)

    def test_q1_can_read_received_q3_suffix_but_not_unsent_slots_q0_or_other_blocks(self):
        model=self.model();decoder=model.channel_decoder
        for block in decoder.denoisers: torch.nn.init.normal_(block.readout.weight,std=.1)
        pointwise=copy.deepcopy(decoder)
        for block in pointwise.denoisers: block.mode='pointwise'
        z=torch.randn(2,4,96);q=torch.tensor([[1,3,1,0],[1,3,1,0]])
        xyz=torch.tensor([[[0.,0.,0.],[.01,0.,0.],[.02,0.,0.],[float('nan'),0.,0.]]]).expand(2,-1,-1).clone()
        mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
        a=decoder(z,mask,q,10.,delivered_xyz=xyz)
        altered=z.clone();altered[0,1,48:]+=2
        b=decoder(altered,mask,q,10.,delivered_xyz=xyz)
        self.assertGreater(float((a[0,0]-b[0,0]).detach().abs().max()),1e-7)
        torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
        own_a=pointwise(z,mask,q,10.,delivered_xyz=xyz)
        own_b=pointwise(altered,mask,q,10.,delivered_xyz=xyz)
        torch.testing.assert_close(own_a[:,0],own_b[:,0],rtol=0,atol=0)
        poisoned=z.masked_fill(~mask.bool(),float('nan'))
        torch.testing.assert_close(a,decoder(poisoned,mask,q,10.,delivered_xyz=xyz),rtol=0,atol=0)
        self.assertTrue((a[q==0]==0).all())
        # Availability affects the cooperative result, never the q1 base readout.
        stages=decoder.decode_stages(z,mask,q,10.,delivered_xyz=xyz)
        altered_stages=decoder.decode_stages(altered,mask,q,10.,delivered_xyz=xyz)
        torch.testing.assert_close(stages[...,0,:],altered_stages[...,0,:],rtol=0,atol=0)
        empty_q=torch.zeros_like(q)
        empty=decoder(torch.full_like(z,float('nan')),torch.zeros_like(mask),empty_q,10.,
                      delivered_xyz=torch.full_like(xyz,float('nan')))
        self.assertTrue(torch.equal(empty,torch.zeros_like(empty)))

    def test_receiver_transport_and_resume_use_final_refinement(self):
        from gaussian_jscc.modular_checkpoint import save_training_checkpoint,restore_training_checkpoint,capture_rng
        from gaussian_jscc.modular_training import configure_phase
        from gaussian_jscc.transport import load_checkpoint,transmit,receive
        from test_gaussian_jscc import fixture
        model=self.model()
        for block in model.channel_decoder.denoisers: torch.nn.init.normal_(block.readout.weight,std=.02)
        opt=configure_phase(model,'channel_bootstrap',{},freeze_semantic=True,denoiser_only=True)
        progress=dict(phase='channel_bootstrap',step=0,freeze_semantic=True,channel_denoiser_only=True,module_lrs={},train_snrs=[10.])
        raw,_=fixture(19,1);q=torch.arange(19)%4
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);save_training_checkpoint(root/'codec.pt',model,opt,progress,capture_rng(),{})
            loaded,_,*_=restore_training_checkpoint(root/'codec.pt','cpu')
            self.assertEqual(loaded.cfg.channel_denoiser_scope,'final')
            stats=transmit(model,raw,q,10.,'none',42,root/'packet')
            rx=load_checkpoint(root/'codec.pt','cpu')
            rx.semantic_encoder.forward=lambda *a: (_ for _ in ()).throw(AssertionError('receiver accessed source'))
            recovered=receive(rx,root/'packet')
            expected=receive(model,root/'packet')
            torch.testing.assert_close(recovered,expected,rtol=5e-5,atol=1e-6)
            self.assertTrue(torch.isfinite(recovered).all())
            self.assertEqual(stats['payload_complex_symbols'],sum((0,8,24,48)[int(x)] for x in q))

    def test_cooperative_diagnostics_report_mixed_target_tiers_without_reblocking(self):
        from gaussian_jscc.latent_error_diagnostics import diagnose
        from test_modular_training import setup
        from test_modular_evaluation import scene_fixture
        _,g,f,_=setup();model=self.model();model.set_scene_geometry(g)
        scene=scene_fixture(model,g,f)
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as t:
            a=diagnose(model,scene,Path(t)/'a',points=16,trials=2,batch_size=3)
            b=diagnose(model,scene,Path(t)/'b',points=16,trials=2,batch_size=11)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(a['mixed'],b['mixed'])
        mixed=a['mixed'];self.assertEqual(sum(v['points'] for v in mixed['by_tier']),a['sample_points'])
        weighted=sum(v['points']*v['summary']['noisy_mse'] for v in mixed['by_tier'])/a['sample_points']
        self.assertAlmostEqual(weighted,mixed['overall']['summary']['noisy_mse'],places=10)

    def test_three_arm_suite_matches_final_capacity_freezes_source_and_resumes_exactly(self):
        import test_modular_experiment as fixture
        import test_render_first as rendering
        from gaussian_jscc.modular_experiment import main as train
        from gaussian_jscc.channel_inverse_controls import parser,build_plan,run_arm,summarize
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=rendering.RenderFirstTests().cameras()):
            root=Path(t);source=fixture.source_fixture(root);helper=fixture.ModularExperimentTests()
            a=helper.args(source,root/'teacher');a.channel_bootstrap_steps=a.channel_render_steps=0;train(a)
            a=helper.args(source,root/'channel');a.init_semantic=str(root/'teacher/checkpoints/end_semantic_render/codec.pt')
            a.freeze_semantic=True;a.train_snrs=[10.];a.rates=[8,24,48];a.channel_variant='progressive';a.channel_hidden=32
            a.channel_bootstrap_steps=4;a.channel_render_steps=0;train(a)
            opts=parser().parse_args(['--run',str(root/'channel'),'--out',str(root/'prefix'),'--suite','prefix',
                '--device','cpu','--steps','4','--validate-every','4','--save-every','2',
                '--diagnostic-points','8','--diagnostic-trials','2','--diagnostic-batch','4'])
            run_arm(opts,'full_prefix10')
            source_path=root/'prefix/full_prefix10';original=(source_path/'checkpoints/final/codec.pt').read_bytes()
            saved=torch.load(source_path/'checkpoints/final/codec.pt',weights_only=True)
            opts=parser().parse_args(['--run',str(root/'prefix'),'--out',str(root/'coop'),'--suite','cooperative',
                '--device','cpu','--steps','4','--validate-every','4','--save-every','2',
                '--diagnostic-points','8','--diagnostic-trials','2','--diagnostic-batch','4',
                '--denoiser-hidden','16','--denoiser-neighbors','3'])
            names=['stage_spatial10','final_pointwise10','final_spatial10']
            plan=build_plan(opts);self.assertEqual([j['arm'] for j in plan['jobs']],names)
            for arm in names:
                run_arm(opts,arm)
                final=load_checkpoint(root/'coop'/arm/'checkpoints/final/codec.pt','cpu')
                for n,v in saved['state_dict'].items():torch.testing.assert_close(v,final.state_dict()[n],rtol=0,atol=0)
            starts=[torch.load(root/'coop'/arm/'checkpoints/initial/codec.pt',weights_only=True) for arm in names[1:]]
            for n,v in starts[0]['state_dict'].items():torch.testing.assert_close(v,starts[1]['state_dict'][n],rtol=0,atol=0)
            rows=[[json.loads(l) for l in (root/'coop'/arm/'metrics/loss.jsonl').read_text().splitlines()] for arm in names]
            for other in rows[1:]:
                self.assertEqual([(r['batch_index'],r['layout'],r['latent_target_power']) for r in rows[0]],
                                 [(r['batch_index'],r['layout'],r['latent_target_power']) for r in other])
            report=summarize(root/'coop');self.assertTrue(report['matched_initial_function'])
            self.assertIn('final_spatial_minus_final_pointwise_psnr_db',report['comparison'][0])
            self.assertIn('mixed_latent_by_tier',report)
            opts.out=str(root/'resumed');run_arm(opts,'final_spatial10',max_updates=3)
            opts.resume_suite=True;run_arm(opts,'final_spatial10')
            uninterrupted=load_checkpoint(root/'coop/final_spatial10/checkpoints/final/codec.pt','cpu')
            resumed=load_checkpoint(root/'resumed/final_spatial10/checkpoints/final/codec.pt','cpu')
            for n,v in uninterrupted.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[n],rtol=0,atol=0)
            self.assertEqual((source_path/'checkpoints/final/codec.pt').read_bytes(),original)


if __name__=='__main__': unittest.main()
