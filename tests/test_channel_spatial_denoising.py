"""Receiver context must use delivered geometry and preserve prefix boundaries."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import torch

from gaussian_jscc.codec import prefix_mask
from gaussian_jscc.modular_codec import ModularGaussianCodec
from gaussian_jscc.modular_config import ModularCodecConfig


def config(**kwargs):
    return ModularCodecConfig(hidden=16,grid_dim=4,levels=(2,),block_size=8,decoder_window=4,
        channel_variant='progressive',channel_prefix_mode='full_prefix',channel_hidden=32,
        rates=(0,8,24,48),**kwargs)


class SpatialChannelTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(123)

    def test_optional_config_preserves_legacy_serialization(self):
        self.assertNotIn('channel_denoiser_mode',config().to_dict())
        enabled=config(channel_denoiser_mode='spatial',channel_denoiser_hidden=16,channel_denoiser_neighbors=3)
        self.assertEqual(ModularCodecConfig.from_dict(enabled.to_dict()),enabled)
        for extra in ({'channel_denoiser_mode':'unknown'}, {'channel_denoiser_neighbors':0},
                      {'channel_denoiser_hidden':0}):
            with self.assertRaises(ValueError): config(**extra)
        with self.assertRaises(ValueError): ModularCodecConfig(channel_denoiser_mode='spatial')

    def test_zero_upgrade_matches_then_learns_without_changing_frozen_weights(self):
        from gaussian_jscc.channel_spatial import enable_channel_denoising
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step,finish_update
        from test_modular_training import setup
        _,geometry,f,q=setup()
        model=ModularGaussianCodec(config(sh_degree=1));model.set_scene_geometry(geometry)
        choices=torch.nn.functional.one_hot(q,4).to(f)
        torch.manual_seed(8); before_y=model.forward_channel_batches(f,f[...,:3],choices,10.)[1]
        original={n:v.clone() for n,v in model.state_dict().items()}
        enable_channel_denoising(model,'spatial',hidden=16,neighbors=3)
        model.set_scene_geometry(geometry)
        torch.manual_seed(8); after_y=model.forward_channel_batches(f,f[...,:3],choices,10.)[1]
        torch.testing.assert_close(after_y,before_y,rtol=0,atol=0)
        opt=configure_phase(model,'channel_bootstrap',{},freeze_semantic=True,denoiser_only=True)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        bootstrap_step(model,f,q,geometry,10.,'awgn','channel_bootstrap')
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.channel_decoder.denoisers.parameters()))
        finish_update(model,opt,before)
        for name,value in original.items(): torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)

    def test_world_metric_neighbors_ignore_padding_and_preserve_pointwise_control(self):
        from gaussian_jscc.channel_spatial import ChannelSpatialDenoiser
        from types import SimpleNamespace
        # In normalized XYZ point 2 is nearer; in world metric point 1 is nearer.
        xyz=torch.tensor([[[0.,0.,0.],[.04,0.,0.],[0.,.02,0.],[float('nan'),0.,0.]]])
        active=torch.tensor([[True,True,True,False]])
        geometry=SimpleNamespace(span=torch.tensor([1.,100.,1.]))
        cfg=config(channel_denoiser_mode='spatial',channel_denoiser_hidden=16,channel_denoiser_neighbors=2)
        block=ChannelSpatialDenoiser(cfg,1);block.set_scene_geometry(geometry)
        ids,valid,_=block.neighborhood(xyz,active)
        self.assertEqual(ids[0,0].tolist(),[0,1]);self.assertTrue(valid[0,0].all())
        self.assertFalse(valid[0,3].any())
        cfg.channel_denoiser_mode='pointwise'
        own=ChannelSpatialDenoiser(cfg,1);own.set_scene_geometry(geometry)
        own_ids,own_valid,_=own.neighborhood(xyz,active)
        self.assertTrue((own_ids[own_valid]==torch.arange(4)[None,:,None].expand_as(own_ids)[own_valid]).all())

    def test_spatial_branch_reacts_to_neighbors_but_not_other_blocks_or_q0(self):
        from gaussian_jscc.channel_spatial import ChannelSpatialDenoiser
        from types import SimpleNamespace
        cfg=config(channel_denoiser_mode='spatial',channel_denoiser_hidden=16,channel_denoiser_neighbors=3)
        spatial=ChannelSpatialDenoiser(cfg,1);spatial.set_scene_geometry(SimpleNamespace(span=torch.ones(3)))
        torch.nn.init.normal_(spatial.readout.weight,std=.1)
        own=copy.deepcopy(spatial);own.mode='pointwise'
        y=torch.randn(2,4,48);signal=torch.randn(2,4,96)
        active=torch.tensor([[True,True,True,False],[True,True,True,False]])
        xyz=torch.randn(2,4,3);mask=prefix_mask(active.long().flatten(),cfg.rates).reshape_as(signal)
        altered=signal.clone();altered[0,1,:16]+=2
        for block,must_change in ((spatial,True),(own,False)):
            a=block(y,signal,mask,xyz,active,10.);b=block(y,altered,mask,xyz,active,10.)
            self.assertEqual(bool((a[0,0]!=b[0,0]).any()),must_change)
            torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
            poisoned=signal.clone();poisoned[:,3]=float('nan')
            torch.testing.assert_close(a,block(y,poisoned,mask,xyz,active,10.),rtol=0,atol=0)

    def test_coincident_centers_keep_query_as_a_neighbor(self):
        from gaussian_jscc.channel_spatial import ChannelSpatialDenoiser
        from types import SimpleNamespace
        cfg=config(channel_denoiser_mode='spatial',channel_denoiser_hidden=16,channel_denoiser_neighbors=3)
        block=ChannelSpatialDenoiser(cfg,1);block.set_scene_geometry(SimpleNamespace(span=torch.ones(3)))
        xyz=torch.zeros(1,20,3);active=torch.ones(1,20,dtype=torch.bool)
        ids,valid,relative=block.neighborhood(xyz,active)
        for i in range(20):self.assertIn(i,ids[0,i].tolist())
        self.assertTrue(valid.all());self.assertTrue(torch.isfinite(relative).all())

    def test_mixed_high_tiers_cannot_change_lower_stage_results(self):
        from gaussian_jscc.channel_spatial import enable_channel_denoising
        from types import SimpleNamespace
        model=ModularGaussianCodec(config());enable_channel_denoising(model,'spatial',hidden=16,neighbors=3)
        model.set_scene_geometry(SimpleNamespace(span=torch.ones(3)))
        for block in model.channel_decoder.denoisers: torch.nn.init.normal_(block.readout.weight,std=.1)
        z=torch.randn(1,5,96);xyz=torch.rand(1,5,3);q=torch.tensor([[1,2,3,3,0]])
        mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
        a=model.channel_decoder.decode_stages(z,mask,q,10.,delivered_xyz=xyz)
        b=z.clone();b[...,48:]+=999
        lower=model.channel_decoder.decode_stages(b,mask,q,10.,delivered_xyz=xyz)
        torch.testing.assert_close(a[...,:2,:],lower[...,:2,:],rtol=0,atol=0)
        b=z.clone();b[...,16:]-=999
        lower=model.channel_decoder.decode_stages(b,mask,q,10.,delivered_xyz=xyz)
        torch.testing.assert_close(a[...,0,:],lower[...,0,:],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'XYZ'): model.channel_decoder(z,mask,q,10.)

    def test_denoiser_checkpoint_restores_only_new_optimizer_and_receiver_transport(self):
        from gaussian_jscc.channel_spatial import enable_channel_denoising
        from gaussian_jscc.modular_checkpoint import save_training_checkpoint,restore_training_checkpoint,capture_rng
        from gaussian_jscc.modular_training import configure_phase
        from gaussian_jscc.transport import load_checkpoint,transmit,receive
        from test_gaussian_jscc import fixture
        model=ModularGaussianCodec(config(sh_degree=1));enable_channel_denoising(model,'spatial',hidden=16,neighbors=3)
        for block in model.channel_decoder.denoisers: torch.nn.init.normal_(block.readout.weight,std=.01)
        opt=configure_phase(model,'channel_bootstrap',{},freeze_semantic=True,denoiser_only=True)
        progress=dict(phase='channel_bootstrap',step=0,freeze_semantic=True,channel_denoiser_only=True,module_lrs={},train_snrs=[10.])
        raw,_=fixture(19,1);q=torch.arange(19)%4
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);save_training_checkpoint(root/'codec.pt',model,opt,progress,capture_rng(),{})
            loaded,restored,*_=restore_training_checkpoint(root/'codec.pt','cpu')
            self.assertEqual({n for n,p in model.named_parameters() if p.requires_grad},
                             {n for n,p in loaded.named_parameters() if p.requires_grad})
            self.assertTrue(all('denoisers.' in n for n,p in loaded.named_parameters() if p.requires_grad))
            stats=transmit(model,raw,q,10.,'none',42,root/'packet')
            rx=load_checkpoint(root/'codec.pt','cpu')
            rx.semantic_encoder.forward=lambda *args: (_ for _ in ()).throw(AssertionError('source read'))
            recovered=receive(rx,root/'packet')
            self.assertEqual(len(recovered),int((q>0).sum()))
            self.assertTrue(torch.isfinite(recovered).all())
            self.assertEqual(stats['payload_complex_symbols'],sum((0,8,24,48)[int(x)] for x in q))

    def test_diagnostics_keeps_full_blocks_before_point_sampling(self):
        from gaussian_jscc.channel_spatial import enable_channel_denoising
        from gaussian_jscc.latent_error_diagnostics import diagnose
        from gaussian_jscc.modular_training import latent_reconstruction_loss
        from test_modular_training import setup
        from test_modular_evaluation import scene_fixture
        _,g,f,q=setup();model=ModularGaussianCodec(config(sh_degree=1))
        enable_channel_denoising(model,'spatial',hidden=16,neighbors=3);model.set_scene_geometry(g)
        for b in model.channel_decoder.denoisers:torch.nn.init.normal_(b.readout.weight,std=.1)
        scene=scene_fixture(model,g,f);rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as t:
            # Changing point sample chunk size must not change the neighborhood.
            a=diagnose(model,scene,Path(t)/'a',points=16,trials=2,batch_size=3)
            b=diagnose(model,scene,Path(t)/'b',points=16,trials=2,batch_size=11)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(a['tiers'],b['tiers'])
        choices=torch.nn.functional.one_hot(torch.full_like(q,3),4).to(f)
        with torch.no_grad(): y,yh,tier=model.forward_channel_batches(f,f[...,:3],choices,10.,'none')
        expected=latent_reconstruction_loss(y,yh,tier)[0]
        self.assertAlmostEqual(a['tiers'][2]['summary']['noiseless_mse'],float(expected),places=6)

    def test_spatial_suite_freezes_base_matches_initials_and_resumes_exactly(self):
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
            source_path=root/'prefix/full_prefix10';saved=torch.load(source_path/'checkpoints/final/codec.pt',weights_only=True)
            original=(source_path/'checkpoints/final/codec.pt').read_bytes()
            opts=parser().parse_args(['--run',str(source_path),'--out',str(root/'context'),'--suite','spatial',
                '--device','cpu','--steps','4','--validate-every','4','--save-every','2',
                '--diagnostic-points','8','--diagnostic-trials','2','--diagnostic-batch','4',
                '--denoiser-hidden','16','--denoiser-neighbors','3'])
            plan=build_plan(opts);self.assertEqual(plan['source_step'],8)
            self.assertEqual([j['arm'] for j in plan['jobs']],['pointwise10','spatial10'])
            for arm in ('pointwise10','spatial10'):
                run_arm(opts,arm)
                final=load_checkpoint(root/'context'/arm/'checkpoints/final/codec.pt','cpu')
                for n,v in saved['state_dict'].items():torch.testing.assert_close(v,final.state_dict()[n],rtol=0,atol=0)
            starts=[torch.load(root/'context'/arm/'checkpoints/initial/codec.pt',weights_only=True) for arm in ('pointwise10','spatial10')]
            for n,v in starts[0]['state_dict'].items():torch.testing.assert_close(v,starts[1]['state_dict'][n],rtol=0,atol=0)
            logs=[[json.loads(l) for l in (root/'context'/arm/'metrics/loss.jsonl').read_text().splitlines()] for arm in ('pointwise10','spatial10')]
            self.assertEqual([(r['batch_index'],r['layout'],r['latent_target_power']) for r in logs[0]],
                             [(r['batch_index'],r['layout'],r['latent_target_power']) for r in logs[1]])
            report=summarize(root/'context');self.assertTrue(report['matched_initial_function'])
            opts.out=str(root/'resumed');run_arm(opts,'spatial10',max_updates=3)
            opts.resume_suite=True;run_arm(opts,'spatial10')
            uninterrupted=load_checkpoint(root/'context/spatial10/checkpoints/final/codec.pt','cpu')
            resumed=load_checkpoint(root/'resumed/spatial10/checkpoints/final/codec.pt','cpu')
            for n,v in uninterrupted.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[n],rtol=0,atol=0)
            self.assertEqual((source_path/'checkpoints/final/codec.pt').read_bytes(),original)


if __name__=='__main__': unittest.main()
