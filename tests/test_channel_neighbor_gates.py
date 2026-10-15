"""An independent gated neighbor correction preserves a trained self base."""
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


class NeighborGateTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(18)

    def base(self):
        from gaussian_jscc.channel_spatial import enable_channel_denoising
        model=ModularGaussianCodec(config(sh_degree=1))
        enable_channel_denoising(model,'pointwise',hidden=16,neighbors=3)
        model.set_scene_geometry(SimpleNamespace(span=torch.ones(3)))
        return model

    def test_legacy_config_and_invalid_neighbor_settings(self):
        self.assertNotIn('channel_neighbor_mode',config().to_dict())
        for extra in ({'channel_neighbor_mode':'unknown'}, {'channel_neighbor_hidden':0}, {'channel_neighbor_neighbors':0}):
            with self.assertRaises(ValueError):config(**extra)
        enabled=config(channel_neighbor_mode='feature')
        self.assertEqual(ModularCodecConfig.from_dict(enabled.to_dict()),enabled)
        with self.assertRaises(ValueError):ModularCodecConfig(channel_neighbor_mode='feature')

    def test_zero_upgrade_then_only_neighbor_trains_with_featurewise_gates(self):
        from gaussian_jscc.channel_spatial import enable_neighbor_refinement
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step,finish_update
        from test_modular_training import setup
        _,g,f,q=setup();model=self.base();model.set_scene_geometry(g)
        choices=torch.nn.functional.one_hot(q,4).to(f)
        torch.manual_seed(2);before=model.forward_channel_batches(f,f[...,:3],choices,10.)[1]
        original={n:v.clone() for n,v in model.state_dict().items()}
        enable_neighbor_refinement(model,'feature',hidden=16,neighbors=3);model.set_scene_geometry(g)
        torch.manual_seed(2);after=model.forward_channel_batches(f,f[...,:3],choices,10.)[1]
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        opt=configure_phase(model,'channel_bootstrap',{},freeze_semantic=True,neighbor_only=True)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            snapshot={n:p.detach().clone() for n,p in model.named_parameters()}
            bootstrap_step(model,f,q,g,10.,'awgn','channel_bootstrap');finish_update(model,opt,snapshot)
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable and all(n.startswith('channel_decoder.neighbor_refiner.') for n in trainable))
        self.assertTrue(model.channel_decoder.neighbor_refiner.feature_gate.weight.grad.abs().sum()>0)
        for n,v in original.items():torch.testing.assert_close(v,model.state_dict()[n],rtol=0,atol=0)
        gates=model.channel_decoder.neighbor_refiner.last_gates
        self.assertEqual(gates.shape,(*q.shape,48));self.assertTrue(((gates>=0)&(gates<=1)).all())
        self.assertTrue((gates[q==0]==0).all())

    def test_neighbor_full_prefix_is_available_but_unsent_slots_and_q0_are_not(self):
        from gaussian_jscc.channel_spatial import enable_neighbor_refinement
        model=self.base();enable_neighbor_refinement(model,'feature',hidden=16,neighbors=3)
        model.set_scene_geometry(SimpleNamespace(span=torch.ones(3)))
        block=model.channel_decoder.neighbor_refiner
        torch.nn.init.normal_(block.readout.weight,std=.1)
        q=torch.tensor([[1,3,1,0],[1,3,1,0]])
        z=torch.randn(2,4,96);xyz=torch.randn(2,4,3)
        mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
        a=model.channel_decoder(z,mask,q,10.,delivered_xyz=xyz)
        changed=z.clone();changed[0,1,48:]+=3
        b=model.channel_decoder(changed,mask,q,10.,delivered_xyz=xyz)
        self.assertTrue((a[0,0]!=b[0,0]).any());torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
        poisoned=z.masked_fill(~mask,float('nan'))
        torch.testing.assert_close(a,model.channel_decoder(poisoned,mask,q,10.,delivered_xyz=xyz),rtol=0,atol=0)

    def test_gated_suite_freezes_self_base_pairs_data_and_resumes_receiver_checkpoint(self):
        import test_modular_experiment as fixture
        import test_render_first as rendering
        from gaussian_jscc.modular_experiment import main as train
        from gaussian_jscc.channel_inverse_controls import parser,build_plan,run_arm,summarize,run_source_geometry
        from gaussian_jscc.transport import load_checkpoint,transmit,receive
        from test_gaussian_jscc import fixture as ply_fixture
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=rendering.RenderFirstTests().cameras()):
            root=Path(t);source=fixture.source_fixture(root);helper=fixture.ModularExperimentTests()
            a=helper.args(source,root/'teacher');a.channel_bootstrap_steps=a.channel_render_steps=0;train(a)
            a=helper.args(source,root/'channel');a.init_semantic=str(root/'teacher/checkpoints/end_semantic_render/codec.pt')
            a.freeze_semantic=True;a.train_snrs=[10.];a.rates=[8,24,48];a.channel_variant='progressive';a.channel_hidden=32
            a.channel_bootstrap_steps=4;a.channel_render_steps=0;train(a)
            common=['--device','cpu','--steps','4','--validate-every','4','--save-every','2',
                '--diagnostic-points','8','--diagnostic-trials','2','--diagnostic-batch','4']
            opts=parser().parse_args(['--run',str(root/'channel'),'--out',str(root/'prefix'),'--suite','prefix',*common])
            run_arm(opts,'full_prefix10')
            opts=parser().parse_args(['--run',str(root/'prefix'),'--out',str(root/'self'),'--suite','spatial',*common,
                '--denoiser-hidden','16','--denoiser-neighbors','3'])
            run_arm(opts,'pointwise10')
            path=root/'self/pointwise10/checkpoints/final/codec.pt';original=path.read_bytes()
            saved=torch.load(path,weights_only=True)
            opts=parser().parse_args(['--run',str(root/'self'),'--out',str(root/'gates'),'--suite','gated',*common,
                '--denoiser-hidden','16','--denoiser-neighbors','3','--sensitivity-blocks','1','--sensitivity-views','2'])
            plan=build_plan(opts);self.assertEqual(plan['source_step'],12)
            self.assertEqual([j['arm'] for j in plan['jobs']],['neighbor_plain10','neighbor_gated10'])
            self.assertIn('--probe-source',plan['geometry_job']['command'])
            run_source_geometry(opts,plan)
            self.assertTrue((root/'gates/source_geometry/error_geometry.json').is_file())
            marker=(root/'gates/source_geometry/source_complete.json').read_bytes()
            run_source_geometry(opts,plan)
            self.assertEqual(marker,(root/'gates/source_geometry/source_complete.json').read_bytes())
            for arm in ('neighbor_plain10','neighbor_gated10'):
                run_arm(opts,arm)
                final=load_checkpoint(root/'gates'/arm/'checkpoints/final/codec.pt','cpu')
                for n,v in saved['state_dict'].items():torch.testing.assert_close(v,final.state_dict()[n],rtol=0,atol=0)
                self.assertTrue((root/'gates'/arm/'diagnostics/final/geometry/error_geometry.json').is_file())
            report=summarize(root/'gates');self.assertTrue(report['matched_initial_function'])
            self.assertIn('gated_minus_plain_psnr_db',report['comparison'][0])
            log=[json.loads(l) for l in (root/'gates/neighbor_gated10/metrics/loss.jsonl').read_text().splitlines()]
            self.assertIn('neighbor_gates_by_tier',log[-1])
            starts=[torch.load(root/'gates'/arm/'checkpoints/initial/codec.pt',weights_only=True) for arm in ('neighbor_plain10','neighbor_gated10')]
            for n,v in starts[0]['state_dict'].items():torch.testing.assert_close(v,starts[1]['state_dict'][n],rtol=0,atol=0)
            opts.out=str(root/'resumed');run_arm(opts,'neighbor_gated10',max_updates=3)
            opts.resume_suite=True;run_arm(opts,'neighbor_gated10')
            uninterrupted=load_checkpoint(root/'gates/neighbor_gated10/checkpoints/final/codec.pt','cpu')
            resumed=load_checkpoint(root/'resumed/neighbor_gated10/checkpoints/final/codec.pt','cpu')
            for n,v in uninterrupted.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[n],rtol=0,atol=0)
            raw,_=ply_fixture(19,1);q=torch.arange(19)%4
            stats=transmit(resumed,raw,q,10.,'none',42,root/'packet')
            resumed.semantic_encoder.forward=lambda *a: (_ for _ in ()).throw(AssertionError('receiver accessed source'))
            result=receive(resumed,root/'packet')
            self.assertTrue(torch.isfinite(result).all());self.assertEqual(len(result),int((q>0).sum()))
            self.assertEqual(stats['payload_complex_symbols'],sum((0,8,24,48)[int(v)] for v in q))
            self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':unittest.main()
