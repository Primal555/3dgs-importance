"""Variable-budget channels over one frozen semantic teacher, no real rasterizer."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_codec import ModularGaussianCodec
from test_modular_training import setup

RATES=(0,4,8,12,16,24,32,48,64)


class VariableRateChannelTests(unittest.TestCase):
    def config(self,**kw):
        return ModularCodecConfig(sh_degree=1,hidden=16,grid_dim=4,levels=(2,),block_size=8,
            decoder_window=4,channel_variant='progressive',channel_hidden=16,channel_prefix_mode='full_prefix',
            experimental_multirate=True,rates=RATES,**kw)

    def test_opt_in_roundtrip_and_all_eight_latent_losses(self):
        cfg=self.config(); self.assertEqual(ModularCodecConfig.from_dict(cfg.to_dict()),cfg)
        self.assertNotIn('experimental_multirate',ModularCodecConfig().to_dict())
        with self.assertRaises(ValueError): ModularCodecConfig(rates=RATES)
        _,g,f,_=setup(); model=ModularGaussianCodec(cfg)
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step
        configure_phase(model,'channel_bootstrap',{},freeze_semantic=True)
        q=torch.tensor([[1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,0]])
        loss,metrics=bootstrap_step(model,f,q,g,10,'awgn','channel_bootstrap')
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(set(metrics['latent_mse_by_tier']),set(map(str,range(1,9))))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.channel_encoder.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in model.semantic_encoder.parameters()))

    def test_rate_validation_schedule_and_receiver_only_packet(self):
        from gaussian_jscc.channel_rate_sweep import schedule
        for rates in ((0,), (0,4,4), (1,4), (0,True), (0,4,3)):
            with self.subTest(rates=rates),self.assertRaises(ValueError):
                ModularCodecConfig(experimental_multirate=True,channel_variant='progressive',rates=rates)
        updates=list(schedule(RATES[1:],2,3))
        self.assertEqual(len(updates),19)
        self.assertEqual(sum(layout=='mixed' for layout,_,_ in updates),3)
        for budget in RATES[1:]: self.assertEqual(sum(b==budget for _,b,_ in updates),2)
        from test_gaussian_jscc import fixture
        from gaussian_jscc.data import prepare,to_features,to_raw
        from gaussian_jscc.transport import transmit,receive,save_checkpoint,load_checkpoint,decode_metadata
        raw,_=fixture(19,1); model=ModularGaussianCodec(self.config())
        q=torch.arange(19)%9
        ordered,geometry,oq=prepare(raw,model.cfg.morton_bits,q)
        expected=[]
        for start in range(0,len(raw),model.cfg.block_size):
            f,x=to_features(ordered[start:start+model.cfg.block_size],geometry,model)
            tier=oq[start:start+model.cfg.block_size]
            expected.append(to_raw(model(f,x,tier,10,'none')[tier>0],geometry,model))
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t); save_checkpoint(folder/'codec.pt',model,0)
            stats=transmit(model,raw,q,10,'none',7,folder/'packet')
            receiver=load_checkpoint(folder/'codec.pt','cpu')
            receiver.semantic_encoder.forward=lambda *a: (_ for _ in ()).throw(AssertionError('receiver read source encoder'))
            actual=receive(receiver,folder/'packet')
            header,received_tiers=decode_metadata((folder/'packet/metadata.bin').read_bytes())
            self.assertEqual(header['tier_id_bits'],4)
            self.assertEqual(received_tiers.tolist(),oq.tolist())
            self.assertEqual(stats['payload_complex_symbols'],sum(RATES[int(x)] for x in oq))
            torch.testing.assert_close(actual,torch.cat(expected),atol=1e-6,rtol=1e-5)

    def test_highest_refinement_reads_all_received_symbols_and_shared_noise_slots(self):
        from gaussian_jscc.codec import prefix_mask
        cfg=self.config(channel_denoiser_mode='pointwise',channel_denoiser_scope='final',channel_neighbor_mode='plain')
        model=ModularGaussianCodec(cfg)
        self.assertEqual(model.channel_decoder.denoisers[0].symbols,64)
        self.assertEqual(model.channel_decoder.neighbor_refiner.symbols,64)
        _,g,f,_=setup(); model.set_scene_geometry(g)
        q=(torch.arange(f.shape[0]*f.shape[1]).reshape(f.shape[:2])%8)+1
        mask=prefix_mask(q.flatten(),cfg.rates).reshape(*q.shape,128)
        z=torch.randn(*q.shape,128)
        torch.testing.assert_close(model.channel_decoder(z,mask,q,10,delivered_xyz=f[...,:3]),
            model.channel_decoder(z.masked_fill(~mask,1e6),mask,q,10,delivered_xyz=f[...,:3]))
        noises=[]
        for budgets in ((0,8,24,48),RATES):
            config=copy.deepcopy(cfg); config.rates=budgets
            net=ModularGaussianCodec(config); net.set_scene_geometry(g)
            q=torch.ones(f.shape[:2],dtype=torch.long)
            captured={}
            a=net.channel_encoder.register_forward_hook(lambda m,i,o:captured.update(z=o.detach()))
            b=net.channel_decoder.register_forward_pre_hook(lambda m,i:captured.update(received=i[0].detach()))
            torch.manual_seed(123)
            net.forward_channel_batches(f,f[...,:3],torch.nn.functional.one_hot(q,len(budgets)).float(),10,
                paired_noise=True,noise_reference_symbols=64)
            a.remove(); b.remove()
            noises.append((captured['received']-captured['z'])[...,:8])
        torch.testing.assert_close(noises[0],noises[1],rtol=1e-5,atol=1e-6)

    def test_evaluation_covers_eight_rates_and_charges_payload(self):
        from gaussian_jscc.modular_evaluation import evaluate_paths
        from test_modular_evaluation import scene_fixture
        from test_render_first import synthetic_render
        _,g,f,_=setup(); model=ModularGaussianCodec(self.config()); scene=scene_fixture(model,g,f)
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            rows=evaluate_paths(model,scene,step=0,phase='initial',snrs=[10],trials=2,seed=42,
                save_images=False,out=Path(t),noise_reference_symbols=64)
        awgn=[r for r in rows if r['path']=='awgn']
        self.assertEqual({r['layout'] for r in awgn},set(map(str,range(1,9)))|{'mixed'})
        self.assertEqual([r['payload_complex_symbols'] for r in awgn if r['layout']!='mixed'],
                         [64,128,192,256,384,512,768,1024])
        self.assertTrue(all(r['noise_reference_complex_symbols']==64 for r in rows))


class RateSweepExperimentTests(unittest.TestCase):
    def test_frozen_teacher_balanced_updates_resume_and_exports(self):
        from gaussian_jscc.channel_rate_sweep import parser,main
        from gaussian_jscc.modular_experiment import main as fit
        from gaussian_jscc.transport import load_checkpoint
        import test_modular_experiment as fixture
        import test_render_first as rendering
        cameras=rendering.RenderFirstTests().cameras()
        def camera_split(*args):
            return cameras[:1] if args[-1]=='test' else cameras
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',side_effect=camera_split):
            root=Path(t); source=fixture.source_fixture(root); teacher=root/'teacher'
            opts=fixture.ModularExperimentTests().args(source,teacher)
            opts.channel_bootstrap_steps=opts.channel_render_steps=0
            fit(opts)
            checkpoint=teacher/'checkpoints/final/codec.pt'; original=checkpoint.read_bytes()
            reference=load_checkpoint(checkpoint,'cpu')
            def options(out,default_arms=False):
                argv=['--run',str(teacher),'--out',str(out),'--device','cpu',
                    '--updates-per-rate','1','--validate-every','100','--save-every','2','--channel-hidden','16',
                    '--no-lpips','--no-save-images','--test-views','1','--trials','1']
                # Historical two-arm resume remains explicitly reproducible.
                if not default_arms: argv+=['--arms','three_tier','fine_tiers']
                return parser().parse_args(argv)
            summary=main(options(root/'full'))
            self.assertEqual(summary['status'],'complete')
            self.assertTrue((root/'full/comparison/rate_distortion.csv').exists())
            self.assertTrue((root/'full/comparison/common_budget_comparison.csv').exists())
            self.assertTrue((root/'full/comparison/quality_vs_symbols.png').exists())
            for arm,count in (('three_tier',3),('fine_tiers',8)):
                run=root/'full'/arm
                loss=[json.loads(s) for s in (run/'metrics/loss.jsonl').read_text().splitlines()]
                self.assertEqual(len(loss),count)
                self.assertEqual({r['symbols'] for r in loss},set((8,24,48) if count==3 else RATES[1:]))
                model=load_checkpoint(run/'checkpoints/final/codec.pt','cpu')
                for key,value in reference.state_dict().items():
                    if key.startswith(('semantic_encoder.','semantic_decoder.')) or key in ('attr_mean','attr_std'):
                        torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
                self.assertTrue(all(r['updates']['semantic_decoder']['update_norm']==0 for r in loss))
            for budget in (8,24,48):
                paired=[]
                for arm in ('three_tier','fine_tiers'):
                    rows=[json.loads(s) for s in (root/'full'/arm/'metrics/loss.jsonl').read_text().splitlines()]
                    paired.append([(r['sample_seed'],r['training_group_index']) for r in rows if r['symbols']==budget])
                self.assertEqual(*paired)
            main(options(root/'resumed'),max_updates=5)
            resume=options(root/'resumed'); resume.resume=True; main(resume)
            for arm in ('three_tier','fine_tiers'):
                full=load_checkpoint(root/f'full/{arm}/checkpoints/final/codec.pt','cpu')
                resumed=load_checkpoint(root/f'resumed/{arm}/checkpoints/final/codec.pt','cpu')
                for key,value in full.state_dict().items():
                    torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(checkpoint.read_bytes(),original)
            bad=options(root/'resumed'); bad.resume=True; bad.lr=2e-4
            with self.assertRaisesRegex(ValueError,'settings'): main(bad)
            bad=options(root/'resumed'); bad.dry_run=True; bad.eval_snrs=[10.,10.]
            with self.assertRaisesRegex(ValueError,'settings'): main(bad)
            from test_perceptual_metrics import backend
            with patch('gaussian_jscc.perceptual_metrics._load_lpips_model',side_effect=backend):
                perceptual=options(root/'perceptual',default_arms=True); perceptual.lpips=True
                report=main(perceptual)
            self.assertFalse((root/'perceptual/three_tier').exists())
            self.assertEqual(report['plan']['expected_updates'],{'fine_tiers':8})
            self.assertEqual(len((root/'perceptual/fine_tiers/metrics/loss.jsonl').read_text().splitlines()),8)
            self.assertTrue((root/'perceptual/comparison/training_history.png').exists())
            self.assertTrue(all(r['source_lpips'] is not None and r['photo_lpips'] is not None for r in report['final_test']))


if __name__=='__main__': unittest.main()
