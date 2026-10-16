"""One-pair noisy color training, objective-local layouts and exact continuation."""
import copy
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F

import test_modular_experiment as experiment_fixture
import test_render_first as rendering
from test_gaussian_jscc import fixture
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import prepare, to_features


def setup():
    torch.set_num_threads(1); torch.manual_seed(27)
    raw, _ = fixture(16, 1); raw, geometry, _ = prepare(raw, 16)
    model = GaussianCodec(CodecConfig(sh_degree=1, hidden=16, grid_dim=4, levels=(2,),
        block_size=8, depth=2, decoder_window=4, rates=(0,12,24,48),
        position_delivery='quantized', position_bits=16, position_compression='delta_zlib',
        prefix_mode='progressive', decoder_mode='residual'))
    model.attr_mean.copy_(raw[:,3:].mean(0)); model.attr_std.copy_(raw[:,3:].std(0).clamp_min(.01))
    f, _ = to_features(raw, geometry, model)
    return model, geometry, f.reshape(2,8,-1), torch.full((2,8),3,dtype=torch.long)


class IntegratedTrainingTests(unittest.TestCase):
    def test_paired_evaluation_keeps_rng_and_measures_packet_side_costs(self):
        from types import SimpleNamespace
        from gaussian_jscc.integrated_evaluation import evaluate
        from gaussian_jscc.data import to_raw
        from gaussian_jscc.modular_checkpoint import capture_rng
        m,g,f,q=setup()
        scene=SimpleNamespace(groups=[f],group_ids=[torch.arange(16).reshape(2,8)],
            geometry=g,raw=to_raw(f.reshape(16,-1),g,m).detach(),degree=1,
            val_cameras=rendering.RenderFirstTests().cameras(),reference=rendering.Reference(),
            white_background=False,net_bits_per_use=2.)
        before=capture_rng()
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render):
            rows=evaluate(m,scene,out=t,step=0,phase='initial',save_images=True)
            again=evaluate(m,scene,out=t,step=1,phase='bootstrap')
            for a,b in zip(rows,again):
                self.assertEqual((a['source_mse'],a['source_color_r_bias']),(b['source_mse'],b['source_color_r_bias']))
            self.assertEqual({r['path'] for r in rows},{'awgn','finite_noiseless'})
            self.assertEqual(len(list((Path(t)/'images/initial').glob('*.png'))),8)
            self.assertFalse((Path(t)/'images/bootstrap').exists())
            for row in rows:
                self.assertGreater(row['xyz_bytes'],0);self.assertGreater(row['metadata_bytes'],0)
                self.assertEqual(row['total_channel_uses'],row['payload_complex_symbols']+4*row['xyz_bytes']+4*row['metadata_bytes'])
        after=capture_rng()
        self.assertEqual(before['python'],after['python']);self.assertEqual(before['numpy'],after['numpy'])
        torch.testing.assert_close(before['torch'],after['torch'],rtol=0,atol=0)

    def test_objective_local_layouts_cover_all_tiers_without_color_alias(self):
        from gaussian_jscc.integrated_training import update_spec
        counts = Counter(update_spec('bootstrap', s) for s in range(1,5001))
        self.assertEqual([counts['local_response',q] for q in ('1','2','3','mixed')],[938,938,937,937])
        self.assertEqual([counts['direction_color',q] for q in ('1','2','3','mixed')],[312,312,313,313])
        for interval in (0,3,4,5):
            rows=[update_spec('bootstrap',s,interval) for s in range(1,54)]
            for objective in {x[0] for x in rows}:
                histogram=Counter(q for obj,q in rows if obj==objective)
                self.assertLessEqual(max(histogram.get(q,0) for q in ('1','2','3','mixed'))-
                                     min(histogram.get(q,0) for q in ('1','2','3','mixed')),1)
        self.assertEqual(Counter(update_spec('render',s) for s in range(1,1001)),
                         Counter({('render',q):250 for q in ('1','2','3','mixed')}))
        for phase,step,interval in [('bad',1,4),('bootstrap',0,4),('bootstrap',1,-1)]:
            with self.assertRaises(ValueError):update_spec(phase,step,interval)

    def test_noisy_color_updates_encoder_and_color_heads_not_noncolor_adam_momentum(self):
        from gaussian_jscc.integrated_training import make_optimizer,bootstrap_step,finish_update
        m,g,f,q=setup(); opt=make_optimizer(m,1e-3)
        centers=torch.tensor([[-3.,-2.,-4.],[3.,-2.,4.]])
        for _ in range(2):
            opt.zero_grad(set_to_none=True); bootstrap_step(m,f,q,g,centers)
            finish_update(m,opt,{n:p.detach().clone() for n,p in m.named_parameters()})
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        opt.zero_grad(set_to_none=True)
        loss,stats=bootstrap_step(m,f,q,g,centers,color=True)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(stats['objective'],'directional_color_mse_unclipped_v1')
        self.assertGreater(float(m.learned.symbol_head.weight.grad.abs().sum()),0)
        finish_update(m,opt,before)
        for n,p in m.named_parameters():
            if '.heads.' in n and n.split('.heads.')[1].split('.')[0] not in ('dc','sh'):
                torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        for heads in [m.learned.heads]+[r.heads for r in m.learned.refinements]:
            self.assertIsNotNone(heads['dc'].weight.grad)
        self.assertTrue(any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if '.heads.dc.' in n))

    def test_replay_matches_checkpoint_and_prefixes_ignore_future_slots(self):
        from gaussian_jscc.integrated_training import make_optimizer
        from gaussian_jscc.training import full_scene_step
        from gaussian_jscc.codec import channel,prefix_mask
        m,g,f,q=setup(); other=copy.deepcopy(m)
        values=[]
        for mode,net in [('replay',m),('checkpoint',other)]:
            make_optimizer(net,1e-4);torch.manual_seed(19)
            values.append(full_scene_step(net,[(f[:1],q[:1]),(f[1:],q[1:])],g,10,'awgn',
                          lambda raw:raw.square().mean(),attr_weight=0,mode=mode)[0])
        torch.testing.assert_close(*values)
        for (n,p),(_,ref) in zip(m.named_parameters(),other.named_parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,ref.grad,rtol=2e-4,atol=2e-6,msg=n)
        # Make enhancements nonzero so this tests actual prefix dependency.
        for r in m.learned.refinements:
            for h in r.heads.values():torch.nn.init.normal_(h.weight,std=.01)
        z=m.learned.encode_full(f,f[...,:3],q>0,10)
        received=channel(z.reshape(-1,2),10,'awgn').reshape_as(z)
        stages,_=m.learned.decode_stages(received,q,10)
        damaged=received.clone();damaged[...,48:]+=100
        others,_=m.learned.decode_stages(damaged,q,10)
        torch.testing.assert_close(stages[1],others[1],rtol=0,atol=0)
        self.assertGreater(float((stages[2]-others[2]).abs().sum().detach()),0)
        q[0,0]=0
        mask=prefix_mask(q.flatten(),m.cfg.rates).reshape_as(z)
        safe=m.learned.decode(received.masked_fill(~mask,0),q,10)
        self.assertTrue(torch.isfinite(safe).all());self.assertTrue((safe[0,0]==0).all())


class IntegratedExperimentTests(unittest.TestCase):
    def options(self,source,out):
        from gaussian_jscc.integrated_experiment import parser
        return parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu',
            '--bootstrap-steps','8','--render-steps','4','--validate-every','8','--save-every','4'])

    def cameras(self):
        cams=rendering.RenderFirstTests().cameras()
        for c in cams:
            c.camera_center=torch.tensor([-3.,-2.,-4.])
            c.FoVx=1.;c.world_view_transform=torch.eye(4)
        return cams

    def test_random_pipeline_exact_resume_at_both_phases_and_boundary(self):
        from gaussian_jscc.integrated_experiment import main
        from gaussian_jscc.transport import load_checkpoint,transmit,receive
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()):
            root=Path(t);source=experiment_fixture.source_fixture(root)
            original=(source/'codec.pt').read_bytes();a=root/'full'
            main(self.options(source,a))
            model=load_checkpoint(a/'checkpoints/final/codec.pt','cpu')
            self.assertEqual(model.cfg.rates,(0,12,24,48));self.assertEqual(model.cfg.decoder_mode,'residual')
            rows=[json.loads(l) for l in (a/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),12)
            self.assertEqual([r['layout'] for r in rows if r['objective']=='directional_color_mse_unclipped_v1'],['3','mixed'])
            self.assertTrue(all(r['aux_loss']==0 for r in rows if r['phase']=='render'))
            self.assertEqual((source/'codec.pt').read_bytes(),original)
            self.assertTrue((a/'complete.json').is_file());self.assertTrue((a/'charts/quality.png').is_file())
            self.assertEqual(len(list((a/'images/initial').glob('*.png'))),4)
            self.assertEqual(len(list((a/'images/final').glob('*.png'))),4)
            for cut in (5,8,10):
                b=root/f'cut{cut}';opts=self.options(source,b);main(opts,max_updates=cut)
                self.assertFalse((b/'complete.json').exists())
                opts.resume=str(b/'checkpoints/latest/codec.pt');main(opts)
                recovered=load_checkpoint(b/'checkpoints/final/codec.pt','cpu')
                for n,v in model.state_dict().items():torch.testing.assert_close(v,recovered.state_dict()[n],rtol=0,atol=0)
                original_state=torch.load(a/'checkpoints/final/codec.pt',weights_only=True)['training']
                recovered_state=torch.load(b/'checkpoints/final/codec.pt',weights_only=True)['training']
                torch.testing.assert_close(original_state['optimizer'],recovered_state['optimizer'],rtol=0,atol=0)
                torch.testing.assert_close(original_state['rng']['torch'],recovered_state['rng']['torch'],rtol=0,atol=0)
                rb=[json.loads(l) for l in (b/'metrics/loss.jsonl').read_text().splitlines()]
                self.assertEqual([(r['objective'],r['layout'],r['loss']) for r in rows],[(r['objective'],r['layout'],r['loss']) for r in rb])
            # Receiver-only roundtrip with q0 holes and no source encoder access.
            raw,_=fixture(19,1);q=torch.arange(19)%4
            stats=transmit(model,raw,q,10,'awgn',42,root/'packet')
            model.learned.encode=lambda *a: (_ for _ in ()).throw(AssertionError('receiver used encoder'))
            result=receive(model,root/'packet');self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(stats['payload_complex_symbols'],sum((0,12,24,48)[int(v)] for v in q))

    def test_resume_rejects_changed_objective_and_evaluation_preserves_rng(self):
        from gaussian_jscc.integrated_experiment import main
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()):
            root=Path(t);source=experiment_fixture.source_fixture(root);out=root/'run'
            opts=self.options(source,out);main(opts,max_updates=4)
            opts.resume=str(out/'checkpoints/latest/codec.pt');opts.color_every=0
            with self.assertRaisesRegex(ValueError,'settings'):main(opts)
            opts.color_every=4;opts.rates=[8,24,48]
            with self.assertRaisesRegex(ValueError,'settings'):main(opts)
            opts.rates=[12,24,48]
            scene_path=source/'scenes/truck/scene.json'
            record=json.loads(scene_path.read_text());record['fingerprint']='changed'
            scene_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'identity'):main(opts)

    def test_final_diagnostics_ignore_missing_unused_existence_prior(self):
        from gaussian_jscc.integrated_experiment import main
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()):
            root=Path(t);source=experiment_fixture.source_fixture(root)
            manifest=root/'manifest.json';spec=json.loads(manifest.read_text())
            spec['scenes'][0]['prior']='unavailable_prior.npy';manifest.write_text(json.dumps(spec))
            opts=self.options(source,root/'run');opts.bootstrap_steps=1;opts.render_steps=0
            main(opts)
            self.assertTrue((root/'run/complete.json').exists())
            self.assertTrue((root/'run/stage_diagnostics.json').exists())

    def test_exact_resume_rejects_changed_camera_data_and_accounting(self):
        from gaussian_jscc.integrated_experiment import main
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()) as load:
            root=Path(t);source=experiment_fixture.source_fixture(root);out=root/'run'
            opts=self.options(source,out);main(opts,max_updates=1)
            opts.resume=str(out/'checkpoints/latest/codec.pt')
            for field in ('camera_center','original_image','FoVx','world_view_transform'):
                cameras=self.cameras();setattr(cameras[0],field,getattr(cameras[0],field)+.1)
                load.return_value=cameras
                with self.assertRaisesRegex(ValueError,'scene|camera'):main(opts,max_updates=1)
            load.return_value=self.cameras()
            path=source/'training.json';metadata=json.loads(path.read_text());metadata['net_bits_per_use']=3.
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'scene|settings'):main(opts,max_updates=1)

    def test_completed_outputs_and_older_checkpoints_cannot_be_rewound(self):
        from gaussian_jscc.integrated_experiment import main
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render',side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()):
            root=Path(t);source=experiment_fixture.source_fixture(root);out=root/'done'
            opts=self.options(source,out);opts.bootstrap_steps=1;opts.render_steps=0;main(opts)
            opts.resume=str(out/'checkpoints/initial/codec.pt')
            with self.assertRaisesRegex(ValueError,'complete'):main(opts,max_updates=1)
            out=root/'incomplete';opts=self.options(source,out);main(opts,max_updates=10)
            opts.resume=str(out/'checkpoints/end_bootstrap/codec.pt')
            with self.assertRaisesRegex(ValueError,'older|rewind'):main(opts,max_updates=1)


if __name__=='__main__':unittest.main()
