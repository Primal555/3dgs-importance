"""Real CPU codecs/Adam with synthetic rendering, not convergence tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_gaussian_jscc import fixture
from test_render_first import synthetic_render
from gaussian_jscc.data import write_ply


def scene_fixture(root):
    specs=[]
    for i,name in enumerate(('train','truck','playroom')):
        raw,_=fixture(8+i*4,1)
        raw[:,:3]+=i*3
        raw[:,11:14]+=i*.4
        write_ply(root/f'{name}.ply',raw,1)
        (root/name).mkdir()
        specs.append(dict(name=name,role='train',ply=f'{name}.ply',source=name,prior='none'))
    specs.append(dict(name='drjohnson',role='heldout',ply='not_loaded.ply',source='not_loaded'))
    path=root/'manifest.json';path.write_text(json.dumps(dict(scenes=specs)))
    config=root/'codec.json'
    config.write_text(json.dumps(dict(sh_degree=1,hidden=16,grid_dim=4,levels=[2],block_size=8,decoder_window=4,
        context_variant='both',context_centers=[4,2],context_neighbors=[2,3],context_radii=[2.,4.],
        channel_variant='progressive',channel_hidden=16,channel_prefix_mode='full_prefix',
        channel_denoiser_mode='pointwise',channel_denoiser_hidden=8,
        channel_neighbor_mode='plain',channel_neighbor_hidden=8,channel_neighbor_neighbors=2,
        rates=[0,8,24,48])))
    return path,config


def cameras(*args):
    split=args[-1]
    return [SimpleNamespace(factor=v,original_image=torch.full((3,8,8),.5),
                            image_name=f'{split}_{i}',camera_center=torch.tensor([v,1.,2.]))
            for i,v in enumerate((.4,1.3,2.2) if split=='train' else (.8,1.8))]


def options(root,out):
    from gaussian_jscc.modular_multiscene import parser
    return parser().parse_args(['--manifest',str(root/'manifest.json'),'--codec-config',str(root/'codec.json'),
        '--out',str(out),'--device','cpu','--no-lpips','--cpu-threads','1',
        '--blocks-per-batch','2','--views-per-step','1','--validation-views','1',
        '--semantic-bootstrap-steps','4','--semantic-render-steps','1',
        '--channel-bootstrap-steps','4','--channel-render-steps','1',
        '--validate-every','4','--save-every','2','--trials','1'])


class ModularMultisceneTests(unittest.TestCase):
    def test_inputs_exclude_heldout_and_reject_missing_or_heldout_selection(self):
        from gaussian_jscc.modular_multiscene import prepare_inputs
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);scene_fixture(root);args=options(root,root/'run')
            specs,cfg=prepare_inputs(args)
            self.assertEqual([s['name'] for s in specs],['train','truck','playroom'])
            self.assertEqual(cfg.rates,(0,8,24,48))
            args.scenes=['drjohnson']
            with self.assertRaisesRegex(ValueError,'train'):prepare_inputs(args)
            args.scenes=['train','train']
            with self.assertRaisesRegex(ValueError,'unique'):prepare_inputs(args)
            args.scenes=['train'];(root/'train.ply').unlink()
            with self.assertRaises(FileNotFoundError):prepare_inputs(args)
            self.assertFalse((root/'run').exists())

    def test_real_balanced_four_stage_training_and_random_shared_statistics(self):
        from gaussian_jscc.modular_multiscene import main
        from gaussian_jscc.data import read_ply
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
                patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(t);scene_fixture(root);out=root/'run'
            result=main(options(root,out))
            self.assertEqual(result['step'],30)
            done=json.loads((out/'complete.json').read_text())
            self.assertEqual(done['status'],'complete')
            self.assertEqual(done['scene_completed']['truck'],dict(semantic_bootstrap=4,semantic_render=1,channel_bootstrap=4,channel_render=1))
            rows=[json.loads(x) for x in (out/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['scene'] for r in rows],['train','truck','playroom']*10)
            self.assertEqual([r['step'] for r in rows],list(range(1,31)))
            for name in ('train','truck','playroom'):
                bootstrap=[r for r in rows if r['scene']==name and r['phase']=='semantic_bootstrap']
                self.assertEqual([r['objective'] for r in bootstrap],['local_response_mse']*3+['directional_color_mse_unclipped_v1'])
                channel=[r for r in rows if r['scene']==name and r['phase']=='channel_bootstrap']
                self.assertEqual([r['layout'] for r in channel],['1','2','3','mixed'])
            for r in rows:
                self.assertTrue(r['frozen_parameters_unchanged'])
                if r['phase'] in ('semantic_bootstrap','semantic_render'):
                    self.assertEqual(r['module_lrs']['semantic_decoder'],1e-4)
                    self.assertEqual(r['updates']['channel_encoder']['update_norm'],0)
                if r['phase']=='channel_render':
                    self.assertEqual(r['module_lrs']['semantic_decoder'],1e-5)
                    self.assertEqual(r['module_lrs']['channel_decoder'],1e-4)
                    self.assertEqual(r['updates']['semantic_encoder']['update_norm'],0)
            initial=torch.load(out/'checkpoints/initial/codec.pt',weights_only=True)
            final=torch.load(out/'checkpoints/final/codec.pt',weights_only=True)
            phase_end=torch.load(out/'checkpoints/end_semantic_bootstrap/codec.pt',weights_only=True)
            # The first semantic parameter participates in all 12 shared updates.
            # Resetting Adam per scene would leave this at 1 instead of 12.
            self.assertEqual(int(phase_end['training']['optimizer']['state'][0]['step']),12)
            raw=[read_ply(root/f'{n}.ply')[0][:,3:].double() for n in ('train','truck','playroom')]
            mean=sum(x.mean(0) for x in raw)/3
            std=(sum(x.square().mean(0) for x in raw)/3-mean.square()).clamp_min(0).sqrt().clamp_min(.01)
            torch.testing.assert_close(initial['state_dict']['attr_mean'],mean.float())
            torch.testing.assert_close(initial['state_dict']['attr_std'],std.float())
            self.assertFalse(torch.equal(initial['state_dict']['semantic_encoder.readout.weight'],
                                         final['state_dict']['semantic_encoder.readout.weight']))
            record=json.loads((out/'training.json').read_text())
            self.assertTrue(record['random_initialization']);self.assertFalse(record['source_weights_loaded'])
            self.assertEqual(record['step_budgets_scope'],'per_scene')
            self.assertEqual(record['statistics_scenes'],['train','truck','playroom'])
            self.assertFalse(list(out.rglob('*allocation*.pt')))
            for name in ('train','truck','playroom'):
                child=out/'scenes'/name
                self.assertEqual({p.name for p in (child/'images').iterdir()},{'initial','final'})
                tests=[json.loads(x) for x in (child/'metrics/test.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['views']==2 for r in tests))
                self.assertTrue(all('source_ssim' in r and 'photo_ssim' in r for r in tests))
            self.assertTrue((out/'metrics/macro_validation.jsonl').exists())
            self.assertEqual(json.loads((out/'progress.json').read_text())['status'],'complete')

    def test_exact_midround_and_phase_boundary_resume_preserves_adam_and_rng(self):
        from gaussian_jscc.modular_multiscene import main
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
                patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(t);scene_fixture(root)
            opts=options(root,root/'full')
            opts.semantic_bootstrap_steps=1;opts.channel_bootstrap_steps=1
            main(opts)
            full=torch.load(root/'full/checkpoints/final/codec.pt',weights_only=True)
            expected=[json.loads(x) for x in (root/'full/metrics/loss.jsonl').read_text().splitlines()]
            for cut in (2,3,8,11):
                out=root/f'cut{cut}';opts.out=str(out);opts.resume=None
                main(opts,max_updates=cut)
                checkpoint=out/'checkpoints/latest/codec.pt'
                saved=torch.load(checkpoint,weights_only=True)
                self.assertEqual(saved['training']['progress']['next_scene'],cut%3)
                if cut==2:
                    # Later writes were not checkpointed: archive before replaying.
                    with (out/'metrics/loss.jsonl').open('a') as f:f.write(json.dumps(dict(step=999))+'\n')
                opts.resume=str(checkpoint);main(opts)
                resumed=torch.load(out/'checkpoints/final/codec.pt',weights_only=True)
                for name,value in full['state_dict'].items():
                    torch.testing.assert_close(value,resumed['state_dict'][name],rtol=0,atol=0,msg=f'cut={cut}, {name}')
                actual=[json.loads(x) for x in (out/'metrics/loss.jsonl').read_text().splitlines()]
                for a,b in zip(expected,actual):
                    self.assertEqual((a['scene'],a['layout'],a['training_view_indices']),
                                     (b['scene'],b['layout'],b['training_view_indices']))
                    self.assertEqual(a['loss'],b['loss'])
                self.assertEqual(len(actual),12)
                if cut==2:self.assertTrue(list((out/'metrics').glob('loss.uncheckpointed_*.jsonl')))

    def test_resume_rejects_identity_or_objective_change_without_rewriting_scene_record(self):
        from gaussian_jscc.modular_multiscene import main
        from gaussian_jscc.data import read_ply
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
                patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(t);scene_fixture(root);out=root/'run';opts=options(root,out)
            main(opts,max_updates=1)
            opts.resume=str(out/'checkpoints/latest/codec.pt');opts.image_aux_weight=.5
            with self.assertRaisesRegex(ValueError,'protocol'):main(opts)
            opts.image_aux_weight=1/3
            record=out/'scenes/train/scene.json';before=record.read_bytes()
            raw,degree=read_ply(root/'train.ply');raw[0,0]+=.1;write_ply(root/'train.ply',raw,degree)
            with self.assertRaisesRegex(ValueError,'scene'):main(opts)
            self.assertEqual(record.read_bytes(),before)

    def test_lpips_measurement_does_not_change_training_weights(self):
        from gaussian_jscc.modular_multiscene import main
        from test_perceptual_metrics import backend
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
                patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras),\
                patch('gaussian_jscc.perceptual_metrics._load_lpips_model',side_effect=backend):
            root=Path(t);scene_fixture(root);opts=options(root,root/'no_metric')
            opts.semantic_bootstrap_steps=0;opts.semantic_render_steps=0;opts.channel_bootstrap_steps=1;opts.channel_render_steps=0
            main(opts);opts.lpips=True;opts.out=str(root/'metric');main(opts)
            a=torch.load(root/'no_metric/checkpoints/final/codec.pt',weights_only=True)['state_dict']
            b=torch.load(root/'metric/checkpoints/final/codec.pt',weights_only=True)['state_dict']
            for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
            rows=[json.loads(x) for x in (root/'metric/metrics/macro_test.jsonl').read_text().splitlines()]
            self.assertTrue(all('source_lpips' in r and 'photo_lpips' in r for r in rows))

    def test_macro_is_equal_scene_weight_not_point_or_view_weight(self):
        from gaussian_jscc.modular_multiscene_report import macro_metrics
        row=dict(step=12,phase='test',path='awgn',layout='2',actual_snr=10.,source_psnr=20.,photo_psnr=19.,views=100)
        rows=macro_metrics({'train':[row],'truck':[dict(row,source_psnr=30.,photo_psnr=21.,views=2)]})
        self.assertEqual(rows[0]['source_psnr'],25.)
        self.assertEqual(rows[0]['photo_psnr'],20.)
        self.assertNotIn('source_lpips',rows[0])
        with self.assertRaisesRegex(ValueError,'every scene'):
            macro_metrics({'train':[row],'truck':[dict(row,layout='3')]})

    def test_resume_cannot_import_another_runs_checkpoint(self):
        from gaussian_jscc.modular_multiscene import main
        import shutil
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
                patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(t);scene_fixture(root);opts=options(root,root/'run')
            main(opts,max_updates=1)
            external=root/'other_run.pt'
            shutil.copy2(root/'run/checkpoints/latest/codec.pt',external)
            opts.resume=str(external)
            with self.assertRaisesRegex(ValueError,'original output'):main(opts)


if __name__=='__main__':unittest.main()
