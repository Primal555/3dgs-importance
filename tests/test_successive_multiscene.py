"""Shared Joint B adaptation and zero-shot heldout boundaries, real CPU codecs."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gaussian_jscc.data import read_ply,write_ply
from test_gaussian_jscc import fixture
from test_render_first import synthetic_render
from test_successive_joint import source_run,cameras


def source_and_manifest(root):
    from gaussian_jscc.successive_joint import main,parser
    parent=source_run(root,context_variant='both');source=root/'jointB'
    main(parser().parse_args(['--run',str(parent),'--out',str(source),'--steps','1',
        '--device','cpu','--test-views','2','--render-trials','1','--no-lpips','--train-semantic-encoder']))
    manifest=root/'manifest.json';record=json.loads(manifest.read_text())
    for index,name in enumerate(('train','playroom','drjohnson'),1):
        raw,_=fixture(8+index*4,1);raw[:,:3]+=index*3;raw[:,11:14]+=index*.1
        write_ply(root/f'{name}.ply',raw,1);(root/name).mkdir()
        record['scenes'].append(dict(name=name,role='heldout' if name=='drjohnson' else 'train',
                                   ply=f'{name}.ply',source=name,prior='none'))
    manifest.write_text(json.dumps(record))
    return source,manifest


class SuccessiveMultisceneTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(27)

    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.successive_multiscene'),
                             'missing current-successive multi-scene/zero-shot entry')
        from gaussian_jscc import successive_multiscene
        return successive_multiscene

    def options(self,source,manifest,out):
        return self.entry().parser().parse_args(['--run',str(source),'--manifest',str(manifest),
            '--out',str(out),'--device','cpu','--steps-per-scene','4','--validation-views','1',
            '--validate-every','2','--save-every','1','--test-views','2','--render-trials','1',
            '--cpu-threads','1','--no-lpips'])

    def test_reject_heldout_training_and_cross_manifest_aliases_before_writing(self):
        entry=self.entry()
        with tempfile.TemporaryDirectory() as folder,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(folder);source,manifest=source_and_manifest(root);args=self.options(source,manifest,root/'out')
            args.dry_run=True
            plan=entry.main(args);self.assertEqual(plan['total_updates'],12)
            self.assertEqual(plan['heldout_scene']['name'],'drjohnson');self.assertFalse((root/'out').exists())
            metadata_path=source/'training.json';metadata=json.loads(metadata_path.read_text())
            changed=dict(metadata,selected_scenes=[dict(name='truck',role='train'),dict(name='drjohnson',role='train')])
            metadata_path.write_text(json.dumps(changed))
            try:
                with self.assertRaisesRegex(ValueError,'Truck-only'):entry.main(args)
            finally:metadata_path.write_text(json.dumps(metadata))
            args.scenes=['train','truck','drjohnson']
            with self.assertRaisesRegex(ValueError,'train'):entry.main(args)
            args.scenes=['train','truck','playroom']
            other=root/'heldout.json';other.write_text(json.dumps(dict(scenes=[dict(name='drjohnson',role='heldout',
                ply='truck.ply',source='source_images')])))
            args.heldout_manifest=str(other)
            with self.assertRaisesRegex(ValueError,'overlap|alias'):entry.main(args)
            self.assertFalse((root/'out').exists())

    def test_lpips_failure_leaves_resumable_initial_checkpoint(self):
        entry=self.entry()
        from test_perceptual_metrics import backend
        with tempfile.TemporaryDirectory() as folder,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(folder);source,manifest=source_and_manifest(root);args=self.options(source,manifest,root/'out')
            args.steps_per_scene=1;args.lpips=True
            with patch('gaussian_jscc.perceptual_metrics._load_lpips_model',side_effect=RuntimeError('LPIPS unavailable')):
                with self.assertRaisesRegex(RuntimeError,'pretrained LPIPS'):entry.main(args)
            latest=root/'out/checkpoints/latest/codec.pt'
            self.assertTrue(latest.exists(),'first evaluation failure must leave an exact-resume checkpoint')
            saved=torch.load(latest,weights_only=True);self.assertEqual(saved['step'],0)
            args.resume=True
            with patch('gaussian_jscc.perceptual_metrics._load_lpips_model',side_effect=backend):
                result=entry.main(args)
            self.assertEqual(result['step'],3)

    def test_server_launcher_dry_run_uses_6000_updates_and_separate_heldout_manifest(self):
        entry=self.entry()
        bash=Path('C:/softwares/Git/bin/bash.exe')
        if not bash.exists():self.skipTest('Git Bash unavailable for Windows launcher integration')
        with tempfile.TemporaryDirectory() as folder,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(folder);source,manifest=source_and_manifest(root);out=root/'launch'
            record=json.loads(manifest.read_text());heldout=root/'heldout_manifest.json'
            heldout.write_text(json.dumps(dict(scenes=[s for s in record['scenes'] if s['role']=='heldout'])))
            manifest.write_text(json.dumps(dict(scenes=[s for s in record['scenes'] if s['role']=='train'])))
            script=Path('scripts/train_successive_multiscene.sh').resolve()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHON_BIN=Path(sys.executable).as_posix(),
                     HELDOUT_MANIFEST=heldout.as_posix(),PYTHONUTF8='1')
            result=subprocess.run([str(bash),script.as_posix(),source.as_posix(),manifest.as_posix(),out.as_posix(),
                '--dry-run','--no-lpips'],cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=45)
            self.assertEqual(result.returncode,0,result.stderr)
            plan=json.loads(result.stdout[result.stdout.index('{'):])
            self.assertEqual(plan['total_updates'],6000);self.assertEqual(plan['heldout_scene']['name'],'drjohnson')
            self.assertEqual(Path(plan['settings']['heldout_manifest']),heldout)
            self.assertFalse(out.exists())

    def test_balanced_updates_preserve_stats_and_measure_unadapted_fourth_scene(self):
        entry=self.entry()
        with tempfile.TemporaryDirectory() as folder,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(folder);source,manifest=source_and_manifest(root);path=source/'checkpoints/final/codec.pt'
            before=path.read_bytes();source_state=torch.load(path,weights_only=True)['state_dict'];out=root/'out'
            done=entry.main(self.options(source,manifest,out))
            self.assertEqual(done['status'],'complete');self.assertEqual(done['step'],12)
            self.assertEqual(done['scene_completed'],dict(train=4,truck=4,playroom=4))
            losses=[json.loads(s) for s in (out/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['scene'] for r in losses],['train','truck','playroom']*4)
            self.assertEqual([r['layout'] for r in losses],['1']*3+['2']*3+['3']*3+['mixed']*3)
            self.assertTrue(all(r['module_lrs']['semantic_encoder']==1e-5 for r in losses))
            final=torch.load(out/'checkpoints/final/codec.pt',weights_only=True)
            initial=torch.load(out/'checkpoints/initial/codec.pt',weights_only=True)
            for n,v in source_state.items():torch.testing.assert_close(initial['state_dict'][n],v,rtol=0,atol=0)
            for n in ('attr_mean','attr_std'):torch.testing.assert_close(final['state_dict'][n],source_state[n],rtol=0,atol=0)
            for module in ('semantic_encoder','semantic_decoder','channel_encoder','channel_decoder'):
                self.assertTrue(any(not torch.equal(v,final['state_dict'][n]) for n,v in source_state.items() if n.startswith(module+'.')),module)
            self.assertEqual(int(final['training']['optimizer']['state'][0]['step']),12)
            self.assertEqual(before,path.read_bytes())
            heldout=out/'scenes/drjohnson'
            self.assertEqual({p.name for p in (heldout/'evaluation').iterdir()},{'initial','final'})
            self.assertEqual({p.name for p in (out/'scenes/train/evaluation').iterdir()},
                             {'initial','round_000002','final'})
            for phase in ('initial','final'):
                rows=[json.loads(s) for s in (heldout/f'evaluation/{phase}/metrics/validation.jsonl').read_text().splitlines()]
                self.assertEqual({r['layout'] for r in rows if r['path']=='awgn'},{'1','2','3','mixed'})
                self.assertTrue(all(r['views']==2 for r in rows))
                self.assertTrue(all(r.get('scale_stream_bytes',0)>0 for r in rows if r['path']=='awgn'))
            report=json.loads((out/'comparison/summary.json').read_text())
            self.assertEqual(len(report['rows']),72)
            self.assertTrue(all(r['generalization']=='zero_shot_unseen_scene' for r in report['rows'] if r['scene']=='drjohnson'))
            self.assertTrue((out/'charts/heldout_quality.png').exists())
            self.assertFalse(list(out.rglob('allocation.pt')))

    def test_exact_midround_and_evaluation_resume_keep_shared_adam_and_rng(self):
        entry=self.entry()
        with tempfile.TemporaryDirectory() as folder,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(folder);source,manifest=source_and_manifest(root)
            full_args=self.options(source,manifest,root/'full');full_args.steps_per_scene=2
            entry.main(full_args);full=torch.load(root/'full/checkpoints/final/codec.pt',weights_only=True)
            for cut in (2,'evaluation'):
                args=self.options(source,manifest,root/str(cut));args.steps_per_scene=2
                if cut=='evaluation':
                    real=entry.evaluate_saved
                    def fail(*a,**kw):real(*a,**kw);raise RuntimeError('evaluation interrupted')
                    with patch.object(entry,'evaluate_saved',side_effect=fail):
                        with self.assertRaisesRegex(RuntimeError,'evaluation interrupted'):entry.main(args)
                else:entry.main(args,max_updates=cut)
                args.resume=True;entry.main(args)
                resumed=torch.load(Path(args.out)/'checkpoints/final/codec.pt',weights_only=True)
                for n,v in full['state_dict'].items():torch.testing.assert_close(v,resumed['state_dict'][n],rtol=0,atol=0)
                for key,state in full['training']['optimizer']['state'].items():
                    for n,v in state.items():torch.testing.assert_close(v,resumed['training']['optimizer']['state'][key][n],rtol=0,atol=0)
            # Changing only heldout target attributes must leave every learned
            # weight and optimizer moment exactly unchanged.
            raw,degree=read_ply(root/'drjohnson.ply');raw[:,11:14]+=2.
            write_ply(root/'drjohnson.ply',raw,degree)
            args=self.options(source,manifest,root/'other_heldout');args.steps_per_scene=2
            entry.main(args)
            other=torch.load(root/'other_heldout/checkpoints/final/codec.pt',weights_only=True)
            for n,v in full['state_dict'].items():torch.testing.assert_close(v,other['state_dict'][n],rtol=0,atol=0)
            for key,state in full['training']['optimizer']['state'].items():
                for n,v in state.items():torch.testing.assert_close(v,other['training']['optimizer']['state'][key][n],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
