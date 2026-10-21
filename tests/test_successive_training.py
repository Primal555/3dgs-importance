"""Common-loss preservation, phase scopes and exact three-phase resume."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class SuccessiveTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_color_balance_has_known_gradient_ratio_and_retains_common_loss(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.successive_objective'))
        from gaussian_jscc.successive_objective import add_color
        value=torch.tensor([1.,2.],requires_grad=True)
        common=value.square().mean();color=16*common;state={}
        loss,stats=add_color(common,color,[value],state,step=600,every=4,start=500,stat_start=250,
                             relative=.25)
        self.assertAlmostEqual(stats['color_weight'],.25/16,places=7)
        self.assertAlmostEqual(float(loss.detach()),3.125,places=6)  # 2.5 + .25*2.5
        loss.backward();torch.testing.assert_close(value.grad,torch.tensor([1.25,2.5]))
        value=torch.ones(1,requires_grad=True)
        common=value.square().mean();color=common*16
        loss,stats=add_color(common,color,[value],{},step=100,every=4,start=500,stat_start=250,relative=.25)
        torch.testing.assert_close(loss,common);self.assertEqual(stats['color_weight'],0.)

    def test_phase_scopes_freeze_teacher_and_reset_adam_at_boundaries(self):
        from gaussian_jscc.successive_training import configure_optimizer
        from test_successive_channel import SuccessiveChannelTests
        model=SuccessiveChannelTests().model()
        rates=dict(main=1e-4,neighbor=1e-4,cooperative=1e-5)
        for phase,expected in [('main',{'channel_encoder','channel_main'}),
                               ('neighbor',{'channel_neighbor'}),
                               ('cooperative',{'channel_encoder','channel_main','channel_neighbor'})]:
            opt=configure_optimizer(model,phase,rates)
            self.assertEqual({g['name'] for g in opt.param_groups},expected)
            self.assertEqual(len(opt.state),0)
            self.assertEqual(bool(model.channel_decoder.neighbor_enabled),phase!='main')
            self.assertTrue(all(not p.requires_grad for p in model.semantic_encoder.parameters()))
            self.assertTrue(all(not p.requires_grad for p in model.semantic_decoder.parameters()))

    def test_short_run_wire_rate_and_exact_resume_across_phase_boundaries(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.successive_experiment'))
        from gaussian_jscc import successive_experiment as entry
        import test_modular_experiment as fixture
        import test_render_first as renders
        from gaussian_jscc.modular_experiment import main as train_teacher
        from gaussian_jscc.channel_normalization_controls import parser as power_parser,run_arm
        def cameras(*args,**kwargs):
            result=renders.RenderFirstTests().cameras()
            for i,camera in enumerate(result):
                camera.camera_center=torch.tensor([5.+i,6.,7.])
                if args[-1]=='test':camera.image_name='test_'+str(i)
            return result
        with tempfile.TemporaryDirectory() as folder, \
             patch('gaussian_jscc.rendering.render',side_effect=renders.synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(folder);source=fixture.source_fixture(root)
            opts=fixture.ModularExperimentTests().args(source,root/'teacher')
            opts.channel_variant='progressive';opts.channel_hidden=16;opts.rates=[8,24,48]
            opts.channel_bootstrap_steps=opts.channel_render_steps=0
            train_teacher(opts)
            args=power_parser().parse_args(['--run',str(root/'teacher'),'--out',str(root/'power'),
                '--steps','1','--device','cpu','--save-every','1','--validate-every','1',
                '--diagnostic-blocks','1','--response-blocks','1','--trials','2','--render-trials','1','--no-lpips'])
            run_arm(args,'block_awgn10')
            parent=root/'power/block_awgn10';endpoint=parent/'checkpoints/end_channel_bootstrap/codec.pt'
            endpoint.parent.mkdir();shutil.copyfile(parent/'checkpoints/final/codec.pt',endpoint)
            source_bytes=endpoint.read_bytes()
            def options(out):
                return entry.parser().parse_args(['--run',str(parent),'--out',str(out),
                    '--main-steps','4','--neighbor-steps','2','--cooperative-steps','4',
                    '--device','cpu','--validate-every','5','--save-every','1','--test-views','2',
                    '--render-trials','1','--response-directions','2','--color-start','0',
                    '--color-stat-start','0','--no-lpips'])
            full=root/'full';result=entry.main(options(full));self.assertEqual(result['step'],10)
            saved=torch.load(full/'checkpoints/final/codec.pt',weights_only=True)
            from gaussian_jscc.modular_checkpoint import restore_training_checkpoint
            for label in ('end_main','end_neighbor','final'):
                restored,opt,prog,_,_=restore_training_checkpoint(full/f'checkpoints/{label}/codec.pt','cpu')
                self.assertEqual({g['name'] for g in opt.param_groups},
                    {'channel_encoder','channel_main'} if label=='end_main' else
                    {'channel_neighbor'} if label=='end_neighbor' else
                    {'channel_encoder','channel_main','channel_neighbor'})
            rows=[json.loads(line) for line in (full/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['main']*4+['neighbor']*2+['cooperative']*4)
            self.assertTrue(any(r['color_weight']>0 for r in rows))
            for row in rows:
                self.assertGreaterEqual(row['common_loss'],row['latent_mse'])
                self.assertEqual(row['updates']['semantic_encoder']['update_norm'],0.)
                self.assertEqual(row['updates']['semantic_decoder']['update_norm'],0.)
                self.assertGreater(row['scale_stream_bytes'],0)
                expected=({'channel_encoder','channel_main'} if row['phase']=='main' else
                          {'channel_neighbor'} if row['phase']=='neighbor' else
                          {'channel_encoder','channel_main','channel_neighbor'})
                self.assertEqual(set(row['optimizer_group_updates']),expected)
                for value in row['optimizer_group_updates'].values():
                    self.assertGreater(value['update_norm'],0.)
            tests=[json.loads(line) for line in (full/'evaluation/final/metrics/validation.jsonl').read_text().splitlines()]
            for row in tests:
                if row['path']!='clean':
                    self.assertGreater(row['scale_stream_bytes'],0)
                    self.assertEqual(row['total_channel_uses'],row['payload_complex_symbols']+
                        row['coordinate_channel_uses']+row['metadata_channel_uses']+row['scale_channel_uses'])
            self.assertTrue((full/'diagnostics/final.json').exists())
            for interrupted in (3,4,6,9):
                args=options(root/f'resume_{interrupted}')
                entry.main(args,max_updates=interrupted);args.resume=True;entry.main(args)
                other=torch.load(Path(args.out)/'checkpoints/final/codec.pt',weights_only=True)
                for key,tensor in saved['state_dict'].items():
                    torch.testing.assert_close(tensor,other['state_dict'][key],rtol=0,atol=0)
                self.assertEqual(saved['training']['progress'],other['training']['progress'])
                for key,state in saved['training']['optimizer']['state'].items():
                    for field,value in state.items():
                        torch.testing.assert_close(value,other['training']['optimizer']['state'][key][field],rtol=0,atol=0)
            self.assertEqual(endpoint.read_bytes(),source_bytes)
            dry=options(root/'dry');dry.dry_run=True;entry.main(dry);self.assertFalse(Path(dry.out).exists())
            bash=Path('C:/softwares/Git/bin/bash.exe') if os.name=='nt' else shutil.which('bash')
            if bash and Path(bash).exists():
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
                launch=subprocess.run([str(bash),'scripts/test_successive_channel_truck.sh',parent.as_posix(),
                    (root/'shell_dry').as_posix(),'--dry-run','--device','cpu'],env=env,capture_output=True,
                    text=True,encoding='utf8',timeout=30)
                self.assertEqual(launch.returncode,0,launch.stdout+launch.stderr)
                self.assertFalse((root/'shell_dry').exists())


if __name__=='__main__':unittest.main()
