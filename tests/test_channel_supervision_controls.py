"""Independent objectives, balanced task/tier schedules, and real exact resume."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch


class ChannelSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.channel_supervision_controls'),
                             'independent channel supervision entry is missing')
        from gaussian_jscc import channel_supervision_controls as c
        return c

    def fixture(self):
        from test_modular_training import setup
        from gaussian_jscc.channel_response_training import frozen_teacher
        from gaussian_jscc.modular_training import configure_phase
        from gaussian_jscc.position_delivery import delivered_positions
        m,g,f,q=setup(); m.set_scene_geometry(g)
        configure_phase(m,'channel_bootstrap',{})
        teacher=frozen_teacher(m)
        with torch.no_grad(): y=m.semantic_encoder(f,f[...,:3],q>0)
        xyz=delivered_positions(f[...,:3],q,m.cfg)
        return m,g,f,q,teacher,y,xyz

    def test_each_task_sees_every_tier_and_default_budget_counts_are_explicit(self):
        c=self.entry()
        for variant,wanted in [('color',{'response':6000,'color':2000}),
                               ('separated',{'latent':2000,'shape':2000,'opacity':2000,'color':2000})]:
            schedule=[c.scheduled_task(variant,i) for i in range(1,8001)]
            self.assertEqual(dict(Counter(t for t,q,n in schedule)),wanted)
            for task,count in wanted.items():
                self.assertEqual(Counter(q for t,q,n in schedule if t==task),
                                 Counter({'1':count//4,'2':count//4,'3':count//4,'mixed':count//4}))
                self.assertEqual([n for t,q,n in schedule if t==task],list(range(1,count+1)))
            self.assertEqual([i for i,(t,_,_) in enumerate(schedule,1) if t=='color'],list(range(4,8001,4)))
        with self.assertRaises(ValueError): c.scheduled_task('color',0)

    def test_portable_objective_metadata_matches_the_actual_task_cycle(self):
        c=self.entry()
        for variant,sequence in [('color',['response','response','response','color']),
                                 ('separated',['latent','shape','opacity','color'])]:
            args=c.parser().parse_args(['--run','unused','--out','unused_out','--variant',variant])
            args.response_weight=.1 if variant=='color' else 0.
            plan={}; c.Supervision(args).add_to_plan(plan)
            self.assertIn('training_objective',plan,'checkpoint objective must describe actual supervision')
            self.assertEqual(plan['training_objective']['schedule'],sequence)
            self.assertEqual(plan['training_objective']['variant'],variant)

    def test_shape_is_independent_and_preserves_absolute_size(self):
        c=self.entry(); m,g,f,q,teacher,y,xyz=self.fixture()
        target=f.flatten(0,1).detach().requires_grad_()
        pred=target.detach().clone(); pred[:,:4]+=7; pred[:,11:]-=20
        pred.requires_grad_(); views=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        loss=c.shape_footprint_loss(pred,target,g,m,views)
        self.assertLess(float(loss.detach()),1e-12)
        larger=target.detach().clone(); larger[:,4:7]+=.5
        larger.requires_grad_(); loss=c.shape_footprint_loss(larger,target,g,m,views)
        self.assertGreater(float(loss.detach()),1e-5)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(larger.grad[:,4:7].abs().sum()),0)
        self.assertEqual(float(larger.grad[:,:4].abs().sum()),0)
        self.assertEqual(float(larger.grad[:,11:].abs().sum()),0)
        signed=target.detach().clone(); signed[:,7:11]*=-1
        torch.testing.assert_close(c.shape_footprint_loss(signed,target,g,m,views),torch.tensor(0.),atol=1e-12,rtol=0)

    def test_opacity_uses_activated_values_only(self):
        c=self.entry(); m,g,f,q,teacher,y,xyz=self.fixture()
        from gaussian_jscc.data import to_raw
        target=f.flatten(0,1).detach().requires_grad_()
        pred=target.detach().clone(); pred[:,3]+=.4; pred[:,4:]+=10; pred.requires_grad_()
        raw_p=to_raw(pred,g,m); raw_t=to_raw(target,g,m)
        expected=(raw_p[:,3].sigmoid()-raw_t[:,3].sigmoid()).square().mean()
        loss=c.opacity_loss(pred,target,m); torch.testing.assert_close(loss,expected)
        loss.backward(); self.assertIsNone(target.grad)
        self.assertGreater(float(pred.grad[:,3].abs().sum()),0)
        self.assertEqual(float(pred.grad[:,:3].abs().sum()+pred.grad[:,4:].abs().sum()),0)

    def test_very_thin_shapes_have_finite_loss_and_gradients(self):
        c=self.entry(); m,g,f,q,teacher,y,xyz=self.fixture()
        target=f.flatten(0,1).detach()
        pred=target.clone(); pred[:,4:7]=torch.tensor([-70.,-70.,8.]); pred.requires_grad_()
        loss=c.shape_footprint_loss(pred,target,g,m,torch.tensor([[1.,1.,1.],[1.,0.,0.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(pred.grad).all())

    def test_color_directions_use_only_supplied_training_centers_and_replay_exactly(self):
        c=self.entry(); m,g,f,q,teacher,y,xyz=self.fixture()
        centers=torch.tensor([[5.,6.,7.]])
        unit=xyz[q>0]
        dirs,ids=c.color_directions(unit,g,centers,8,71)
        expected=torch.nn.functional.normalize(g.denormalize(unit)-centers[0],dim=-1)
        torch.testing.assert_close(dirs[:,:4],expected[:,None].expand(-1,4,-1))
        self.assertEqual(ids,[0,0,0,0])
        again,_=c.color_directions(unit,g,centers,8,71)
        torch.testing.assert_close(dirs,again,rtol=0,atol=0)
        with self.assertRaises(ValueError): c.color_directions(unit,g,centers,3,71)

    def test_all_four_objectives_reach_channels_but_never_teacher_and_ignore_padding(self):
        c=self.entry()
        from gaussian_jscc.channel_response_training import transmit,directions
        for task in ['response','color','latent','shape','opacity']:
            m,g,f,q,teacher,y,xyz=self.fixture(); q[:,-1]=0
            y=y.detach().requires_grad_(); recovered=transmit(m,y,q,xyz,17)
            views=directions(4,51,'cpu')
            color_views,_=c.color_directions(xyz[q>0],g,torch.tensor([[5.,6.,7.]]),8,62)
            loss,stats=c.supervision_loss(m,teacher,y,recovered,q,xyz,g,views,color_views,task,.1)
            self.assertTrue(torch.isfinite(loss))
            expected={'response':stats['latent_mse']+.1*stats['response_mse'],
                      'color':stats['directional_color_mse'],'latent':stats['latent_mse'],
                      'shape':stats['shape_mse'],'opacity':stats['opacity_mse']}[task]
            self.assertAlmostEqual(float(loss.detach()),expected,places=6)
            loss.backward(); self.assertIsNone(y.grad)
            for name in ['channel_encoder','channel_decoder']:
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(m,name).parameters() if p.grad is not None),0,task)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertTrue(all(p.grad is None for p in m.semantic_encoder.parameters()))
            changed_y=y.detach().clone(); changed_hat=recovered.detach().clone()
            changed_y[:,-1]=100; changed_hat[:,-1]=-100
            other,_=c.supervision_loss(m,teacher,changed_y,changed_hat,q,xyz,g,views,color_views,task,.1)
            torch.testing.assert_close(loss.detach(),other.detach(),rtol=0,atol=0)

    def test_two_random_variants_resume_exactly_and_never_load_source_channels(self):
        c=self.entry()
        import test_modular_experiment as fixture
        import test_render_first as renders
        from gaussian_jscc.modular_experiment import main as train_teacher
        from gaussian_jscc.channel_normalization_controls import parser as power_parser,run_arm
        def cameras(*args,**kwargs):
            values=renders.RenderFirstTests().cameras()
            for i,cam in enumerate(values): cam.camera_center=torch.tensor([5.+i,6.,7.])
            if args[-1]=='test':
                for i,cam in enumerate(values): cam.image_name='test_'+str(i)
            return values
        with tempfile.TemporaryDirectory() as temp, \
             patch('gaussian_jscc.rendering.render',side_effect=renders.synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(temp); source=fixture.source_fixture(root)
            opts=fixture.ModularExperimentTests().args(source,root/'teacher')
            opts.channel_variant='progressive'; opts.channel_hidden=16; opts.rates=[8,24,48]
            opts.context_variant='both'; opts.context_centers=[4,2]; opts.context_neighbors=[4,4]
            opts.channel_bootstrap_steps=opts.channel_render_steps=0
            train_teacher(opts)
            power=power_parser().parse_args(['--run',str(root/'teacher'),'--out',str(root/'power'),
                '--steps','1','--device','cpu','--save-every','1','--validate-every','1',
                '--diagnostic-blocks','1','--response-blocks','1','--trials','2','--render-trials','1','--no-lpips'])
            run_arm(power,'block_awgn10')
            parent=root/'power/block_awgn10'; endpoint=parent/'checkpoints/end_channel_bootstrap/codec.pt'
            endpoint.parent.mkdir(); shutil.copyfile(parent/'checkpoints/final/codec.pt',endpoint)
            source_bytes=endpoint.read_bytes(); initials=[]
            for variant in ['color','separated']:
                args=c.parser().parse_args(['--run',str(parent),'--out',str(root/variant),'--variant',variant,
                    '--steps','16','--device','cpu','--validate-every','8','--save-every','1',
                    '--test-views','2','--render-trials','1','--no-lpips'])
                result=c.main(args); self.assertEqual(result['step'],16)
                self.assertEqual(result['supervision_updates']['color'],4)
                full=torch.load(root/variant/'checkpoints/final/codec.pt',weights_only=True)
                self.assertEqual(full['training']['progress']['training_objective']['variant'],variant)
                initial=torch.load(root/variant/'checkpoints/initial/codec.pt',weights_only=True)
                initials.append(initial['state_dict'])
                self.assertEqual(initial['training']['optimizer']['state'],{})
                rows=[json.loads(l) for l in (root/variant/'metrics/loss.jsonl').read_text().splitlines()]
                self.assertEqual(Counter(r['task'] for r in rows),
                    Counter({'response':12,'color':4}) if variant=='color' else Counter({'latent':4,'shape':4,'opacity':4,'color':4}))
                for r in rows:
                    self.assertEqual(r['updates']['semantic_encoder']['update_norm'],0)
                    self.assertEqual(r['updates']['semantic_decoder']['update_norm'],0)
                args.out=str(root/(variant+'_resume')); c.main(args,max_updates=7)
                args.resume=True; c.main(args)
                resumed=torch.load(Path(args.out)/'checkpoints/final/codec.pt',weights_only=True)
                for key,value in full['state_dict'].items(): torch.testing.assert_close(value,resumed['state_dict'][key],rtol=0,atol=0)
                self.assertEqual(full['training']['progress']['supervision_updates'],resumed['training']['progress']['supervision_updates'])
                for key,state in full['training']['optimizer']['state'].items():
                    for field,value in state.items(): torch.testing.assert_close(value,resumed['training']['optimizer']['state'][key][field],rtol=0,atol=0)
                args.variant='separated' if variant=='color' else 'color'
                args.response_weight=None
                with self.assertRaisesRegex(ValueError,'resume settings/source differ'): c.main(args)
            for key,value in initials[0].items(): torch.testing.assert_close(value,initials[1][key],rtol=0,atol=0)
            self.assertEqual(endpoint.read_bytes(),source_bytes)
            # Exercise the REAL parallel shell wrapper and real parsers without
            # CUDA/training. A dry-run must not be mistaken for a completed run.
            bash=Path('C:/softwares/Git/bin/bash.exe') if os.name=='nt' else shutil.which('bash')
            if bash and Path(bash).exists():
                env=dict(os.environ,GPU_COLOR='0',GPU_SEPARATED='1',
                         PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
                suite_out=root/'parallel'
                command=[str(bash),'scripts/test_channel_supervision_parallel.sh',parent.as_posix(),
                         suite_out.as_posix(),'--dry-run','--device','cpu']
                run=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf-8',timeout=40)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                self.assertEqual(json.loads((suite_out/'exit_status.json').read_text()),
                                 {'color_exit_code':0,'separated_exit_code':0})
                self.assertFalse((suite_out/'complete.json').exists())
                self.assertIn('GPU: 0; variant: color',(suite_out/'color.log').read_text())
                self.assertIn('GPU: 1; variant: separated',(suite_out/'separated.log').read_text())
                again=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf-8',timeout=40)
                self.assertNotEqual(again.returncode,0)  # Existing logs must not be overwritten.


if __name__=='__main__': unittest.main()
