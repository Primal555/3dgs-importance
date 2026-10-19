"""Shared bootstrap -> independent perceptual representation/channel training."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class BootstrapInitializerTests(unittest.TestCase):
    def test_only_completed_bootstrap_is_accepted_and_channel_starts_fresh(self):
        from gaussian_jscc import modular_checkpoint as checkpoints
        from test_modular_allocation import small_model
        self.assertTrue(hasattr(checkpoints,'initialize_bootstrap_run'),
                        'need an initializer BEFORE semantic render, not the end-semantic-render initializer')
        torch.set_num_threads(1)
        _,_,_,teacher = small_model()
        with torch.no_grad():
            for p in teacher.parameters(): p.fill_(.125)
        progress = dict(phase='semantic_bootstrap',phase_complete=True,step=4,channel_trained=False,
                        completed=dict(semantic_bootstrap=4,semantic_render=0,channel_bootstrap=0,channel_render=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bootstrap.pt'
            checkpoints.save_training_checkpoint(path,teacher,None,progress,checkpoints.capture_rng(),{'fingerprint':'abc'})
            torch.manual_seed(17); a,source,scene = checkpoints.initialize_bootstrap_run(path,'cpu')
            torch.manual_seed(17); b,_,_ = checkpoints.initialize_bootstrap_run(path,'cpu')
            self.assertEqual(source['step'],4); self.assertEqual(scene['fingerprint'],'abc')
            for name,value in a.state_dict().items():
                torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
                if name.startswith(('semantic_encoder.','semantic_decoder.')) or name in ('attr_mean','attr_std'):
                    torch.testing.assert_close(value,teacher.state_dict()[name],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(value,teacher.state_dict()[name]) for name,value in a.state_dict().items()
                                if name.startswith('channel_encoder.')))
            for change in (dict(phase_complete=False),dict(channel_trained=True),
                           dict(completed=dict(progress['completed'],semantic_render=1))):
                checkpoints.save_training_checkpoint(path,teacher,None,dict(progress,**change),checkpoints.capture_rng(),scene)
                with self.assertRaisesRegex(ValueError,'semantic_bootstrap'):
                    checkpoints.initialize_bootstrap_run(path,'cpu')


class ParallelTrainingLauncherTests(unittest.TestCase):
    def test_both_arms_are_launched_concurrently_and_have_separate_logs_and_pids(self):
        from gaussian_jscc import perceptual_training_controls as suite
        self.assertTrue(hasattr(suite,'run_jobs'),'same-GPU arms must launch before waiting for either one')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            # Real subprocess rendezvous: either sequential launch would time out.
            code=('import pathlib,sys,time\n'
                  'own,other=map(pathlib.Path,sys.argv[1:])\n'
                  'own.touch()\n'
                  'end=time.monotonic()+5\n'
                  'while not other.exists() and time.monotonic()<end: time.sleep(.02)\n'
                  'print(own.name,flush=True)\n'
                  'raise SystemExit(0 if other.exists() else 7)\n')
            jobs=[dict(name=name,command=[sys.executable,'-c',code,str(root/name),str(root/other)])
                  for name,other in [('ssim','lpips'),('lpips','ssim')]]
            suite.run_jobs(jobs,root,resume=False,parallel=True)
            state=json.loads((root/'processes.json').read_text())
            self.assertEqual({r['returncode'] for r in state.values()},{0})
            self.assertNotEqual(state['ssim']['pid'],state['lpips']['pid'])
            self.assertEqual((root/'ssim.log').read_text().strip(),'ssim')
            self.assertEqual((root/'lpips.log').read_text().strip(),'lpips')

    def test_failed_arm_does_not_abort_healthy_arm(self):
        from gaussian_jscc.perceptual_training_controls import run_jobs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);marker=root/'healthy_finished'
            jobs=[dict(name='bad',command=[sys.executable,'-c','raise SystemExit(7)']),
                  dict(name='healthy',command=[sys.executable,'-c',
                    'import pathlib,sys,time;time.sleep(.15);pathlib.Path(sys.argv[1]).touch()',str(marker)])]
            with self.assertRaisesRegex(RuntimeError,'Failed training arms'):run_jobs(jobs,root)
            self.assertTrue(marker.exists())
            states=json.loads((root/'processes.json').read_text())
            self.assertEqual(states['healthy']['returncode'],0)
            self.assertEqual(states['bad']['returncode'],7)

    def test_second_spawn_exception_still_waits_for_first_child(self):
        from gaussian_jscc import perceptual_training_controls as suite
        import subprocess
        children=[]; real_popen=subprocess.Popen
        def launch(*args,**kwargs):
            child=real_popen(*args,**kwargs);children.append(child);return child
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);marker=root/'healthy_finished'
            jobs=[dict(name='healthy',command=[sys.executable,'-c',
                    'import pathlib,sys,time;time.sleep(.15);pathlib.Path(sys.argv[1]).touch()',str(marker)]),
                  dict(name='missing',command=[str(root/'nonexistent_executable')])]
            try:
                with patch('gaussian_jscc.perceptual_training_controls.subprocess.Popen',side_effect=launch):
                    with self.assertRaisesRegex(RuntimeError,'Failed training arms'):suite.run_jobs(jobs,root)
                self.assertTrue(marker.exists())
                states=json.loads((root/'processes.json').read_text())
                self.assertEqual(states['healthy']['returncode'],0)
                self.assertEqual(states['missing']['status'],'launch_failed')
            finally:
                for child in children:child.wait(timeout=5)

    def test_pid_publication_failure_does_not_abandon_started_children(self):
        from gaussian_jscc import perceptual_training_controls as suite
        import subprocess
        children=[]; real_popen=subprocess.Popen; real_write=Path.write_text
        def launch(*args,**kwargs):
            child=real_popen(*args,**kwargs);children.append(child);return child
        def write(path,*args,**kwargs):
            if path.name=='bad.pid':raise OSError('injected PID write failure')
            return real_write(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            jobs=[dict(name=name,command=[sys.executable,'-c',
                    'import pathlib,sys,time;time.sleep(.15);pathlib.Path(sys.argv[1]).touch()',str(root/(name+'_finished'))])
                  for name in ('healthy','bad')]
            try:
                with patch('gaussian_jscc.perceptual_training_controls.subprocess.Popen',side_effect=launch),patch.object(Path,'write_text',write):
                    with self.assertRaisesRegex(RuntimeError,'Failed training arms'):suite.run_jobs(jobs,root)
                self.assertTrue((root/'healthy_finished').exists())
                self.assertTrue((root/'bad_finished').exists())
                states=json.loads((root/'processes.json').read_text())
                self.assertEqual(states['healthy']['returncode'],0)
                self.assertEqual(states['bad']['returncode'],0)
                self.assertIn('PID write failure',states['bad']['supervision_error'])
            finally:
                for child in children:child.wait(timeout=5)


class PerceptualTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def source(self,root):
        from test_fixed_deployment_finetune import FixedDeploymentTests
        # Full metadata contains the current progressive/full-prefix refiners.
        return FixedDeploymentTests().source(root)[0]

    def cameras(self):
        from test_render_first import RenderFirstTests
        cameras=RenderFirstTests().cameras()
        for camera in cameras:camera.camera_center=torch.tensor([0.,0.,3.])
        return cameras

    def options(self,source,out,bootstrap,kind='mse_ssim'):
        from gaussian_jscc.modular_experiment import parser
        return parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu',
            '--use-source-config','--init-bootstrap',str(bootstrap),'--image-objective',kind,
            '--semantic-bootstrap-steps','0','--semantic-render-steps','2',
            '--channel-bootstrap-steps','2','--channel-render-steps','2',
            '--validate-every','2','--save-every','2','--train-snrs','10',
            '--semantic-decoder-lr','1e-5','--no-lpips'])

    def bootstrap(self,run,root):
        from gaussian_jscc.modular_experiment import parser,main
        opts=parser().parse_args(['--run',str(run),'--out',str(root/'bootstrap'),'--device','cpu',
            '--use-source-config','--clean-only','--semantic-bootstrap-steps','2','--semantic-render-steps','0',
            '--channel-bootstrap-steps','0','--channel-render-steps','0','--validate-every','2','--save-every','2',
            '--semantic-color-every','2','--semantic-color-directions','2','--no-lpips'])
        main(opts)
        return root/'bootstrap/checkpoints/end_semantic_bootstrap/codec.pt'

    def renderer(self,raw,camera,degree,white=False,existence=None,colors=None):
        from test_render_first import synthetic_render
        if colors is not None: return colors.mean(0)[:,None,None].expand(3,8,8)
        return synthetic_render(raw,camera)

    def test_semantic_and_channel_render_use_new_loss_but_bootstrap_keeps_latent_target(self):
        from gaussian_jscc.modular_experiment import main
        from test_render_first import RenderFirstTests
        with tempfile.TemporaryDirectory() as directory, \
             patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()), \
             patch('gaussian_jscc.rendering.render',side_effect=self.renderer), \
             patch('gaussian_jscc.modular_plots.plot_run'):
            root=Path(directory); run=self.source(root); bootstrap=self.bootstrap(run,root)
            out=root/'ssim'; main(self.options(run,out,bootstrap))
            cfg=json.loads((out/'training.json').read_text())
            self.assertEqual(cfg['initializer_kind'],'semantic_bootstrap')
            self.assertEqual(cfg['phase_budgets'],dict(semantic_bootstrap=0,semantic_render=2,channel_bootstrap=2,channel_render=2))
            self.assertEqual(cfg['codec_config']['rates'],[0,8,24,48])
            self.assertEqual(cfg['codec_config']['channel_prefix_mode'],'full_prefix')
            self.assertEqual(cfg['codec_config']['channel_denoiser_mode'],'pointwise')
            rows=[json.loads(line) for line in (out/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),6)
            for row in rows:
                phase=row['phase']
                if phase=='channel_bootstrap':
                    self.assertEqual(row['objective'],'latent_mse')
                    self.assertNotIn('image_ssim_loss',row)
                else:
                    self.assertEqual(row['image_objective'],'mse_ssim')
                    self.assertAlmostEqual(row['render_loss'],row['image_mse']+row['image_aux_weighted'],places=6)
                    self.assertAlmostEqual(row['loss'],row['render_loss']+
                        (row['weighted_latent_loss'] if phase=='channel_render' else 0),places=6)
                    self.assertIn('mse_ssim',row['objective'])
            self.assertTrue(all(row['updates']['semantic_encoder']['update_norm']==0 for row in rows if row['phase'].startswith('channel')))
            self.assertFalse((out/'deployment').exists())

    def test_new_loss_fork_exact_resume_and_changed_weight_rejection(self):
        from gaussian_jscc.modular_experiment import main
        from gaussian_jscc.transport import load_checkpoint
        from test_render_first import RenderFirstTests
        with tempfile.TemporaryDirectory() as directory, \
             patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()), \
             patch('gaussian_jscc.rendering.render',side_effect=self.renderer), \
             patch('gaussian_jscc.modular_plots.plot_run'):
            root=Path(directory); run=self.source(root); bootstrap=self.bootstrap(run,root)
            main(self.options(run,root/'full',bootstrap))
            main(self.options(run,root/'resumed',bootstrap),max_updates=3)
            opts=self.options(run,root/'resumed',bootstrap)
            opts.init_bootstrap=None; opts.resume=str(root/'resumed/checkpoints/latest/codec.pt')
            bad=copy.deepcopy(opts);bad.image_aux_weight=.02
            with self.assertRaisesRegex(ValueError,'objective'):main(bad)
            main(opts)
            a=load_checkpoint(root/'full/checkpoints/final/codec.pt','cpu')
            b=load_checkpoint(root/'resumed/checkpoints/final/codec.pt','cpu')
            for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
            cfg=json.loads((root/'resumed/training.json').read_text())
            self.assertEqual(cfg['phase_budgets']['semantic_render'],2)

    def test_suite_plans_only_two_arms_with_same_bootstrap_and_full_template(self):
        from test_render_first import RenderFirstTests
        with tempfile.TemporaryDirectory() as directory, \
             patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()), \
             patch('gaussian_jscc.rendering.render',side_effect=self.renderer), \
             patch('gaussian_jscc.modular_plots.plot_run'):
            root=Path(directory); run=self.source(root); bootstrap=self.bootstrap(run,root)
            from gaussian_jscc.perceptual_training_controls import parser,build_plan
            opts=parser().parse_args(['--run',str(run),'--bootstrap',str(bootstrap),'--out',str(root/'suite')])
            plan=build_plan(opts)
            self.assertEqual(len(plan['jobs']),2)
            self.assertEqual([j['name'] for j in plan['jobs']],['mse_ssim','mse_lpips'])
            for job in plan['jobs']:
                command=job['command']
                self.assertEqual(command[command.index('--init-bootstrap')+1],str(bootstrap.resolve()))
                self.assertIn('--use-source-config',command)
                self.assertEqual(command[command.index('--semantic-render-steps')+1],'1000')
                self.assertEqual(command[command.index('--channel-bootstrap-steps')+1],'2000')
                self.assertEqual(command[command.index('--channel-render-steps')+1],'2000')
                self.assertNotIn('--fixed-deployment',command)
            self.assertEqual(plan['codec_config']['rates'],[0,8,24,48])
            # Disk round-trip must not invalidate the original suite settings.
            from gaussian_jscc.perceptual_training_controls import main as suite_main
            out=Path(opts.out);out.mkdir()
            (out/'suite.json').write_text(json.dumps(plan))
            for job in plan['jobs']:
                (out/job['name']).mkdir();(out/job['name']/'complete.json').write_text('{}')
            opts.resume_suite=True
            with patch('gaussian_jscc.perceptual_training_controls.run_jobs',side_effect=AssertionError('completed arms must not rerun')):
                suite_main(opts)
            self.assertEqual(json.loads((out/'complete.json').read_text())['completed'],['mse_ssim','mse_lpips'])

    @unittest.skipUnless(all((Path(torch.hub.get_dir())/'checkpoints'/name).exists()
                             for name in ('vgg16-397923af.pth','vgg.pth')),
                         'real LPIPS path requires cached pretrained weights; never download in tests')
    def test_real_lpips_trains_both_render_stages_without_changing_channel_bootstrap(self):
        from gaussian_jscc.modular_experiment import main
        cameras=self.cameras()
        for camera in cameras:
            camera.original_image=torch.nn.functional.interpolate(camera.original_image[None],size=(32,32),mode='nearest')[0]
        def render(*args,**kwargs):
            return torch.nn.functional.interpolate(self.renderer(*args,**kwargs)[None],size=(32,32),mode='nearest')[0]
        with tempfile.TemporaryDirectory() as directory, \
             patch('gaussian_jscc.rendering.load_cameras',return_value=cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=render), \
             patch('gaussian_jscc.modular_plots.plot_run'):
            root=Path(directory);run=self.source(root);bootstrap=self.bootstrap(run,root)
            out=root/'lpips';opts=self.options(run,out,bootstrap,'mse_lpips');opts.test_snrs=[10.]
            main(opts)
            rows=[json.loads(line) for line in (out/'metrics/loss.jsonl').read_text().splitlines()]
            for row in rows:
                if row['phase'].endswith('render'):
                    self.assertEqual(row['image_objective'],'mse_lpips')
                    self.assertGreater(row['image_lpips_loss'],0)
                    self.assertAlmostEqual(row['image_aux_weighted'],row['image_lpips_loss'],places=7)
                else:self.assertEqual(row['objective'],'latent_mse')
            tests=[json.loads(line) for line in (out/'metrics/test.jsonl').read_text().splitlines()]
            self.assertEqual({r['actual_snr'] for r in tests if r['path']=='awgn'},{10.})


if __name__ == '__main__': unittest.main()
