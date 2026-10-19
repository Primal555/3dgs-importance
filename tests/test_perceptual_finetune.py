"""Perceptual finetune contracts: real losses/autograd, no CUDA scene training."""
import copy
import importlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class PerceptualObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def objective(self, kind, weight=.01):
        import gaussian_jscc
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.perceptual_objective'),
                             'missing differentiable perceptual training objective')
        from gaussian_jscc.perceptual_objective import PerceptualImageObjective
        return PerceptualImageObjective(kind, weight, 'cpu')

    def test_ssim_weighted_value_and_gradient_not_just_mse(self):
        from utils.loss_utils import ssim
        objective = self.objective('mse_ssim', .02)
        image = torch.linspace(.1, .9, 3*16*16).reshape(3,16,16).requires_grad_()
        target = torch.full_like(image, .4, requires_grad=True)
        loss, terms = objective(image, target)
        mse = (image-target.detach()).square().mean()
        extra = 1-ssim(image, target.detach())
        torch.testing.assert_close(loss, mse+.02*extra)
        torch.testing.assert_close(terms['image_aux_weighted'], .02*extra)
        mse_grad = torch.autograd.grad(mse, image, retain_graph=True)[0]
        loss.backward()
        self.assertGreater(float((image.grad-mse_grad).abs().sum()), 0)
        self.assertIsNone(target.grad)

    def test_paper_relative_weights_are_resolved_and_provenance_is_recorded(self):
        ssim_objective = self.objective('mse_ssim', None)
        # Weight selection needs no downloaded feature extractor; its actual
        # gradient/input contract is exercised by the cached-pretrained test.
        with patch('gaussian_jscc.perceptual_metrics._load_lpips_model',return_value=torch.nn.Identity()):
            lpips_objective = self.objective('mse_lpips', None)
        self.assertAlmostEqual(ssim_objective.weight, 1/3)
        self.assertEqual(lpips_objective.weight, 1.)
        self.assertIn('2510.20067',ssim_objective.protocol()['weight_reference']['url'])
        self.assertIn('2211.13772',lpips_objective.protocol()['weight_reference']['url'])

    def test_mse_stays_unclipped_when_structural_branch_clips_rgb(self):
        objective = self.objective('mse_ssim')
        image = torch.full((3,16,16), 2., requires_grad=True)
        target = torch.full_like(image, .5)
        loss, terms = objective(image, target)
        self.assertEqual(float(terms['image_mse'].detach()), 2.25)
        loss.backward()
        torch.testing.assert_close(image.grad, torch.full_like(image, 3/image.numel()))

    def test_bad_weights_and_inputs_rejected(self):
        for kind, weight in [('mse_ssim',0), ('mse_lpips',-1), ('mse_ssim',float('nan')),
                             ('mse_ssim',float('inf')), ('unknown',.01)]:
            with self.assertRaises(ValueError): self.objective(kind, weight)
        objective = self.objective('mse_ssim')
        for image,target in [(torch.zeros(1,8,8),torch.zeros(1,8,8)),
                             (torch.zeros(3,8,8),torch.zeros(3,4,4)),
                             (torch.full((3,8,8),float('nan')),torch.zeros(3,8,8))]:
            with self.assertRaises(ValueError): objective(image,target)

    @unittest.skipUnless(all((Path(torch.hub.get_dir())/'checkpoints'/name).exists()
                             for name in ('vgg16-397923af.pth','vgg.pth')),
                         'requires cached pretrained VGG; no test-time download')
    def test_real_lpips_adds_input_gradient_but_never_trains_vgg(self):
        objective = self.objective('mse_lpips', .01)
        image = torch.linspace(.1,.9,3*32*32).reshape(3,32,32).requires_grad_()
        target = image.detach().flip(-1).clone().requires_grad_()
        loss,terms = objective(image,target)
        from gaussian_jscc.perceptual_metrics import LPIPSEvaluator
        metric = LPIPSEvaluator('cpu')
        self.assertAlmostEqual(float(terms['image_lpips_loss'].detach()), metric(image,target), places=6)
        gradient = torch.autograd.grad(terms['image_lpips_loss'], image, retain_graph=True)[0]
        self.assertGreater(float(gradient.norm()),0)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in objective.parameters()))
        self.assertFalse(objective.lpips_model.training)

    @unittest.skipUnless(all((Path(torch.hub.get_dir())/'checkpoints'/name).exists()
                             for name in ('vgg16-397923af.pth','vgg.pth')),
                         'requires cached pretrained VGG; no test-time download')
    def test_real_lpips_streamed_replay_matches_direct_codec_gradient(self):
        objective = self.objective('mse_lpips', 1.)
        from gaussian_jscc.perceptual_objective import PerceptualMultiViewRenderTask
        from gaussian_jscc.training import full_scene_step
        from test_learned_joint import setup
        from test_render_first import synthetic_render, RenderFirstTests
        class Reference:
            def get(self, camera, device): return torch.full((3,32,32),.6,device=device)
        def render(scene,camera,*args):
            return torch.nn.functional.interpolate(synthetic_render(scene,camera)[None],size=(32,32),mode='nearest')[0]
        _,geometry,f,model = setup(16)
        other = copy.deepcopy(model)
        batches = [(f[None],torch.ones(1,16,dtype=torch.long))]
        losses = []
        with patch('gaussian_jscc.rendering.render',side_effect=render):
            for mode,codec in [('replay',model),('checkpoint',other)]:
                torch.manual_seed(125)
                task = PerceptualMultiViewRenderTask(RenderFirstTests().cameras(),Reference(),0,False,objective)
                loss,_ = full_scene_step(codec,batches,geometry,10,'awgn',task,attr_weight=0,mode=mode)
                losses.append(loss)
                self.assertGreater(task.stats['image_lpips_loss'],0)
        torch.testing.assert_close(*losses)
        for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4,msg=name)

    def test_replay_multiview_gradient_matches_direct_graph_and_logs_actual_mse(self):
        objective = self.objective('mse_ssim', .02)
        from gaussian_jscc.perceptual_objective import PerceptualMultiViewRenderTask
        from gaussian_jscc.training import full_scene_step
        from test_learned_joint import setup
        from test_render_first import synthetic_render, RenderFirstTests, Reference
        _,geometry,f,model = setup(16)
        other = copy.deepcopy(model)
        batches = [(f[None], torch.ones(1,16,dtype=torch.long))]
        values = []
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            for mode, codec in [('replay',model),('checkpoint',other)]:
                torch.manual_seed(125)
                task = PerceptualMultiViewRenderTask(RenderFirstTests().cameras(),Reference(),0,False,objective)
                loss,stats = full_scene_step(codec,batches,geometry,10,'awgn',task,attr_weight=0,mode=mode)
                values.append(loss)
                self.assertAlmostEqual(task.stats['image_mse'],sum(task.stats['view_mse'])/2)
                self.assertAlmostEqual(float(loss), task.stats['image_mse']+task.stats['image_aux_weighted'],places=6)
                self.assertEqual(task.stats['image_objective'],'mse_ssim')
        torch.testing.assert_close(*values)
        for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4,msg=name)


class PerceptualFinetuneTests(unittest.TestCase):
    def args(self, run, out):
        from gaussian_jscc.modular_allocation import parser
        return parser().parse_args(['--run',str(run),'--out',str(out),'--fixed-deployment',
            '--device','cpu','--allocation-steps','0','--joint-steps','4','--beta','.005',
            '--prefix-anchor-every','2','--validate-every','2','--save-every','2',
            '--trials','1','--test-views','1','--image-views','1','--no-lpips',
            '--image-objective','mse_ssim','--image-aux-weight','.02'])

    def test_new_objectives_require_fixed_map_without_new_allocation(self):
        from gaussian_jscc.modular_allocation import parser,main
        args = parser().parse_args(['--run','missing','--out','unused','--image-objective','mse_ssim'])
        with self.assertRaisesRegex(ValueError,'fixed-deployment'): main(args)

    def test_finetune_anchor_uses_same_loss_and_resume_preserves_settings_and_weights(self):
        from test_fixed_deployment_finetune import FixedDeploymentTests
        from test_render_first import RenderFirstTests, synthetic_render
        from gaussian_jscc.modular_allocation import main
        from gaussian_jscc.transport import load_checkpoint
        def render(raw,camera,degree,white=False,existence=None,colors=None):
            if colors is not None: return colors.mean(0)[:,None,None].expand(3,8,8)
            return synthetic_render(raw,camera)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, \
             patch('gaussian_jscc.rendering.load_cameras',return_value=RenderFirstTests().cameras()), \
             patch('gaussian_jscc.rendering.render',side_effect=render):
            root = Path(directory)
            run,q = FixedDeploymentTests().source(root)
            main(self.args(run,root/'full'))
            main(self.args(run,root/'resumed'),max_updates=2)
            args = self.args(run,root/'resumed'); args.resume=True; main(args)
            a = load_checkpoint(root/'full/checkpoints/final/codec.pt','cpu')
            b = load_checkpoint(root/'resumed/checkpoints/final/codec.pt','cpu')
            for name,value in a.state_dict().items():
                torch.testing.assert_close(value,b.state_dict()[name],atol=0,rtol=0)
            rows = [json.loads(s) for s in (root/'full/metrics/loss.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['image_objective']=='mse_ssim' for r in rows))
            for row in rows:
                self.assertAlmostEqual(row['render_loss'],row['image_mse']+.02*row['image_ssim_loss'],places=6)
                if row['prefix_anchor_applied']:
                    self.assertEqual(row['prefix_anchor_image_objective'],'mse_ssim')
                    self.assertAlmostEqual(row['optimization_image_loss'],
                        (row['render_loss']+row['prefix_anchor_render_loss'])/2,places=6)
            cfg = json.loads((root/'full/training.json').read_text())
            self.assertEqual(cfg['render_objective']['kind'],'mse_ssim')
            changed = self.args(run,root/'resumed'); changed.resume=True; changed.image_aux_weight=.03
            with self.assertRaisesRegex(ValueError,'mismatch'): main(changed)


class PerceptualLauncherTests(unittest.TestCase):
    def test_launcher_runs_only_two_arms_from_same_source_and_accepts_single_arm(self):
        git_bash = Path('C:/softwares/Git/bin/bash.exe')
        bash = str(git_bash) if git_bash.is_file() else shutil.which('bash')
        if not bash: self.skipTest('Bash unavailable')
        if not Path(bash).is_file(): self.skipTest('Git Bash unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'source'; (run/'checkpoints/final').mkdir(parents=True)
            (run/'deployment/final').mkdir(parents=True)
            for name in ('complete.json','training.json','checkpoints/final/codec.pt','deployment/final/tiers.npy'):
                (run/name).touch()
            capture = root/'calls'
            env = dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN='capture_python',
                       CAPTURE=capture.as_posix(),LOSS='both')
            # Replace only the expensive Python training process; run the real launcher.
            command = ('capture_python() { printf "%s\\0" "$@" >> "$CAPTURE"; '
                       'printf "\\n" >> "$CAPTURE"; }; export -f capture_python; '
                       'bash scripts/test_perceptual_finetune.sh "$1" "$2" --dry-run')
            result = subprocess.run([bash,'-c',command,'test',run.as_posix(),(root/'out').as_posix()],
                                    env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls = [line.split('\0')[:-1] for line in capture.read_text().splitlines()]
            self.assertEqual(len(calls),2)
            for args,kind,weight in zip(calls,('mse_ssim','mse_lpips'),(1/3,1.)):
                self.assertEqual(args[args.index('--run')+1],run.as_posix())
                self.assertEqual(args[args.index('--image-objective')+1],kind)
                self.assertAlmostEqual(float(args[args.index('--image-aux-weight')+1]),weight)
                self.assertIn('--fixed-deployment',args)
                self.assertEqual(args[args.index('--allocation-steps')+1],'0')
            env['LOSS']='mse_lpips';env['STEPS']='12';capture.unlink()
            result = subprocess.run([bash,'-c',command,'test',run.as_posix(),(root/'out2').as_posix()],
                                    env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls = capture.read_text().splitlines()
            self.assertEqual(len(calls),1)
            self.assertIn('mse_lpips',calls[0])
            self.assertIn('--joint-steps\x0012\x00',calls[0])


if __name__ == '__main__': unittest.main()
