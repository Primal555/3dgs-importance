"""GAN gradient isolation, replay, checkpoint continuation and independent entry."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class _PixelDistance(torch.nn.Module):
    def forward(self, a, b):
        return (a-b).square().mean().reshape(1, 1, 1, 1)


class AdversarialObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def objective(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.adversarial_objective'),
                             'independent differentiable GAN objective is missing')
        from gaussian_jscc.adversarial_objective import AdversarialImageObjective
        # Only the pretrained external feature extractor is replaced in these
        # unit tests; real VGG integration is exercised separately below.
        with patch('gaussian_jscc.perceptual_objective.LPIPSEvaluator') as evaluator:
            evaluator.return_value.model = _PixelDistance()
            return AdversarialImageObjective(1., kwargs.pop('gan_weight', .001), 'cpu', **kwargs)

    def test_generator_has_input_gradients_but_cannot_change_discriminator(self):
        torch.manual_seed(43)
        rng = torch.get_rng_state().clone()
        objective = self.objective()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        before = copy.deepcopy(objective.discriminator.state_dict())
        image = torch.full((3, 32, 32), .4, requires_grad=True)
        target = torch.full_like(image, .7)
        objective.begin_generator()
        loss, terms = objective(image, target)
        self.assertAlmostEqual(float(loss.detach()), float((terms['image_mse'] + terms['image_lpips_weighted']
                                                   + terms['image_gan_weighted']).detach()), places=7)
        # Isolate the GAN gradient, not the already existing MSE/LPIPS path.
        gradient = torch.autograd.grad(terms['image_gan_weighted'], image, retain_graph=True)[0]
        self.assertGreater(float(gradient.norm()), 0)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in objective.discriminator.parameters()))
        for key, value in before.items():
            torch.testing.assert_close(value, objective.discriminator.state_dict()[key], rtol=0, atol=0)
        stats = objective.update_discriminator([(image.detach(), target.detach())])
        self.assertGreater(stats['discriminator_grad_norm'], 0)
        self.assertGreater(stats['discriminator_update_norm'], 0)
        self.assertEqual(stats['discriminator_updates'], 1)
        self.assertTrue(any(not torch.equal(value, objective.discriminator.state_dict()[key])
                            for key, value in before.items()))

    def test_zero_gan_weight_matches_lpips_and_rejects_invalid_protocol(self):
        objective = self.objective(gan_weight=0)
        image = torch.full((3, 32, 32), .4, requires_grad=True)
        target = torch.full_like(image, .7)
        loss, terms = objective(image, target)
        self.assertAlmostEqual(float(loss.detach()), .45, places=6)  # .09 MSE + .36 feature distance
        self.assertEqual(float(terms['image_gan_weighted'].detach()), 0)
        for kwargs in ({'gan_weight':-1}, {'gan_weight':float('nan')}, {'discriminator_lr':0}):
            with self.assertRaises(ValueError): self.objective(**kwargs)
        with self.assertRaisesRegex(ValueError, '32'):
            objective(image[:, :16, :16], target[:, :16, :16])
        with self.assertRaisesRegex((ValueError, RuntimeError), 'finite'):
            objective.update_discriminator([(image.detach()*float('nan'), target)])
        self.assertEqual(objective.updates, 0)

    def test_spectral_normalization_is_initialized_before_first_generator_step(self):
        torch.manual_seed(42)
        objective=self.objective()
        # Without initial power iteration the random u/v singular estimate can
        # be near zero. Initial eval-only G then sees a catastrophically scaled D.
        for layer in objective.discriminator.modules():
            if hasattr(layer,'weight_orig'):
                matrix=layer.weight_orig.detach().flatten(1)
                estimate=layer.weight_u @ matrix @ layer.weight_v
                self.assertGreater(float(estimate/matrix.norm()), .05)
        fake=torch.full((3,32,32),.4,requires_grad=True)
        _,terms=objective(fake,torch.full_like(fake,.7))
        self.assertLess(float(terms['image_gan_loss'].detach()),10)

    def test_streamed_scene_gradient_matches_direct_and_d_update_is_separate(self):
        from gaussian_jscc.adversarial_objective import AdversarialMultiViewRenderTask
        objective = self.objective()
        base = torch.linspace(.2, .7, 12).reshape(4, 3)
        targets = {0:torch.full((3, 32, 32), .6), 1:torch.full((3, 32, 32), .3)}
        class Reference:
            def get(self, camera, device): return targets[camera].to(device)
        def renderer(scene, camera, degree, white=False):
            return (scene.mean(0)*(.8 + camera*.1))[:, None, None].expand(3, 32, 32)
        with patch('gaussian_jscc.rendering.render', side_effect=renderer):
            scene = base.clone().requires_grad_()
            task = AdversarialMultiViewRenderTask([0, 1], Reference(), 0, False, objective)
            loss = task(scene)
            direct = torch.autograd.grad(loss, scene)[0]
            streamed = base.clone().requires_grad_()
            replay = AdversarialMultiViewRenderTask([0, 1], Reference(), 0, False, objective)
            replay_loss = replay.backward_scene(streamed)
            torch.testing.assert_close(replay_loss, loss.detach())
            torch.testing.assert_close(streamed.grad, direct, rtol=1e-5, atol=1e-7)
            self.assertEqual(objective.updates, 0)
            self.assertEqual(len(replay.pairs), 2)
            self.assertTrue(all(not fake.requires_grad and fake.device.type=='cpu'
                                for fake, real in replay.pairs))
            gradient = streamed.grad.clone()
            stats = replay.finish_discriminator()
            self.assertEqual(stats['discriminator_updates'], 1)
            self.assertEqual(len(replay.pairs), 0)
            torch.testing.assert_close(streamed.grad, gradient, rtol=0, atol=0)

    def test_discriminator_optimizer_resume_is_exact_and_missing_state_is_rejected(self):
        objective = self.objective()
        pairs = [(torch.full((3, 32, 32), .4), torch.full((3, 32, 32), .7))]
        objective.update_discriminator(pairs)
        saved = copy.deepcopy(objective.training_state())
        objective.update_discriminator(pairs)
        restored = self.objective()
        restored.restore_training_state(saved)
        restored.update_discriminator(pairs)
        self.assertEqual(restored.updates, 2)
        for key, value in objective.discriminator.state_dict().items():
            torch.testing.assert_close(value, restored.discriminator.state_dict()[key], rtol=0, atol=0)
        for bad in (None, {}):
            with self.assertRaisesRegex(ValueError, 'adversarial'):
                restored.restore_training_state(bad)
        changed = self.objective(gan_weight=.1)
        with self.assertRaisesRegex(ValueError, 'protocol'):
            changed.restore_training_state(saved)


class AdversarialPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_independent_plan_does_not_launch_other_loss_arms(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.adversarial_training_controls'),
                             'independent GAN entry is missing')
        from gaussian_jscc.adversarial_training_controls import parser, build_plan
        from test_perceptual_training import PerceptualTrainingTests
        helper = PerceptualTrainingTests()
        with tempfile.TemporaryDirectory() as directory, \
             patch('gaussian_jscc.rendering.load_cameras', return_value=helper.cameras()), \
             patch('gaussian_jscc.rendering.render', side_effect=helper.renderer), \
             patch('gaussian_jscc.modular_plots.plot_run'):
            root=Path(directory); run=helper.source(root); bootstrap=helper.bootstrap(run, root)
            options=parser().parse_args(['--run',str(run),'--out',str(root/'gan'),'--bootstrap',str(bootstrap)])
            plan=build_plan(options)
            self.assertEqual(len(plan['jobs']),1)
            command=plan['jobs'][0]['command']
            self.assertEqual(command[command.index('--out')+1],str((root/'gan').resolve()))
            self.assertEqual(command[command.index('--image-objective')+1],'mse_gan')
            self.assertEqual(command[command.index('--semantic-render-steps')+1],'1000')
            self.assertEqual(command[command.index('--channel-bootstrap-steps')+1],'2000')
            self.assertEqual(command[command.index('--channel-render-steps')+1],'2000')
            self.assertEqual(command[command.index('--gan-weight')+1],'0.001')

    def test_saved_curves_separate_gan_terms_and_discriminator_diagnostics(self):
        from gaussian_jscc.modular_plots import plot_run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'metrics').mkdir()
            row=dict(step=1,phase_step=1,phase='semantic_render',loss=.02,step_seconds=1.,
                     image_mse=.01,image_lpips_loss=.005,image_lpips_weighted=.005,
                     image_gan_loss=.7,image_gan_weighted=.0007,discriminator_loss=.69,
                     discriminator_real_logit=.1,discriminator_fake_logit=-.1,
                     discriminator_grad_norm=.5,discriminator_update_norm=.01,
                     generator_seconds=.8,discriminator_seconds=.2,updates={},gradient_groups={})
            (root/'metrics/loss.jsonl').write_text(json.dumps(row)+'\n')
            paths=plot_run(root)
            self.assertIn('adversarial_diagnostics.png',[p.name for p in paths])
            self.assertTrue(all(p.stat().st_size>1000 for p in paths))

    @unittest.skipUnless(all((Path(torch.hub.get_dir())/'checkpoints'/name).exists()
                             for name in ('vgg16-397923af.pth','vgg.pth')),
                         'real LPIPS requires cached pretrained weights')
    def test_real_training_saves_both_optimizers_and_exactly_resumes_across_phases(self):
        from gaussian_jscc.modular_experiment import parser, main
        from test_perceptual_training import PerceptualTrainingTests
        helper=PerceptualTrainingTests(); cameras=helper.cameras()
        for camera in cameras:
            camera.original_image=torch.nn.functional.interpolate(camera.original_image[None],size=(32,32))[0]
        def render(*args,**kwargs):
            return torch.nn.functional.interpolate(helper.renderer(*args,**kwargs)[None],size=(32,32))[0]
        def options(run,out,bootstrap):
            args=helper.options(run,out,bootstrap,'mse_lpips')
            args.image_objective='mse_gan';args.test_snrs=[10.]
            return args
        with tempfile.TemporaryDirectory() as directory, \
             patch('gaussian_jscc.rendering.load_cameras',return_value=cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=render), \
             patch('gaussian_jscc.modular_plots.plot_run'):
            # CLI must accept this as an opt-in third arm, never the default.
            accepted=parser().parse_args(['--run','r','--out','o','--image-objective','mse_gan'])
            self.assertEqual(accepted.image_objective,'mse_gan')
            root=Path(directory);run=helper.source(root);bootstrap=helper.bootstrap(run,root)
            # Exercise the independent public entry itself, not only its plan.
            from gaussian_jscc.adversarial_training_controls import parser as entry_parser, main as entry_main
            entry_args=entry_parser().parse_args(['--run',str(run),'--out',str(root/'entry'),
                '--bootstrap',str(bootstrap),'--device','cpu','--semantic-render-steps','2',
                '--channel-bootstrap-steps','2','--channel-render-steps','2'])
            entry_main(entry_args,max_updates=1)
            entry=torch.load(root/'entry/checkpoints/latest/codec.pt',map_location='cpu',weights_only=True)
            self.assertEqual(entry['training']['adversarial']['updates'],1)
            self.assertEqual(entry['training']['progress']['completed']['semantic_render'],1)
            main(options(run,root/'full',bootstrap))
            main(options(run,root/'resume',bootstrap),max_updates=3)
            resumed=options(run,root/'resume',bootstrap)
            resumed.init_bootstrap=None;resumed.resume=str(root/'resume/checkpoints/latest/codec.pt')
            bad=copy.deepcopy(resumed);bad.gan_weight=.1
            with self.assertRaisesRegex(ValueError,'objective'): main(bad)
            main(resumed)
            full=torch.load(root/'full/checkpoints/final/codec.pt',map_location='cpu',weights_only=True)
            final=torch.load(root/'resume/checkpoints/final/codec.pt',map_location='cpu',weights_only=True)
            for key,value in full['state_dict'].items():
                torch.testing.assert_close(value,final['state_dict'][key],rtol=0,atol=0)
            a=full['training']['adversarial'];b=final['training']['adversarial']
            self.assertEqual(a['updates'],4)
            self.assertEqual(b['updates'],4)
            for key,value in a['discriminator'].items():
                torch.testing.assert_close(value,b['discriminator'][key],rtol=0,atol=0)
            self.assertTrue(b['optimizer']['state'])
            rows=[json.loads(line) for line in (root/'resume/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),6)
            for row in rows:
                if row['phase'].endswith('render'):
                    self.assertIn('image_gan_loss',row)
                    self.assertGreater(row['discriminator_update_norm'],0)
                else:
                    self.assertNotIn('discriminator_loss',row)
                    self.assertEqual(row['objective'],'latent_mse')


if __name__=='__main__': unittest.main()
