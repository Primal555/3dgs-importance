"""Paired appearance controls: real codec/autograd; CPU renders are plumbing only."""
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc import local_response, multiscene
from gaussian_jscc.data import prepare, to_features, write_ply
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.render_objective import MultiViewRenderTask
from gaussian_jscc.training import full_scene_step
from test_gaussian_jscc import fixture
from test_render_first import Reference, synthetic_render
import test_render_first as render_fixture


class AppearanceControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model_fixture(self):
        torch.manual_seed(71)
        raw, _ = fixture(12, degree=1)
        raw, geometry, _ = prepare(raw, 16)
        model = GaussianCodec(CodecConfig(architecture='learned_joint', loss_profile='learned_v1',
            sh_degree=1, hidden=16, depth=1, grid_dim=4, levels=(2,), block_size=8,
            decoder_window=4, prefix_mode='progressive', position_delivery='quantized'))
        model.attr_mean.copy_(raw[:, 3:].mean(0))
        model.attr_std.copy_(raw[:, 3:].std(0, unbiased=False).clamp_min(.01))
        features, _ = to_features(raw, geometry, model)
        return raw, geometry, features, model

    def test_camera_direction_is_per_point_and_matches_renderer_sign(self):
        function = getattr(local_response, 'camera_directions', None)
        self.assertTrue(callable(function), 'missing per-point camera direction sampler')
        points = torch.tensor([[1., 0., 0.], [0., 1., 0.]], requires_grad=True)
        centers = torch.tensor([[0., 0., 0.], [0., 0., 1.]], requires_grad=True)
        result = function(points, centers)
        expected = torch.tensor([[[1., 0., 0.], [2**-.5, 0., -2**-.5]],
                                 [[0., 1., 0.], [0., 2**-.5, -2**-.5]]])
        torch.testing.assert_close(result, expected)
        self.assertFalse(result.requires_grad)
        with self.assertRaises(ValueError):
            function(points.detach(), points[:1].detach())

    def test_per_point_response_matches_individual_and_has_finite_gradients(self):
        _, geometry, features, model = self.model_fixture()
        pred = features.clone(); pred[:, 11:] += .3
        pred.requires_grad_()
        directions = torch.randn(len(pred), 3, 3)
        try:
            loss, _ = local_response.local_response_loss(pred, features, geometry, model, directions=directions)
        except ValueError as exc:
            self.fail(f'per-point directions must be accepted: {exc}')
        separate = torch.stack([local_response.local_response_loss(
            p[None], t[None], geometry, model, directions=d)[0]
            for p, t, d in zip(pred, features, directions)]).mean()
        torch.testing.assert_close(loss, separate, rtol=1e-5, atol=1e-7)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad[:, 11:].norm()), 0)
        self.assertEqual(float(pred.grad[:, :3].abs().sum()), 0.)

    def test_shared_directions_remain_equivalent_to_repeated_per_point_directions(self):
        _, geometry, features, model = self.model_fixture()
        pred = features.clone(); pred[:, 4:7] += .1
        directions = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        a = local_response.local_response_loss(pred, features, geometry, model, directions=directions)[0]
        try:
            b = local_response.local_response_loss(pred, features, geometry, model,
                directions=directions[None].expand(len(pred), -1, -1))[0]
        except ValueError as exc:
            self.fail(f'per-point directions must be accepted: {exc}')
        torch.testing.assert_close(a, b)

    def test_color_only_real_replay_updates_only_dc_and_sh(self):
        configure = getattr(multiscene, 'configure_codec_training', None)
        self.assertTrue(callable(configure), 'missing audited color-head freezing')
        _, geometry, features, model = self.model_fixture()
        before = copy.deepcopy(model.state_dict())
        names = configure(model, 'color_heads')
        self.assertEqual(set(names), {'learned.heads.dc.weight', 'learned.heads.dc.bias',
                                      'learned.heads.sh.weight', 'learned.heads.sh.bias'})
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
            full_scene_step(model, [(features[None], torch.full((1, len(features)), 2))],
                geometry, 10, 'awgn', MultiViewRenderTask(render_fixture.RenderFirstTests().cameras(), Reference(), 1),
                attr_weight=0., mode='replay')
        optimizer.step()
        changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
        self.assertEqual(changed, set(names))
        configure(model, 'all')
        self.assertTrue(model.learned.symbol_head.weight.requires_grad)
        self.assertFalse(model.learned.heads['xyz'].weight.requires_grad)

    def test_color_metrics_roi_does_not_hide_signed_color_error(self):
        try:
            metrics = importlib.import_module('gaussian_jscc.appearance_metrics')
        except ModuleNotFoundError:
            self.fail('missing appearance metrics and predeclared ROI support')
        target = torch.zeros(3, 4, 4)
        image = target.clone(); image[0, :2, :2] = .2
        result = metrics.appearance_metrics(image, target, {'red_patch': [0, 0, .5, .5]})
        self.assertAlmostEqual(result['r_bias'], .05, places=6)
        self.assertAlmostEqual(result['regions']['red_patch']['r_bias'], .2, places=6)
        self.assertAlmostEqual(result['regions']['red_patch']['rg_error_rmse'], .2, places=6)
        with self.assertRaises(ValueError):
            metrics.appearance_metrics(image, target, {'invalid': [.5, 0, .4, 1]})

    def test_render_only_rejects_accidental_allocator_training(self):
        options = [a.option_strings for a in multiscene.parser()._actions]
        self.assertIn(['--render-only'], options, 'missing safe render-only initializer mode')
        args = multiscene.parser().parse_args(['--out', 'unused', '--render-only', '--checkpoint', 'unused'])
        with self.assertRaisesRegex(ValueError, 'render-only'):
            multiscene.train(args)


class AppearancePlanTests(unittest.TestCase):
    def test_paired_jobs_keep_codec_budgets_and_training_streams_matched(self):
        try:
            controls = importlib.import_module('gaussian_jscc.appearance_controls')
        except ModuleNotFoundError:
            self.fail('missing appearance control command planner')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'; source.mkdir()
            folder = source/'checkpoints/end_render'; folder.mkdir(parents=True)
            (folder/'codec.pt').touch()
            (source/'scenes/truck').mkdir(parents=True)
            from gaussian_jscc.allocation import scene_fingerprint
            raw, _ = fixture(12, degree=1)
            write_ply(root/'truck.ply', raw, 1)
            # Match the identity of the canonical PLY read back from disk.
            from gaussian_jscc.data import read_ply
            canonical, _ = read_ply(root/'truck.ply')
            (source/'scenes/truck/scene.json').write_text(json.dumps({
                'fingerprint': scene_fingerprint(canonical), 'source_gaussians':len(raw),
                'train_view_names':['train'], 'validation_view_names':['validation']}))
            (root/'images').mkdir()
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'scenes': [
                {'name': 'truck', 'role': 'train', 'ply': 'truck.ply', 'source': 'images'}]}))
            (source/'training.json').write_text(json.dumps({'manifest': str(manifest),
                'selected_scenes': [{'name': 'truck', 'role': 'train'}], 'rates': [0, 4, 12, 24],
                'seed': 42, 'snr': 10., 'blocks_per_batch': 64}))
            args = controls.parser().parse_args(['--run', str(source), '--out', str(root/'controls')])
            plan = controls.build_plan(args, python='python')
            jobs = [j for j in plan['jobs'] if j['kind'] == 'fit']
            self.assertEqual(len(jobs), 4)
            for job in jobs:
                c = job['command']
                self.assertEqual(c[c.index('--allocation-steps')+1], '0')
                self.assertEqual(c[c.index('--joint-steps')+1], '0')
                self.assertIn('--paired-training-rng', c)
                self.assertIn('--color-metrics', c)
            colors = [j['command'] for j in jobs if j['section'] == 'color']
            self.assertEqual(len(colors), 2)
            for c in colors:
                self.assertIn('--render-only', c)
                self.assertEqual(c[c.index('--uniform-tier')+1], '2')
                self.assertEqual(c[c.index('--drop')+1], '0')
                self.assertEqual(c[c.index('--checkpoint')+1], str(folder/'codec.pt'))
            directions = [j['command'] for j in jobs if j['section'] == 'directions']
            self.assertEqual({c[c.index('--bootstrap-directions')+1] for c in directions}, {'random', 'camera'})
            self.assertTrue(all('--checkpoint' not in c for c in directions))
            # The diagnostic is a multiscene-aware read-only oracle, not a new training group.
            diagnostic = next(j['command'] for j in plan['jobs'] if j['kind'] == 'diagnose')
            self.assertIn('--scene', diagnostic)
            # Exercise the actual Bash launcher, including GPU guard and argv forwarding.
            bash = shutil.which('bash') if os.name != 'nt' else 'C:/softwares/Git/bin/bash.exe'
            if bash and Path(bash).is_file():
                launch = [bash, 'scripts/test_appearance_controls.sh', str(source), str(root/'launcher'), '--dry-run']
                env = dict(os.environ, PYTHON_BIN=sys.executable.replace('\\','/'), SUITE='directions', SCENES='truck')
                env.pop('CUDA_VISIBLE_DEVICES', None)
                failed = subprocess.run(launch, env=env, capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn('Set CUDA_VISIBLE_DEVICES', failed.stderr)
                env['CUDA_VISIBLE_DEVICES'] = '2'
                ready = subprocess.run(launch, env=env, capture_output=True, text=True)
                self.assertEqual(ready.returncode, 0, ready.stderr)
                self.assertIn('directions_camera:', ready.stdout)
                self.assertNotIn('color_initial:', ready.stdout)
                self.assertFalse((root/'launcher').exists())
            # Skipping the oracle must NOT skip the source-data identity check.
            args.skip_oracle = True
            raw[0, 0] += 1
            write_ply(root/'truck.ply', raw, 1)
            with self.assertRaisesRegex(ValueError, 'PLY differs'):
                controls.build_plan(args, python='python')

    def test_complete_paired_suites_and_color_oracle_on_multiscene_record(self):
        try:
            controls = importlib.import_module('gaussian_jscc.appearance_controls')
        except ModuleNotFoundError:
            self.fail('missing appearance suite')
        from gaussian_jscc.color_ablation import run as oracle_run
        from gaussian_jscc.multiscene_experiments import evaluate, parser as evaluation_parser
        from gaussian_jscc.transport import model_id, load_checkpoint
        from types import SimpleNamespace
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, _ = fixture(12, degree=1)
            write_ply(root/'truck.ply', raw, 1)
            (root/'images').mkdir()
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'scenes': [
                {'name':'truck','role':'train','ply':'truck.ply','source':'images'}]}))
            source = root/'source'
            options = multiscene.parser().parse_args(['--manifest',str(manifest),'--out',str(source),
                '--device','cpu','--bootstrap-steps','1','--render-steps','1','--allocation-steps','0',
                '--joint-steps','0','--hidden','16','--depth','1','--block-size','8','--decoder-window','4',
                '--blocks-per-batch','2','--views-per-step','1','--validation-views','1','--validation-trials','1',
                '--output-profile','compact'])
            cameras = render_fixture.RenderFirstTests().cameras()
            for index, camera in enumerate(cameras):
                camera.camera_center = torch.tensor([10., 10., 10.+index])
            out = root/'controls'
            args = controls.parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu',
                '--color-steps','2','--bootstrap-steps','2','--render-steps','1','--validate-every','1',
                '--test-views','1','--trials','1','--image-views','1'])
            current_job, streams = {'name':None}, {}
            original_forward = GaussianCodec.forward_tier_batches
            from gaussian_jscc.codec import channel as original_channel
            def observe_forward(model, features, xyz, choices, *positional, **keywords):
                if torch.is_grad_enabled() and current_job['name'] in ('directions_random','directions_camera'):
                    streams.setdefault(current_job['name']+'_inputs', []).append(
                        (features.detach().clone(), choices.detach().clone()))
                return original_forward(model, features, xyz, choices, *positional, **keywords)
            def observe_channel(symbols, *positional, **keywords):
                if torch.is_grad_enabled() and current_job['name'] in ('directions_random','directions_camera'):
                    streams.setdefault(current_job['name']+'_noise', []).append(
                        (tuple(symbols.shape), torch.get_rng_state().clone()))
                return original_channel(symbols, *positional, **keywords)
            def in_process(command, check):
                module = command[command.index('-m')+1]
                arguments = command[command.index('-m')+2:]
                if module == 'gaussian_jscc.multiscene':
                    current_job['name'] = Path(arguments[arguments.index('--out')+1]).parts[-2]+'_'+Path(arguments[arguments.index('--out')+1]).name
                    multiscene.train(multiscene.parser().parse_args(arguments))
                elif module == 'gaussian_jscc.multiscene_experiments':
                    current_job['name'] = None
                    evaluate(evaluation_parser().parse_args(arguments))
                elif module == 'gaussian_jscc.color_ablation':
                    def option(name):
                        return arguments[arguments.index(name)+1]
                    oracle_run(SimpleNamespace(training=option('--training'), checkpoint=option('--checkpoint'),
                        scene=option('--scene'), ply=option('--ply'), source=option('--source'), out=option('--out'),
                        tiers=[2], trials=1, device='cpu', image_views=1))
                else:
                    self.fail(module)
            with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
                 patch('gaussian_jscc.rendering.load_cameras', return_value=cameras), \
                 patch.object(GaussianCodec, 'forward_tier_batches', new=observe_forward), \
                 patch('gaussian_jscc.codec.channel', side_effect=observe_channel), \
                 patch('gaussian_jscc.appearance_controls.subprocess.run', side_effect=in_process):
                multiscene.train(options)
                controls.main(args)
            complete = json.loads((out/'complete.json').read_text())
            self.assertFalse(complete['allocator_trained'])
            self.assertEqual(set(complete['paired_initializers_verified']), {'color','directions'})
            results = json.loads((out/'comparison.json').read_text())
            self.assertEqual(len(results), 15)  # five tests, three positive prefixes each
            self.assertTrue(all('source_color_b_bias' in r for r in results))
            self.assertTrue(all(r['allocation_fit']=='no table' for r in results))
            self.assertTrue((out/'charts/truck_color.png').is_file())
            self.assertTrue((out/'charts/truck_color_final_test.png').is_file())
            self.assertTrue((out/'charts/truck_directions_loss.png').is_file())
            before = load_checkpoint(source/'checkpoints/end_render/codec.pt','cpu')
            heads = load_checkpoint(out/'color/color_heads/checkpoints/final/codec.pt','cpu')
            for name, value in before.state_dict().items():
                if not name.startswith(('learned.heads.dc.', 'learned.heads.sh.')):
                    torch.testing.assert_close(value, heads.state_dict()[name], atol=0, rtol=0)
            initial_a = load_checkpoint(out/'directions/random/checkpoints/initial/codec.pt','cpu')
            initial_b = load_checkpoint(out/'directions/camera/checkpoints/initial/codec.pt','cpu')
            self.assertEqual(model_id(initial_a), model_id(initial_b))
            for kind in ('inputs','noise'):
                a, b = streams['directions_random_'+kind], streams['directions_camera_'+kind]
                self.assertTrue(a)
                self.assertEqual(len(a), len(b))
                for pair_a, pair_b in zip(a,b):
                    for value_a, value_b in zip(pair_a,pair_b):
                        if torch.is_tensor(value_a):
                            torch.testing.assert_close(value_a,value_b,atol=0,rtol=0)
                        else:
                            self.assertEqual(value_a,value_b)
            for branch in ('random','camera'):
                run = out/'directions'/branch
                self.assertTrue(json.loads((run/'training.json').read_text())[
                    'reference_scene_identity_and_splits_verified'])
                rows = [json.loads(line) for line in (run/'loss.jsonl').read_text().splitlines()]
                self.assertEqual([r['phase'] for r in rows], ['bootstrap','bootstrap','render'])
                if branch=='camera':
                    self.assertTrue(all(r['bootstrap_camera_names']==['1.3']*4 for r in rows[:2]))
                images = list((run/'scenes/truck/validation_images').iterdir())
                self.assertEqual({p.name for p in images}, {'000000','000003'})
            losses = []
            for scope in ('color_heads','all'):
                rows = [json.loads(line) for line in (out/f'color/{scope}/loss.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['retained_gaussians']==len(raw) and r['layout']=='2' for r in rows))
                losses.append(rows)
            self.assertEqual([r['training_view_indices'] for r in losses[0]],
                             [r['training_view_indices'] for r in losses[1]])


if __name__ == '__main__':
    unittest.main()
