import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import Geometry, fit_feature_statistics, to_features, write_ply
from gaussian_jscc.center_attribute_codec import center_loss
from gaussian_jscc.center_attribute_train import add_parser, train, make_batches, render_step, check_args, center_ready, attribute_ready
from gaussian_jscc.center_attribute_validation import validate_render
from gaussian_jscc.transport import load_checkpoint, save_checkpoint
from test_representation_codec import setup as source_setup


def setup(n=24):
    raw, g, _, _ = source_setup(n)
    cfg = CodecConfig(architecture='learned_split_logcov', context_mode='multiscale_self',
        encoder_attention='geometric_point', decoder_attention='transformer_trunk',
        sh_degree=0, hidden=16, depth=1, decoder_depth=3, attention_heads=2,
        block_size=8, representation_dim=16, center_latent_dim=8)
    model = GaussianCodec(cfg)
    fit_feature_statistics(raw, model)
    f, _ = to_features(raw, g, model)
    return raw, g, f, model


def args_for(ply, out, extra=()):
    p = argparse.ArgumentParser()
    add_parser(p.add_subparsers(dest='command'))
    return p.parse_args(['train-center-attributes', '--ply', str(ply), '--out', str(out), '--device', 'cpu',
        '--hidden', '16', '--depth', '1', '--decoder-depth', '3', '--attention-heads', '2',
        '--latent-dim', '16', '--center-latent-dim', '8', '--block-size', '8',
        '--blocks-per-batch', '2', '--render-blocks-per-batch', '2', '--validation-region-size', '8',
        '--validation-blocks', '1', '--validation-views', '1', '--cpu-threads', '1',
        '--center-steps', '3', '--attribute-steps', '0', '--joint-steps', '0',
        '--validate-every', '1', '--render-every', '2', '--save-every', '1', '--profile-every', '1', *extra])


class CenterAttributeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legacy_identity_and_invalid_config(self):
        self.assertNotIn('center_latent_dim', CodecConfig().to_dict())
        for value in (-1, 1.5, 32):
            with self.assertRaises(ValueError):
                CodecConfig(center_latent_dim=value)

    def test_center_cannot_see_attributes(self):
        _, _, f, model = setup()
        f = f.reshape(3, 8, -1)
        active = torch.ones(3, 8, dtype=torch.bool)
        base = model.reconstruct_clean(f)
        changed = f.clone()
        changed[..., 3:] = torch.randn_like(changed[..., 3:])*100
        torch.testing.assert_close(model.reconstruct_clean(changed)[..., :3], base[..., :3], atol=0, rtol=0)
        with patch.object(model.learned.attribute_encoder, 'forward', side_effect=AssertionError('attributes used')):
            model.learned.centers(f[..., :3], active).sum().backward()
        for name, params in model.learned.module_parameters().items():
            self.assertEqual(any(p.grad is not None for p in params), name.startswith('center'))

    def test_attribute_decoder_receives_predicted_not_source_xyz(self):
        _, _, f, model = setup(8)
        f = f[None]
        seen = []
        handle = model.learned.attribute_decoder.register_forward_pre_hook(lambda m, a: seen.append(a[1]))
        try:
            pred = model.reconstruct_clean(f)
        finally:
            handle.remove()
        torch.testing.assert_close(seen[0], pred[..., :3], atol=0, rtol=0)
        self.assertFalse(torch.allclose(seen[0], f[..., :3]))

    def test_three_phase_updates_and_joint_condition_gradient(self):
        _, g, f, model = setup(8)
        active = torch.ones(1, 8, dtype=torch.bool)
        for phase in ('center', 'attribute', 'joint'):
            model.learned.set_phase(phase)
            model.zero_grad(set_to_none=True)
            before = {k: p.clone() for k, p in model.named_parameters()}
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-4)
            if phase == 'center':
                xyz = model.learned.centers(f[None, :, :3], active)
                loss = center_loss(xyz[0], f[:, :3], g, .001)
            else:
                # Attribute-only output loss must also reach center via conditioning in C.
                loss = model.reconstruct_clean(f)[:, 3:].square().mean()
            loss.backward()
            opt.step()
            for name, module in model.learned.named_children():
                changed = any(not torch.equal(p, before[f'learned.{name}.{key}']) for key, p in module.named_parameters())
                expected = phase == 'joint' or name.startswith('center') == (phase == 'center')
                self.assertEqual(changed, expected, (phase, name))

    def test_center_loss_world_gradient_and_no_teacher_gradient(self):
        g = Geometry([0, 0, 0], [1, 1, 1], 16)
        target = torch.zeros(3, 3, requires_grad=True)
        pred = torch.tensor([[0., 0, 0], [100., 100, 100], [.001, 0, 0]], requires_grad=True)
        loss = center_loss(pred, target, g, .001)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertLessEqual(float(pred.grad.norm(dim=-1).max()), 1/3+1e-7)
        torch.testing.assert_close(pred.grad[0], torch.zeros(3))
        same, _ = torch.autograd.functional.jvp(lambda p: center_loss(p, target, g, .001), pred, torch.ones_like(pred))
        torch.testing.assert_close(same, loss)

    def test_checkpoint_roundtrip_and_no_fake_communication(self):
        _, _, f, model = setup(8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'codec.pt'
            save_checkpoint(path, model, 0)
            recovered = load_checkpoint(path, 'cpu')
            torch.testing.assert_close(recovered.reconstruct_clean(f), model.reconstruct_clean(f))
        with self.assertRaisesRegex(ValueError, 'clean-only'):
            model.encode(f, f[:, :3], torch.full((8,), 3), 10)

    def test_padding_and_center_permutation(self):
        _, _, f, model = setup(8)
        active = torch.tensor([[True]*6+[False]*2, [False]*8])
        features = f[None].expand(2, -1, -1).clone()
        features[~active] = float('nan')
        pred = model.reconstruct_clean(features, active)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(float(pred[~active].abs().sum().detach()), 0.)
        z = torch.randn(1, 8, 8)
        mask = torch.ones(1, 8, dtype=torch.bool)
        perm = torch.randperm(8)
        torch.testing.assert_close(model.learned.center_decoder(z[:, perm], mask),
                                   model.learned.center_decoder(z, mask)[:, perm], atol=2e-6, rtol=2e-5)

    def test_replay_direct_gradients_attribute_and_joint(self):
        _, geometry, f, source = setup(19)
        batches = make_batches(list(f.split(8)), 2)
        for phase in ('attribute', 'joint'):
            reference = None
            for mode in ('direct', 'replay'):
                model = copy.deepcopy(source)
                model.learned.set_phase(phase)
                loss, stats = render_step(model, batches, geometry, lambda scene: (scene*.1).square().mean(), mode)
                gradients = {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
                if reference is None:
                    reference = (loss, gradients)
                else:
                    torch.testing.assert_close(loss, reference[0])
                    for n, grad in gradients.items():
                        if grad is not None:
                            torch.testing.assert_close(grad, reference[1][n], atol=2e-6, rtol=2e-4)
                        else:
                            self.assertIsNone(reference[1][n])
                self.assertEqual(any(v is not None and v.abs().sum() > 0 for n, v in gradients.items() if 'center_' in n), phase == 'joint')

    def test_render_validation_diagnostic_images(self):
        raw, g, f, model = setup(16)
        camera = SimpleNamespace(image_name='held', original_image=torch.zeros(3, 12, 12))
        reference = SimpleNamespace(get=lambda c, d: torch.ones(3, 12, 12)*.4)
        args = args_for('none', 'none')
        scenes = []
        def raster(scene, *a, **kw):
            scenes.append(scene.clone())
            return torch.ones(3, 12, 12)*.3
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.rendering.render', raster):
            result = validate_render(model, make_batches(list(f.split(8)), 2), raw, g, [camera], reference, args, 500, tmp, True)
            for name in ('full', 'center_only', 'quantized12', 'source', 'photo', 'comparison'):
                self.assertTrue((Path(tmp)/'images/000500/view_00'/f'{name}.png').exists())
            torch.testing.assert_close(scenes[1][:, 3:], raw[:, 3:])
            torch.testing.assert_close(scenes[1][:, :3], scenes[0][:, :3])
            torch.testing.assert_close(scenes[2][:, :3], g.denormalize((g.normalize(raw[:, :3])*4095).round()/4095))
            self.assertIn('source_psnr', result['full'])

    def test_gates_and_camera_requirement(self):
        args = args_for('none', 'none')
        self.assertFalse(center_ready({'render': None}, args))
        self.assertTrue(center_ready({'render': {'center_gap_db': 2.}}, args))
        self.assertFalse(center_ready({'render': {'center_gap_db': 4.}}, args))
        self.assertTrue(attribute_ready(1, .9, args)[0])
        self.assertFalse(attribute_ready(1, 1, args)[0])
        args.attribute_steps = 2
        with self.assertRaisesRegex(ValueError, 'cameras'):
            check_args(args)

    def test_center_trainer_exact_resume(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            train(args_for(root/'source.ply', root/'full'))
            actual = torch.optim.Adam.step
            calls = [0]
            def crash(opt, *a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('interruption')
                return actual(opt, *a, **kw)
            with patch.object(torch.optim.Adam, 'step', crash):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    train(args_for(root/'source.ply', root/'resume'))
            args = args_for(root/'source.ply', root/'resume')
            args.resume = str(root/'resume/training_state.pt')
            train(args)
            full = load_checkpoint(root/'full/codec.pt', 'cpu')
            restored = load_checkpoint(root/'resume/codec.pt', 'cpu')
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
            summary = json.loads((root/'full/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['completed'], {'center': 3, 'attribute': 0, 'joint': 0})
            self.assertEqual(summary['status'], 'complete')

    def test_full_three_phase_trainer_and_render_resume(self):
        raw, _, _, _ = setup(40)
        train_camera = SimpleNamespace(image_name='train', original_image=torch.zeros(3, 12, 12))
        test_camera = SimpleNamespace(image_name='test', original_image=torch.zeros(3, 12, 12))
        def cameras(source, resolution, white_background, images, split):
            return [train_camera] if split == 'train' else [test_camera]
        def raster(scene, *a, **kw):
            # Differentiable mock, NOT a CUDA splatting quality test.
            rgb = (scene[:, :3].mean(0)*.02 + scene[:, 3].sigmoid().mean()).sigmoid()
            return rgb[:, None, None].expand(3, 12, 12)
        options = ['--source', 'mock', '--attribute-steps', '3', '--joint-steps', '3',
                   '--center-max-gap-db', '1000', '--attribute-min-improvement', '0']
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'), \
                patch('gaussian_jscc.rendering.load_cameras', cameras), patch('gaussian_jscc.rendering.render', raster):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            train(args_for(root/'source.ply', root/'full', options))
            original = torch.optim.Adam.step
            # Check recovery inside both frozen-center and joint replay training.
            for interrupt_at in (5, 8):
                directory = root/f'resume_{interrupt_at}'
                calls = [0]
                def crash(opt, *a, **kw):
                    calls[0] += 1
                    if calls[0] == interrupt_at:
                        raise RuntimeError('render interruption')
                    return original(opt, *a, **kw)
                with patch.object(torch.optim.Adam, 'step', crash):
                    with self.assertRaisesRegex(RuntimeError, 'render interruption'):
                        train(args_for(root/'source.ply', directory, options))
                args = args_for(root/'source.ply', directory, options)
                args.resume = str(directory/'training_state.pt')
                train(args)
                a = load_checkpoint(root/'full/codec.pt', 'cpu')
                b = load_checkpoint(directory/'codec.pt', 'cpu')
                for name, value in a.state_dict().items():
                    torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
                self.assertEqual(len((directory/'loss.jsonl').read_text().splitlines()), 9)
            logs = [json.loads(line) for line in (root/'full/loss.jsonl').read_text().splitlines()]
            for row in logs:
                center_grad = row['module_grad_norms']['center_decoder']
                if row['phase'] == 'attribute':
                    self.assertEqual(center_grad, 0)
                    self.assertEqual(row['module_updates']['center_decoder'], 0)
                else:
                    self.assertGreater(center_grad, 0)
            summary = json.loads((root/'full/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['completed'], dict(center=3, attribute=3, joint=3))
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['export_selected_step'], summary['best_steps']['joint'])
            # Raw history steps must not be replaced by earlier selected weights
            # at a phase boundary or when exporting the validation-best codec.
            raw_last = torch.load(root/'full/codec_9.pt', weights_only=True)
            selected = torch.load(root/'full/codec.pt', weights_only=True)
            self.assertEqual(raw_last['step'], 9)
            self.assertEqual(selected['step'], summary['best_steps']['joint'])

    def test_failed_center_gate_prevents_attribute_training(self):
        raw, _, _, _ = setup(40)
        camera = SimpleNamespace(image_name='test', original_image=torch.zeros(3, 12, 12))
        other = SimpleNamespace(image_name='train', original_image=torch.zeros(3, 12, 12))
        def cameras(*a):
            return [other] if a[-1] == 'train' else [camera]
        def raster(scene, *a, **kw):
            return scene[:, :3].mean(0)[:, None, None].expand(3, 12, 12)
        options = ['--source', 'mock', '--attribute-steps', '2', '--center-max-gap-db', '0']
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'), \
                patch('gaussian_jscc.rendering.load_cameras', cameras), patch('gaussian_jscc.rendering.render', raster):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            train(args_for(root/'source.ply', root/'run', options))
            summary = json.loads((root/'run/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['status'], 'gate_stopped')
            self.assertEqual(summary['completed']['attribute'], 0)

    def test_early_transition_resume_keeps_decision(self):
        raw, _, _, _ = setup(40)
        def cameras(*a):
            return [SimpleNamespace(image_name=a[-1], original_image=torch.zeros(3, 12, 12))]
        def raster(scene, *a, **kw):
            return (scene[:, :3].mean(0)*.01+scene[:, 3].sigmoid().mean()).sigmoid()[:, None, None].expand(3, 12, 12)
        options = ['--source', 'mock', '--center-max-gap-db', '1000', '--min-center-steps', '1',
                   '--attribute-steps', '2', '--joint-steps', '2', '--attribute-min-improvement', '0']
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.center_attribute_train.plot_run'), \
                patch('gaussian_jscc.rendering.load_cameras', cameras), patch('gaussian_jscc.rendering.render', raster):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            train(args_for(root/'source.ply', root/'full', options))
            with patch('gaussian_jscc.center_attribute_train.load_checkpoint', side_effect=RuntimeError('boundary interruption')):
                with self.assertRaisesRegex(RuntimeError, 'boundary interruption'):
                    train(args_for(root/'source.ply', root/'resume', options))
            state = torch.load(root/'resume/training_state.pt', weights_only=True)
            self.assertEqual(state['progress']['end_reason'], 'center_render_gate')
            args = args_for(root/'source.ply', root/'resume', options)
            args.resume = str(root/'resume/training_state.pt')
            train(args)
            a, b = [load_checkpoint(root/x/'codec.pt', 'cpu') for x in ('full', 'resume')]
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
            summary = json.loads((root/'resume/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['completed'], dict(center=1, attribute=2, joint=2))


if __name__ == '__main__':
    unittest.main()
