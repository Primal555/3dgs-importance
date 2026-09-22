import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from gaussian_jscc.codec import CodecConfig, GaussianCodec, pack, unpack
from gaussian_jscc.data import Geometry, fit_feature_statistics, to_features, write_ply
from gaussian_jscc.transport import save_checkpoint, load_checkpoint
from gaussian_jscc.representation_train import add_parser, train, phase_objective, check_args, transition_gate
from gaussian_jscc.representation_validation import paths, validate_images


def setup(n=29):
    torch.manual_seed(72)
    cfg = CodecConfig(sh_degree=0, hidden=16, depth=1, attention_heads=2,
                      architecture='learned_split_logcov', context_mode='multiscale_self',
                      encoder_attention='geometric_point', decoder_attention='transformer_trunk',
                      representation_dim=16, communication_depth=1, block_size=8)
    model = GaussianCodec(cfg)
    raw = torch.randn(n, 14)*.1
    raw[:, :3] = torch.rand(n, 3)*2
    raw[:, 4:7] = -2+torch.randn(n, 3)*.2
    raw[:, 7:11] = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    geometry = Geometry.fit(raw[:, :3], 16)
    fit_feature_statistics(raw, model)
    features, _ = to_features(raw, geometry, model)
    return raw, geometry, features, model


def args_for(ply, out, extra=()):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command')
    add_parser(sub)
    return parser.parse_args(['train-representation', '--ply', str(ply), '--out', str(out),
        '--device', 'cpu', '--hidden', '16', '--depth', '1', '--attention-heads', '2',
        '--latent-dim', '16', '--communication-depth', '1', '--block-size', '8',
        '--validation-region-size', '8', '--validation-blocks', '1', '--blocks-per-batch', '2',
        '--representation-steps', '2', '--adapter-steps', '2', '--joint-steps', '2',
        '--max-clean-loss', '100000', '--max-adapter-loss-ratio', '100000',
        '--save-every', '1', '--validate-every', '2', '--render-every', '2', '--profile-every', '1',
        '--cpu-threads', '1', *extra])


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_clean_bypasses_channel_and_gradients(self):
        _, _, f, model = setup()
        core = model.learned
        with patch.object(core.channel_encoder, 'forward', side_effect=AssertionError('channel touched')):
            pred = model.reconstruct_clean(f)
            pred.square().mean().backward()
        for name, params in core.module_parameters().items():
            nonzero = any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
            self.assertEqual(nonzero, name.startswith('representation'))
        self.assertFalse(any('tier' in n or 'snr' in n for n, _ in core.representation_encoder.named_parameters()))

    def test_phase_freezing_and_joint_terms(self):
        _, g, f, model = setup()
        args = args_for('unused', 'unused')
        f = f[None]
        active = torch.ones(f.shape[:2], dtype=torch.bool)
        for phase in ('representation', 'adapter', 'joint'):
            model.zero_grad(set_to_none=True)
            model.learned.set_phase(phase)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-4)
            loss, terms, _ = phase_objective(model, f, active, g, args, phase, torch.randn(4, 3))
            if phase == 'joint':
                torch.testing.assert_close(loss, terms['communication']+terms['weighted_clean'])
            loss.backward()
            opt.step()
            for name, module in model.learned.named_children():
                expected = phase == 'joint' or name.startswith('representation') == (phase == 'representation')
                changed = any(not torch.equal(p, before['learned.'+name+'.'+n]) for n, p in module.named_parameters())
                self.assertEqual(changed, expected, (phase, name))

    def test_protocol_checkpoint_and_receiver_only(self):
        _, _, f, model = setup()
        q = torch.arange(len(f)) % 4
        z = model.encode(f, f[:, :3], q, 10)
        self.assertEqual(len(z), int(torch.tensor(model.cfg.rates)[q].sum()))
        decoded = model.decode(z, q, 10)
        torch.testing.assert_close(decoded, model(f, f[:, :3], q, 10, 'none'))
        self.assertEqual(float(decoded[q == 0].detach().abs().sum()), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'codec.pt'
            save_checkpoint(path, model, 5)
            loaded = load_checkpoint(path, 'cpu')
            torch.testing.assert_close(loaded.decode(z.detach(), q, 10), decoded)
            torch.testing.assert_close(loaded.reconstruct_clean(f), model.reconstruct_clean(f))
        # Sender modules are not needed at the receiver.
        with patch.object(model.learned.representation_encoder, 'forward', side_effect=AssertionError):
            model.decode(z, q, 10)

    def test_padding_permutation_and_power(self):
        _, _, f, model = setup(16)
        f = f.reshape(2, 8, -1)
        active = torch.ones(2, 8, dtype=torch.bool)
        active[0, -2:] = False
        active[1] = False
        poisoned = f.clone()
        poisoned[~active] = float('nan')
        clean, comm, _, _ = paths(model, poisoned, active, 10, 'none')
        self.assertTrue(torch.isfinite(clean).all() and torch.isfinite(comm).all())
        self.assertEqual(float((clean[~active].abs().sum()+comm[~active].abs().sum()).detach()), 0)
        q = active.long()*3
        z = model.learned.encode(poisoned, poisoned[..., :3], q, 10)
        self.assertTrue((z.square().sum(-1)/32 <= 1.00001).all())
        perm = torch.randperm(8)
        decoded = model.learned.decode(z, q, 10)
        torch.testing.assert_close(model.learned.decode(z[:, perm], q[:, perm], 10), decoded[:, perm], atol=2e-6, rtol=2e-5)

    def test_gate_requires_explicit_criteria(self):
        args = args_for('unused', 'unused')
        check_args(args)
        b = {'clean': {'loss': 5}, 'communication': {'loss': 20}}
        args.max_clean_loss = 4
        self.assertFalse(transition_gate('adapter', b, None, args)['passed'])
        args.max_clean_loss = 6
        args.max_adapter_loss_ratio = 2
        self.assertFalse(transition_gate('joint', b, None, args)['passed'])
        args.max_clean_loss = None
        with self.assertRaises(ValueError):
            check_args(args)

    def test_trainer_all_phases_and_exact_resume(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ply = root/'source.ply'
            write_ply(ply, raw, 0)
            full, interrupted = root/'full', root/'interrupted'
            with patch('gaussian_jscc.representation_plots.plot_run'):
                train(args_for(ply, full, ['--clip-norm', '1', '--save-every', '2']))
                original = torch.optim.Adam.step
                calls = [0]
                def crash(opt, *args, **kwargs):
                    calls[0] += 1
                    if calls[0] == 4:
                        raise RuntimeError('simulated interruption')
                    return original(opt, *args, **kwargs)
                with patch.object(torch.optim.Adam, 'step', crash):
                    with self.assertRaisesRegex(RuntimeError, 'simulated'):
                        train(args_for(ply, interrupted, ['--clip-norm', '1', '--save-every', '2']))
                resume = args_for(ply, interrupted)
                resume.resume = str(interrupted/'training_state.pt')
                train(resume)
            a = load_checkpoint(full/'codec.pt', 'cpu')
            b = load_checkpoint(interrupted/'codec.pt', 'cpu')
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
            summary = json.loads((full/'summary.json').read_text())
            self.assertEqual(summary['completed'], dict(representation=2, adapter=2, joint=2))
            self.assertEqual(len((interrupted/'loss.jsonl').read_text().splitlines()), 6)
            self.assertTrue(list(interrupted.glob('interrupted_tail_*')))
            rows = [json.loads(line) for line in (full/'loss.jsonl').read_text().splitlines()]
            for row in rows:
                if row['phase'] == 'adapter':
                    self.assertEqual(row['module_grad_norms']['representation_encoder'], 0)
                    self.assertEqual(row['module_grad_norms']['representation_decoder'], 0)

    def test_failed_gate_does_not_train_adapters(self):
        raw, _, _, _ = setup(40)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.representation_plots.plot_run'):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            args = args_for(root/'source.ply', root/'run', ['--max-clean-loss', '0'])
            train(args)
            summary = json.loads((root/'run/summary.json').read_text())
            self.assertFalse(summary['stopped_by_gate']['passed'])
            self.assertEqual(summary['completed'], dict(representation=2, adapter=0, joint=0))

    def test_existing_transport_and_replay_contracts(self):
        import test_learned_joint as contracts
        for name in ('test_new_receiver_has_only_packet_and_weights', 'test_drop_inputs_do_not_leak_into_other_outputs',
                     'test_replay_equals_checkpoint', 'test_reject_fake_st_gradients'):
            with self.subTest(name=name), patch('test_learned_joint.setup', setup):
                getattr(contracts.LearnedTests(name), name)()

    def test_render_outputs_with_mocked_rasterizer(self):
        from types import SimpleNamespace
        raw, geometry, f, model = setup(16)
        camera = SimpleNamespace(image_name='heldout', original_image=torch.zeros(3, 12, 12))
        reference = SimpleNamespace(get=lambda camera, device: torch.ones(3, 12, 12)*.5)
        args = args_for('unused', 'unused')
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.rendering.render', return_value=torch.ones(3, 12, 12)*.4):
            result = validate_images(model, list(f.split(8)), raw, geometry, [camera], reference, args, 500, 'representation', tmp)
            self.assertIn('photo_psnr', result['clean'])
            for name in ('clean', 'communication', 'source', 'photo', 'comparison'):
                self.assertTrue((Path(tmp)/'images/000500/view_00'/f'{name}.png').exists())


if __name__ == '__main__':
    unittest.main()
