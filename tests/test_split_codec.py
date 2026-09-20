"""Information boundary, gradient, transport and training checks for split codec."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import test_learned_joint as baseline_tests
from test_learned_joint import setup
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import write_ply
from gaussian_jscc.optimization import clip_codec_gradients
from gaussian_jscc.spatial_response import spatial_response_loss


def split_setup(n=17):
    raw, g, f, baseline = setup(n)
    cfg = baseline.cfg.to_dict()
    cfg['architecture'] = 'learned_split'
    m = GaussianCodec(CodecConfig.from_dict(cfg))
    m.attr_mean.copy_(baseline.attr_mean)
    m.attr_std.copy_(baseline.attr_std)
    return raw, g, f, m


class SplitCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_existing_transport_and_mixed_tier_contracts(self):
        # Exercise identical invariants without changing baseline tests/schema.
        for name in ('test_no_hand_geometry_or_split_budget',
                     'test_packed_batched_noise_and_gradient_match',
                     'test_power_budget_empty_windows_and_singletons',
                     'test_new_receiver_has_only_packet_and_weights',
                     'test_xyz_responds_to_received_symbols_and_not_clamped',
                     'test_drop_inputs_do_not_leak_into_other_outputs',
                     'test_replay_equals_checkpoint', 'test_reject_fake_st_gradients',
                     'test_discrete_joint_omits_zero_rows_and_backpropagates'):
            with self.subTest(name=name), patch('test_learned_joint.setup', split_setup):
                case = baseline_tests.LearnedTests(name)
                getattr(case, name)()

    def test_sender_relative_geometry_changes_attention(self):
        _, _, f, m = split_setup(8)
        block = m.learned.enc_geometry_blocks[0]
        active = torch.ones(1, 8, dtype=torch.bool)
        h = torch.randn(1, 8, 16)
        xyz = f[None, :, :3].clone().requires_grad_()
        out = block(h, xyz, active)
        out.square().mean().backward()
        self.assertGreater(float(xyz.grad.norm()), 0)
        # Relative bias does not depend on global translation.
        torch.testing.assert_close(out, block(h, xyz.detach()+2, active), atol=2e-6, rtol=2e-5)
        changed = xyz.detach().clone()
        changed[:, 1] += .5
        self.assertFalse(torch.allclose(out, block(h, changed, active)))

    def test_receiver_geometry_is_not_driven_by_appearance_branch(self):
        _, _, f, m = split_setup(8)
        q = torch.arange(8) % 3 + 1
        z = m.encode(f, f[:, :3], q, 10).detach()
        before = m.decode(z, q, 10).detach()
        with torch.no_grad():
            for name, p in m.learned.named_parameters():
                if name.startswith(('dec_appearance', 'dec_exchange')):
                    p.add_(torch.randn_like(p))
        after = m.decode(z, q, 10).detach()
        torch.testing.assert_close(before[:, :3], after[:, :3])
        torch.testing.assert_close(before[:, 4:11], after[:, 4:11])
        self.assertFalse(torch.allclose(before[:, 11:], after[:, 11:]))

    def test_sh3_and_all_dropped_batched_windows(self):
        m = GaussianCodec(CodecConfig(architecture='learned_split', hidden=16, depth=2,
                                     sh_degree=3, block_size=8, decoder_window=4))
        f = torch.randn(2, 8, m.cfg.attr_dim + 3)
        q = torch.tensor([[1, 0, 2, 3, 1, 0, 2, 3], [0]*8])
        pred = m.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).float(), 10, 'awgn')[0]
        self.assertEqual(pred.shape, f.shape)
        self.assertEqual(float(pred[q == 0].detach().abs().sum()), 0.)
        pred[q > 0].square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        self.assertGreater(float(m.learned.heads['sh'].weight.grad.norm()), 0.)

    def test_all_streams_have_finite_gradients_and_can_fit(self):
        torch.manual_seed(42)
        _, g, f, m = split_setup(8)
        q = torch.arange(8) % 3 + 1
        optimizer = torch.optim.Adam(m.parameters(), lr=2e-4)
        def objective():
            pred = m(f, f[:, :3], q, 10, 'none')
            return spatial_response_loss(pred, f, g, m, directions=torch.eye(3))[0]
        initial = float(objective().detach())
        for step in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            _, stats = clip_codec_gradients(m, mode='none')
            if step == 0:
                for key in ('geometry_encoder', 'appearance_encoder', 'geometry_decoder',
                            'appearance_decoder', 'encoder_exchange', 'decoder_exchange', 'xyz_head'):
                    self.assertGreater(stats['gradient_groups'][key]['before'], 0, key)
            optimizer.step()
        self.assertLess(float(objective().detach()), initial * .8)

    def test_cli_random_split_checkpoint_and_architecture_guard(self):
        from gaussian_jscc.cli import main
        from gaussian_jscc.transport import load_checkpoint
        raw, _, _, _ = split_setup(17)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_ply(root/'input.ply', raw, 0)
            argv = ['gaussian_jscc', 'train-learned', '--architecture', 'learned_split',
                    '--ply', str(root/'input.ply'), '--out', str(root/'run'), '--device', 'cpu',
                    '--bootstrap-objective', 'spatial-response', '--bootstrap-steps', '2',
                    '--render-steps', '0', '--joint-steps', '0', '--hidden', '16', '--depth', '1',
                    '--block-size', '4', '--decoder-window', '4', '--blocks-per-batch', '2',
                    '--validation-blocks', '1', '--validation-trials', '1', '--validate-every', '1']
            with patch('sys.argv', argv), patch('gaussian_jscc.rendering.load_cameras', side_effect=AssertionError), \
                 patch('gaussian_jscc.plots.safe_plot'):
                main()
            model = load_checkpoint(root/'run/codec.pt', 'cpu')
            self.assertEqual(model.cfg.architecture, 'learned_split')
            rows = [json.loads(s) for s in (root/'run/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertIn('geometry_decoder', rows[-1]['gradient_groups'])
            # An explicitly requested architecture cannot be silently overridden.
            bad = argv.copy()
            bad[bad.index('learned_split')] = 'learned_joint'
            bad[bad.index(str(root/'run'))] = str(root/'bad')
            bad += ['--init', str(root/'run/codec.pt')]
            with patch('sys.argv', bad), self.assertRaisesRegex(ValueError, 'architecture mismatch'):
                main()


if __name__ == '__main__':
    unittest.main()
