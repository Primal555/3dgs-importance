"""Main-path decoder: masking, dependency, transport, gradients and CLI."""
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
import test_multiscale_codec as contracts
import test_q3_noiseless as q3_contracts
import test_position_path_diagnosis as diagnosis
from test_point_attention import setup as old_setup
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.spatial_response import spatial_response_loss
from gaussian_jscc.optimization import parameter_group


def setup(n=17):
    raw, g, f, old = old_setup(n)
    cfg = old.cfg.to_dict(); cfg['decoder_attention'] = 'transformer_trunk'
    model = GaussianCodec(CodecConfig.from_dict(cfg))
    model.attr_mean.copy_(old.attr_mean); model.attr_std.copy_(old.attr_std)
    return raw, g, f, model


class TransformerTrunkCLI(q3_contracts.Q3NoiselessTests):
    decoder_attention = 'transformer_trunk'


class TransformerTrunkDiagnosis(diagnosis.PositionPathTests):
    decoder_attention = 'transformer_trunk'


class TransformerTrunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_encoder_identical_and_no_unused_receiver(self):
        _, _, f, old = old_setup()
        cfg = old.cfg.to_dict()
        torch.manual_seed(42); a = GaussianCodec(CodecConfig.from_dict(cfg))
        cfg['decoder_attention'] = 'transformer_trunk'
        torch.manual_seed(42); b = GaussianCodec(CodecConfig.from_dict(cfg))
        q = torch.full((1, len(f)), 3, dtype=torch.long)
        torch.testing.assert_close(a.learned.encode(f[None], f[None, :, :3], q, 10),
                                   b.learned.encode(f[None], f[None, :, :3], q, 10), atol=0, rtol=0)
        self.assertFalse(hasattr(b.learned, 'context_heads'))
        self.assertFalse(hasattr(b.learned, 'dec_geometry_gate'))
        self.assertEqual(b.learned.dec_trunk.tap_indices, (0, 1, 3))
        self.assertNotIn('decoder_depth', a.cfg.to_dict())
        self.assertEqual(CodecConfig.from_dict(b.cfg.to_dict()).decoder_depth, 4)
        for changes in ({'decoder_depth': 2}, {'xyz_decoder': 'block_center'}, {'context_mode': 'window'}):
            with self.assertRaises(ValueError):
                GaussianCodec(CodecConfig.from_dict(dict(cfg, **changes)))

    def test_receiver_permutation_masking_and_distant_dependency(self):
        _, _, _, m = setup()
        q = torch.full((2, 65), 3, dtype=torch.long); q[0, 1] = 0; q[1] = 0
        z = torch.randn(2, 65, 2*m.cfg.rates[-1], requires_grad=True)
        y = m.learned.decode(z, q, 10)
        perm = torch.randperm(65)
        torch.testing.assert_close(m.learned.decode(z[:, perm], q[:, perm], 10), y[:, perm], atol=2e-6, rtol=2e-5)
        self.assertEqual(float(y[q == 0].detach().abs().sum()), 0)
        altered = z.detach().clone(); altered[q == 0] = float('nan')
        torch.testing.assert_close(m.learned.decode(altered, q, 10), y)
        y[0, 0, :3].sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(float(z.grad[q == 0].abs().sum()), 0)
        self.assertGreater(float(z.grad[0, -1].norm()), 0)
        empty = m.learned.decode(z[:, :0], q[:, :0], 10)
        self.assertEqual(empty.shape, (2, 0, m.cfg.attr_dim+3))

    def test_all_heads_and_depths_receive_physical_loss_gradients(self):
        _, g, f, m = setup(17)
        q = torch.arange(17)%4
        pred = m(f, f[:, :3], q, 10, 'none')
        loss, _ = spatial_response_loss(pred[q > 0], f[q > 0], g, m, directions=torch.eye(3))
        loss.backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        trunk = m.learned.dec_trunk
        for layer in trunk.blocks:
            self.assertGreater(float(layer.attention.in_proj_weight.grad.norm()), 0)
        for norm in trunk.xyz_norms:
            self.assertGreater(float(norm.weight.grad.norm()), 0)
        for head in trunk.heads.values():
            self.assertGreater(float(head.weight.grad.norm()), 0)
        groups = {parameter_group(name) for name, _ in m.named_parameters()}
        self.assertIn('xyz_multidepth_head', groups)
        self.assertIn('decoder_transformer_layer_3', groups)
        baseline = GaussianCodec(CodecConfig.from_dict(dict(m.cfg.to_dict(), decoder_attention='window')))
        baseline.attr_mean.copy_(m.attr_mean); baseline.attr_std.copy_(m.attr_std)
        reference = spatial_response_loss(pred[q > 0], f[q > 0], g, baseline, directions=torch.eye(3))[0]
        torch.testing.assert_close(loss, reference, atol=0, rtol=0)

    def test_transport_short_fit_and_backward_modes(self):
        for name in ('test_transport_power_drop_and_replay_contracts', 'test_short_fit_and_all_backward_modes'):
            with self.subTest(name=name), patch('test_multiscale_codec.setup', setup):
                getattr(contracts.MultiScaleTests(name), name)()

    def test_full_block_sh3_and_all_dropped_batch(self):
        cfg = CodecConfig(architecture='learned_split_logcov', context_mode='multiscale_self',
                          encoder_attention='geometric_point', decoder_attention='transformer_trunk')
        m = GaussianCodec(cfg)
        f = torch.randn(2, 256, cfg.attr_dim+3)
        q = torch.randint(0, 4, (2, 256)); q[1] = 0
        out = m.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).float(), 10, 'awgn')[0]
        self.assertEqual(out.shape, f.shape)
        self.assertEqual(float(out[1].detach().abs().sum()), 0)
        out.square().mean().backward()
        self.assertGreater(float(m.learned.dec_trunk.heads['sh'].weight.grad.norm()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
