import copy
import unittest

import torch

from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_core import ChannelEncoder


class ChannelPowerNormalizationTests(unittest.TestCase):
    def normalizer(self, mode):
        from gaussian_jscc.channel_power import SegmentPowerNormalizer
        return SegmentPowerNormalizer((0, 1, 2), mode, floor=.01, momentum=.1)

    def test_statistical_scale_preserves_point_amplitude_and_freezes_at_eval(self):
        norm = self.normalizer('statistical')
        norm.set_population_energy(torch.tensor([4., 9.]))
        norm.eval()
        z = torch.tensor([[[1., 0., 2., 0.], [3., 0., 6., 0.]]])
        active = torch.ones(1, 2, dtype=torch.bool)
        before = copy.deepcopy(norm.state_dict())
        output = norm(z, active)
        torch.testing.assert_close(output[0, 1], output[0, 0]*3)
        for key, value in before.items():
            torch.testing.assert_close(norm.state_dict()[key], value, rtol=0, atol=0)
        other = self.normalizer('statistical').eval()
        other.load_state_dict(norm.state_dict())
        torch.testing.assert_close(other(z, active), output, rtol=0, atol=0)

    def test_block_energy_excludes_padding_and_other_batch_blocks(self):
        norm = self.normalizer('block')
        z = torch.tensor([[[1., 0., 2., 0.], [3., 0., 6., 0.], [999., 999., 999., 999.]],
                          [[2., 0., 4., 0.], [2., 0., 4., 0.], [2., 0., 4., 0.]]])
        active = torch.tensor([[True, True, False], [True, True, True]])
        out = norm(z, active)
        self.assertAlmostEqual(float(out[0, 0, 0]), 1/(5.01**.5), places=6)
        self.assertEqual(float(out[0, 2].abs().sum()), 0.)
        torch.testing.assert_close(out[:1], norm(z[:1], active[:1]), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(norm(z, torch.zeros_like(active))).all())
        changed = z.clone(); changed[1] *= 100
        torch.testing.assert_close(out[0], norm(changed, active)[0], rtol=0, atol=0)

    def test_statistical_training_ema_ignores_padding_and_has_finite_gradient(self):
        norm = self.normalizer('statistical')
        norm.set_population_energy(torch.ones(2))
        z = torch.tensor([[[2., 0., 4., 0.], [100., 100., 100., 100.]]], requires_grad=True)
        out = norm(z, torch.tensor([[True, False]]))
        torch.testing.assert_close(norm.running_energy, torch.tensor([1.3, 2.5]))
        out.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(float(z.grad[0, 1].abs().sum()), 0.)

    def test_default_config_and_encoder_remain_identical_to_historical_formula(self):
        cfg = ModularCodecConfig(hidden=16, channel_variant='residual', channel_hidden=16)
        self.assertNotIn('channel_power_mode', cfg.to_dict())
        self.assertNotIn('channel_power_momentum', cfg.to_dict())
        model = ChannelEncoder(cfg)
        self.assertFalse(any('normalizer' in name for name in model.state_dict()))
        y = torch.randn(2, 8, 48); active = torch.ones(2, 8, dtype=torch.bool)
        raw = {}
        hook = model.readout.register_forward_hook(lambda m, a, o: raw.update(z=o))
        try: output = model(y, active, 10.)
        finally: hook.remove()
        want = []
        for start, end in zip(cfg.rates[:-1], cfg.rates[1:]):
            segment = raw['z'][..., 2*start:2*end]
            want.append(segment/(segment.square().sum(-1, keepdim=True)/(end-start)+.01).sqrt())
        torch.testing.assert_close(output, torch.cat(want, -1), rtol=0, atol=0)

    def test_opt_in_normalizer_reloads_through_standard_codec_factory(self):
        from gaussian_jscc.model_factory import build_codec, config_from_dict
        cfg = ModularCodecConfig(hidden=16, channel_variant='progressive', channel_hidden=16,
                                 channel_power_mode='statistical', channel_power_momentum=.03)
        model = build_codec(cfg)
        model.channel_encoder.normalizer.set_population_energy(torch.tensor([2., 3., 4.]))
        reloaded = build_codec(config_from_dict(cfg.to_dict()))
        reloaded.load_state_dict(model.state_dict())
        self.assertEqual(reloaded.cfg.channel_power_mode, 'statistical')
        torch.testing.assert_close(reloaded.channel_encoder.normalizer.running_energy,
                                   torch.tensor([2., 3., 4.]))


if __name__ == '__main__': unittest.main()
