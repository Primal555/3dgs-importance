"""Our LPIPS boundary, without downloading a pretrained backbone in unit tests."""
import unittest
from pathlib import Path
from unittest.mock import patch
import torch


class PairDistance(torch.nn.Module):
    """Small external-backend stand-in; the adapter's RGB contract stays real."""
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, image, target):
        if self.training or torch.is_grad_enabled():
            raise AssertionError('evaluation must be frozen and no-grad')
        if image.shape[0] != 1 or image.dtype != torch.float32:
            raise AssertionError('LPIPS backend requires single float32 RGB pairs')
        return self.scale * (image - target).square().mean()


def backend():
    # A real pretrained model constructor also initializes tensors before load.
    torch.rand(7)
    return PairDistance()


class PerceptualMetricTests(unittest.TestCase):
    @unittest.skipUnless(all((Path(torch.hub.get_dir())/'checkpoints'/name).exists()
                             for name in ('vgg16-397923af.pth','vgg.pth')),
                         'real pretrained LPIPS smoke requires cached VGG weights; no test-time download')
    def test_cached_pretrained_vgg_smoke(self):
        from gaussian_jscc.perceptual_metrics import LPIPSEvaluator
        torch.set_num_threads(1)
        metric=LPIPSEvaluator('cpu')
        image=torch.zeros(3,32,32)
        self.assertEqual(metric(image,image),0.)
        self.assertGreater(metric(image,torch.ones_like(image)),0.)

    def test_rgb_conversion_no_grad_and_rng_neutral_initialization(self):
        from gaussian_jscc.perceptual_metrics import LPIPSEvaluator
        rng = torch.get_rng_state().clone()
        with patch('gaussian_jscc.perceptual_metrics._load_lpips_model', side_effect=backend):
            metric = LPIPSEvaluator('cpu')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(not p.requires_grad for p in metric.model.parameters()))
        image = torch.full((3, 16, 16), 1.5, dtype=torch.float64, requires_grad=True)
        target = torch.full_like(image, -.5)
        self.assertEqual(metric(image, target), 4.)  # clamp -> +1 and -1, squared difference 4
        self.assertEqual(metric(image, image), 0.)
        self.assertIsNone(image.grad)
        self.assertEqual(metric.protocol['network'], 'vgg')
        self.assertEqual(metric.protocol['version'], '0.1')
        self.assertEqual(metric.protocol['input_range'], [-1, 1])

    def test_bad_inputs_and_missing_weights_fail_instead_of_fake_scores(self):
        from gaussian_jscc.perceptual_metrics import LPIPSEvaluator
        with patch('gaussian_jscc.perceptual_metrics._load_lpips_model', side_effect=backend):
            metric = LPIPSEvaluator('cpu')
        for image, target in ((torch.zeros(1, 16, 16), torch.zeros(1, 16, 16)),
                              (torch.zeros(3, 16, 16), torch.zeros(3, 8, 8)),
                              (torch.full((3, 16, 16), float('nan')), torch.zeros(3, 16, 16))):
            with self.assertRaises(ValueError):
                metric(image, target)
        with patch('gaussian_jscc.perceptual_metrics._load_lpips_model', side_effect=OSError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'pretrained.*weights'):
                LPIPSEvaluator('cpu')


if __name__ == '__main__':
    unittest.main()
