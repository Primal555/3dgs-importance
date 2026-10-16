"""Evaluation-only LPIPS: frozen VGG, explicit RGB range, no codec changes."""
import math
import torch
from .optimization import preserved_rng


def _load_lpips_model():
    # Reuse the repository implementation, not its functional helper which
    # constructs/downloads a new network for EVERY image pair.
    from lpipsPyTorch.modules.lpips import LPIPS
    return LPIPS(net_type='vgg', version='0.1')


def image_metric_protocol(lpips=False):
    return {
        'psnr': {'input': 'unclipped renderer RGB', 'peak': 1., 'higher_is_better': True},
        'ssim': {'input': 'RGB clamped to [0,1]', 'implementation': 'utils.loss_utils.ssim',
                 'higher_is_better': True},
        'lpips': ({'implementation': 'lpipsPyTorch', 'network': 'vgg', 'version': '0.1',
                   'input': 'RGB clamped to [0,1], then 2*x-1; no resize or alignment',
                   'input_range': [-1, 1], 'lower_is_better': True, 'evaluation_only': True}
                  if lpips else {'enabled': False}),
        'aggregation': 'mean per view, then mean per noise trial; std across trial view-means',
    }


class LPIPSEvaluator:
    """One pretrained network per evaluation pass, reused for all image pairs.

    Never attached to the codec or its optimizer/checkpoint. Construct inside
    evaluate_paths so it is released before training resumes. Missing weights
    are fatal when requested: an absent metric must never masquerade as zero.
    """
    def __init__(self, device):
        self.device = torch.device(device)
        self.protocol = image_metric_protocol(True)['lpips']
        try:
            with preserved_rng(self.device):
                self.model = _load_lpips_model().to(device=self.device, dtype=torch.float32)
                self.model.eval().requires_grad_(False)
        except Exception as exc:
            raise RuntimeError(
                'Cannot load pretrained LPIPS/VGG weights. Ensure torchvision matches torch '
                'and allow the first weight download (torch hub cache), or explicitly use '
                '--no-lpips. No substitute score was produced. Preflight: '
                'python -m gaussian_jscc.perceptual_metrics --device cpu'
            ) from exc

    @torch.no_grad()
    def __call__(self, image, target):
        if image.shape != target.shape or image.ndim != 3 or image.shape[0] != 3 or not image.numel():
            raise ValueError('LPIPS requires equally sized, nonempty 3xHxW RGB images')
        if not torch.isfinite(image).all() or not torch.isfinite(target).all():
            raise ValueError('LPIPS input RGB must be finite')
        # Vendored LPIPS expects [-1,1], and its reduction is single-pair only.
        a = image.detach().to(self.device, dtype=torch.float32).clamp(0, 1).mul(2).sub(1)[None]
        b = target.detach().to(self.device, dtype=torch.float32).clamp(0, 1).mul(2).sub(1)[None]
        value = self.model(a, b)
        if value.numel() != 1 or not math.isfinite(float(value)):
            raise RuntimeError('LPIPS produced a nonfinite or nonscalar score')
        return float(value)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download/check frozen LPIPS VGG weights before a long run')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    metric = LPIPSEvaluator(args.device)
    image = torch.zeros(3, 32, 32, device=args.device)
    print({'protocol': metric.protocol, 'identical_image_lpips': metric(image, image)})
