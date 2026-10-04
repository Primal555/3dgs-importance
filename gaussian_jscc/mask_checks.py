"""Real-CUDA forward and inactive-mask checks for local allocation training."""
import torch

from .rendering import render
from .render_objective import image_distortion


def check_masked_renderer(raw, camera, degree, white_background=False, candidates=256,
                          max_forward_error=1e-3):
    """A hard-zero shadow must not change the deployed ordinary render."""
    if raw.device.type != 'cuda' or not len(raw):
        raise ValueError('mask renderer check needs a nonempty CUDA scene')
    try:
        import mask_diff_gaussian_rasterization  # noqa: F401
    except ImportError as error:
        raise RuntimeError('joint allocation needs the MaskGaussian CUDA rasterizer; '
                           'build/install submodules/mask-diff-gaussian-rasterization') from error
    count = min(candidates, len(raw))
    with torch.no_grad():
        deployed = render(raw, camera, degree, white_background)
    shadow = torch.cat((raw, raw[:count]), 0)
    zeros = raw.new_zeros(count, requires_grad=True)
    existence = torch.cat((raw.new_ones(len(raw)), zeros))
    masked = render(shadow, camera, degree, white_background, existence)
    difference = (masked.detach() - deployed).abs()
    image_target = camera.original_image[:3].to(raw.device)
    image_loss = image_distortion(masked, image_target)
    shadow_gradient = torch.autograd.grad(image_loss, zeros)[0]
    stats = {'ordinary_vs_masked_max_abs': float(difference.max()),
             'ordinary_vs_masked_mse': float(difference.square().mean()),
             'zero_mask_gradient_norm': float(shadow_gradient.norm()),
             'zero_mask_nonzero_count': int((shadow_gradient != 0).sum()),
             'checked_zero_mask_candidates': count,
             'note': 'Nonzero gradient count depends on whether sampled candidates are visible; synthetic CPU tests cover the derivative path.'}
    if stats['ordinary_vs_masked_max_abs'] > max_forward_error:
        raise RuntimeError('MaskGaussian renderer changes hard-forward image beyond tolerance: '
                           f'{stats}; do not train with a mismatched visual objective')
    return stats
