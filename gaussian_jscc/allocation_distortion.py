"""Fixed, calibrated image objective for the structured allocation experiment.

Reporting RGB MSE is unchanged. Structural comparisons use double-precision
local maps, including paired finite differences; no subtraction of rounded
whole-image SSIM scores.
"""
from functools import lru_cache
import math
import torch
from torch.nn import functional as F
from .allocation_finite import PairedMaskedRenderTask


@lru_cache(maxsize=16)
def _window(channels, device):
    # Match the existing reporting window coefficients before promotion.
    from utils.loss_utils import create_window
    return create_window(11, channels).to(device=device, dtype=torch.float64)


def _target_stats(target):
    y = target.detach().clamp(0, 1).double()
    w = _window(y.shape[-3], str(y.device))
    my = F.conv2d(y[None], w, padding=5, groups=y.shape[-3])[0]
    vy = F.conv2d((y*y)[None], w, padding=5, groups=y.shape[-3])[0]-my*my
    return my, vy


def ssim_map(image, target, target_stats=None):
    x, y = image.clamp(0, 1).double(), target.detach().clamp(0, 1).double()
    w = _window(x.shape[-3], str(x.device))
    def blur(t): return F.conv2d(t.unsqueeze(0), w, padding=5, groups=x.shape[-3])[0]
    mx = blur(x)
    my, vy = _target_stats(target) if target_stats is None else target_stats
    vx, cov = blur(x*x)-mx*mx, blur(x*y)-mx*my
    return ((2*mx*my+.01**2)*(2*cov+.03**2)/
            ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))


def calibrate(mse, structure, *, mode='mse_ssim', share=.2, lambda_s=None):
    if (mode not in ('mse', 'mse_ssim') or not 0 < share < 1 or
            any(not math.isfinite(v) or v < 0 for v in (mse, structure))):
        raise ValueError('invalid distortion calibration')
    if mode == 'mse':
        if lambda_s not in (None, 0.): raise ValueError('MSE mode requires zero structural weight')
        weight, method = 0., 'mse'
    elif lambda_s is not None:
        if not math.isfinite(lambda_s) or lambda_s < 0: raise ValueError('invalid lambda_s')
        weight, method = lambda_s, 'manual'
    else:
        if min(mse, structure) <= 1e-12:
            raise ValueError('degenerate q3 reference; specify --lambda-s explicitly')
        weight, method = share/(1-share)*mse/structure, 'q3_calibration_share'
    raw = mse+weight*structure
    return dict(distortion=mode, lambda_s=weight, lambda_method=method,
        requested_structure_share=share, mse_ref=mse, structure_ref=structure,
        structure_reference_fraction=weight*structure/max(raw, 1e-12),
        d_ref_raw=raw, d_ref=max(raw, 1e-12), d_ref_floor=1e-12,
        ssim_protocol='clamped RGB; 11x11 sigma1.5; zero padding; float64 maps')


class ImageDistortion:
    def __init__(self, mode='mse_ssim', lambda_s=0.):
        if mode not in ('mse', 'mse_ssim') or not math.isfinite(lambda_s) or lambda_s < 0:
            raise ValueError('invalid image distortion')
        if mode == 'mse' and lambda_s != 0: raise ValueError('MSE structural weight must be zero')
        self.mode, self.lambda_s = mode, lambda_s

    def identity(self): return dict(distortion=self.mode, lambda_s=self.lambda_s, protocol='ssim_map64_v1')

    def components(self, image, target):
        mse = (image.double()-target.detach().double()).square().mean()
        structure = (1-ssim_map(image, target).mean()).clamp_min(0) if self.mode != 'mse' else mse*0
        return torch.stack((mse, structure))

    def combine(self, components): return components[..., 0]+self.lambda_s*components[..., 1]

    def prepare(self, base, target):
        if self.mode == 'mse': return None
        stats = _target_stats(target)
        return dict(target_stats=stats, base_map=ssim_map(base, target, stats).detach())

    def delta(self, image, base, target, prepared=None):
        delta = image.double()-base.double()
        mse = (2*(base.double()-target.double())*delta+delta.square()).mean()
        if self.mode == 'mse': return torch.stack((mse, mse*0))
        cached = self.prepare(base, target) if prepared is None else prepared
        old = cached['base_map']
        new = ssim_map(image, target, cached['target_stats'])
        # SSIM <= 1 theoretically. Match the clamp in components also at equality.
        ds = (old-new).mean()
        if old.mean() > 1 or new.mean() > 1:
            ds = (1-new.mean()).clamp_min(0)-(1-old.mean()).clamp_min(0)
        return torch.stack((mse, ds))


class ComponentRenderTask(PairedMaskedRenderTask):
    def __init__(self, *args, distortion, **kwargs):
        super().__init__(*args, **kwargs)
        self.distortion = distortion

    def component_gradient(self, scene, timings=None):
        from .rendering import render
        from .budgeted_allocation_kernel import timing
        gradients = [torch.zeros_like(scene), torch.zeros_like(scene)]
        values, self.baseline = [], []
        for i, camera in enumerate(self.cameras):
            with timing(timings, f'view_{i}_render_seconds'):
                image = render(scene[:, :-1], camera, self.degree, self.white_background, scene[:, -1])
                target = self.reference.get(camera, scene.device)
                parts = self.distortion.components(image, target)
            if not torch.isfinite(parts).all(): raise RuntimeError('nonfinite image components')
            with timing(timings, f'view_{i}_backward_seconds'):
                for j in range(2 if self.distortion.mode != 'mse' else 1):
                    grad, = torch.autograd.grad(parts[j]+scene.sum()*0, scene,
                        retain_graph=(j == 0 and self.distortion.mode != 'mse'))
                    gradients[j].add_(grad, alpha=1/len(self.cameras))
            base = image.detach(); target = target.detach()
            with torch.no_grad(): prepared = self.distortion.prepare(base, target)
            self.baseline.append((base, target, prepared)); values.append(parts.detach())
        mean = torch.stack(values).mean(0)
        self.component_upstream = gradients
        self.stats = dict(view_mse=[float(v[0]) for v in values], views_per_step=len(values),
            source_mse=float(mean[0]), source_structure_error=float(mean[1]),
            source_distortion=float(self.distortion.combine(mean)), lambda_s=self.distortion.lambda_s)
        return float(self.distortion.combine(mean)), gradients[0]+self.distortion.lambda_s*gradients[1]

    @torch.no_grad()
    def delta_components(self, scene):
        from .rendering import render
        if len(self.baseline) != len(self.cameras): raise RuntimeError('incomplete baseline window')
        values = []
        for camera, (base, target, prepared) in zip(self.cameras, self.baseline):
            image = render(scene[:, :-1], camera, self.degree, self.white_background, scene[:, -1])
            values.append(self.distortion.delta(image, base, target, prepared))
        result = torch.stack(values).mean(0)
        if not torch.isfinite(result).all(): raise RuntimeError('nonfinite paired components')
        return result
