"""Physical attribute supervision for the 8/24/48-symbol channel layers."""
import math
from statistics import fmean

import torch
from torch.nn.functional import pad

from utils.sh_utils import C0, eval_sh

from .local_response import camera_directions
from .semantic_attribute_codec import shape_opacity_errors


FIELDS = ('shape', 'opacity', 'base_rgb', 'high_order_rgb')
WEIGHTS = {
    1: dict(zip(FIELDS, (.35, .25, .30, .10))),
    2: dict(zip(FIELDS, (.30, .20, .25, .25))),
    3: dict(zip(FIELDS, (.25, .20, .20, .35))),
}
EQUAL_WEIGHTS = {tier: dict.fromkeys(FIELDS, 0.25) for tier in WEIGHTS}
SCALE_FLOOR = 1e-4
SAMPLE_LIMIT = 32768


def attribute_errors(pred, target, directions):
    errors = shape_opacity_errors(pred, target)
    delta = pred[:, 11:] - target[:, 11:]
    errors['base_rgb'] = (C0 * delta[:, :3]).square().mean(-1)
    # The degree-0 difference has its own loss. The remaining SH terms are linear.
    high_order = pad(delta[:, 3:].reshape(-1, 3, 15), (1, 0))
    errors['high_order_rgb'] = eval_sh(3, high_order[:, None], directions).square().mean((-1, -2))
    return errors


def grouped_objective(pred, target, directions, scales, tier, *, equal_weights=False):
    errors = attribute_errors(pred, target.detach(), directions)
    terms = {name: errors[name].mean() / scales[name] for name in FIELDS}
    weights = EQUAL_WEIGHTS if equal_weights else WEIGHTS
    if isinstance(tier, int):
        loss = sum(weights[tier][name] * value for name, value in terms.items())
    else:
        table = pred.new_tensor([[0.] * len(FIELDS), *[[row[name] for name in FIELDS] for row in weights.values()]])
        point_weights = table[tier]
        loss = sum((point_weights[:, i] * errors[name]).mean() / scales[name] for i, name in enumerate(FIELDS))
    return loss, terms


def sample_ids(count):
    return torch.linspace(0, count-1, min(count, SAMPLE_LIMIT), dtype=torch.float64).long()


@torch.no_grad()
def fit_color_scales(train_raws):
    base, high = [], []
    for raw in train_raws:
        coeff = raw[sample_ids(len(raw)), 11:].double()
        base.append(float((C0 * coeff[:, :3]).var(0, unbiased=False).mean()))
        sh = coeff[:, 3:].reshape(-1, 3, 15)
        high.append(float(sh.var(0, unbiased=False).sum(-1).mean() / (4 * math.pi)))
    return {'base_rgb': max(fmean(base), SCALE_FLOOR), 'high_order_rgb': max(fmean(high), SCALE_FLOOR)}


@torch.no_grad()
def assess_attributes(pred, target, cameras, device, *, detailed=False):
    ids = sample_ids(len(target))
    centers = torch.stack([camera.camera_center.to(device) for camera in cameras])
    sums = dict.fromkeys(FIELDS, 0.)
    for chunk in ids.split(4096):
        raw = target[chunk].to(device)
        errors = attribute_errors(pred[chunk].to(device), raw, camera_directions(raw[:, :3], centers))
        for key, value in errors.items():
            sums[key] += float(value.double().sum())
    result = dict(points=len(ids), views=len(cameras), mse={key: value / len(ids) for key, value in sums.items()})
    if detailed:
        from .benchmark import parameter_metrics
        source, recovered = target[ids].to(device), pred[ids].to(device)
        parameters = parameter_metrics(source, recovered)
        details = {key: parameters[key] for key in ('sorted_log_scale_rmse', 'log_volume_bias',
                   'rotation_angle_mean_deg', 'rotation_angle_p95_deg', 'opacity_alpha_mae')}
        # Three color channels, each with 3, 5 and 7 coefficients for SH degrees 1, 2 and 3.
        sh = (recovered[:, 14:] - source[:, 14:]).reshape(-1, 3, 15)
        for degree, band in enumerate(sh.split((3, 5, 7), dim=-1), start=1):
            details[f'sh_degree_{degree}_coefficient_mse'] = float(band.double().square().mean())
        result['details'] = details
    return result


def summarize_attribute_recovery(clean, rows, actual):
    """Average each metric over trials for the same layout, points and camera directions."""
    def metrics(errors):
        return {**{k+'_mse': v for k, v in errors['mse'].items()}, **errors['details']}

    tiers = {}
    for tier in dict.fromkeys(r['layer'] for r in rows):
        finite, = [r for r in rows if r['layer'] == tier and r['path'] == 'finite_noiseless']
        baseline = metrics(finite['attribute_errors'])
        noisy = [r for r in rows if r['layer'] == tier and r['path'] == 'awgn']
        used = [metrics(r['attribute_errors']) for r in noisy
                if not actual or r['actual_delivery']['status'] == 'success']
        mean = {name: fmean(trial[name] for trial in used) for name in baseline} if used else None
        tiers[str(tier)] = dict(finite_noiseless=baseline, awgn_mean=mean,
            awgn_minus_finite={k: v-baseline[k] for k, v in mean.items()} if mean else None,
            awgn_trials_used=len(used), awgn_trials_attempted=len(noisy))
    return dict(reference='source PLY attributes', sampling='fixed evenly spaced indices in scene Morton order',
        points=clean['points'], views=clean['views'], clean=metrics(clean), tiers=tiers,
        awgn_basis='successful actual deliveries' if actual else 'reliable side files')
