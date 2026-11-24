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
def assess_attributes(pred, target, cameras, device):
    ids = sample_ids(len(target))
    centers = torch.stack([camera.camera_center.to(device) for camera in cameras])
    sums = dict.fromkeys(FIELDS, 0.)
    for chunk in ids.split(4096):
        raw = target[chunk].to(device)
        errors = attribute_errors(pred[chunk].to(device), raw, camera_directions(raw[:, :3], centers))
        for key, value in errors.items():
            sums[key] += float(value.double().sum())
    return dict(points=len(ids), views=len(cameras), mse={key: value / len(ids) for key, value in sums.items()})
