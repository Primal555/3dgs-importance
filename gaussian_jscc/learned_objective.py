"""Explicit, bounded-slope auxiliaries for the fully learned codec.

These are engineering objectives, not a reproduced paper's tuned coefficients.
No inverse tiny Gaussian covariance, logarithmic XYZ tail, or decoder seed loss.
"""
import torch
from torch.nn import functional as F


def learned_terms(pred, target, model):
    def huber(x, y, beta=1.):
        return F.smooth_l1_loss(x, y, beta=beta, reduction='none').mean(-1)
    # .05 bbox units is a saved, configurable supervision scale, not a
    # transmitted coordinate or an assumed sufficient positional accuracy.
    geometry = huber(pred[:, :3]/model.cfg.xyz_loss_scale,
                     target[:, :3]/model.cfg.xyz_loss_scale, beta=.1)
    pa = pred[:, 3:] * model.attr_std + model.attr_mean
    ta = target[:, 3:] * model.attr_std + model.attr_mean
    # Quaternion sign ambiguity is handled without acos singularities.
    pq = F.normalize(pa[:, 4:8], dim=-1, eps=1e-4)
    tq = F.normalize(ta[:, 4:8], dim=-1, eps=1e-4)
    shape = torch.minimum((pq-tq).square().sum(-1), (pq+tq).square().sum(-1))
    terms = dict(geometry=geometry, shape=shape,
                 scale=huber(pred[:, 4:7], target[:, 4:7]),
                 opacity=huber(pa[:, :1].sigmoid(), ta[:, :1].sigmoid(), .1)
                         + .1*huber(pred[:, 3:4], target[:, 3:4]),
                 dc=huber(pred[:, 11:14], target[:, 11:14]))
    if pred.shape[-1] > 14:
        terms['sh'] = huber(pred[:, 14:], target[:, 14:])
    return terms


def projection_loss(raw, target, camera, near=.01):
    """Training-only source camera projection; safe depth and frustum gating.

    The source defines valid rows. Predicted negative depth cannot hide a row:
    positive-depth penalty remains, while the denominator is bounded below.
    Coordinate residual is measured in normalized image-plane units.
    """
    from math import tan
    matrix = camera.world_view_transform.to(raw)
    def view(x):
        return torch.cat((x[:, :3], torch.ones_like(x[:, :1])), -1) @ matrix
    src, dst = view(target).detach(), view(raw)
    fov = raw.new_tensor([tan(camera.FoVx/2), tan(camera.FoVy/2)])
    src_uv = src[:, :2] / src[:, 2:3].clamp_min(near) / fov
    valid = (src[:, 2] > near) & (src_uv.abs() < 1).all(-1)
    if not valid.any():
        return raw.sum()*0
    # Bound both slope and evaluation domain; do not drop invalid predictions.
    source_depth = src[:, 2:3].clamp_min(near)
    denominator = torch.maximum(dst[:, 2:3], .1*source_depth).clamp_min(near)
    uv = dst[:, :2] / denominator / fov
    xy = F.smooth_l1_loss(uv[valid], src_uv[valid], beta=.02)
    relative_depth = dst[:, 2] / source_depth[:, 0]
    depth = F.smooth_l1_loss(relative_depth[valid].clamp_max(.1),
                             torch.full_like(relative_depth[valid], .1), beta=.1)
    return xy + depth
