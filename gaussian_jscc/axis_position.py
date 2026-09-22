"""Experimental teacher-axis position objective; no scene-scale denominator.

Teacher axes and the once-fitted floor are training supervision only. Decoder
inputs, architecture, communication payload and attribute objectives are unchanged.
"""
import math
import numpy as np
import torch
from .covariance import unpack_symmetric


@torch.no_grad()
def fit_axis_floor(raw, percentile=1., override=None):
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError('axis floor percentile must be in [0,100]')
    axes = raw[:, 4:7].detach().double().exp().cpu().numpy()
    if not np.isfinite(axes).all() or (axes <= 0).any():
        raise ValueError('source Gaussian axis scales must be finite and positive')
    # NumPy supports large scenes exceeding torch.quantile's element limit.
    floor = float(np.quantile(axes, percentile/100)) if override is None else float(override)
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('axis floor must be finite and strictly positive')
    clipped = axes < floor
    return {'world': floor, 'percentile': percentile, 'method': 'explicit_world' if override is not None else 'source_axis_percentile',
            'source': 'all original PLY axis standard deviations, fitted once; never predicted axes or batch quantiles',
            'clamped_axis_fraction': float(clipped.mean()), 'affected_point_fraction': float(clipped.any(1).mean()),
            'source_axis_min_world': float(axes.min()), 'source_axis_max_world': float(axes.max()),
            'source_axis_median_world': float(np.median(axes)),
            'warning': 'engineering floor; neither pixel accuracy nor bounded network parameter gradients is guaranteed'}


def teacher_axis_position(pred, target, geometry, model, floor_world):
    if model.cfg.architecture != 'learned_split_logcov':
        raise ValueError('teacher-axis experiment requires logcov representation')
    if not math.isfinite(floor_world) or floor_world <= 0:
        raise ValueError('fit a fixed positive axis floor before training')
    with torch.no_grad():
        # Diagonalize log covariance, NOT exp(logcov): stable for thin splats.
        packed = target[:, 4:10].detach().double()*model.attr_std[1:7].detach().double()+model.attr_mean[1:7].detach().double()
        spectrum, axes = torch.linalg.eigh(unpack_symmetric(packed))
        scales = (.5*spectrum).exp()
        effective = scales.clamp_min(floor_world)
        if not torch.isfinite(effective).all() or (scales <= 0).any():
            raise FloatingPointError('invalid teacher scales')
    # Subtract normalized centers first to avoid world-origin cancellation.
    delta = (pred[:, :3].double()-target[:, :3].detach().double())*geometry.span.to(pred).double()
    local = torch.einsum('nij,ni->nj', axes, delta)
    u = local/effective
    squared = u.square().sum(-1)
    # Rationalized pseudo-Huber: zero-safe, linear tail, no norm-at-zero singularity.
    per_point = squared/((1+squared).sqrt()+1)
    loss = per_point.mean().to(pred.dtype)
    with torch.no_grad():
        radius = squared.sqrt()
        native_radius = (local/scales).square().sum(-1).sqrt()
        clipped = scales < floor_world
        stats = {'axis_position_response': float(loss.detach()),
                 'axis_floor_world': float(floor_world),
                 'axis_clamped_fraction': float(clipped.double().mean()),
                 'axis_affected_point_fraction': float(clipped.any(-1).double().mean()),
                 'axis_effective_radius_p50': float(radius.median()),
                 'axis_effective_radius_p95': float(torch.quantile(radius, .95)),
                 'axis_native_radius_p50': float(native_radius.median()),
                 'axis_native_radius_p95': float(torch.quantile(native_radius, .95)),
                 'axis_effective_within_one_fraction': float((radius <= 1).double().mean()),
                 'axis_native_within_one_fraction': float((native_radius <= 1).double().mean()),
                 'axis_thin_effective_abs_p50': float(u[:, 0].abs().median()),
                 'axis_thin_native_abs_p50': float((local[:, 0]/scales[:, 0]).abs().median()),
                 # Individual XYZ output gradient, BEFORE averaging points and /3.
                 'axis_world_gradient_bound': 1./float(floor_world)}
    return loss, stats
