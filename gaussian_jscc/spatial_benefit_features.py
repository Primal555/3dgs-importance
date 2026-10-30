"""Spatially paired block tokens and cached, ordered footprint observations."""
import time

import torch
from torch.nn import functional as F

from .physical_benefit_features import PhysicalWindow, candidate_features, _state_attributes, _summary
from .pixel_benefit_features import exact_values

CONTEXT_DIM = 73
STATE_DIM = 26


def image_samples(fields, xy):
    """Bilinear image samples at pixel centers; retain the order of all samples."""
    h, w = fields.shape[-2:]
    shape = xy.shape[:-1]
    scale = xy.new_tensor([w, h])
    grid = (2 * (xy + .5) / scale - 1).reshape(1, 1, -1, 2)
    sampled = F.grid_sample(fields[None], grid, align_corners=False, padding_mode='zeros')
    return sampled[0, :, 0].T.reshape(*shape, fields.shape[0])


@torch.no_grad()
def spatial_inputs(static, stats, raw, projected, point, tile, target, changed, d_ref, lambda_s,
                   cache_key=None):
    """Use every block point, preserving location/candidate/residual pairing.

    The ordered 3x3 baseline profile and individual visibility describe the
    same point as each candidate token. Candidate center samples account for
    shifted footprints. Neither sampling nor pooling defines the exact label.
    """
    features = candidate_features(static, stats, raw, projected, point,
                                 tile.height, tile.width, d_ref, lambda_s)
    fields = getattr(tile, '_spatial_fields', None)
    if fields is None:
        fields = torch.cat((target.to(tile.image), tile.image-target.to(tile.image)), 0).contiguous()
        tile._spatial_fields = fields
    baseline = tile.projected[changed]
    cache = getattr(tile, '_spatial_profiles', None)
    if cache is None:
        cache = {}; tile._spatial_profiles = cache
    profile = cache.get(cache_key) if cache_key is not None else None
    if profile is None:
        axis = baseline.new_tensor([-.5, 0., .5])
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        offsets = torch.stack((xx.flatten(), yy.flatten()), -1)
        xy = baseline[:, None, :2] + baseline[:, None, 11:12].clamp_min(0)*offsets[None]
        profile = image_samples(fields, xy).flatten(1)
        if cache_key is not None:
            cache[cache_key] = profile
    size = baseline.new_tensor([tile.width, tile.height])
    flag = baseline.new_zeros(len(baseline), 1); flag[point] = 1
    context = torch.cat((_summary(stats, tile.height*tile.width), baseline[:, :2]/size,
        (baseline[:, :2]-baseline[point, :2])/size,
        (baseline[:, 2:3]-baseline[point, 2:3]).sign() *
            (baseline[:, 2:3]-baseline[point, 2:3]).abs().log1p(),
        baseline[:, 11:12]/max(tile.height, tile.width), flag, profile), -1)
    exists = projected[..., 7:8]
    states = torch.cat((_state_attributes(raw, projected), projected[..., :2]/size*exists,
                        image_samples(fields, projected[..., :2])*exists), -1)
    # Fixed baseline weights, shared by all four states. Uniform backup is
    # essential for candidate contributors with zero baseline visibility.
    weights = stats[:, 0].clamp_min(0) + .05*stats[:, 4].clamp_min(0)
    weights = .9*weights/weights.sum().clamp_min(1e-12) + .1/max(1, len(weights))
    if (context.shape != (len(baseline), CONTEXT_DIM) or states.shape != (4, len(baseline), STATE_DIM)
            or not all(torch.isfinite(v).all() for v in (context, states, weights))):
        raise ValueError('invalid spatially paired input')
    return dict(features=features, context=context, states=states, weights=weights)


class SpatialWindow(PhysicalWindow):
    def build_view(self, tile, changed, projected, target, *, source, block_ids, raw, with_labels):
        start = time.perf_counter()
        point = int((block_ids == source).nonzero(as_tuple=True)[0].item())
        inputs = spatial_inputs(self.static[source], tile._physical_stats[changed], raw, projected, point,
            tile, target, changed, self.reference['d_ref'], self.reference['lambda_s'],
            cache_key=tuple(block_ids.tolist()))
        result = dict(inputs=inputs, height=tile.height, width=tile.width, d_ref=self.reference['d_ref'],
                      tile_count=0, feature_seconds=time.perf_counter()-start)
        if with_labels:
            start = time.perf_counter()
            result['values'] = exact_values(tile, changed, projected, target,
                self.reference['lambda_s'], self.reference['d_ref']).cpu()
            result['tile_count'] = len(tile.affected_tiles(changed, projected, halo_tiles=1))
            result['teacher_seconds'] = time.perf_counter()-start
        return result
