"""Candidate-region observations from known projected attributes and baseline RGB."""
import time

import torch

from .aligned_spatial_inputs import AlignedSpatialWindow
from .pixel_local_renderer import _rectangles
from .spatial_benefit_features import SpatialWindow, image_samples

PATCH = 9
CHANNELS = 15


def _alpha(p, xy, height, width):
    delta = p[:, None, :2]-xy[None]
    power = -.5*(p[:, None, 3]*delta[..., 0].square()+p[:, None, 5]*delta[..., 1].square())
    power -= p[:, None, 4]*delta[..., 0]*delta[..., 1]
    alpha = (p[:, None, 6]*power.exp()).clamp_max(.99)*p[:, None, 7]
    low, high = _rectangles(p, height, width)
    tile = torch.floor(xy/16).long()
    inside = ((tile[None] >= low[:, None]) & (tile[None] < high[:, None])).all(-1)
    return torch.where(inside & (power <= 0) & (alpha >= 1/255), alpha, 0.)


@torch.no_grad()
def footprint_inputs(tile, target, projected, changed, point):
    """Two aligned region grids: target support and the whole changed block.

    RGB and residuals are sampled from the unchanged baseline/reference.
    Alpha/RGB coverage channels are analytic descriptors, not intervention renders.
    """
    h, w = tile.height, tile.width
    baseline = tile.projected[changed]
    all_states = torch.cat((baseline[None], projected), 0)
    key = (target.data_ptr(), target._version, tile.image._version)
    if getattr(tile, '_footprint_fields_key', None) != key:
        tile._footprint_fields = torch.cat((target.to(tile.image), tile.image-target.to(tile.image)), 0)
        tile._footprint_fields_key = key
    fields = tile._footprint_fields
    grids, validity, areas = [], [], []
    for p in (all_states[:, point:point+1], all_states):
        flat = p.reshape(-1, 12)
        low, high = _rectangles(flat, h, w)
        good = (high > low).all(-1) & (flat[:, 7] > 0)
        if good.any():
            lo = (low[good].amin(0)*16-5).clamp_min(0)
            hi = torch.minimum(high[good].amax(0)*16+5, low.new_tensor([w, h]))
        else:
            lo, hi = low.new_zeros(2), low.new_zeros(2)
        extent = hi-lo
        axis = (torch.arange(PATCH, device=flat.device, dtype=flat.dtype)+.5)/PATCH
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = lo.to(flat)+torch.stack((xx, yy), -1)*extent.clamp_min(1).to(flat)-.5
        valid = ((grid[..., 0] >= -.5) & (grid[..., 0] < w-.5) &
                 (grid[..., 1] >= -.5) & (grid[..., 1] < h-.5) & (extent > 0).all())
        grids.append(grid); validity.append(valid)
        areas.append((extent.prod().to(flat)/(PATCH*PATCH*h*w)).expand(PATCH, PATCH, 1))
    xy = torch.stack(grids)
    valid = torch.stack(validity)
    observed = image_samples(fields, xy)
    coordinates = xy/xy.new_tensor([w, h])
    geometry = torch.cat((coordinates, torch.stack(areas)), -1)
    flat_xy = xy.reshape(-1, 2)
    base_alpha = _alpha(baseline, flat_xy, h, w).sum(0).reshape(2, PATCH, PATCH)
    maps = []
    for state in projected:
        alpha = _alpha(state, flat_xy, h, w)
        colour = (alpha[..., None]*state[:, None, 8:11]).sum(0).reshape(2, PATCH, PATCH, 3)
        mass = alpha.sum(0).reshape(2, PATCH, PATCH, 1)
        value = torch.cat((observed, colour, mass, base_alpha[..., None], valid[..., None], geometry), -1)
        maps.append((value*valid[..., None]).permute(0, 3, 1, 2))
    result = torch.stack(maps)
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite candidate-region features')
    return result


class FootprintWindow(SpatialWindow):
    # Preserve original PLY-row identities in both static descriptors and probes.
    prepare_camera = AlignedSpatialWindow.prepare_camera
    capture_inputs = AlignedSpatialWindow.capture_inputs

    def build_view(self, tile, changed, projected, target, *, source, block_ids, raw, with_labels):
        view = super().build_view(tile, changed, projected, target, source=source,
            block_ids=block_ids, raw=raw, with_labels=with_labels)
        started = time.perf_counter()
        point = int((block_ids == source).nonzero(as_tuple=True)[0].item())
        view['inputs']['footprints'] = footprint_inputs(tile, target, projected, changed, point)
        view['feature_seconds'] += time.perf_counter()-started
        return view


def verify_footprints(device):
    from .pixel_local_renderer import TileScene
    points = torch.tensor([[7,8,1,.4,0,.4,.7,1,.3,.4,.5,5],
                           [28,4,2,.2,.01,.2,.6,1,.2,.7,.8,7]], dtype=torch.float32)
    candidates = points[None].repeat(4,1,1)
    candidates[0,0,7] = 0
    candidates[2,0,11] = 9
    candidates[3,1,8:11] += .2
    target = torch.linspace(0,1,3*17*31).reshape(3,17,31)
    results = []
    for selected in ('cpu', device):
        tile = TileScene.from_projected(points.to(selected),17,31,backend=torch.device(selected).type)
        results.append(footprint_inputs(tile, target.to(selected), candidates.to(selected),
                                       torch.arange(2,device=selected),0).cpu())
    torch.testing.assert_close(results[0],results[1],rtol=2e-4,atol=2e-4)
    return dict(footprint_cpu_device_max_abs=float((results[0]-results[1]).abs().max()),
                footprint_backend=torch.device(device).type)
