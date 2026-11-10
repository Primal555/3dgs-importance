"""Opt-in PointNet++-style MSG/feature propagation inside bounded Morton blocks.

All sampling is internal: one output remains for each input Gaussian. Geometry
is detached, never inferred from attributes. No custom CUDA extension is needed.
"""
import torch
from torch import nn
from contextlib import contextmanager


def gather(values, indices):
    batch = torch.arange(len(values), device=values.device)
    return values[batch.reshape((-1,) + (1,)*(indices.ndim-1)), indices]


def metric_coordinates(unit_xyz, span, active):
    # A single scalar preserves world-space neighbor ordering, unlike per-axis
    # bbox normalization. Translation is irrelevant to distances and offsets.
    return unit_xyz.detach().masked_fill(~active[..., None], 0) * (span / span.max())


@torch.no_grad()
def farthest_points(xyz, active, count):
    """Deterministic FPS, distinct active slots; surplus anchors are masked."""
    batch, n, _ = xyz.shape
    count = min(count, n)
    center = xyz.sum(1) / active.sum(1).clamp_min(1)[:, None]
    distance = (xyz-center[:, None]).square().sum(-1)
    remaining = active.clone()
    minimum = torch.full_like(distance, float('inf'))
    indices, valid = [], []
    for _ in range(count):
        index = distance.masked_fill(~remaining, -1).argmax(-1)
        valid.append(remaining.any(-1)); indices.append(index)
        remaining.scatter_(1, index[:, None], False)
        selected = gather(xyz, index[:, None])
        minimum = torch.minimum(minimum, (xyz-selected).square().sum(-1))
        distance = minimum
    return torch.stack(indices, 1), torch.stack(valid, 1)


@torch.no_grad()
def spacing(xyz, active):
    """Median nearest distinct-position distance at this resolution.

    Ignore coincident centers when estimating radius. Fully coincident/empty
    groups use a finite floor; masks still decide whether they contribute.
    """
    distance = torch.cdist(xyz.float(), xyz.float(), compute_mode='donot_use_mm_for_euclid_dist')
    distance = distance.masked_fill(~active[:, None, :] | (distance <= 1e-8), float('inf'))
    nearest = distance.amin(-1).masked_fill(~active, float('inf'))
    nearest = nearest.masked_fill(~torch.isfinite(nearest), float('nan'))
    return nearest.nanmedian(-1).values.nan_to_num(nan=1e-6).clamp_min(1e-6)


@torch.no_grad()
def radius_neighbors(xyz, centers, active, center_active, radius, count):
    distance = torch.cdist(centers.float(), xyz.float(), compute_mode='donot_use_mm_for_euclid_dist')
    distance = distance.masked_fill(~active[:, None, :], float('inf'))
    chosen, indices = distance.topk(min(count, xyz.shape[1]), dim=-1, largest=False)
    valid = (chosen <= radius[..., None]) & center_active[..., None] & torch.isfinite(chosen)
    relative = (gather(xyz, indices)-centers[:, :, None]) / radius[..., None, None].clamp_min(1e-8)
    return indices, valid, relative.masked_fill(~valid[..., None], 0)


def mlp(in_dim, hidden, out_dim=None):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim or hidden))


class SetAbstraction(nn.Module):
    def __init__(self, hidden, centers, neighbors, radii):
        super().__init__()
        self.centers, self.neighbors, self.radii = centers, neighbors, radii
        self.branches = nn.ModuleList(mlp(hidden+4, hidden) for _ in radii)
        self.merge = mlp(len(radii)*hidden, hidden)

    def geometry(self, xyz, active):
        ids, valid_centers = farthest_points(xyz, active, self.centers)
        centers = gather(xyz, ids).masked_fill(~valid_centers[..., None], 0)
        base_radius = spacing(xyz, active)[:, None].expand_as(valid_centers)
        neighbors = [radius_neighbors(xyz, centers, active, valid_centers, base_radius*m, k)
                     for m, k in zip(self.radii, self.neighbors)]
        return centers, valid_centers, neighbors

    def forward(self, h, xyz, active, geometry=None):
        centers, valid_centers, neighbors = self.geometry(xyz, active) if geometry is None else geometry
        scales = []
        for branch, (idx, valid, relative) in zip(self.branches, neighbors):
            features = torch.cat((gather(h, idx), relative, relative.norm(dim=-1, keepdim=True)), -1)
            values = branch(features).masked_fill(~valid[..., None], -torch.finfo(h.dtype).max)
            pooled = values.amax(2).masked_fill(~valid.any(-1)[..., None], 0)
            scales.append(pooled)
        return self.merge(torch.cat(scales, -1))*valid_centers[..., None], centers, valid_centers


class FeaturePropagation(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fuse = mlp(2*hidden, hidden)

    def geometry(self, xyz, active, centers, center_active):
        with torch.no_grad():
            distance = torch.cdist(xyz.float(), centers.float(), compute_mode='donot_use_mm_for_euclid_dist')
            distance = distance.masked_fill(~center_active[:, None, :], float('inf'))
            d, ids = distance.topk(min(3, centers.shape[1]), dim=-1, largest=False)
            weights = d.clamp_min(1e-8).reciprocal() * active[..., None]
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return ids, weights

    def forward(self, fine, xyz, active, coarse, centers, center_active, geometry=None):
        ids, weights = self.geometry(xyz, active, centers, center_active) if geometry is None else geometry
        interpolated = (gather(coarse, ids)*weights[..., None]).sum(2)
        return self.fuse(torch.cat((fine, interpolated), -1))*active[..., None]


class SpatialPointContext(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sa1 = SetAbstraction(cfg.hidden, cfg.context_centers[0], cfg.context_neighbors, cfg.context_radii)
        self.sa2 = SetAbstraction(cfg.hidden, cfg.context_centers[1], cfg.context_neighbors, cfg.context_radii)
        self.fp1, self.fp0 = FeaturePropagation(cfg.hidden), FeaturePropagation(cfg.hidden)
        self.gate = nn.Parameter(torch.full((cfg.hidden,), .1))
        self.register_buffer('scene_span', torch.ones(3), persistent=False)
        self.geometry_ready = False

    def set_scene_geometry(self, geometry):
        self.scene_span.copy_(geometry.span.to(self.scene_span))
        self.geometry_ready = True

    def forward(self, h, unit_xyz, active):
        if not self.geometry_ready:
            raise ValueError('PointNet context requires scene bbox geometry from metadata')
        h = h.masked_fill(~active[..., None], 0)
        if h.shape[1] == 0:
            return h
        xyz = metric_coordinates(unit_xyz, self.scene_span, active)
        cache = getattr(self, '_allocation_geometry_cache', None)
        plan = cache.get(self, xyz, active) if cache is not None else (None,)*4
        h1, x1, a1 = self.sa1(h, xyz, active, plan[0])
        h2, x2, a2 = self.sa2(h1, x1, a1, plan[1])
        up1 = self.fp1(h1, x1, a1, h2, x2, a2, plan[2])
        up0 = self.fp0(h, xyz, active, up1, x1, a1, plan[3])
        return (h + self.gate.tanh()*up0)*active[..., None]


class _AllocationGeometryCache:
    """One candidate batch's detached geometry; never reused across updates."""
    def __init__(self, block_rows):
        self.block_rows = block_rows
        self.entries = {}
        self.builds = self.hits = 0

    @torch.no_grad()
    def get(self, module, xyz, active):
        n = min(self.block_rows, len(xyz))
        repeat = len(xyz)//n
        base_x, base_a = xyz[:n], active[:n]
        if len(xyz)%n or not torch.equal(xyz, base_x.repeat(repeat, 1, 1)) or not torch.equal(active, base_a.repeat(repeat, 1)):
            # Different keep sets/positions must receive their own neighborhood.
            n, repeat, base_x, base_a = len(xyz), 1, xyz, active
        previous = self.entries.get(module)
        if previous is not None and torch.equal(previous[0], base_x) and torch.equal(previous[1], base_a):
            plan = previous[2]; self.hits += 1
        else:
            sa1 = module.sa1.geometry(base_x, base_a)
            sa2 = module.sa2.geometry(sa1[0], sa1[1])
            plan = (sa1, sa2, module.fp1.geometry(sa1[0], sa1[1], sa2[0], sa2[1]),
                    module.fp0.geometry(base_x, base_a, sa1[0], sa1[1]))
            self.entries[module] = (base_x.clone(), base_a.clone(), plan)
            self.builds += 1
        def expand(value):
            if isinstance(value, torch.Tensor):
                return value if repeat == 1 else value.repeat((repeat,)+(1,)*(value.ndim-1))
            return type(value)(expand(v) for v in value)
        return expand(plan)


@contextmanager
def allocation_geometry_reuse(model, block_rows):
    if torch.is_grad_enabled() or model.training:
        raise ValueError('allocation geometry reuse requires eval and no_grad')
    cache = _AllocationGeometryCache(block_rows)
    modules = [m for m in model.modules() if isinstance(m, SpatialPointContext)]
    previous = [getattr(m, '_allocation_geometry_cache', None) for m in modules]
    try:
        for m in modules: m._allocation_geometry_cache = cache
        yield cache
    finally:
        for m, old in zip(modules, previous):
            if old is None: delattr(m, '_allocation_geometry_cache')
            else: m._allocation_geometry_cache = old


@torch.no_grad()
def geometry_diagnostics(unit_xyz, active, geometry, cfg):
    """One-off sampled-block audit; no host synchronization in training forward."""
    xyz=metric_coordinates(unit_xyz,geometry.span.to(unit_xyz),active)
    rows=[]
    for stage,count in enumerate(cfg.context_centers,1):
        ids,center_active=farthest_points(xyz,active,count)
        centers=gather(xyz,ids).masked_fill(~center_active[...,None],0)
        base=spacing(xyz,active)[:,None].expand_as(center_active)
        distances=torch.cdist(centers.float(),xyz.float(),compute_mode='donot_use_mm_for_euclid_dist')
        for radius_multiplier,cap in zip(cfg.context_radii,cfg.context_neighbors):
            radius=base*radius_multiplier
            available=((distances<=radius[...,None]) & active[:,None,:]).sum(-1)[center_active]
            rows.append({'stage':stage,'radius_multiplier':radius_multiplier,'neighbor_cap':cap,
                'active_anchors':int(center_active.sum()),
                'mean_neighbors':float(available.clamp_max(cap).float().mean()) if available.numel() else 0.,
                'singleton_fraction':float((available<=1).float().mean()) if available.numel() else 0.,
                'cap_fraction':float((available>cap).float().mean()) if available.numel() else 0.,
                'mean_radius_world':float(radius[center_active].mean()*geometry.span.max()) if available.numel() else 0.})
        xyz,active=centers,center_active
    return rows
