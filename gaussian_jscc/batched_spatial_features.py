"""Batch the unchanged four-state projection and paired spatial descriptors."""
import torch

from .physical_benefit_features import _state_attributes, _summary, FEATURE_DIM
from .spatial_benefit_features import image_samples, CONTEXT_DIM, STATE_DIM


@torch.no_grad()
def project_candidates(tile, raws, tier_maps):
    if not raws or len(raws) != len(tier_maps):
        raise ValueError('nonempty paired candidate batches required')
    if any(r.ndim != 3 or r.shape[0] != 4 or t.shape != r.shape[:2] or not r.shape[1]
           for r, t in zip(raws, tier_maps)):
        raise ValueError('candidate states must be [4,K,A] with matching tiers')
    sizes = [4*r.shape[1] for r in raws]
    raw = torch.cat([r.reshape(-1, r.shape[-1]) for r in raws])
    existence = torch.cat([(t > 0).reshape(-1) for t in tier_maps]).to(raw)
    result = tile.project(raw, existence)
    return [p.reshape(4, r.shape[1], 12) for p, r in zip(result.split(sizes), raws)]


def _baseline_batch(tile, target, stats, changed, cache_keys):
    fields = getattr(tile, '_spatial_fields', None)
    if fields is None:
        fields = torch.cat((target.to(tile.image), tile.image-target.to(tile.image)), 0).contiguous()
        tile._spatial_fields = fields
    cache = getattr(tile, '_batched_spatial_baselines', None)
    if cache is None:
        cache = {}; tile._batched_spatial_baselines = cache
    missing = {}
    for i, key in enumerate(cache_keys):
        if key not in cache:
            missing.setdefault(key, i)
    if missing:
        indices = list(missing.values())
        baseline = tile.projected[changed[indices]]
        s = stats[indices]
        axis = baseline.new_tensor([-.5, 0., .5])
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        offsets = torch.stack((xx.flatten(), yy.flatten()), -1)
        xy = baseline[:, :, None, :2]+baseline[:, :, None, 11:12].clamp_min(0)*offsets
        profiles = image_samples(fields, xy).flatten(-2)
        summary = _summary(s, tile.height*tile.width)
        feature_weights = s[..., 0].clamp_min(0)+.05*s[..., 4].clamp_min(0)
        feature_weights = feature_weights/feature_weights.sum(-1, keepdim=True).clamp_min(1e-12)
        spatial_weights = .9*feature_weights+.1/baseline.shape[1]
        for j, key in enumerate(missing):
            cache[key] = dict(baseline=baseline[j], stats_summary=summary[j], profile=profiles[j],
                              feature_weights=feature_weights[j], weights=spatial_weights[j])
    packed = {name: torch.stack([cache[key][name] for key in cache_keys]) for name in
              ('baseline', 'stats_summary', 'profile', 'feature_weights', 'weights')}
    return fields, packed


@torch.no_grad()
def spatial_inputs_batch(static, stats, raw, projected, points, tile, target, changed,
                         d_ref, lambda_s, cache_keys):
    """Equal-length complete blocks; callers partition only ragged block lengths."""
    b, states, k, _ = raw.shape
    if (states != 4 or k < 1 or static.shape != (b, 24) or stats.shape != (b, k, 12) or
            projected.shape != (b, 4, k, 12) or points.shape != (b,) or
            changed.shape != (b, k) or len(cache_keys) != b or
            ((points < 0) | (points >= k)).any()):
        raise ValueError('invalid complete spatial batch dimensions')
    fields, cached = _baseline_batch(tile, target, stats, changed, cache_keys)
    index = torch.arange(b, device=raw.device)
    attributes = _state_attributes(raw, projected)
    delta = attributes-attributes[:, :1]
    w = cached['feature_weights'][:, None, :, None]
    common = torch.cat((static.to(raw), cached['stats_summary'][index, points],
                        _summary(stats.sum(1), tile.height*tile.width),
                        raw.new_tensor([d_ref, lambda_s]).expand(b, -1)), -1)
    features = torch.cat((common[:, None].expand(-1, 4, -1), attributes[index, :, points],
        (attributes*w).sum(2), delta[index, :, points], (delta*w).sum(2),
        (delta.abs()*w).sum(2), delta.abs().amax(2)), -1)
    baseline = cached['baseline']; size = raw.new_tensor([tile.width, tile.height])
    flag = raw.new_zeros(b, k, 1); flag[index, points] = 1
    depth_delta = baseline[..., 2:3]-baseline[index, points, 2:3][:, None]
    context = torch.cat((cached['stats_summary'], baseline[..., :2]/size,
        (baseline[..., :2]-baseline[index, points, :2][:, None])/size,
        depth_delta.sign()*depth_delta.abs().log1p(),
        baseline[..., 11:12]/max(tile.height, tile.width), flag, cached['profile']), -1)
    exists = projected[..., 7:8]
    state = torch.cat((attributes, projected[..., :2]/size*exists,
                       image_samples(fields, projected[..., :2])*exists), -1)
    weights = cached['weights']
    finite = torch.stack([torch.isfinite(v).all() for v in (features, context, state, weights)]).all()
    if (features.shape != (b, 4, FEATURE_DIM) or context.shape != (b, k, CONTEXT_DIM) or
            state.shape != (b, 4, k, STATE_DIM) or not finite):
        raise ValueError('batched spatial inputs must have finite complete descriptors')
    return [dict(features=features[i], context=context[i], states=state[i], weights=weights[i])
            for i in range(b)]
