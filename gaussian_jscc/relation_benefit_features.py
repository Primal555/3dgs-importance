"""Frozen sender-side relation inputs; every cached row uses original PLY IDs.

Candidate attributes come from complete uniform-tier finite-symbol decodes.
The only image reference is a capture of the source PLY. No photograph or
realized communication noise enters this cache.
"""
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import to_raw
from .optimization import preserved_rng
from .physical_benefit_features import static_features, visibility_statistics, _summary
from .pixel_local_renderer import capture
from .transport import model_id
from .relation_benefit_context import background_configurations, context_identity


VERSION = 'relation_benefit_bank_context_v2'


def _array(value, columns):
    value = torch.as_tensor(value).detach().cpu()
    if value.ndim != 2 or value.shape[1] not in columns or not torch.isfinite(value).all():
        raise ValueError('finite coordinate matrix with supported columns required')
    return value.double().numpy()


def _graph_options(k, workers):
    if not isinstance(k, int) or k < 0 or workers == 0 or workers < -1:
        raise ValueError('nonnegative integer k and positive workers (or -1) required')


def build_spatial_graph(xyz, k=16, workers=1):
    """Euclidean KNN across encoding blocks, without self, with -1 padding."""
    from scipy.spatial import cKDTree
    _graph_options(k, workers)
    xyz = _array(xyz, (3,))
    n = len(xyz)
    output = np.full((n, k), -1, dtype=np.int64)
    count = min(k, max(n-1, 0))
    if not count:
        return torch.from_numpy(output)
    tree = cKDTree(xyz)
    for first in range(0, n, 8192):
        last = min(first+8192, n)
        _, indices = tree.query(xyz[first:last], k=count+1, workers=workers)
        own = np.arange(first, last)[:, None]
        # Coincident points may displace self from the first query results.
        order = np.argsort(indices == own, axis=1, kind='stable')[:, :count]
        output[first:last, :count] = np.take_along_axis(indices, order, axis=1)
    return torch.from_numpy(output)


def build_projection_graph(projected, k=16, workers=1):
    """Coverage-overlap neighbors, balanced across front/back depth layers.

    Accept the renderer's [N,12] records or [u,v,depth,radius] [N,4]. Radius
    zero denotes an invalid/off-image point. Radius-bucketed KD trees provide
    a complete broad phase even when a huge disk has a distant center. Disk
    intersection then filters the conservative renderer bounding footprints.
    Query batches bound broad-phase temporaries independently of scene size.
    Widespread overlap can still produce O(N^2) broad-phase candidates; only
    retained graph storage is O(N*k).
    """
    from scipy.spatial import cKDTree
    _graph_options(k, workers)
    values = _array(projected, (4, 12))
    n = len(values)
    output = np.full((n, k), -1, dtype=np.int64)
    if not k or n < 2:
        return torch.from_numpy(output)
    center, depth = values[:, :2], values[:, 2]
    radius = values[:, -1]
    valid = (radius > 0) & (depth > 0)
    ids = np.flatnonzero(valid)
    if len(ids) < 2:
        return torch.from_numpy(output)
    # Powers of two limit each bucket's radius ratio while retaining all hits.
    exponent = np.floor(np.log2(radius[ids]))
    buckets = []
    for scale in np.unique(exponent):
        members = ids[exponent == scale]
        buckets.append((members, cKDTree(center[members]), radius[members].max()))
    front_budget = (k+1)//2
    for first in range(0, len(ids), 256):
        roots = ids[first:first+256]
        choices = [[] for _ in roots]
        for members, tree, largest in buckets:
            # Avoid a chunk*N Python-list allocation for heavily overlapping views.
            chunk = max(1, min(len(roots), 262144//max(len(members), 1)))
            for start in range(0, len(roots), chunk):
                selected = roots[start:start+chunk]
                matches = tree.query_ball_point(center[selected], radius[selected]+largest,
                                                workers=workers, return_sorted=True)
                for offset, hits in enumerate(matches):
                    root = int(selected[offset])
                    other = members[np.asarray(hits, dtype=np.int64)]
                    distance = np.linalg.norm(center[other]-center[root], axis=1)
                    keep = (other != root) & (distance <= radius[root]+radius[other])
                    other, distance = other[keep], distance[keep]
                    if not len(other):
                        continue
                    # Normalized penetration ranks shared coverage; distance breaks ties.
                    overlap = 1-distance/(radius[root]+radius[other])
                    for front in (True, False):
                        side = depth[other] <= depth[root] if front else depth[other] > depth[root]
                        candidates = other[side]
                        rank = np.lexsort((candidates, distance[side], -overlap[side]))[:k]
                        choices[start+offset].extend(zip(candidates[rank].tolist(),
                            overlap[side][rank].tolist(), distance[side][rank].tolist()))
        for slot, root in enumerate(roots):
            ranked = sorted(choices[slot], key=lambda row: (-row[1], row[2], row[0]))
            front = [row for row in ranked if depth[row[0]] <= depth[root]]
            back = [row for row in ranked if depth[row[0]] > depth[root]]
            chosen = front[:front_budget]+back[:k-front_budget]
            used = {row[0] for row in chosen}
            chosen += [row for row in ranked if row[0] not in used][:k-len(chosen)]
            output[root, :len(chosen)] = [row[0] for row in chosen]
    return torch.from_numpy(output)


def _scatter(packet, order):
    output = torch.empty_like(packet, device='cpu')
    output[order] = packet.detach().cpu()
    return output


def _attributes(raw, model, origin, scale):
    return torch.cat(((raw[..., :3]-origin)/scale,
        (raw[..., 3:]-model.attr_mean.detach().to(raw)) /
        model.attr_std.detach().to(raw).clamp_min(1e-6)), -1)


def _projection_features(projected, width, height):
    p = projected.detach().cpu().float()
    result = p.clone()
    result[:, :2] /= p.new_tensor([width, height])
    result[:, 2] = torch.sign(p[:, 2])*torch.log1p(p[:, 2].abs())
    result[:, 3:6] *= p.new_tensor([width*width, width*height, height*height])
    result[:, 11] = torch.log1p(p[:, 11].clamp_min(0)/max(width, height))
    return result


def _image_profile(source, baseline, projected):
    """3x3 RGB/source-residual samples, retaining the context of hidden points."""
    height, width = source.shape[-2:]
    p = projected.to(source)
    axis = torch.linspace(-1, 1, 3, device=source.device, dtype=source.dtype)
    dy, dx = torch.meshgrid(axis, axis, indexing='ij')
    offset = torch.stack((dx, dy), -1).reshape(1, 9, 2)
    coordinates = p[:, None, :2]+offset*p[:, None, 11:12]
    grid = (coordinates+.5)/p.new_tensor([width, height])*2-1
    maps = torch.cat((source, baseline-source), 0)[None]
    sampled = F.grid_sample(maps, grid[None], padding_mode='border', align_corners=False)
    return sampled[0].permute(1, 0, 2).flatten(1).cpu()


def _view_geometry(projected, visibility, width, height):
    return torch.stack((projected[:, 0]/width, projected[:, 1]/height,
        torch.sign(projected[:, 2])*torch.log1p(projected[:, 2].abs()),
        projected[:, 11]/max(width, height), projected[:, 6],
        torch.log1p(visibility[:, 0].clamp_min(0)),
        torch.log1p(visibility[:, 1].clamp_min(0))), -1)


def _extend_coverage(graph, source_projection, additional):
    """Enclose every active additional disk around the original source center."""
    if additional.ndim == 2:
        additional = additional[:, None]
    active = (additional[..., 11] > 0) & (additional[..., 2] > 0)
    distance = (additional[..., :2]-source_projection[:, None, :2]).norm(dim=-1)
    radius = torch.where(active, additional[..., 11]+distance, 0).amax(1)
    graph[:, 11] = torch.maximum(graph[:, 11], radius)
    depth = additional[..., 2].masked_fill(~active, torch.inf).amin(1)
    graph[:, 2] = torch.where(graph[:, 2] > 0, graph[:, 2],
                             torch.where(active.any(1), depth, 0))


def _camera_identity(cameras):
    digest = hashlib.sha256()
    for camera in cameras:
        digest.update(str(getattr(camera, 'image_name', '')).encode())
        for field in ('image_width', 'image_height', 'FoVx', 'FoVy',
                      'world_view_transform', 'full_proj_transform', 'camera_center'):
            value = getattr(camera, field)
            digest.update(field.encode())
            if torch.is_tensor(value):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(str(value).encode())
    return digest.hexdigest()


@torch.no_grad()
def prepare_bank(model, scene, cameras, args):
    """Build CPU float32 inputs and int64 graphs in original PLY row order.

    ``cameras`` is the caller's deterministic training-only preparation window;
    its uniform weights must also be used by the paired teacher.
    """
    cameras = list(cameras)
    n, dim = scene.raw.shape
    batch = int(getattr(args, 'feature_batch', 8192))
    ks = int(getattr(args, 'spatial_neighbors', 16))
    kv = int(getattr(args, 'projection_neighbors', 16))
    workers = int(getattr(args, 'cpu_threads', 1))
    context_seed = int(getattr(args, 'context_seed', getattr(args, 'seed', 42)))
    _graph_options(ks, workers)
    _graph_options(kv, workers)
    if n < 1 or not cameras or batch < 1:
        raise ValueError('nonempty source/cameras and positive feature batch required')
    if (tuple(model.cfg.rates) != (0, 8, 24, 48) or
            model.cfg.position_delivery != 'quantized' or model.cfg.position_bits != 16 or
            float(getattr(args, 'snr', 10.)) != 10.):
        raise ValueError('fixed 10dB, 0/8/24/48 and reliable 16bit XYZ required')
    order = torch.as_tensor(scene.order).long().cpu()
    if order.shape != (n,) or not torch.equal(order.sort().values, torch.arange(n)):
        raise ValueError('packet-to-original order must be a complete permutation')
    raw = scene.raw.detach().cpu().float()
    if not torch.isfinite(raw).all():
        raise ValueError('finite source attributes required')
    source = _scatter(raw, order)
    origin = source[:, :3].amin(0)
    scale = (source[:, :3].amax(0)-origin).amax().clamp_min(1e-6)
    xyz = (source[:, :3]-origin)/scale
    reference = _attributes(source, model, origin, scale)
    # q0 retains the complete source footprint and signed source appearance.
    candidates = source[:, None].expand(n, 4, dim).clone()
    configurations = background_configurations(n, context_seed)
    block_stats = raw.new_empty(n, 2*dim+1)
    all_ids = []
    for features, group_ids in zip(scene.groups, scene.group_ids):
        if features.shape[:2] != group_ids.shape:
            raise ValueError('feature groups and original IDs must align')
        for ids in group_ids:
            selected = ids[ids >= 0].long().cpu()
            all_ids.append(selected)
            if len(selected):
                values = reference[selected]
                stats = torch.cat((values.mean(0), values.std(0, unbiased=False),
                                   values.new_tensor([len(selected)/model.cfg.block_size])))
                block_stats[selected] = stats
    complete_ids = torch.cat(all_ids) if all_ids else torch.empty(0, dtype=torch.long)
    # Codec groups own their explicit original IDs. Their order can differ from
    # the Morton order used by source rendering without changing cache row IDs.
    if not torch.equal(complete_ids.sort().values, torch.arange(n)):
        raise ValueError('encoding groups must contain all original IDs exactly once')
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        model.set_scene_geometry(scene.geometry)
        with preserved_rng(device):
            for tier in (1, 2, 3):
                print(f'Preparing shared finite-noiseless q{tier} candidates...', flush=True)
                for features, ids in zip(scene.groups, scene.group_ids):
                    features = features.to(device)
                    valid = ids >= 0
                    q = torch.where(valid, tier, 0).to(device)
                    # Run complete codec blocks: semantic context, reliable XYZ,
                    # finite stage symbols and block scale delivery all stay active.
                    prediction = model.forward_tier_batches(features, features[..., :3],
                        F.one_hot(q, 4).to(features), 10., 'none', paired_noise=True)[0]
                    decoded = to_raw(prediction[valid.to(device)], scene.geometry, model)
                    candidates[ids[valid].long().cpu(), tier] = decoded.detach().cpu().float()
            # Uniform backgrounds alias the already reconstructed candidate raw
            # tensors. Only the mixed background needs an extra complete decode.
            background_raw = [candidates[:, tier] for tier in (1, 2, 3)]
            mixed_raw = source.clone()
            print('Preparing finite-noiseless mixed background...', flush=True)
            for features, ids in zip(scene.groups, scene.group_ids):
                valid = ids >= 0
                q = configurations[3]['tiers'][ids.clamp_min(0).cpu()].masked_fill(~valid.cpu(), 0).to(device)
                features = features.to(device)
                prediction = model.forward_tier_batches(features, features[..., :3],
                    F.one_hot(q, 4).to(features), 10., 'none', paired_noise=True)[0]
                active = valid.to(device) & (q > 0)
                # Absent nodes retain source descriptors, with existence encoded
                # explicitly by the q0 tier. They never contribute to captures.
                mixed_raw[ids[active.cpu()].long().cpu()] = to_raw(
                    prediction[active], scene.geometry, model).detach().cpu().float()
            background_raw.append(mixed_raw)
            contexts = []
            for configuration, decoded in zip(configurations, background_raw):
                attributes = _attributes(decoded, model, origin, scale)
                tiers = configuration['tiers']
                histogram = raw.new_empty(n, 4)
                for ids in all_ids:
                    if len(ids):
                        histogram[ids] = torch.bincount(tiers[ids], minlength=4).to(raw)/len(ids)
                contexts.append(dict(name=configuration['name'], tiers=tiers,
                    node=torch.cat((attributes, attributes-reference, F.one_hot(tiers, 4).to(raw), histogram), -1),
                    view_node=raw.new_empty(len(cameras), n, 78),
                    view_geom=raw.new_empty(len(cameras), n, 7)))
            rap = _scatter(static_features(raw, neighbors=max(1, ks), workers=workers), order)
            candidate = raw.new_empty(n, 4, 2*dim+2)
            node = raw.new_empty(n, 4*dim+24+2*dim+1)
            for first in range(0, n, batch):
                last = min(first+batch, n)
                attributes = _attributes(candidates[first:last], model, origin, scale)
                difference = attributes-reference[first:last, None]
                candidate[first:last, :, :dim] = attributes
                candidate[first:last, :, dim:2*dim] = difference
                candidate[first:last, :, 2*dim] = torch.tensor([0., 1., 1., 1.])
                candidate[first:last, :, 2*dim+1] = torch.tensor([0., 8/48, 24/48, 1.])
                positive = difference[:, 1:]
                node[first:last] = torch.cat((reference[first:last], positive.mean(1),
                    positive.abs().mean(1), positive.abs().amax(1), rap[first:last],
                    block_stats[first:last]), -1)
            view_dim = 12*3+54+4*12*2
            view_node = raw.new_empty(len(cameras), n, view_dim)
            view_geom = raw.new_empty(len(cameras), n, 7)
            view_index = torch.empty(len(cameras), n, kv, dtype=torch.long)
            coverage_profiles = []
            for view, camera in enumerate(cameras):
                view_started = time.perf_counter()
                print(f'Preparing relation view {view+1}/{len(cameras)} ({camera.image_name})...', flush=True)
                source_packet = raw.to(device)
                existence = torch.ones(n, device=device)
                tile = capture(source_packet, camera, scene.degree, existence,
                               scene.white_background, backend=device.type)
                q2_packet = candidates[order, 2].to(device)
                baseline = capture(q2_packet, camera, scene.degree, existence,
                                   scene.white_background, backend=device.type)
                source_stats = _scatter(visibility_statistics(tile, tile.image), order)
                q2_stats = _scatter(visibility_statistics(baseline, tile.image), order)
                projected = _scatter(tile.projected, order)
                graph_projected = projected.clone()
                source_projection = _projection_features(projected, tile.width, tile.height)
                geometry = _view_geometry(projected, source_stats, tile.width, tile.height)
                view_geom[view] = geometry
                for first in range(0, n, batch):
                    last = min(first+batch, n)
                    count = last-first
                    cp = tile.project(candidates[first:last].reshape(-1, dim).to(device),
                                      torch.ones(count*4, device=device)).cpu()
                    physical = cp.reshape(count, 4, 12)
                    _extend_coverage(graph_projected[first:last], projected[first:last], physical)
                    cp = _projection_features(cp, tile.width, tile.height).reshape(count, 4, 12)
                    cp[:, 0, 7] = 0
                    delta = cp-source_projection[first:last, None]
                    profile = _image_profile(tile.image, baseline.image, projected[first:last])
                    view_node[view, first:last] = torch.cat((source_projection[first:last],
                        _summary(source_stats[first:last], tile.height*tile.width),
                        _summary(q2_stats[first:last], tile.height*tile.width), profile,
                        cp.flatten(1), delta.flatten(1)), -1)
                for context, decoded in zip(contexts, background_raw):
                    if context['name'] == 'q2':
                        context_capture, context_stats = baseline, q2_stats
                    else:
                        masks = (context['tiers'][order] > 0).to(device=device, dtype=raw.dtype)
                        context_capture = capture(decoded[order].to(device), camera, scene.degree,
                            masks, scene.white_background, backend=device.type)
                        context_stats = _scatter(visibility_statistics(context_capture, tile.image), order)
                    context_projected = _scatter(context_capture.projected, order)
                    context['view_geom'][view] = _view_geometry(context_projected, context_stats,
                                                              tile.width, tile.height)
                    _extend_coverage(graph_projected, projected, context_projected)
                    for first in range(0, n, batch):
                        last = min(first+batch, n)
                        context['view_node'][view, first:last] = torch.cat((
                            _projection_features(context_projected[first:last], tile.width, tile.height),
                            _summary(context_stats[first:last], tile.height*tile.width),
                            _image_profile(tile.image, context_capture.image, projected[first:last])), -1)
                    del context_capture, context_stats, context_projected
                print(f'Building coverage graph for view {view+1}/{len(cameras)}...', flush=True)
                view_index[view] = build_projection_graph(graph_projected, kv, workers)
                coverage_profiles.append(dict(
                    source_max_radius=float(projected[:, 11].amax()),
                    union_max_radius=float(graph_projected[:, 11].amax()),
                    expanded_nodes=int((graph_projected[:, 11] > projected[:, 11]).sum()),
                    candidate_only_visible_nodes=int(((projected[:, 11] <= 0) &
                        (graph_projected[:, 11] > 0)).sum()),
                    seconds=time.perf_counter()-view_started))
    finally:
        for module, mode in modes:
            module.training = mode
    bank = dict(node=node, candidate=candidate, xyz=xyz.float(),
        spatial_index=build_spatial_graph(xyz, ks, workers), view_node=view_node,
        view_geom=view_geom, view_index=view_index,
        view_weights=torch.full((len(cameras),), 1/len(cameras), dtype=torch.float32))
    for name, value in bank.items():
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite relation bank '+name)
    metadata = dict(version=VERSION, row_order='original_ply', codec_id=model_id(model),
        scene_fingerprint=str(getattr(scene, 'fingerprint', '')),
        scene=str(getattr(scene, 'name', getattr(scene, 'spec', {}).get('name', ''))),
        camera_names=[str(getattr(camera, 'image_name', index)) for index, camera in enumerate(cameras)],
        camera_geometry_sha256=_camera_identity(cameras),
        candidate_protocol=dict(path='finite_noiseless', noise_input='none', snr_db=10.,
            uniform_tiers=[1, 2, 3], complete_encoding_blocks=True,
            position_delivery='reliable_16bit', q0='source_descriptor_with_existence_zero',
            signed_attribute_difference=True, image_reference='source_ply_capture'),
        parameters=dict(spatial_neighbors=ks, projection_neighbors=kv, cpu_threads=workers,
            feature_batch=batch, view_weighting='fixed_uniform',
            projection_overlap='source_candidate_context_union_bounding_disk',
            projection_complexity='O(N*K) retained graph; O(N^2) broadphase worst case'),
        projection_coverage_profiles=coverage_profiles,
        xyz_normalization=dict(origin=origin.tolist(), isotropic_scale=float(scale)),
        candidate_layout=dict(attributes=[0, dim], difference=[dim, 2*dim],
                              existence=2*dim, rate=2*dim+1),
        node_layout=dict(source=[0, dim], candidate_signed_mean=[dim, 2*dim],
            candidate_absolute_mean=[2*dim, 3*dim], candidate_absolute_max=[3*dim, 4*dim],
            rap=[4*dim, 4*dim+24], block_statistics=[4*dim+24, node.shape[-1]]),
        view_layout=dict(source_projection=[0, 12], source_visibility=[12, 24],
            q2_visibility_residual=[24, 36], source_residual_profile=[36, 90],
            candidate_projection=[90, 138], candidate_projection_difference=[138, 186]),
        dimensions={name: list(value.shape) for name, value in bank.items()},
        tensor_bytes=sum(value.numel()*value.element_size() for value in bank.values()) +
            sum(value.numel()*value.element_size() for context in contexts for value in context.values()
                if torch.is_tensor(value)),
        contexts=dict(context_identity(configurations, context_seed),
            node_layout=dict(attributes=[0, dim], difference=[dim, 2*dim],
                             tier_one_hot=[2*dim, 2*dim+4], block_tier_histogram=[2*dim+4, 2*dim+8]),
            view_layout=dict(projection=[0, 12], visibility=[12, 24], residual_profile=[24, 78]),
            absent_attributes='source_descriptor', raw_reconstruction='shared_uniform_candidates_plus_one_mixed_decode',
            dimensions=[dict(node=list(c['node'].shape), view_node=list(c['view_node'].shape),
                             view_geom=list(c['view_geom'].shape)) for c in contexts]))
    # Enforce portable checkpoint metadata without relying on pickle objects.
    json.dumps(metadata, allow_nan=False)
    bank['metadata'] = metadata
    for context in contexts:
        if any(not torch.isfinite(value).all() for value in context.values() if torch.is_tensor(value)):
            raise ValueError('nonfinite relation context '+context['name'])
    bank['contexts'] = contexts
    return bank
