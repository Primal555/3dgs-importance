"""Fail-closed CUDA, frozen-codec and dense-label checks on a real scene."""
import math
import time

import torch
from torch.nn import functional as F

from .allocation_distortion import ImageDistortion
from .budgeted_allocation_kernel import raw_batches
from .optimization import preserved_rng
from .pixel_benefit import integrate_density
from .pixel_benefit_features import build_group, pixel_coordinates
from .pixel_local_renderer import capture, _members


def _sync(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def assert_close(name, actual, expected, *, atol, rtol):
    """Return measured tolerances, raising for any out-of-tolerance element."""
    if actual.shape != expected.shape:
        raise RuntimeError(f'{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}')
    a, b = actual.detach().double(), expected.detach().to(actual.device).double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError(f'{name}: nonfinite comparison')
    difference = (a-b).abs()
    limit = atol + rtol*b.abs()
    maximum = float(difference.max()) if difference.numel() else 0.
    rms = float(difference.square().mean().sqrt()) if difference.numel() else 0.
    if (difference > limit).any():
        violations = int((difference > limit).sum())
        raise RuntimeError(f'{name}: {violations} values exceed atol={atol:g}, rtol={rtol:g}; '
                           f'max_absolute_error={maximum:.9g}, rms_error={rms:.9g}')
    return dict(max_absolute_error=maximum, rms_error=rms, atol=float(atol), rtol=float(rtol))


def stitch_tiles(image, tiles, rendered):
    """Create one full comparison image while retaining untouched pixels."""
    output = image.clone()
    x, y = pixel_coordinates(tiles, image.shape[-1], halo=0)
    valid = (x < image.shape[-1]) & (y < image.shape[-2])
    output[:, y[valid], x[valid]] = rendered.permute(1, 0, 2, 3).reshape(3, -1)[:, valid.flatten()]
    return output


@torch.no_grad()
def verify_expansion_probe(device, white_background, full_renderer):
    """Exercise new tile entries with a deterministic, visible footprint change.

    The probe has its own camera and raw tensors. Real decoded blocks can
    already cover every image tile, so their union cannot test this boundary.
    The caller supplies the original renderer for the two independent checks.
    """
    from types import SimpleNamespace
    from utils.graphics_utils import getProjectionMatrix
    from utils.sh_utils import RGB2SH
    device = torch.device(device)
    camera = SimpleNamespace(image_height=64, image_width=64, FoVx=math.pi/2,
        FoVy=math.pi/2, world_view_transform=torch.eye(4, device=device),
        full_proj_transform=getProjectionMatrix(.01, 100., math.pi/2, math.pi/2).T.to(device).contiguous(),
        camera_center=torch.zeros(3, device=device))
    raw = torch.zeros(3, 14, device=device)
    raw[:, :3] = raw.new_tensor([[-.5, -.5, 2.], [0., 0., 3.], [.8, -.7, 4.]])
    raw[:, 3] = 2.
    raw[:, 4:7] = raw.new_tensor([.02, .4, .2]).log()[:, None]
    raw[:, 7] = 1.
    raw[:, 11:14] = RGB2SH(raw.new_tensor([[.9, .15, .05], [.05, .2, .9], [.1, .8, .15]]))
    masks = raw.new_ones(3)
    cached = capture(raw, camera, 0, masks, white_background, backend=device.type)
    baseline = assert_close('expansion probe baseline versus full renderer', cached.image,
        full_renderer(raw, camera, 0, white_background, masks), atol=3e-5, rtol=3e-5)
    changed = torch.tensor([0], device=device)
    candidate = raw[:1].clone()
    multiplier = 16.
    candidate[:, 4:7] += math.log(multiplier)
    projected = cached.project(candidate, masks[:1])
    original_tiles = _members(cached.projected[changed], 64, 64)[0].unique()
    enlarged_tiles = _members(projected, 64, 64)[0].unique()
    new_tiles = enlarged_tiles[~torch.isin(enlarged_tiles, original_tiles)]
    if not len(new_tiles):
        raise RuntimeError('expansion probe failed to construct new tile entries')
    affected = cached.affected_tiles(changed, projected[None].expand(4, -1, -1))
    output = stitch_tiles(cached.image, affected, cached.render_tiles(changed, projected, affected))
    branch = raw.clone()
    branch[changed] = candidate
    full_image = full_renderer(branch, camera, 0, white_background, masks)
    image_check = assert_close('expanded footprint local image versus full renderer', output,
                              full_image, atol=3e-5, rtol=3e-5)
    x, y = pixel_coordinates(new_tiles, 64, halo=0)
    new_tile_change = float((full_image[:, y, x]-cached.image[:, y, x]).abs().max())
    if new_tile_change <= .001:
        raise RuntimeError('expansion probe must produce a visible change in newly entered tiles')
    return dict(scope='deterministic_expansion_probe', height=64, width=64, points=3,
        scale_multiplier=multiplier, original_tiles=len(original_tiles), new_tiles=len(new_tiles),
        affected_tiles=len(affected), new_tile_max_change=new_tile_change,
        baseline=baseline, image=image_check, full_render_count=2)


@torch.no_grad()
def verify_group(tile_scene, ids, projected, target, full_renderer, *, lambda_s, d_ref):
    """Compare four local states and dense integrals with sequential full renders.

    ``full_renderer(q)`` must render the whole altered scene independently. It
    returns one image at a time; this function retains only scalar components.
    """
    _sync(tile_scene.image)
    started = time.perf_counter()
    group = build_group(tile_scene, ids, projected, target, lambda_s, d_ref, with_labels=True)
    integrated = integrate_density(group['density'], group['inputs']['valid'],
        tile_scene.height, tile_scene.width, d_ref).double()
    # Release token features before full-scene renderer scratch allocations.
    label_tiles = len(group['tiles'])
    del group
    affected = tile_scene.affected_tiles(ids, projected)
    objective = ImageDistortion('mse_ssim' if lambda_s else 'mse', lambda_s)
    states, components = [], []
    for q in range(4):
        rendered = tile_scene.render_tiles(ids, projected[q], affected)
        local_image = stitch_tiles(tile_scene.image, affected, rendered)
        del rendered
        full_image = full_renderer(q)
        measured = assert_close(f'local state q{q} versus full renderer', local_image, full_image,
                                atol=3e-5, rtol=3e-5)
        components.append(objective.components(full_image, target).double())
        states.append(dict(tier=q, image=measured))
        del full_image, local_image
    values = torch.stack(components).double()
    expected = (values[1:]-values[:1])*values.new_tensor([1., lambda_s])
    physical = integrated*float(d_ref)
    # Absolute tolerance is in physical distortion units. Normalized tolerance
    # is derived from the same bound, rather than relaxing for small d_ref.
    absolute, relative = 5e-11, 2e-5
    physical_check = assert_close('dense integral versus full-image component difference',
        physical, expected, atol=absolute, rtol=relative)
    normalized_check = assert_close('normalized dense integral versus full-image difference',
        integrated, expected/float(d_ref), atol=absolute/float(d_ref), rtol=relative)
    _sync(tile_scene.image)
    return dict(states=states, changed_points=len(ids), affected_tiles=len(affected),
        label_tiles=label_tiles, rendered_pixels=len(affected)*256*4,
        dense_vs_full_physical=physical_check, dense_vs_full_normalized=normalized_check,
        integrated=integrated.cpu().tolist(), full_image_difference=(expected/float(d_ref)).cpu().tolist(),
        seconds=time.perf_counter()-started)


def _decode_batch(model, scene, batch, seed, snr, device):
    f = scene.groups[batch].to(device)
    ids = scene.group_ids[batch].to(device)
    valid = ids >= 0
    q = ((ids.clamp_min(0)+seed) % 4).long().masked_fill(~valid, 0)
    generator = torch.Generator(device=device).manual_seed(seed+104729*(batch+1))
    noise = torch.randn((*q.shape, 2*model.cfg.rates[-1]), generator=generator, device=device, dtype=f.dtype)
    noise *= math.sqrt(10**(-float(snr)/10)/2)
    semantic = model.semantic_encoder(f, f[..., :3], q > 0)
    actual = model.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).to(f), snr,
        'awgn', paired_noise=True, _semantic_y=semantic, _channel_noise=noise)[0]
    return f, ids, q, semantic, actual, noise


@torch.no_grad()
def verify_renderer(model, scene, args, reference, camera=None):
    """Mandatory real CUDA startup check using two actual frozen-codec blocks."""
    from .pixel_benefit_window import decode_group
    from .rendering import render
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('renderer preflight requires a fully frozen eval-mode codec')
    device = next(model.parameters()).device
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('renderer preflight requires actual CUDA; CPU checks cannot certify the extension')
    if getattr(args, 'renderer_backend', 'cuda') != 'cuda':
        raise ValueError('real-scene preflight requires renderer_backend=cuda')
    if len(scene.groups) != len(scene.group_ids):
        raise ValueError('scene groups and row IDs must align')
    camera = scene.cameras[0] if camera is None else camera
    lambda_s = float(reference['lambda_s'])
    # Exercise SSIM even for an MSE-only experiment, reporting the diagnostic
    # weight explicitly. Training's objective/reference remain unchanged.
    diagnostic_weight = lambda_s if lambda_s > 0 else .01
    d_ref = float(reference['d_ref'])
    if not math.isfinite(d_ref) or d_ref <= 0:
        raise ValueError('preflight requires a positive finite d_ref')
    seed, snr = int(getattr(args, 'seed', 0))+72817, float(args.snr)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with preserved_rng(device):
        parts, blocks, offset = [], [], 0
        for batch in range(len(scene.groups)):
            f, ids, q, semantic, actual, noise = _decode_batch(model, scene, batch, seed, snr, device)
            valid = ids >= 0
            decoded = raw_batches(actual, scene.geometry, model)
            parts.append(torch.cat((decoded[valid], (q[valid] > 0).to(decoded)[:, None]), -1))
            for row, count in enumerate(valid.sum(-1).cpu().tolist()):
                if count:
                    blocks.append((batch, row, offset, offset+count))
                offset += count
            del f, ids, q, semantic, actual, noise, decoded
        if len(blocks) < 2:
            raise RuntimeError('preflight requires at least two actual codec blocks')
        mixed = torch.cat(parts)
        del parts
        torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter()-started
        capture_started = time.perf_counter()
        cached = capture(mixed[:, :-1], camera, scene.degree, mixed[:, -1],
            scene.white_background, backend='cuda')
        torch.cuda.synchronize(device)
        capture_seconds = time.perf_counter()-capture_started
        original = render(mixed[:, :-1], camera, scene.degree, scene.white_background, mixed[:, -1])
        baseline_check = assert_close('captured baseline versus original MaskGaussian', cached.image,
                                     original, atol=3e-5, rtol=3e-5)
        del original
        target = scene.reference.get(camera, device)
        # One GPU reduction ranks actual blocks without per-block CUDA syncs.
        counts = torch.tensor([end-begin for _, _, begin, end in blocks], device=device)
        membership = torch.repeat_interleave(torch.arange(len(blocks), device=device), counts)
        scores = mixed.new_zeros(len(blocks))
        scores.scatter_add_(0, membership, cached.projected[:, 11].square()*
                            cached.projected[:, 6]*cached.projected[:, 7])
        del membership
        order = scores.argsort(descending=True).cpu().tolist()
        scores_cpu = scores.cpu()
        selected = [i for i in order if float(scores_cpu[i]) > 0][:2]
        if len(selected) < 2:
            raise RuntimeError('preflight camera must see at least two populated codec blocks')
        reports = []
        for index in selected:
            batch, row, begin, end = blocks[index]
            f, ids, q, semantic, actual, noise = _decode_batch(model, scene, batch, seed, snr, device)
            valid = ids[row] >= 0
            columns = valid.nonzero(as_tuple=True)[0]
            score = cached.projected[begin:end, 11].square()*cached.projected[begin:end, 7]
            column = int(columns[score.argmax()])
            alternatives, tiers = decode_group(model, f, q, row, column, actual, semantic, noise, snr)
            codec_checks = []
            for tier in (0, 1, 3):
                full_q = q.clone()
                full_q[row, column] = tier
                # Re-encode the full original batch independently. This checks
                # q0 context changes and preservation of the original noise slot.
                independently_decoded = model.forward_tier_batches(f, f[..., :3],
                    F.one_hot(full_q, 4).to(f), snr, 'awgn', paired_noise=True,
                    _channel_noise=noise)[0]
                codec_checks.append(dict(tier=tier, **assert_close('selected block versus original paired-noise batch',
                    alternatives[tier, valid], independently_decoded[row, valid], atol=2e-5, rtol=2e-5)))
                del independently_decoded
            raw = raw_batches(alternatives, scene.geometry, model)[:, valid]
            masks = (tiers[:, valid] > 0).to(raw)
            del alternatives, actual, semantic, noise, f
            changed = torch.arange(begin, end, device=device)
            projected = torch.stack([cached.project(raw[tier], masks[tier]) for tier in range(4)])

            def full_renderer(tier):
                branch = mixed.clone()
                branch[changed, :-1] = raw[tier]
                branch[changed, -1] = masks[tier]
                return render(branch[:, :-1], camera, scene.degree, scene.white_background, branch[:, -1])

            report = verify_group(cached, changed, projected, target, full_renderer,
                lambda_s=diagnostic_weight, d_ref=d_ref)
            report.update(batch=batch, block=row, source_id=int(ids[row, column]),
                          original_tier=int(q[row, column]), codec_checks=codec_checks)
            reports.append(report)
            del raw, masks, projected, ids, q, tiers
        expansion_report = verify_expansion_probe(device, scene.white_background, render)
        torch.cuda.synchronize(device)
        return dict(version=1, status='passed', backend='cuda', extension='maskgaussian_pixel_local_v1',
            camera=str(getattr(camera, 'image_name', 'unnamed')), height=cached.height, width=cached.width,
            points=len(mixed), tile_count=len(cached.tile_ranges), complete_tile_entries=len(cached.tile_ids),
            seed=seed, snr=snr, original_lambda_s=lambda_s, diagnostic_lambda_s=diagnostic_weight, d_ref=d_ref,
            actual_decode_seconds=decode_seconds, capture_seconds=capture_seconds,
            total_seconds=time.perf_counter()-started, baseline=baseline_check, groups=reports,
            expansion=expansion_report, full_render_count=1+4*len(reports)+expansion_report['full_render_count'],
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))


def verify_cuda(model, scene, camera, args):
    """Compatibility entry point for callers that calibrate after preflight."""
    weight = getattr(args, 'lambda_s', None)
    objective = getattr(args, 'allocation_image_objective', None)
    if weight is None and objective is not None:
        weight = objective.lambda_s
    return verify_renderer(model, scene, args, dict(lambda_s=.01 if weight is None else weight, d_ref=1.),
                           camera=camera)
