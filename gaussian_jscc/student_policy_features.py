"""Source-only immutable student features, aligned to original PLY rows."""
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .physical_benefit_features import static_features, visibility_statistics
from .pixel_local_renderer import capture
from .student_policy_graph import build_graph

VERSION = 'student_source_features_v1'


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp'); torch.save(value, temporary); temporary.replace(path)


@torch.no_grad()
def retained_latent(model, scene):
    device = next(model.parameters()).device
    result = torch.empty(len(scene.raw), model.cfg.latent_dim)
    for f, ids in zip(scene.groups, scene.group_ids):
        f = f.to(device); valid = ids.to(device) >= 0
        y = model.semantic_encoder(f, f[..., :3], valid)
        result[ids[ids >= 0]] = y[valid].float().cpu()
    return result


@torch.no_grad()
def visibility_from_capture(scene, camera, tile, stats):
    p = tile.projected; raw = scene.raw.to(p)
    area = float(tile.height*tile.width); mass = stats[:, 1].clamp_min(1e-12)
    seen = stats[:, 1] > 0
    determinant = (p[:, 3]*p[:, 5]-p[:, 4].square()).clamp_min(1e-12)
    footprint = 9*torch.pi/determinant.sqrt()/area
    color = stats[:, 5:8]/mass[:, None]
    direction = F.normalize(raw[:, :3]-camera.camera_center.to(raw), dim=-1)
    # 14 channels; all dimensions refer to the source scene, never a future q.
    features = torch.cat((footprint.log1p()[:, None], (mass/area).log1p()[:, None],
        (mass/stats[:, 4].clamp_min(1e-12)).clamp(0, 1)[:, None],
        (stats[:, 2].clamp_min(0).sqrt()/mass)[:, None], stats[:, 3:4], stats[:, 11:12]/area,
        p[:, 2:3]/scene.geometry.span.max().to(p).clamp_min(1e-9), direction,
        color, (p[:, 8:11]-color).norm(dim=-1, keepdim=True)), -1)
    features = features.masked_fill(~seen[:, None], 0)
    original = torch.empty_like(features); original[scene.order.to(p.device)] = features
    visible = torch.empty_like(seen); visible[scene.order.to(p.device)] = seen
    return original.cpu(), visible.cpu()


@torch.no_grad()
def capture_view(model, scene, camera, verify=False):
    device = next(model.parameters()).device
    raw = scene.raw.to(device)
    tile = capture(raw, camera, scene.degree, raw.new_ones(len(raw)), scene.white_background,
                   backend=device.type)
    reference = scene.reference.get(camera, device)
    maximum = float((tile.image-reference).abs().max())
    if verify:
        torch.testing.assert_close(tile.image, reference, atol=3e-4, rtol=3e-4,
                                   msg='student source visibility renderer differs from the source renderer')
    stats = visibility_statistics(tile, tile.image)
    features, visible = visibility_from_capture(scene, camera, tile, stats)
    return dict(features=features, visible=visible, max_renderer_error=maximum)


def summarize_visibility(features, visible):
    if features.ndim != 3 or features.shape[-1] != 14 or visible.shape != features.shape[:2]:
        raise ValueError('per-camera source features must be [views,N,14]')
    valid = visible[..., None]; count = visible.sum(0).clamp_min(1)[:, None]
    values = features.masked_fill(~valid, 0)
    mean = values.sum(0)/count
    variance = ((values-mean[None]).square()*valid).sum(0)/count
    maximum = features.masked_fill(~valid, -torch.inf).amax(0)
    maximum = maximum.masked_fill(~visible.any(0)[:, None], 0)
    return torch.cat((mean, maximum, variance.sqrt(), visible.float().mean(0)[:, None]), -1)


class VisibilityAccumulator:
    """Visible-only Welford statistics with memory independent of camera count."""
    def __init__(self, count):
        self.count = torch.zeros(count, dtype=torch.float64)
        self.mean = torch.zeros(count, 14, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)
        self.maximum = torch.full((count,14), -torch.inf)
        self.views = 0

    def add(self, features, visible):
        x = features.cpu().double(); valid = visible.cpu().bool()
        self.count += valid
        delta = (x-self.mean).masked_fill(~valid[:,None],0)
        self.mean += delta/self.count.clamp_min(1)[:,None]
        self.m2 += delta*(x-self.mean).masked_fill(~valid[:,None],0)
        self.maximum = torch.maximum(self.maximum, features.cpu().masked_fill(~valid[:,None],-torch.inf))
        self.views += 1

    def result(self):
        std = (self.m2/self.count.clamp_min(1)[:,None]).clamp_min(0).sqrt()
        maximum = self.maximum.masked_fill(self.count[:,None]==0,0)
        return torch.cat((self.mean.float(),maximum,std.float(),(self.count/self.views).float()[:,None]),-1)


def prepare_features(model, scene, cameras, identity, out, *, fine_points=16, coarse_points=4, neighbors=16,
                     streaming=False):
    out = Path(out); path = out/'preparation/features.pt'
    protocol = dict(version=VERSION, identity=identity, camera_names=[str(c.image_name) for c in cameras],
                    fine_points=fine_points, coarse_points=coarse_points, neighbors=neighbors)
    if streaming: protocol['visibility_reduction'] = 'visible_welford_float64_v1'
    signature = digest(protocol)
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['signature'] != signature: raise ValueError('student source cache identity mismatch')
        return saved
    started = time.perf_counter()
    raw = torch.empty_like(scene.raw.cpu()); raw[scene.order] = scene.raw.cpu()
    print('Preparing source attribute statistics and immutable spatial groups...', flush=True)
    xyz = raw[:, :3]; span = scene.geometry.span.max().clamp_min(1e-9).cpu()
    xyz = (xyz-xyz.amin(0))/span
    quat = F.normalize(raw[:, 7:11], dim=-1)
    sign = quat.gather(1, quat.abs().argmax(-1, keepdim=True)).sign()
    quat = quat*torch.where(sign == 0, torch.ones_like(sign), sign)
    attributes = torch.cat((xyz, raw[:, 3:4].sigmoid(), raw[:, 4:7]-span.log(), quat,
                            raw[:, 11:], static_features(raw, neighbors=32, workers=1)), -1)
    graph = build_graph(raw[:, :3], fine_points, coarse_points, neighbors)
    print('Preparing all-retained semantic latent in original PLY order...', flush=True)
    latent = retained_latent(model, scene)
    rows = []; maximum_error = 0.
    accumulator = VisibilityAccumulator(len(raw)) if streaming else None
    for i, camera in enumerate(cameras):
        view_path = out/'preparation/visibility'/f'{i:03d}.pt'
        if view_path.is_file():
            record = torch.load(view_path, map_location='cpu', weights_only=True)
            if record['signature'] != signature or record['name'] != str(camera.image_name):
                raise ValueError('source visibility cache identity mismatch')
        else:
            record = dict(capture_view(model, scene, camera, verify=i == 0), signature=signature,
                          name=str(camera.image_name))
            atomic_torch(view_path, record)
        maximum_error = max(maximum_error, record['max_renderer_error'])
        if accumulator is not None: accumulator.add(record['features'], record['visible'])
        else: rows.append(record)
        print(f'Source visibility {i+1}/{len(cameras)}: {camera.image_name}', flush=True)
    visibility = (accumulator.result() if accumulator is not None else
        summarize_visibility(torch.stack([r['features'] for r in rows]), torch.stack([r['visible'] for r in rows])))
    if not all(torch.isfinite(x).all() for x in (attributes, latent, visibility)):
        raise ValueError('nonfinite source student features')
    saved = dict(signature=signature, protocol=protocol, attributes=attributes, latent=latent,
        visibility=visibility, graph=graph, preparation_seconds=time.perf_counter()-started,
        max_renderer_error=maximum_error)
    atomic_torch(path, saved)
    return saved
