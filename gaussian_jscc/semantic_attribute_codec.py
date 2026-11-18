"""Continuous semantic-only codecs with shared or attribute-private latents."""
from dataclasses import replace

import torch
from torch import nn

from .modular_config import ModularCodecConfig
from .modular_core import SemanticDecoder
from .pointnet_context import SpatialPointContext
from .position_delivery import delivered_positions


ARMS = ('shared128', 'shared256', 'split128', 'split256')
FULL_GEOMETRY = 'split_full_geometry128'
GROUPS = {'geometry': (0, 8), 'dc': (8, 11), 'sh': (11, 56)}


class _SharedGeometry:
    """A decoder call shares detached neighborhoods, never learned activations."""
    def __init__(self):
        self.saved = None

    @torch.no_grad()
    def get(self, module, xyz, active):
        if self.saved is not None:
            x, a, plan = self.saved
            if torch.equal(x, xyz) and torch.equal(a, active):
                return plan
        sa1 = module.sa1.geometry(xyz, active)
        sa2 = module.sa2.geometry(sa1[0], sa1[1])
        plan = (sa1, sa2, module.fp1.geometry(sa1[0], sa1[1], sa2[0], sa2[1]),
                module.fp0.geometry(xyz, active, sa1[0], sa1[1]))
        self.saved = (xyz, active, plan)
        return plan


def latent_slices(width, split):
    if width not in (128, 256):
        raise ValueError('semantic experiment supports 128 or 256 real dimensions')
    if not split:
        return {'shared': (0, width)}
    unit = width // 8
    return {'geometry': (0, unit), 'dc': (unit, 2*unit),
            'sh': (2*unit, 6*unit), 'shared': (6*unit, width)}


def make_config(width, hidden=256, block_size=256):
    return ModularCodecConfig(latent_dim=width, hidden=hidden, block_size=block_size,
        rates=(0, 8, 24, 48), context_variant='both')


class AttributeEncoder(nn.Module):
    """Separate DC/high-order embeddings; one shared spatial trunk in all arms."""
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.embeddings = nn.ModuleList(nn.Linear(size, h) for size in (3, 1, 3, 4, 3, 45))
        self.fuse = nn.Sequential(nn.Linear(6*h, h), nn.GELU(), nn.Linear(h, h))
        self.spatial_context = SpatialPointContext(cfg)
        self.readout = nn.Linear(h, cfg.latent_dim)

    def forward(self, features, xyz, active):
        features = features.masked_fill(~active[..., None], 0)
        xyz = xyz.masked_fill(~active[..., None], 0)
        pieces = (features[..., :3]*2-1, features[..., 3:4], features[..., 4:7],
                  features[..., 7:11], features[..., 11:14], features[..., 14:])
        h = self.fuse(torch.cat([layer(x) for layer, x in zip(self.embeddings, pieces)], -1))
        h = self.spatial_context(h*active[..., None], xyz, active)
        return self.readout(h)*active[..., None]


class AttributeDecoder(nn.Module):
    def __init__(self, cfg, split, full_geometry=False):
        super().__init__()
        self.slices = latent_slices(cfg.latent_dim, split)
        self.split = split
        self.full_geometry = full_geometry
        if not split:
            self.shared = SemanticDecoder(cfg)
            self.shared.blocks = nn.ModuleList()
            self.shared.spatial_context = SpatialPointContext(cfg)
        else:
            self.branches = nn.ModuleDict()
            for name, (start, end) in GROUPS.items():
                a, b = self.slices[name]
                c, d = self.slices['shared']
                width = cfg.latent_dim if full_geometry and name == 'geometry' else b-a+d-c
                branch = SemanticDecoder(replace(cfg, latent_dim=width))
                branch.blocks = nn.ModuleList()
                branch.spatial_context = SpatialPointContext(cfg)
                keys = {'geometry': ('opacity', 'scale', 'rotation'), 'dc': ('dc',), 'sh': ('sh',)}[name]
                branch.heads = nn.ModuleDict({k: branch.heads[k] for k in keys})
                self.branches[name] = branch

    def forward(self, z, active, xyz):
        if not self.split:
            return self.shared(z, active, xyz)
        c, d = self.slices['shared']
        result = []
        # Identical received coordinates/masks and spatial hyperparameters in all
        # branches. Plans are detached; sharing them preserves each branch's VJP.
        cache = _SharedGeometry()
        for name, branch in self.branches.items():
            a, b = self.slices[name]
            branch.spatial_context._allocation_geometry_cache = cache
            try:
                inputs = z if self.full_geometry and name == 'geometry' else torch.cat((z[..., a:b], z[..., c:d]), -1)
                result.append(branch(inputs, active, xyz))
            finally:
                del branch.spatial_context._allocation_geometry_cache
        return torch.cat(result, -1)


class SemanticAttributeCodec(nn.Module):
    """Receiver consumes only the latent, active mask and delivered XYZ."""
    def __init__(self, arm, cfg=None):
        super().__init__()
        if arm not in (*ARMS, FULL_GEOMETRY):
            raise ValueError('unknown semantic arm')
        self.arm = arm
        self.cfg = cfg or make_config(int(arm[-3:]))
        if self.cfg.latent_dim != int(arm[-3:]) or self.cfg.sh_degree != 3:
            raise ValueError('arm/config mismatch or unsupported SH degree')
        self.register_buffer('attr_mean', torch.zeros(56))
        self.register_buffer('attr_std', torch.ones(56))
        # Width-independent common encoder modules precede the latent projection.
        self.semantic_encoder = AttributeEncoder(self.cfg)
        self.semantic_decoder = AttributeDecoder(self.cfg, arm.startswith('split'), arm == FULL_GEOMETRY)
        self.slices = latent_slices(self.cfg.latent_dim, arm.startswith('split'))

    def set_scene_geometry(self, geometry):
        for module in self.modules():
            if isinstance(module, SpatialPointContext):
                module.set_scene_geometry(geometry)

    def encode(self, features, active):
        return self.semantic_encoder(features, features[..., :3], active)

    def decode(self, z, active, xyz):
        return torch.cat((xyz, self.semantic_decoder(z, active, xyz)), -1)

    def forward(self, features, active):
        side = delivered_positions(features[..., :3], active.long(), self.cfg)
        return self.decode(self.encode(features, active), active, side)


def log_shape(raw):
    """Exact matrix logarithm of SPD Gaussian covariance; no eigenvector gradient."""
    from .losses import rotation_matrix
    rotation = rotation_matrix(raw[:, 7:11])
    return (rotation * (2*raw[:, 4:7])[:, None, :]) @ rotation.transpose(-1, -2)


def error_rows(pred, target, model, directions):
    """Independent physical/group errors, one row per active Gaussian."""
    from utils.sh_utils import eval_sh
    p, t = pred[:, 3:], target[:, 3:]
    standardized = (p-t) / model.attr_std
    shape = (log_shape(pred)-log_shape(target)).square().mean((-1, -2))

    def color(raw):
        sh = torch.cat((raw[:, 11:14, None], raw[:, 14:].reshape(-1, 3, 15)), -1)
        dirs = directions if directions.ndim == 3 else directions[None]
        return eval_sh(3, sh[:, None], dirs) + .5

    return {'opacity': (p[:, 0].sigmoid()-t[:, 0].sigmoid()).square(),
            'shape': shape, 'dc': standardized[:, 8:11].square().mean(-1),
            'sh': standardized[:, 11:].square().mean(-1),
            'direction_rgb': (color(pred)-color(target)).square().mean((-1, -2))}


def attribute_objective(pred, target, geometry, model, directions, scales):
    from .data import to_raw
    terms = error_rows(to_raw(pred, geometry, model), to_raw(target.detach(), geometry, model),
                       model, directions)
    means = {k: v.mean() / scales[k] for k, v in terms.items()}
    return torch.stack(list(means.values())).mean(), means


@torch.no_grad()
def fit_loss_scales(raw_scenes):
    """Equal scene mass; train-only physical variance, fixed before initialization."""
    opacity, shape, rgb = [], [], []
    for raw in raw_scenes:
        # Bounded deterministic sample; same rows in every arm.
        ids = torch.linspace(0, len(raw)-1, min(len(raw), 32768)).round().long()
        sample = raw[ids].double()
        opacity.append(sample[:, 3].sigmoid().var(unbiased=False))
        shape.append(log_shape(sample).var(0, unbiased=False).mean())
        # Uniform-sphere SH basis energy: sum of coefficient variances / (4*pi).
        coeff = torch.cat((sample[:, 11:14, None], sample[:, 14:].reshape(-1, 3, 15)), -1)
        rgb.append(coeff.var(0, unbiased=False).sum(-1).mean()/(4*torch.pi))
    return {'opacity': float(torch.stack(opacity).mean().clamp_min(1e-4)),
            'shape': float(torch.stack(shape).mean().clamp_min(1e-4)),
            'dc': 1., 'sh': 1.,
            'direction_rgb': float(torch.stack(rgb).mean().clamp_min(1e-4))}
