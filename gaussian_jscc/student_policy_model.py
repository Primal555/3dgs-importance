"""Pointwise detail, two-scale residual context and hierarchical action heads."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .student_policy_graph import pooled_mean


@dataclass
class StudentConfig:
    attribute_dim: int
    latent_dim: int = 48
    visibility_dim: int = 43
    width: int = 128
    coarse_width: int = 256
    blocks: int = 2
    expansion: int = 2
    chunk_size: int = 16384

    def __post_init__(self):
        if min(asdict(self).values()) < 1: raise ValueError('student dimensions must be positive')

    def to_dict(self): return asdict(self)


def mlp(source, hidden, target):
    return nn.Sequential(nn.Linear(source, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, target))


def pool(values, cluster, count):
    mean = pooled_mean(values, cluster, count)
    maximum = values.new_full((count, values.shape[-1]), -torch.inf)
    maximum = maximum.scatter_reduce(0, cluster[:, None].expand_as(values), values, reduce='amax', include_self=True)
    return torch.cat((mean, maximum), -1)


class SpatialResidual(nn.Module):
    def __init__(self, width, expansion):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.spatial = nn.Sequential(nn.Linear(width+3, width), nn.GELU())
        self.reduce = nn.Linear(2*width, width)
        self.pointwise = mlp(width, width*expansion, width)

    def forward(self, h, xyz, neighbors):
        relative = xyz[neighbors]-xyz[:, None]
        radius = relative.norm(dim=-1).amax(-1).clamp_min(1e-6)
        normalized = relative/radius[:, None, None]
        x = self.norm(h)
        messages = self.spatial(torch.cat((x[neighbors]-x[:, None], normalized), -1))
        aggregated = self.reduce(torch.cat((messages.mean(1), messages.amax(1)), -1))
        return h+self.pointwise(x+aggregated)


class StudentPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.attribute_encoder = mlp(cfg.attribute_dim, 64, 64)
        self.latent_encoder = mlp(cfg.latent_dim, 64, 64)
        self.visibility_encoder = mlp(cfg.visibility_dim, 32, 32)
        self.stem = mlp(160, cfg.width, cfg.width)
        self.pool1 = nn.Linear(2*cfg.width, cfg.width)
        self.pool2 = nn.Linear(2*cfg.width, cfg.coarse_width)
        self.local1 = nn.ModuleList(SpatialResidual(cfg.width, cfg.expansion) for _ in range(cfg.blocks))
        self.local2 = nn.ModuleList(SpatialResidual(cfg.coarse_width, cfg.expansion) for _ in range(cfg.blocks))
        self.coarse_readout = nn.Linear(cfg.coarse_width, cfg.width)
        self.global_encoder = nn.Sequential(nn.Linear(2*cfg.coarse_width+4, 64), nn.GELU())
        self.fusion = mlp(3*cfg.width+64, cfg.width, cfg.width)
        self.keep_head = mlp(cfg.width, 64, 2)
        self.tier_head = mlp(cfg.width, 64, 3)
        for key, dimension in (('attribute', cfg.attribute_dim), ('latent', cfg.latent_dim), ('visibility', cfg.visibility_dim)):
            self.register_buffer(key+'_mean', torch.zeros(dimension))
            self.register_buffer(key+'_scale', torch.ones(dimension))
        self.register_buffer('normalization_ready', torch.tensor(False))

    @torch.no_grad()
    def fit_normalizers(self, attributes, latent, visibility):
        if bool(self.normalization_ready): raise ValueError('student normalization is immutable once fitted')
        for key, values in zip(('attribute', 'latent', 'visibility'), (attributes, latent, visibility)):
            if not torch.isfinite(values).all() or len(values) < 1: raise ValueError('invalid normalization data')
            values = values.double()
            getattr(self, key+'_mean').copy_(values.mean(0))
            getattr(self, key+'_scale').copy_(values.std(0, unbiased=False).clamp_min(1e-3))
        self.normalization_ready.fill_(True)

    def _transform(self, values, key):
        values = values.to(self.attribute_mean.device)
        return torch.asinh((values-getattr(self, key+'_mean'))/getattr(self, key+'_scale'))

    def _stem(self, a, y, v):
        return self.stem(torch.cat((self.attribute_encoder(self._transform(a, 'attribute')),
            self.latent_encoder(self._transform(y, 'latent')),
            self.visibility_encoder(self._transform(v, 'visibility'))), -1))

    def _heads(self, h, fine, coarse, global_feature, fine_ids, coarse_ids):
        inputs = torch.cat((h, fine[fine_ids], coarse[coarse_ids], global_feature.expand(len(h), -1)), -1)
        features = self.fusion(inputs)
        return torch.cat((self.keep_head(features), self.tier_head(features)), -1)

    def forward(self, attributes, latent, visibility, graph, controls):
        if not bool(self.normalization_ready): raise ValueError('fit training feature normalizers first')
        count = len(attributes); chunk = self.cfg.chunk_size
        if latent.shape != (count, self.cfg.latent_dim) or visibility.shape != (count, self.cfg.visibility_dim):
            raise ValueError('student input rows/dimensions differ')
        if attributes.shape != (count, self.cfg.attribute_dim) or len(graph['cluster1']) != count:
            raise ValueError('student attribute/graph rows differ')
        device = self.attribute_mean.device
        if graph['cluster1'].device != device: raise ValueError('move the immutable graph to the student device')
        def run(function, *args):
            return checkpoint(function, *args, use_reentrant=False) if self.training and torch.is_grad_enabled() else function(*args)
        h = torch.cat([run(self._stem, attributes[i:i+chunk], latent[i:i+chunk], visibility[i:i+chunk])
                       for i in range(0, count, chunk)])
        fine = self.pool1(pool(h, graph['cluster1'], len(graph['coords1'])))
        for layer in self.local1: fine = run(layer, fine, graph['coords1'], graph['neighbors1'])
        coarse = self.pool2(pool(fine, graph['cluster2'], len(graph['coords2'])))
        for layer in self.local2: coarse = run(layer, coarse, graph['coords2'], graph['neighbors2'])
        summary = torch.cat((coarse.mean(0), coarse.amax(0), controls.to(coarse)))
        global_feature = self.global_encoder(summary)[None]
        coarse = self.coarse_readout(coarse)
        coarse_ids = graph['cluster2'][graph['cluster1']]
        output = torch.cat([run(self._heads, h[i:i+chunk], fine, coarse, global_feature,
                          graph['cluster1'][i:i+chunk], coarse_ids[i:i+chunk]) for i in range(0, count, chunk)])
        return output[:, :2], output[:, 2:]


def hard_tiers(keep_logits, tier_logits):
    if keep_logits.shape != (len(tier_logits), 2) or tier_logits.shape[1:] != (3,):
        raise ValueError('student heads must be [N,2] and [N,3]')
    return torch.where(keep_logits[:, 1] >= keep_logits[:, 0], tier_logits.argmax(-1)+1, 0)


@torch.no_grad()
def teacher_metrics(keep_logits, tier_logits, teacher_q):
    target = teacher_q.to(keep_logits.device)
    q = hard_tiers(keep_logits, tier_logits)
    retained = target > 0
    return dict(teacher_action_accuracy=float((q == target).float().mean()),
        teacher_keep_accuracy=float(((q > 0) == retained).float().mean()),
        teacher_positive_tier_accuracy=float((tier_logits[retained].argmax(-1)+1 == target[retained]).float().mean())
            if retained.any() else None,
        student_tier_counts=torch.bincount(q, minlength=4).tolist(),
        teacher_tier_counts=torch.bincount(target, minlength=4).tolist(),
        action_accuracy_by_teacher_tier=[float((q[target == k] == k).float().mean())
            if (target == k).any() else None for k in range(4)])


def teacher_loss(keep_logits, tier_logits, teacher_q):
    target = teacher_q.to(keep_logits.device)
    keep = F.cross_entropy(keep_logits, (target > 0).long())
    retained = target > 0
    tier = (F.cross_entropy(tier_logits[retained], target[retained]-1) if retained.any()
            else tier_logits.sum()*0)
    return keep+tier, dict(keep_ce=float(keep.detach()), tier_ce=float(tier.detach()))
