"""Sender-only geometric neighbor attention, not a full Point Transformer port.

Exact kNN is restricted to each processing block (also at pooled scales).
No neighbor indices, source centers or relative positions enter the receiver.
"""
import torch
from torch import nn
from .split_codec import mlp


def gather_neighbors(values, indices):
    batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch, indices]


def geometric_neighbors(xyz, active, count):
    """Non-learned sender graph; mask dropped slots, include the point itself.

    Pairwise distances are O(B*N*N) inside a bounded block, not a full scene.
    Return safe indices even for an entirely dropped block.
    """
    with torch.no_grad():
        points = xyz.detach().masked_fill(~active[..., None], 0)
        distances = torch.cdist(points.float(), points.float(), compute_mode='donot_use_mm_for_euclid_dist')
        # Always retain self, including duplicate centers with tied distances.
        distances.diagonal(dim1=-2, dim2=-1).fill_(-1)
        distances = distances.masked_fill(~active[:, None, :], float('inf'))
        indices = distances.topk(min(count, xyz.shape[1]), dim=-1, largest=False).indices
        valid = active[..., None] & gather_neighbors(active[..., None], indices).squeeze(-1)
        offsets = points[:, :, None] - gather_neighbors(points, indices)
        offsets = offsets.masked_fill(~valid[..., None], 0)
        # Local radius normalization is a sender feature transform, NOT metadata.
        radius = offsets.norm(dim=-1).amax(-1, keepdim=True).clamp_min(1e-6)
        relative = offsets / radius[..., None]
    return indices, valid, relative


class GeometricPointBlock(nn.Module):
    """Grouped neighbor attention with geometry in both weights AND values.

    Pre-norm + residual FFN. One weight per head/group and neighbor, computed
    from q_i-k_j plus relative position. No global scene attention or side bits.
    """
    def __init__(self, cfg, shifted=False):
        super().__init__()
        self.neighbors = cfg.encoder_neighbors
        self.groups = cfg.attention_heads
        self.norm = nn.LayerNorm(cfg.hidden)
        self.qkv = nn.Linear(cfg.hidden, 3*cfg.hidden)
        self.position = mlp(4, cfg.hidden)
        self.relation = nn.Sequential(nn.LayerNorm(cfg.hidden), mlp(cfg.hidden, cfg.hidden, self.groups))
        self.projection = nn.Linear(cfg.hidden, cfg.hidden)
        self.ffn_norm = nn.LayerNorm(cfg.hidden)
        self.ffn = mlp(cfg.hidden, 2*cfg.hidden, cfg.hidden)

    def forward(self, h, xyz, active):
        b,n,d = h.shape
        if n == 0:
            return h
        indices, valid, relative = geometric_neighbors(xyz, active, self.neighbors)
        x = self.norm(h.masked_fill(~active[..., None], 0))
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        position = self.position(torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1))
        relation = q[:, :, None] - gather_neighbors(k, indices) + position
        scores = self.relation(relation)
        safe = valid.clone()
        safe[..., 0] |= ~safe.any(-1)
        weights = scores.masked_fill(~safe[..., None], -float('inf')).softmax(dim=2)
        values = (gather_neighbors(v, indices) + position).masked_fill(~valid[..., None], 0)
        attended = (weights[..., None] * values.reshape(b,n,-1,self.groups,d//self.groups)).sum(2).reshape(b,n,d)
        h = h.masked_fill(~active[..., None], 0) + self.projection(attended)*active[..., None]
        return (h + self.ffn(self.ffn_norm(h)))*active[..., None]
