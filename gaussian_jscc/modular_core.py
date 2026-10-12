"""Semantic representation and channel modules with explicit information boundaries."""
import torch
from torch import nn
from .codec import ContextBlock


class SemanticEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.embeddings = nn.ModuleList(nn.Linear(size, h) for size in (3, 1, 3, 4, cfg.attr_dim-8))
        self.fuse = nn.Sequential(nn.Linear(5*h, h), nn.GELU(), nn.Linear(h, h))
        self.condition = nn.Parameter(torch.zeros(h))
        self.blocks = nn.ModuleList(ContextBlock(cfg) for _ in range(cfg.depth))
        self.readout = nn.Linear(h, cfg.latent_dim)

    def forward(self, features, xyz, active):
        features = features.masked_fill(~active[..., None], 0)
        xyz = xyz.masked_fill(~active[..., None], 0)
        pieces = (features[..., :3]*2-1, features[..., 3:4], features[..., 4:7],
                  features[..., 7:11], features[..., 11:])
        h = self.fuse(torch.cat([layer(x) for layer, x in zip(self.embeddings, pieces)], -1))
        condition = self.condition.expand_as(h)
        h = (h + condition) * active[..., None]
        plan = self.blocks[0].grid.geometry_plan(xyz, active.to(features))
        for block in self.blocks:
            h = block(h, xyz, condition, active.to(features), plan) * active[..., None]
        return self.readout(h) * active[..., None]
