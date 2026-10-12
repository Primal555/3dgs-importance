"""Semantic representation and channel modules with explicit information boundaries."""
import math
import torch
from torch import nn
from .codec import ContextBlock
from .learned_codec import LocalFeatureBlock


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


class ChannelEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg.latent_dim, cfg.hidden)
        self.snr = nn.Sequential(nn.Linear(1, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.hidden))
        self.readout = nn.Sequential(nn.GELU(), nn.Linear(cfg.hidden, 2*cfg.rates[-1]))

    def forward(self, y, active, snr):
        if not math.isfinite(float(snr)):
            raise ValueError('SNR must be finite')
        z = self.readout(self.input(y) + self.snr(y.new_full((*active.shape, 1), float(snr)/20)))
        layers = []
        for start, end in zip(self.cfg.rates[:-1], self.cfg.rates[1:]):
            segment = z[..., 2*start:2*end]
            energy = segment.square().sum(-1, keepdim=True)/(end-start)
            layers.append(segment/(energy+self.cfg.power_floor).sqrt())
        return torch.cat(layers, -1)*active[..., None]


class ChannelDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(4*cfg.rates[-1], cfg.hidden)
        self.tier = nn.Embedding(len(cfg.rates), cfg.hidden)
        self.snr = nn.Sequential(nn.Linear(1, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.hidden))
        self.readout = nn.Sequential(nn.GELU(), nn.Linear(cfg.hidden, cfg.latent_dim))

    def forward(self, received, receive_mask, q, snr):
        if received.shape != (*q.shape, 2*self.cfg.rates[-1]) or receive_mask.shape != received.shape:
            raise ValueError('invalid receiver slot/mask shape')
        if not math.isfinite(float(snr)):
            raise ValueError('SNR must be finite')
        masked = received.masked_fill(~receive_mask.bool(), 0)
        condition = self.tier(q) + self.snr(received.new_full((*q.shape, 1), float(snr)/20))
        h = self.input(torch.cat((masked, receive_mask.to(received)), -1)) + condition
        return self.readout(h)*(q>0)[..., None]


class SemanticDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg.latent_dim, cfg.hidden)
        self.blocks = nn.ModuleList(LocalFeatureBlock(cfg, i%2 == 1) for i in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.hidden)
        sizes = {'opacity':1, 'scale':3, 'rotation':4, 'dc':3}
        if cfg.sh_degree:
            sizes['sh'] = cfg.attr_dim-11
        self.heads = nn.ModuleDict({key: nn.Linear(cfg.hidden, size) for key,size in sizes.items()})
        if cfg.decoder_geometry == 'xyz':
            self.xyz_embedding = nn.Sequential(nn.Linear(3, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.hidden))
            nn.init.zeros_(self.xyz_embedding[-1].weight)
            nn.init.zeros_(self.xyz_embedding[-1].bias)

    def forward(self, y_hat, active, delivered_xyz=None):
        xyz = None
        if self.cfg.decoder_geometry == 'xyz':
            if delivered_xyz is None or delivered_xyz.shape != (*active.shape, 3):
                raise ValueError('semantic decoder requires matching delivered XYZ')
            if not torch.isfinite(delivered_xyz[active]).all():
                raise ValueError('nonfinite active delivered XYZ')
            xyz = delivered_xyz.detach().to(y_hat).masked_fill(~active[..., None], 0)
        h = self.input(y_hat.masked_fill(~active[..., None], 0))
        if xyz is not None:
            h = h + self.xyz_embedding(xyz*2-1)
        n, d = h.shape[1:]
        pos = torch.arange(n,device=h.device,dtype=h.dtype)[:,None]
        freq = torch.exp(torch.arange(0,d,2,device=h.device,dtype=h.dtype)*(-math.log(10000)/d))
        encoding = h.new_zeros(n,d)
        encoding[:,0::2] = torch.sin(pos*freq)
        encoding[:,1::2] = torch.cos(pos*freq[:encoding[:,1::2].shape[-1]])
        h = (h + encoding)*active[..., None]
        for block in self.blocks:
            h = block(h,active,xyz)
        h = self.norm(h)
        return torch.cat([head(h) for head in self.heads.values()],-1)*active[...,None]
