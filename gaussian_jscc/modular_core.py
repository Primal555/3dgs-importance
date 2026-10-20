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
        if hasattr(self,'spatial_context'):
            h = self.spatial_context(h,xyz,active)
        else:
            plan = self.blocks[0].grid.geometry_plan(xyz, active.to(features))
            for block in self.blocks:
                h = block(h, xyz, condition, active.to(features), plan) * active[..., None]
        return self.readout(h) * active[..., None]


class ChannelEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h=cfg.channel_hidden if cfg.channel_variant in ('residual','progressive') else cfg.hidden
        self.input = nn.Linear(cfg.latent_dim, h)
        self.snr = nn.Sequential(nn.Linear(1, h), nn.GELU(), nn.Linear(h, h))
        self.readout = nn.Sequential(nn.GELU(), nn.Linear(h, 2*cfg.rates[-1]))
        if cfg.channel_variant in ('residual','progressive'):
            self.blocks=nn.ModuleList(ChannelResidualBlock(h) for _ in range(3))
        if cfg.channel_power_mode != 'point':
            from .channel_power import SegmentPowerNormalizer
            self.normalizer = SegmentPowerNormalizer(cfg.rates, cfg.channel_power_mode,
                floor=cfg.power_floor, momentum=cfg.channel_power_momentum)

    def raw_signal(self, y, active, snr):
        if not math.isfinite(float(snr)):
            raise ValueError('SNR must be finite')
        condition=self.snr(y.new_full((*active.shape, 1), float(snr)/20))
        h=self.input(y)+condition
        for block in getattr(self,'blocks',()): h=block(h,condition)
        return self.readout(h)

    def forward(self, y, active, snr):
        z = self.raw_signal(y, active, snr)
        if hasattr(self, 'normalizer'):
            if y.shape[1] > self.cfg.block_size:
                raise ValueError('power controls require original codec blocks, not regrouped points')
            return self.normalizer(z, active)
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
        h=cfg.channel_hidden if cfg.channel_variant=='residual' else cfg.hidden
        self.input = nn.Linear(4*cfg.rates[-1], h)
        self.tier = nn.Embedding(len(cfg.rates), h)
        self.snr = nn.Sequential(nn.Linear(1, h), nn.GELU(), nn.Linear(h, h))
        self.readout = nn.Sequential(nn.GELU(), nn.Linear(h, cfg.latent_dim))
        if cfg.channel_variant=='residual':
            self.blocks=nn.ModuleList(ChannelResidualBlock(h) for _ in range(3))

    def forward(self, received, receive_mask, q, snr):
        if received.shape != (*q.shape, 2*self.cfg.rates[-1]) or receive_mask.shape != received.shape:
            raise ValueError('invalid receiver slot/mask shape')
        if not math.isfinite(float(snr)):
            raise ValueError('SNR must be finite')
        masked = received.masked_fill(~receive_mask.bool(), 0)
        condition = self.tier(q) + self.snr(received.new_full((*q.shape, 1), float(snr)/20))
        h = self.input(torch.cat((masked, receive_mask.to(received)), -1)) + condition
        for block in getattr(self,'blocks',()): h=block(h,condition)
        return self.readout(h)*(q>0)[..., None]


class ChannelResidualBlock(nn.Module):
    """Pointwise residual mapping; no sample normalization or extra symbols.

    Bounded affine conditioning at every block preserves feature magnitude.
    Small final weights initialize a near-identity path without dead gradients.
    """
    def __init__(self, hidden):
        super().__init__()
        self.modulation=nn.Linear(hidden,2*hidden)
        self.input=nn.Linear(hidden,2*hidden)
        self.output=nn.Linear(2*hidden,hidden)
        nn.init.normal_(self.output.weight,std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self,h,condition):
        scale,shift=self.modulation(condition).chunk(2,-1)
        x=h*(1+.1*scale.tanh())+.1*shift.tanh()
        return h+self.output(torch.nn.functional.gelu(self.input(x)))


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

    def forward(self, y_hat, active, delivered_xyz=None, *, detach_geometry=True):
        xyz = None
        if self.cfg.decoder_geometry == 'xyz':
            if delivered_xyz is None or delivered_xyz.shape != (*active.shape, 3):
                raise ValueError('semantic decoder requires matching delivered XYZ')
            if not torch.isfinite(delivered_xyz[active]).all():
                raise ValueError('nonfinite active delivered XYZ')
            xyz = (delivered_xyz.detach() if detach_geometry else delivered_xyz).to(y_hat).masked_fill(~active[..., None], 0)
        h = self.input(y_hat.masked_fill(~active[..., None], 0))
        if xyz is not None:
            h = h + self.xyz_embedding(xyz*2-1)
        if hasattr(self,'spatial_context'):
            h = self.spatial_context(h,xyz,active)
            return torch.cat([head(self.norm(h)) for head in self.heads.values()],-1)*active[...,None]
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
