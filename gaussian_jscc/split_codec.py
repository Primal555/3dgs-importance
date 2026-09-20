"""Experimental two-stream codec, ONE learned per-primitive JSCC payload.

Source XYZ supplies relative attention bias at the sender only. The receiver
uses sequence windows and never receives source XYZ or source-relative offsets.
The geometry stream retains XYZ/scale/rotation features until channel fusion;
the appearance stream receives learned geometry features, not coordinates.
No analytic anchor, coordinate repetition, or geometry symbol reservation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .codec import prefix_mask
from .learned_codec import LocalFeatureBlock


def mlp(input_dim, hidden, output_dim=None):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden if output_dim is None else output_dim))


class GeometryWindowBlock(nn.Module):
    """Pre-norm residual attention; source-relative bias in Morton windows.

    Memory is O(B*N*window*heads), never O(scene_size**2). Shifted windows
    are padded, not circular. q0 slots do not enter attention or normalization
    of source geometry. Relative offsets are discarded after encoding.
    """
    def __init__(self, cfg, shifted=False):
        super().__init__()
        self.window = cfg.decoder_window
        self.offset = self.window // 2 if shifted else 0
        self.heads = cfg.attention_heads
        self.norm = nn.LayerNorm(cfg.hidden)
        self.qkv = nn.Linear(cfg.hidden, 3 * cfg.hidden)
        self.relative_bias = mlp(3, 16, self.heads)
        self.projection = nn.Linear(cfg.hidden, cfg.hidden)
        self.ffn_norm = nn.LayerNorm(cfg.hidden)
        self.ffn = mlp(cfg.hidden, 2 * cfg.hidden, cfg.hidden)

    def forward(self, h, xyz, active):
        b, n, d = h.shape
        if n == 0:
            return h
        left, w = self.offset, self.window
        right = (-(n + left)) % w
        x = F.pad(self.norm(h), (0, 0, left, right)).reshape(-1, w, d)
        valid = F.pad(active, (left, right)).reshape(-1, w)
        points = F.pad(xyz.masked_fill(~active[..., None], 0),
                       (0, 0, left, right)).reshape(-1, w, 3)
        # Translation-invariant, window-relative sender coordinates. The scale
        # is not metadata, nor a decoded coordinate reference.
        lo = points.masked_fill(~valid[..., None], float('inf')).amin(1, keepdim=True)
        hi = points.masked_fill(~valid[..., None], -float('inf')).amax(1, keepdim=True)
        span = (hi - lo).amax(-1, keepdim=True)
        span = torch.where(valid.any(1)[:, None, None], span, torch.ones_like(span))
        relative = (points[:, :, None] - points[:, None, :]) / span.clamp_min(1e-6)[:, None]
        # Invalid slots have no effect; zero them before the MLP as well.
        pair_valid = valid[:, :, None] & valid[:, None, :]
        relative = relative.masked_fill(~pair_valid[..., None], 0)
        bias = self.relative_bias(relative).permute(0, 3, 1, 2)
        q, k, v = self.qkv(x).reshape(-1, w, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads) + bias
        safe = valid.clone()
        safe[~safe.any(-1), 0] = True
        scores = scores.masked_fill(~safe[:, None, None, :], -float('inf'))
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(-1, w, d)
        attended = (self.projection(attended) * valid[..., None]).reshape(b, -1, d)[:, left:left+n]
        h = h + attended
        return (h + self.ffn(self.ffn_norm(h))) * active[..., None]


class SplitLearnedCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.enc_geometry_in = mlp(10, h)  # XYZ + log scale + quaternion
        self.enc_appearance_in = mlp(cfg.attr_dim - 7, h)  # alpha + all SH
        self.tier = nn.Embedding(4, h)
        self.snr = mlp(1, h)
        self.enc_geometry_blocks = nn.ModuleList(GeometryWindowBlock(cfg, i % 2 == 1) for i in range(cfg.depth))
        self.enc_appearance_blocks = nn.ModuleList(GeometryWindowBlock(cfg, i % 2 == 1) for i in range(cfg.depth))
        self.enc_exchange = nn.ModuleList(mlp(2*h, h) for _ in range(cfg.depth))
        self.enc_geometry_norm = nn.LayerNorm(h)
        self.enc_appearance_norm = nn.LayerNorm(h)
        self.enc_fuse = mlp(2*h, h)
        self.enc_norm = nn.LayerNorm(h)
        self.symbol_head = nn.Linear(h, 2*cfg.rates[-1])
        # Independent learned input projections, fed by exactly the same packet.
        self.dec_geometry_in = mlp(4*cfg.rates[-1], h)
        self.dec_appearance_in = mlp(4*cfg.rates[-1], h)
        self.dec_geometry_blocks = nn.ModuleList(LocalFeatureBlock(cfg, i % 2 == 1) for i in range(cfg.depth))
        self.dec_appearance_blocks = nn.ModuleList(LocalFeatureBlock(cfg, i % 2 == 1) for i in range(cfg.depth))
        self.dec_exchange = nn.ModuleList(mlp(2*h, h) for _ in range(cfg.depth))
        self.dec_geometry_refine = nn.Sequential(nn.LayerNorm(h), mlp(h, h))
        self.dec_appearance_refine = nn.Sequential(nn.LayerNorm(h), mlp(h, h))
        sizes = {'xyz': 3, 'opacity': 1, 'scale': 3, 'rotation': 4, 'dc': 3}
        if cfg.sh_degree:
            sizes['sh'] = cfg.attr_dim - 11
        self.heads = nn.ModuleDict({key: nn.Linear(h, size) for key, size in sizes.items()})
        nn.init.normal_(self.heads['xyz'].weight, std=.02)
        nn.init.constant_(self.heads['xyz'].bias, .5)

    def condition(self, q, snr, dtype):
        value = torch.full((*q.shape, 1), float(snr)/20, device=q.device, dtype=dtype)
        return self.tier(q) + self.snr(value)

    def encode(self, features, xyz, q, snr):
        active = q > 0
        condition = self.condition(q, snr, features.dtype)
        geometry = torch.cat((features[..., :3]*2-1, features[..., 4:11]), -1)
        appearance = torch.cat((features[..., 3:4], features[..., 11:]), -1)
        g = (self.enc_geometry_in(geometry) + condition) * active[..., None]
        a = (self.enc_appearance_in(appearance) + condition) * active[..., None]
        for gb, ab, exchange in zip(self.enc_geometry_blocks, self.enc_appearance_blocks, self.enc_exchange):
            g = gb(g, xyz, active)
            a = ab(a, xyz, active)
            a = (a + exchange(torch.cat((self.enc_appearance_norm(a), self.enc_geometry_norm(g)), -1))) * active[..., None]
        h = self.enc_fuse(torch.cat((self.enc_geometry_norm(g), self.enc_appearance_norm(a)), -1))
        z = self.symbol_head(self.enc_norm(h))
        mask = prefix_mask(q.flatten(), self.cfg.rates).reshape_as(z)
        z = z * mask
        energy = z.square().sum(-1, keepdim=True) / (mask.sum(-1, keepdim=True)/2).clamp_min(1)
        return z / (energy + self.cfg.power_floor).sqrt()

    def decode(self, received, q, snr):
        active = q > 0
        mask = prefix_mask(q.flatten(), self.cfg.rates).reshape_as(received)
        packet = torch.cat((received * mask, mask.to(received)), -1)
        condition = self.condition(q, snr, received.dtype)
        # Deterministic sequence offsets carry neither point IDs nor coordinates.
        pos = torch.arange(q.shape[1], device=q.device, dtype=received.dtype)[:, None]
        freq = torch.exp(torch.arange(0, self.cfg.hidden, 2, device=q.device,
                                     dtype=received.dtype) * (-math.log(10000)/self.cfg.hidden))
        encoding = received.new_zeros(q.shape[1], self.cfg.hidden)
        encoding[:, 0::2] = torch.sin(pos*freq)
        encoding[:, 1::2] = torch.cos(pos*freq[:encoding[:, 1::2].shape[-1]])
        g = (self.dec_geometry_in(packet) + condition + encoding[None]) * active[..., None]
        a = (self.dec_appearance_in(packet) + condition + encoding[None]) * active[..., None]
        for gb, ab, exchange in zip(self.dec_geometry_blocks, self.dec_appearance_blocks, self.dec_exchange):
            g = gb(g, active)
            a = ab(a, active)
            a = (a + exchange(torch.cat((a, g), -1))) * active[..., None]
        g = self.dec_geometry_refine(g)
        a = self.dec_appearance_refine(a)
        return torch.cat([head(g if key in ('xyz', 'scale', 'rotation') else a)
                          for key, head in self.heads.items()], -1) * active[..., None]
