"""Shared attribute/JSCC backbone with adaptive or truly progressive prefixes.

Encoder grids use source XYZ. Decoder windows use ONLY received feature slots,
q, SNR and packet-local sequence offsets. No reconstructed-coordinate grid,
point-ID table, reference waveform, systematic repetition or XYZ sub-budget.
Explicit XYZ delivery is handled by GaussianCodec, outside this core.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .codec import ContextBlock, prefix_mask


class LocalFeatureBlock(nn.Module):
    def __init__(self, cfg, shifted=False):
        super().__init__()
        self.window = cfg.decoder_window
        self.offset = self.window // 2 if shifted else 0
        self.norm1 = nn.LayerNorm(cfg.hidden)
        self.norm2 = nn.LayerNorm(cfg.hidden)
        self.attention = nn.MultiheadAttention(cfg.hidden, cfg.attention_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(cfg.hidden, 2 * cfg.hidden), nn.GELU(),
                                 nn.Linear(2 * cfg.hidden, cfg.hidden))

    def forward(self, h, active):
        b, n, d = h.shape
        if n == 0:
            return h
        left = self.offset
        right = (-(n + left)) % self.window
        x = F.pad(self.norm1(h), (0, 0, left, right)).reshape(-1, self.window, d)
        valid = F.pad(active, (left, right)).reshape(-1, self.window)
        # All-q0 windows must not invoke softmax on an all-masked row.
        safe = valid.clone()
        safe[~safe.any(-1), 0] = True
        out = self.attention(x, x, x, key_padding_mask=~safe, need_weights=False)[0]
        out = (out * valid[..., None]).reshape(b, -1, d)[:, left:left+n]
        h = h + out
        return (h + self.ffn(self.norm2(h))) * active[..., None]


class LearnedCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        # Separate embeddings, NOT independently budgeted channel branches.
        self.embeddings = nn.ModuleList(nn.Linear(size, h) for size in (3, 1, 3, 4, cfg.attr_dim-8))
        self.fuse = nn.Sequential(nn.Linear(5*h, h), nn.GELU(), nn.Linear(h, h))
        self.tier = nn.Embedding(len(cfg.rates), h)
        self.snr = nn.Sequential(nn.Linear(1, h), nn.GELU(), nn.Linear(h, h))
        self.enc_blocks = nn.ModuleList(ContextBlock(cfg) for _ in range(cfg.depth))
        self.enc_norm = nn.LayerNorm(h)
        self.symbol_head = nn.Linear(h, 2*cfg.rates[-1])
        self.dec_in = nn.Linear(4*cfg.rates[-1], h)
        self.dec_blocks = nn.ModuleList(LocalFeatureBlock(cfg, i % 2 == 1) for i in range(cfg.depth))
        self.dec_norm = nn.LayerNorm(h)
        sizes = {'xyz': 3, 'opacity': 1, 'scale': 3, 'rotation': 4, 'dc': 3}
        if cfg.sh_degree:
            sizes['sh'] = cfg.attr_dim-11
        self.heads = nn.ModuleDict({key: nn.Linear(h, size) for key, size in sizes.items()})
        # This is an unconstrained learned affine output, not a residual around
        # a transmitted anchor. Standard small initialization is not zero-init.
        nn.init.normal_(self.heads['xyz'].weight, std=.02)
        nn.init.constant_(self.heads['xyz'].bias, .5)

    def condition(self, q, snr, dtype):
        snr_col = torch.full((*q.shape, 1), float(snr)/20, device=q.device, dtype=dtype)
        return self.tier(q) + self.snr(snr_col)

    def _encode_features(self, features, xyz, active, condition):
        pieces = (features[..., :3]*2-1, features[..., 3:4], features[..., 4:7],
                  features[..., 7:11], features[..., 11:])
        h = self.fuse(torch.cat([layer(value) for layer, value in zip(self.embeddings, pieces)], -1))
        h = (h + condition) * active[..., None]
        plan = self.enc_blocks[0].grid.geometry_plan(xyz, active.to(features))
        for block in self.enc_blocks:
            h = block(h, xyz, condition, active.to(features), plan) * active[..., None]
        return self.symbol_head(self.enc_norm(h))

    def encode_full(self, features, xyz, active, snr):
        """One progressive codeword per point, independent of positive tiers.

        Normalize each incremental layer separately. Adding/removing a suffix
        cannot alter a previously transmitted symbol or its power scaling.
        """
        if self.cfg.prefix_mode != 'progressive':
            raise ValueError('encode_full requires progressive prefix_mode')
        snr_col = features.new_full((*active.shape, 1), float(snr)/20)
        condition = self.snr(snr_col)  # No target tier, including neighbors' tiers.
        z = self._encode_features(features, xyz, active, condition)
        layers = []
        for start, end in zip(self.cfg.rates[:-1], self.cfg.rates[1:]):
            layer = z[..., 2*start:2*end]
            energy = layer.square().sum(-1, keepdim=True) / (end-start)
            layers.append(layer / (energy + self.cfg.power_floor).sqrt())
        return torch.cat(layers, -1) * active[..., None]

    def encode(self, features, xyz, q, snr):
        active = q > 0
        if self.cfg.prefix_mode == 'progressive':
            z = self.encode_full(features, xyz, active, snr)
            return z * prefix_mask(q.flatten(), self.cfg.rates).reshape_as(z)
        condition = self.condition(q, snr, features.dtype)
        z = self._encode_features(features, xyz, active, condition)
        mask = prefix_mask(q.flatten(), self.cfg.rates).reshape_as(z)
        z = z * mask
        energy = z.square().sum(-1, keepdim=True) / (mask.sum(-1, keepdim=True)/2).clamp_min(1)
        # Smooth RMS normalization, mean complex energy <= 1. The positive floor
        # bounds normalization gain; no padding waveform to force equality.
        return z / (energy + self.cfg.power_floor).sqrt()

    def decode(self, received, q, snr):
        active = q > 0
        mask = prefix_mask(q.flatten(), self.cfg.rates).reshape_as(received)
        h = self.dec_in(torch.cat((received * mask, mask.to(received)), -1))
        h = h + self.condition(q, snr, received.dtype)
        # Deterministic offsets restart at every fixed source block. They do not
        # identify a point globally and contain no spatial coordinates.
        pos = torch.arange(q.shape[1], device=q.device, dtype=received.dtype)[:, None]
        freq = torch.exp(torch.arange(0, h.shape[-1], 2, device=q.device,
                                     dtype=received.dtype) * (-math.log(10000)/h.shape[-1]))
        encoding = torch.zeros_like(h[0])
        encoding[:, 0::2] = torch.sin(pos*freq)
        encoding[:, 1::2] = torch.cos(pos*freq[:encoding[:, 1::2].shape[-1]])
        h = (h + encoding[None]) * active[..., None]
        for block in self.dec_blocks:
            h = block(h, active)
        h = self.dec_norm(h)
        return torch.cat([head(h) for head in self.heads.values()], -1) * active[..., None]
