"""Received-only block Transformer with multi-depth XYZ readout.

One token/output per Gaussian. No source coordinates, slot IDs, kNN, pooling,
or coordinate side stream. Attention spans only the caller's codec block.
"""
import torch
from torch import nn


class BlockSelfAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.GELU(), nn.Linear(4*hidden, hidden))

    def forward(self, x, active):
        if x.shape[1] == 0:
            return x
        x = x.masked_fill(~active[..., None], 0)
        # A wholly dropped/padded block must not softmax over all -inf keys.
        # The temporary key contains no source information; its output is masked.
        safe = active.clone()
        safe[:, 0] |= ~active.any(dim=1)
        h = self.norm1(x)
        update = self.attention(h, h, h, key_padding_mask=~safe, need_weights=False)[0]
        x = (x + update).masked_fill(~active[..., None], 0)
        return (x + self.ff(self.norm2(x))).masked_fill(~active[..., None], 0)


class ReceivedTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.input = nn.Sequential(nn.Linear(4*cfg.rates[-1], h), nn.GELU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList(BlockSelfAttention(h, cfg.attention_heads) for _ in range(cfg.decoder_depth))
        # Distinct shallow/middle/deep post-block features; no pre-trunk XYZ bypass.
        self.tap_indices = (0, (cfg.decoder_depth-1)//2, cfg.decoder_depth-1)
        self.xyz_norms = nn.ModuleList(nn.LayerNorm(h) for _ in self.tap_indices)
        self.xyz_readout = nn.Sequential(nn.Linear(3*h, h), nn.GELU(), nn.Linear(h, 3))
        nn.init.normal_(self.xyz_readout[-1].weight, std=.02)
        nn.init.constant_(self.xyz_readout[-1].bias, .5)
        self.final_norm = nn.LayerNorm(h)
        sizes = {'opacity': 1, 'logcov': 6, 'dc': 3}
        if cfg.sh_degree:
            sizes['sh'] = 3*((cfg.sh_degree+1)**2-1)
        self.heads = nn.ModuleDict({key: nn.Linear(h, n) for key, n in sizes.items()})
        # Allocate LAST: shared encoder/trunk/head initialization stays identical.
        if cfg.decoder_localization == 'token_translation':
            self.localization = BlockLocalizationToken(h, cfg.attention_heads, len(self.tap_indices))

    def forward(self, packet, condition, active):
        x = (self.input(packet.masked_fill(~active[..., None], 0)) + condition)
        x = x.masked_fill(~active[..., None], 0)
        taps = []
        for index, block in enumerate(self.blocks):
            x = block(x, active)
            if index in self.tap_indices:
                taps.append(self.xyz_norms[len(taps)](x))
        xyz = self.xyz_readout(torch.cat(taps, dim=-1))
        if hasattr(self, 'localization'):
            xyz = xyz + self.localization(taps, active)
        last = self.final_norm(x)
        return torch.cat([xyz] + [head(last) for head in self.heads.values()], dim=-1).masked_fill(~active[..., None], 0)


class BlockLocalizationToken(nn.Module):
    """Learned query reads received shallow/middle/deep features, never GT XYZ.

    Read-only attention does NOT modify point tokens. One translation is added
    to all active points. Zero-start output preserves the complete base model.
    """
    def __init__(self, hidden, heads, levels):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.normal_(self.query, std=.02)
        self.query_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(levels))
        self.attention = nn.ModuleList(nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.) for _ in range(levels))
        self.ff_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(levels))
        self.ff = nn.ModuleList(nn.Sequential(nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, hidden)) for _ in range(levels))
        self.output_norm = nn.LayerNorm(hidden)
        self.translation = nn.Linear(hidden, 3)
        nn.init.zeros_(self.translation.weight)
        nn.init.zeros_(self.translation.bias)

    def forward(self, taps, active):
        query = self.query.expand(active.shape[0], -1, -1)
        if active.shape[1] == 0:
            return self.translation(self.output_norm(query))*0
        safe = active.clone()
        safe[:, 0] |= ~active.any(dim=1)
        for features, norm, attention, ff_norm, ff in zip(taps, self.query_norms, self.attention, self.ff_norms, self.ff):
            memory = features.masked_fill(~active[..., None], 0)
            query = query + attention(norm(query), memory, memory, key_padding_mask=~safe, need_weights=False)[0]
            query = query + ff(ff_norm(query))
        shift = self.translation(self.output_norm(query))
        return shift.masked_fill(~active.any(dim=1)[:, None, None], 0)
