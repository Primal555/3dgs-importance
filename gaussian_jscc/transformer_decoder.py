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
        # Allocate last so shared weights keep exactly the baseline initialization.
        if cfg.decoder_memory == 'received':
            self.memory_reads = nn.ModuleList(ReceivedMemoryRead(h, cfg.attention_heads)
                                             for _ in range(cfg.decoder_depth))

    def forward(self, packet, condition, active):
        x = (self.input(packet.masked_fill(~active[..., None], 0)) + condition)
        x = x.masked_fill(~active[..., None], 0)
        # Fixed within this forward, NOT detached: encoder/input gradients from
        # every read accumulate normally. No second channel draw or transmission.
        memory = x
        taps = []
        for index, block in enumerate(self.blocks):
            if hasattr(self, 'memory_reads'):
                x = self.memory_reads[index](x, memory, active)
            x = block(x, active)
            if index in self.tap_indices:
                taps.append(self.xyz_norms[len(taps)](x))
        xyz = self.xyz_readout(torch.cat(taps, dim=-1))
        last = self.final_norm(x)
        return torch.cat([xyz] + [head(last) for head in self.heads.values()], dim=-1).masked_fill(~active[..., None], 0)


class ReceivedMemoryRead(nn.Module):
    """Each Gaussian state queries immutable received-payload embeddings.

    No extra localization token, source geometry, slot IDs or coordinate branch.
    Zero-start output projection preserves the original forward and shared grads.
    """
    def __init__(self, hidden, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, x, memory, active):
        if x.shape[1] == 0:
            return x
        x = x.masked_fill(~active[..., None], 0)
        memory = memory.masked_fill(~active[..., None], 0)
        safe = active.clone()
        safe[:, 0] |= ~active.any(dim=1)
        kv = self.memory_norm(memory)
        update = self.attention(self.query_norm(x), kv, kv,
                                key_padding_mask=~safe, need_weights=False)[0]
        return (x + update).masked_fill(~active[..., None], 0)
