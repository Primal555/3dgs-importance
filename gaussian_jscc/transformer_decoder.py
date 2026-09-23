"""Received-only block Transformer with multi-depth XYZ readout.

One token/output per Gaussian. No source coordinates, slot IDs, kNN, pooling,
or coordinate side stream. Attention spans only the caller's codec block.
"""
import torch
from torch import nn
from contextlib import contextmanager


class BlockSelfAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.GELU(), nn.Linear(4*hidden, hidden))

    def forward(self, x, active, bias=None, self_only=False):
        if x.shape[1] == 0:
            return x
        x = x.masked_fill(~active[..., None], 0)
        # A wholly dropped/padded block must not softmax over all -inf keys.
        # The temporary key contains no source information; its output is masked.
        safe = active.clone()
        safe[:, 0] |= ~active.any(dim=1)
        h = self.norm1(x)
        padding = ~safe
        if bias is not None:
            padding = torch.zeros_like(safe, dtype=x.dtype).masked_fill(~safe, float('-inf'))
        if self_only:
            if bias is not None:
                raise ValueError('self-only attention cannot also accept attention bias')
            # Exact diagonal-attention algebra: softmax over one key is 1.
            # Keep V/out projections, FFN, Pre-LN and residuals. Q/K remain in
            # the state for matched initialization but have zero influence.
            width = h.shape[-1]
            value = torch.nn.functional.linear(h, self.attention.in_proj_weight[2*width:],
                                                self.attention.in_proj_bias[2*width:])
            update = self.attention.out_proj(value)
        else:
            update = self.attention(h, h, h, key_padding_mask=padding, attn_mask=bias, need_weights=False)[0]
        x = (x + update).masked_fill(~active[..., None], 0)
        return (x + self.ff(self.norm2(x))).masked_fill(~active[..., None], 0)


class ReceivedTransformerDecoder(nn.Module):
    def __init__(self, cfg, input_dim=None):
        super().__init__()
        h = cfg.hidden
        self.input = nn.Sequential(nn.Linear(4*cfg.rates[-1] if input_dim is None else input_dim, h), nn.GELU(), nn.Linear(h, h))
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


class PositionRefinement(nn.Module):
    """Soft predicted-geometry attention, never ground-truth neighbors.

    XYZ is in global normalized bbox coordinates. No inverse local radius,
    hard kNN, detached coordinates, coordinate clamp or extra loss is used.
    """
    def __init__(self, hidden, heads):
        super().__init__()
        self.head_count = heads
        self.position = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.received = nn.Linear(hidden, hidden)
        self.log_precision = nn.Parameter(torch.zeros(heads))
        self.block = BlockSelfAttention(hidden, heads)
        self.delta = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        # Small but nonzero output: all refinement paths receive gradients at start.
        nn.init.normal_(self.delta[-1].weight, std=.002)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, x, memory, xyz, active):
        xyz = xyz.masked_fill(~active[..., None], 0)
        center = xyz.sum(1, keepdim=True) / active.sum(1, keepdim=True).clamp_min(1)[..., None]
        relative = (xyz-center).masked_fill(~active[..., None], 0)
        x = x + self.received(memory) + self.position(torch.cat([xyz, relative], -1))
        # Translation-invariant pairwise squared distances, no N*N*hidden tensor.
        squared = relative.square().sum(-1)
        distance = (squared[:, :, None]+squared[:, None, :]-2*relative.bmm(relative.transpose(1, 2))).clamp_min(0)
        precision = torch.nn.functional.softplus(self.log_precision)
        bias = (-distance[:, None]*precision[None, :, None, None]).flatten(0, 1)
        x = self.block(x, active, bias)
        xyz = (xyz+self.delta(x)).masked_fill(~active[..., None], 0)
        return x, xyz


class ProgressiveTransformerDecoder(nn.Module):
    """One initial XYZ prediction, then depth-1 learned per-point refinements.

    Each token retains its identity and original received embedding. Final XYZ
    alone is supervised by the existing objective; no new coordinate side stream.
    """
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.input = nn.Sequential(nn.Linear(4*cfg.rates[-1], h), nn.GELU(), nn.Linear(h, h))
        self.initial_block = BlockSelfAttention(h, cfg.attention_heads)
        self.initial_xyz = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))
        nn.init.normal_(self.initial_xyz[-1].weight, std=.02)
        nn.init.constant_(self.initial_xyz[-1].bias, .5)
        self.refiners = nn.ModuleList(PositionRefinement(h, cfg.attention_heads) for _ in range(cfg.decoder_depth-1))
        self.final_norm = nn.LayerNorm(h)
        sizes = {'opacity': 1, 'logcov': 6, 'dc': 3}
        if cfg.sh_degree:
            sizes['sh'] = 3*((cfg.sh_degree+1)**2-1)
        self.heads = nn.ModuleDict({key: nn.Linear(h, n) for key, n in sizes.items()})

    def forward_stages(self, packet, condition, active):
        memory = (self.input(packet.masked_fill(~active[..., None], 0))+condition).masked_fill(~active[..., None], 0)
        x = self.initial_block(memory, active)
        xyz = self.initial_xyz(x).masked_fill(~active[..., None], 0)
        stages = [xyz]
        for refiner in self.refiners:
            x, xyz = refiner(x, memory, xyz, active)
            stages.append(xyz)
        last = self.final_norm(x)
        result = torch.cat([xyz]+[head(last) for head in self.heads.values()], -1).masked_fill(~active[..., None], 0)
        return result, stages

    def forward(self, packet, condition, active):
        return self.forward_stages(packet, condition, active)[0]


@contextmanager
def capture_xyz_stages(model):
    """Optional validation diagnostics without a persistent activation cache."""
    stages, hooks = [], []
    if model.cfg.decoder_refinement == 'progressive':
        decoder = model.learned.dec_trunk
        hooks.append(decoder.initial_xyz.register_forward_hook(lambda m, a, y: stages.append(y.detach())))
        hooks.extend(r.register_forward_hook(lambda m, a, y: stages.append(y[1].detach())) for r in decoder.refiners)
    try:
        yield stages
    finally:
        for hook in hooks:
            hook.remove()
