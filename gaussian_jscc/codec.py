"""ROI-JSCC prefix transport with independently built spatial grid contexts.

The prefix mask/pack/scatter mechanism is adapted from ROI-JSCC (MIT; see
NOTICE.md). Grid splat/query is a PyTorch implementation inspired by FCGS;
no FCGS arithmetic codec, checkpoints or custom CUDA operators are required.
Lengths below always count COMPLEX channel symbols.
"""

from dataclasses import asdict, dataclass
from itertools import product
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CodecConfig:
    sh_degree: int = 3
    hidden: int = 96
    grid_dim: int = 16
    levels: tuple = (4, 8, 16)
    planes: bool = True
    depth: int = 2
    rates: tuple = (0, 8, 16, 32)
    block_size: int = 4096
    morton_bits: int = 16

    def __post_init__(self):
        self.rates, self.levels = tuple(self.rates), tuple(self.levels)
        if len(self.rates) != 4 or self.rates[0] != 0 or any(
            a >= b for a, b in zip(self.rates, self.rates[1:])
        ) or any(int(r) != r for r in self.rates):
            raise ValueError("rates must be four increasing integer complex lengths, starting at 0")
        if not 0 <= self.sh_degree <= 3:
            raise ValueError("supported SH degrees: 0..3")
        if not 1 <= self.morton_bits <= 16 or self.block_size < 1:
            raise ValueError("morton_bits must be 1..16; block_size must be positive")
        if not self.levels or any(int(r) != r or not 2 <= r <= 64 for r in self.levels):
            raise ValueError("grid levels must be integers in 2..64")
        if min(self.hidden, self.grid_dim, self.depth) < 1:
            raise ValueError("hidden, grid_dim and depth must be positive")

    @property
    def attr_dim(self):
        return 8 + 3 * (self.sh_degree + 1) ** 2

    def to_dict(self):
        return asdict(self)


def validate_tiers(q, n):
    if q.ndim != 1 or len(q) != n or q.dtype != torch.long:
        raise ValueError("q must be an int64 vector with one entry per Gaussian")
    if n and ((q < 0).any() or (q > 3).any()):
        raise ValueError("tiers must lie in 0..3")


def prefix_mask(q, rates):
    validate_tiers(q, len(q))
    lengths = torch.as_tensor(rates, device=q.device, dtype=torch.long)[q]
    return torch.arange(2 * rates[-1], device=q.device)[None] < 2 * lengths[:, None]


def pack(z, q, rates):
    """Pack real/imag pairs together, without pairing across Gaussian boundaries."""
    mask = prefix_mask(q, rates)
    if z.shape != mask.shape:
        raise ValueError("latent shape does not match the rate table")
    return z[mask].reshape(-1, 2)


def unpack(symbols, q, rates):
    mask = prefix_mask(q, rates)
    if symbols.ndim != 2 or symbols.shape[1] != 2 or symbols.numel() != int(mask.sum()):
        raise ValueError("received payload length does not match metadata")
    return symbols.new_zeros(mask.shape).masked_scatter(mask, symbols.reshape(-1))


def normalize_power(symbols):
    if not symbols.numel():
        return symbols
    # Unit average energy per complex symbol; no encoder norm passed to receiver.
    energy = symbols.square().sum(-1).mean()
    return symbols / energy.clamp_min(1e-12).sqrt()


def channel(symbols, snr, kind="awgn", generator=None):
    if kind not in ("none", "awgn", "rayleigh"):
        raise ValueError("channel must be none, awgn or rayleigh")
    if kind == "none" or not symbols.numel():
        return symbols
    if not math.isfinite(float(snr)):
        raise ValueError("SNR must be finite")
    variance = 10.0 ** (-float(snr) / 10.0)
    noise = torch.randn(symbols.shape, device=symbols.device, dtype=symbols.dtype,
                        generator=generator) * math.sqrt(variance / 2)
    if kind == "awgn":
        return symbols + noise
    # Flat fading independently per symbol; perfect receiver CSI, MMSE equalizer.
    h = torch.randn(symbols.shape, device=symbols.device, dtype=symbols.dtype,
                    generator=generator) / math.sqrt(2)
    real = h[:, 0] * symbols[:, 0] - h[:, 1] * symbols[:, 1] + noise[:, 0]
    imag = h[:, 0] * symbols[:, 1] + h[:, 1] * symbols[:, 0] + noise[:, 1]
    denom = h.square().sum(-1) + variance
    return torch.stack(((h[:, 0] * real + h[:, 1] * imag) / denom,
                        (h[:, 0] * imag - h[:, 1] * real) / denom), -1)


class GridContext(nn.Module):
    """Weighted splat -> normalize occupied vertices -> interpolate at each point.

    Bounded local dense grids (not a scene-wide dense volume). Both splatting
    and querying include self; the feature gradients flow through index_add.
    Encoder coordinates come from the source; decoder coordinates are predicted.
    Only grid bounds/indices are discrete; interpolation weights carry gradients.
    """

    def __init__(self, hidden, dim, levels, planes):
        super().__init__()
        self.project = nn.Linear(hidden, dim)
        self.levels = levels
        self.axes = [(0, 1, 2)] + ([(0, 1), (0, 2), (1, 2)] if planes else [])
        self.output_dim = dim * len(levels) * len(self.axes)
        # Nonpersistent buffers preserve compatibility with existing codec.pt files.
        for dimensions in (2, 3):
            self.register_buffer(f"corners_{dimensions}",
                                 torch.tensor(list(product((0, 1), repeat=dimensions))),
                                 persistent=False)

    def geometry_plan(self, xyz, active=None):
        """Reusable encoder interpolation plan; never cache predicted decoder xyz.

        Mask-dependent weights retain their graph for joint four-tier training.
        Plans are local to a forward call, so changing choices cannot make them stale.
        """
        single = xyz.ndim == 2
        if single:
            xyz = xyz[None]
            active = None if active is None else active[None]
        with torch.no_grad():
            if active is None:
                lower, upper = xyz.amin(1, keepdim=True), xyz.amax(1, keepdim=True)
            else:
                retained = active.detach() > .5
                retained = retained | ~retained.any(1, keepdim=True)
                lower = xyz.masked_fill(~retained[..., None], float("inf")).amin(1, keepdim=True)
                upper = xyz.masked_fill(~retained[..., None], -float("inf")).amax(1, keepdim=True)
            span = (upper - lower).clamp_min(1e-8)
        unit = ((xyz - lower) / span).clamp(0, 1)
        batches = len(xyz)
        plan = []
        for axes in self.axes:
            corners = getattr(self, f"corners_{len(axes)}")
            for resolution in self.levels:
                p = unit[..., list(axes)] * (resolution - 1)
                base = p.floor().long().clamp(max=resolution - 2)
                frac = p - base
                weight = torch.where(corners.bool(), frac[..., None, :],
                                     1 - frac[..., None, :]).prod(-1)
                contribution = weight if active is None else weight * active[..., None]
                vertex = base[..., None, :] + corners
                index = sum(vertex[..., d] * resolution ** d for d in range(len(axes)))
                vertices = resolution ** len(axes)
                index = index + torch.arange(batches, device=xyz.device)[:, None, None] * vertices
                plan.append((index, weight, contribution, batches * vertices))
        return plan

    def forward(self, h, xyz, active=None, plan=None):
        single = h.ndim == 2
        if plan is None:
            plan = self.geometry_plan(xyz, active)
        if single:
            h = h[None]
        f = self.project(h)
        dim = f.shape[-1]
        contexts = []
        for index, weight, contribution, cells in plan:
            # Accumulate features AND mass in one scatter, for all corners/blocks.
            values = torch.cat((f[..., None, :] * contribution[..., None],
                                contribution[..., None]), -1)
            accumulated = f.new_zeros((cells, dim + 1)).index_add(
                0, index.reshape(-1), values.reshape(-1, dim + 1))
            grid = accumulated[:, :dim] / accumulated[:, dim:].clamp_min(1e-8)
            contexts.append((grid[index] * weight[..., None]).sum(-2))
        result = torch.cat(contexts, -1)
        return result[0] if single else result


class ContextBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden)
        self.grid = GridContext(cfg.hidden, cfg.grid_dim, cfg.levels, cfg.planes)
        self.fuse = nn.Sequential(nn.Linear(cfg.hidden + self.grid.output_dim, cfg.hidden),
                                  nn.GELU(), nn.Linear(cfg.hidden, cfg.hidden))
        self.gate = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.Sigmoid())

    def forward(self, h, xyz, condition, active=None, plan=None):
        x = self.norm(h)
        return h + self.gate(condition) * self.fuse(torch.cat((x, self.grid(x, xyz, active, plan)), -1))


class GaussianCodec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("attr_mean", torch.zeros(cfg.attr_dim))
        self.register_buffer("attr_std", torch.ones(cfg.attr_dim))
        self.tier_emb = nn.Embedding(4, cfg.hidden)
        self.enc_condition = nn.Sequential(nn.Linear(4, cfg.hidden), nn.GELU(),
                                           nn.Linear(cfg.hidden, cfg.hidden))
        self.dec_condition = nn.Sequential(nn.Linear(1, cfg.hidden), nn.GELU(),
                                           nn.Linear(cfg.hidden, cfg.hidden))
        self.enc_in = nn.Linear(3 + cfg.attr_dim, cfg.hidden)
        self.enc_blocks = nn.ModuleList([ContextBlock(cfg) for _ in range(cfg.depth)])
        self.enc_out = nn.Linear(cfg.hidden, 2 * cfg.rates[-1])
        self.dec_in = nn.Linear(4 * cfg.rates[-1], cfg.hidden)
        self.dec_blocks = nn.ModuleList([ContextBlock(cfg) for _ in range(cfg.depth)])
        self.position_seed = nn.Linear(cfg.hidden, 3)
        self.position_updates = nn.ModuleList([nn.Linear(cfg.hidden, 3)
                                               for _ in range(cfg.depth)])
        sizes = {"opacity": 1, "scale": 3, "rotation": 4, "dc": 3}
        if cfg.sh_degree:
            sizes["sh"] = cfg.attr_dim - 11
        self.heads = nn.ModuleDict({k: nn.Linear(cfg.hidden, v) for k, v in sizes.items()})

    def encoder_conditioning(self, xyz, q, snr):
        snr_col = xyz.new_full((len(xyz), 1), float(snr) / 20)
        return self.tier_emb(q) + self.enc_condition(torch.cat((xyz * 2 - 1, snr_col), -1))

    def decoder_conditioning(self, symbols, q, snr):
        snr_col = symbols.new_full((len(q), 1), float(snr) / 20)
        return self.tier_emb(q) + self.dec_condition(snr_col)

    def encode(self, features, xyz, q, snr):
        validate_tiers(q, len(features))
        if (q == 0).any():
            raise ValueError("remove tier-0 Gaussians before building context")
        condition = self.encoder_conditioning(xyz, q, snr)
        h = self.enc_in(features) + condition
        plan = self.enc_blocks[0].grid.geometry_plan(xyz)
        for block in self.enc_blocks:
            h = block(h, xyz, condition, plan=plan)
        return normalize_power(pack(self.enc_out(h), q, self.cfg.rates))

    def decode(self, symbols, q, snr, return_seed=False):
        validate_tiers(q, len(q))
        if (q == 0).any():
            raise ValueError("decoder metadata must contain only retained Gaussians")
        if len(q) == 0:
            empty = symbols.new_empty((0, 3 + self.cfg.attr_dim))
            return (empty, symbols.new_empty((0, 3))) if return_seed else empty
        padded = unpack(symbols, q, self.cfg.rates)
        mask = prefix_mask(q, self.cfg.rates).to(padded.dtype)
        condition = self.decoder_conditioning(symbols, q, snr)
        h = self.dec_in(torch.cat((padded, mask), -1)) + condition
        seed = self.position_seed(h).sigmoid()
        xyz = seed
        for block, update in zip(self.dec_blocks, self.position_updates):
            h = block(h, xyz, condition)
            xyz = update(h).sigmoid()
        outputs = [head(h) for head in self.heads.values()]
        result = torch.cat((xyz, *outputs), -1)
        return (result, seed) if return_seed else result

    def forward(self, features, xyz, q, snr, kind="awgn", return_seed=False):
        symbols = channel(self.encode(features, xyz, q, snr), snr, kind)
        return self.decode(symbols, q, snr, return_seed)

    def forward_tiers(self, features, xyz, choices, snr, kind="awgn"):
        """Hard one-hot forward, straight-through gradients to all four logits.

        Dense latent slots exist only as a training tensor. At hard choices the
        retained outputs equal the packed codec (noiseless / same active noise).
        Dropped nodes contribute neither power, grid features nor rendered alpha.
        """
        if choices.shape != (len(features), 4):
            raise ValueError("choices must have shape [N,4]")
        return tuple(x[0] for x in self.forward_tier_batches(
            features[None], xyz[None], choices[None], snr, kind))

    def forward_tier_batches(self, features, xyz, choices, snr, kind="awgn"):
        """Independent padded blocks [B,N,D]; q0 slots do not enter context/power.

        Each block has its OWN grid bounds and average symbol energy. This is
        not equivalent to concatenating blocks into a larger codec packet.
        Choices may be hard one-hots or straight-through Gumbel choices.
        """
        if choices.shape != (*features.shape[:2], 4) or features.ndim != 3:
            raise ValueError("batched choices must have shape [B,N,4]")
        table = prefix_mask(torch.arange(4, device=features.device), self.cfg.rates).to(features.dtype)
        mask = choices @ table
        active = choices[..., 1:].sum(-1)
        embedding = choices @ self.tier_emb.weight
        snr_col = features.new_full((*features.shape[:2], 1), float(snr) / 20)
        condition = embedding + self.enc_condition(torch.cat((xyz * 2 - 1, snr_col), -1))
        h = self.enc_in(features) + condition
        plan = self.enc_blocks[0].grid.geometry_plan(xyz, active)
        for block in self.enc_blocks:
            h = block(h, xyz, condition, active, plan=plan)
        latent = self.enc_out(h)
        energy = (latent.square() * mask).sum((1, 2), keepdim=True) / (
            mask.sum((1, 2), keepdim=True) / 2).clamp_min(1.)
        # All-dropped packets have no transmitted energy. A finite training-only
        # normalization keeps counterfactual ST derivatives from exploding.
        fallback = latent.detach().square().sum(-1).mean(-1)[:, None, None].clamp_min(1e-4)
        energy = torch.where((active.detach() > .5).any(-1)[:, None, None], energy, fallback)
        normalized = latent / energy.clamp_min(1e-12).sqrt()
        received = channel(normalized.reshape(-1, 2), snr, kind).reshape_as(latent) * mask
        condition = embedding + self.dec_condition(snr_col)
        h = self.dec_in(torch.cat((received, mask), -1)) + condition
        seed = self.position_seed(h).sigmoid()
        position = seed
        for block, update in zip(self.dec_blocks, self.position_updates):
            h = block(h, position, condition, active)
            position = update(h).sigmoid()
        result = torch.cat((position, *(head(h) for head in self.heads.values())), -1)
        return result, seed, active
