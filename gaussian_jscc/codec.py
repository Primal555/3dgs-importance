"""Fully learned Gaussian JSCC configuration, spatial aggregation and transport."""
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
    levels: tuple = (4, 8)
    planes: bool = False
    depth: int = 2
    rates: tuple = (0, 8, 16, 32)
    block_size: int = 256
    morton_bits: int = 16
    architecture: str = "learned_joint"
    loss_profile: str = "learned_v1"
    position_head: str = "learned_affine"
    individual_tiers: bool = True
    decoder_window: int = 32
    attention_heads: int = 4
    power_floor: float = .01
    xyz_loss_scale: float = .05
    geometry_weight: float = 1.
    shape_weight: float = .25
    scale_weight: float = 1.
    opacity_weight: float = 1.
    dc_weight: float = 1.
    sh_weight: float = .25
    # Reserved v4 metadata values: no runtime branch or geometry budget.
    # Keeping these two constants preserves existing learned-v4 packet hashes.
    geometry_rates: tuple = ()
    geometry_floor: float = 1e-4
    # Explicit diagnostic alternative; learned remains the unchanged default.
    position_delivery: str = 'learned'
    position_bits: int = 12

    def __post_init__(self):
        if self.position_delivery not in ('learned', 'float32', 'quantized'):
            raise ValueError('position_delivery must be learned, float32 or quantized')
        if not isinstance(self.position_bits, int) or not 1 <= self.position_bits <= 16:
            raise ValueError('position_bits must be an integer in 1..16')
        self.rates, self.levels = tuple(self.rates), tuple(self.levels)
        self.geometry_rates = tuple(self.geometry_rates)
        if self.architecture not in ('learned_joint', 'learned_split', 'learned_split_logcov'):
            raise ValueError('Supported architectures: learned_joint, learned_split, learned_split_logcov')
        if self.loss_profile != 'learned_v1' or self.position_head != 'learned_affine':
            raise ValueError('learned codecs require learned_v1 and learned_affine')
        if not self.individual_tiers or self.geometry_rates or self.geometry_floor != 1e-4:
            raise ValueError('learned codecs require individual tiers and no geometry sub-budget')
        if len(self.rates) != 4 or self.rates[0] != 0 or any(
            a >= b for a,b in zip(self.rates,self.rates[1:])
        ) or any(int(r) != r for r in self.rates):
            raise ValueError('rates must be four increasing integer complex lengths starting at zero')
        if not 0 <= self.sh_degree <= 3:
            raise ValueError('supported SH degrees: 0..3')
        if not 1 <= self.morton_bits <= 16 or self.block_size < 1:
            raise ValueError('morton_bits must be 1..16; block_size must be positive')
        if not self.levels or any(int(r) != r or not 2 <= r <= 64 for r in self.levels):
            raise ValueError('grid levels must be integers in 2..64')
        if min(self.hidden,self.grid_dim,self.depth) < 1:
            raise ValueError('hidden, grid_dim and depth must be positive')
        if self.decoder_window < 2 or self.attention_heads < 1 or self.hidden % self.attention_heads:
            raise ValueError('invalid local attention window/head dimensions')
        for name in ('power_floor','xyz_loss_scale'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('geometry','shape','scale','opacity','dc','sh'):
            value=getattr(self,name+'_weight')
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}_weight must be finite and nonnegative')

    @property
    def attr_dim(self):
        return (7 if self.architecture == 'learned_split_logcov' else 8) + 3 * (self.sh_degree + 1) ** 2

    def to_dict(self):
        result = asdict(self)
        if self.position_delivery == 'learned' and self.position_bits == 12:
            # Preserve existing v4 shared-model hashes exactly.
            result.pop('position_delivery')
            result.pop('position_bits')
        return result

    @classmethod
    def from_dict(cls, values):
        if values.get('architecture') not in ('learned_joint', 'learned_split', 'learned_split_logcov'):
            raise ValueError('Supported checkpoints: learned_joint, learned_split, learned_split_logcov')
        return cls(**values)


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
    Coordinates come from source points at the sender only.
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
        """Reusable sender interpolation plan.

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
    """Shared transport for learned_joint and the opt-in learned_split experiment."""
    def __init__(self, cfg):
        super().__init__()
        from .learned_codec import LearnedCore
        self.cfg = cfg
        self.register_buffer("attr_mean", torch.zeros(cfg.attr_dim))
        self.register_buffer("attr_std", torch.ones(cfg.attr_dim))
        if cfg.architecture in ('learned_split', 'learned_split_logcov'):
            from .split_codec import SplitLearnedCore
            self.learned = SplitLearnedCore(cfg)
        else:
            self.learned = LearnedCore(cfg)
        if cfg.position_delivery != 'learned':
            # Keep identical initialization/RNG for the paired experiment, but
            # do not train the bypassed XYZ head or claim its gradients improve.
            self.learned.heads['xyz'].requires_grad_(False)

    def encode(self, features, xyz, q, snr):
        validate_tiers(q, len(features))
        return pack(self.learned.encode(features[None], xyz[None], q[None], snr)[0], q, self.cfg.rates)

    def decode(self, symbols, q, snr, return_seed=False, delivered_xyz=None):
        validate_tiers(q, len(q))
        result = self.learned.decode(unpack(symbols, q, self.cfg.rates)[None], q[None], snr)[0]
        result = self.apply_position_delivery(result, q, delivered_xyz)
        # Optional benchmark diagnostic is final XYZ, not a bootstrap decoder.
        return (result, result[:, :3]) if return_seed else result

    def forward(self, features, xyz, q, snr, kind="awgn", return_seed=False):
        from .position_delivery import delivered_positions
        side = delivered_positions(features[..., :3], q, self.cfg)
        return self.decode(channel(self.encode(features, xyz, q, snr), snr, kind), q, snr, return_seed,
                           delivered_xyz=side)

    def apply_position_delivery(self, result, q, delivered_xyz):
        if self.cfg.position_delivery == 'learned':
            if delivered_xyz is not None:
                raise ValueError('learned decoder must not receive source coordinates')
            return result
        if delivered_xyz is None or delivered_xyz.shape != result[..., :3].shape:
            raise ValueError('explicit position decoder requires delivered XYZ with matching slots')
        if not torch.isfinite(delivered_xyz).all():
            raise ValueError('nonfinite delivered XYZ')
        # No positional residual or new context input: isolate position bypass.
        side = delivered_xyz.detach().to(result) * (q > 0)[..., None]
        return torch.cat((side, result[..., 3:]), -1)

    def forward_tiers(self, features, xyz, choices, snr, kind="awgn"):
        if choices.shape != (len(features), 4):
            raise ValueError("choices must have shape [N,4]")
        return tuple(x[0] for x in self.forward_tier_batches(
            features[None], xyz[None], choices[None], snr, kind))

    def forward_tier_batches(self, features, xyz, choices, snr, kind="awgn"):
        if choices.shape != (*features.shape[:2], 4) or features.ndim != 3:
            raise ValueError("batched choices must have shape [B,N,4]")
        if choices.requires_grad or not torch.equal(choices, F.one_hot(choices.argmax(-1), 4).to(choices)):
            raise ValueError('Use hard actions and score-function mask gradients, not ST choices')
        q = choices.argmax(-1)
        latent = self.learned.encode(features, xyz, q, snr)
        mask = prefix_mask(q.flatten(), self.cfg.rates).reshape_as(latent)
        noisy = channel(latent[mask].reshape(-1, 2), snr, kind)
        received = latent.new_zeros(latent.shape).masked_scatter(mask, noisy.flatten())
        result = self.learned.decode(received, q, snr)
        from .position_delivery import delivered_positions
        result = self.apply_position_delivery(result, q, delivered_positions(features[..., :3], q, self.cfg))
        return result, result[..., :3], (q > 0).to(features)
