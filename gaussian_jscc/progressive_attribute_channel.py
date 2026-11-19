"""Independently trainable channel layers around a frozen semantic codec."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import SimpleNamespace

import torch
from torch import nn

from .channel_spatial import ChannelSpatialDenoiser
from .codec import channel
from .modular_config import ModularCodecConfig
from .position_delivery import delivered_positions
from .semantic_attribute_codec import SemanticAttributeCodec
from .successive_channel import ChannelMapping, normalize_segment


EIGHT_RATES = tuple(range(0, 65, 8))
THREE_RATES = (0, 8, 24, 48)
EIGHT_FAMILY = 'progressive_attribute_eight_v1'
THREE_FAMILY = 'progressive_attribute_three_v1'


@dataclass(frozen=True)
class LayerConfig:
    rates: tuple = EIGHT_RATES
    hidden: int = 256
    power_floor: float = .01
    neighbor_hidden: int = 64
    neighbors: int = 16

    def __post_init__(self):
        object.__setattr__(self, 'rates', tuple(self.rates))
        if (self.rates not in (EIGHT_RATES, THREE_RATES) or
                any(type(x) is not int or x < 1 for x in (self.hidden, self.neighbor_hidden, self.neighbors)) or
                not math.isfinite(self.power_floor) or self.power_floor <= 0):
            raise ValueError('supported progressive rate table and positive channel dimensions required')

    def to_dict(self):
        return asdict(self)

    @property
    def layer_count(self):
        return len(self.rates)-1

    @property
    def family(self):
        return EIGHT_FAMILY if self.rates == EIGHT_RATES else THREE_FAMILY


def validate_tiers(q, layers=8):
    if q.ndim != 2 or q.dtype != torch.long or ((q < 0) | (q > layers)).any():
        raise ValueError(f'expected padded int64 tiers in q0..q{layers}')


def state_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def load_semantic_slot(path):
    """Accept either semantic family once its fixed final checkpoint is ready."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('family') != 'semantic_attribute_controls_v1':
        raise ValueError('expected semantic_attribute_controls_v1 checkpoint')
    settings = saved['plan']['settings']
    if saved['progress']['step'] != settings['bootstrap_steps'] + settings['render_steps']:
        raise ValueError('semantic slot requires the completed fixed-final weights')
    semantic = SemanticAttributeCodec(saved['arm'], ModularCodecConfig.from_dict(saved['config']))
    semantic.load_state_dict(saved['model'], strict=True)
    semantic.eval().requires_grad_(False)
    return semantic, saved


class LayeredAttributeCodec(nn.Module):
    def __init__(self, semantic, layer_config=None):
        super().__init__()
        self.semantic = semantic.eval().requires_grad_(False)
        self.cfg = semantic.cfg  # Geometry/attribute conventions remain those of the semantic source.
        self.layer_config = layer_config or LayerConfig()
        c, d = self.layer_config, self.cfg.latent_dim
        self.prefix_widths = tuple(2*rate for rate in c.rates)
        self.segment_widths = tuple(b-a for a, b in zip(self.prefix_widths, self.prefix_widths[1:]))
        self.encoders = nn.ModuleList(ChannelMapping(d if j == 0 else 2*d, width, c.hidden)
                                      for j, width in enumerate(self.segment_widths))
        self.decoders = nn.ModuleList(ChannelMapping(2*rate+j+(d if j > 1 else 0),
            d, c.hidden, residual=j > 1) for j, rate in enumerate(c.rates[1:], 1))
        spatial = SimpleNamespace(latent_dim=d, rates=c.rates, channel_denoiser_mode='spatial',
            channel_denoiser_neighbors=c.neighbors, channel_denoiser_hidden=c.neighbor_hidden)
        self.neighbor = ChannelSpatialDenoiser(spatial, c.layer_count)
        self.register_buffer('neighbor_enabled', torch.tensor(False))
        self.active_stage = None

    @property
    def attr_mean(self):
        return self.semantic.attr_mean

    @property
    def attr_std(self):
        return self.semantic.attr_std

    def train(self, mode=True):
        super().train(mode)
        self.semantic.eval()
        return self

    def set_scene_geometry(self, geometry):
        self.semantic.set_scene_geometry(geometry)
        self.neighbor.set_scene_geometry(geometry)

    def identity(self):
        spec = dict(semantic=self.cfg.to_dict(), arm=self.semantic.arm, channel=self.layer_config.to_dict())
        return hashlib.sha256((json.dumps(spec, sort_keys=True)+state_digest(self.state_dict())).encode()).hexdigest()

    def advance(self, layer, received, scales, previous, active, snr):
        prefix = received[..., :self.prefix_widths[layer]]
        gains = torch.cat([scales[:, j:j+1].expand(-1, width)
                           for j, width in enumerate(self.segment_widths[:layer])], dim=1)
        inputs = [prefix*gains[:, None], scales[:, None, :layer].log2().expand(-1, active.shape[1], -1)]
        if layer > 1:
            inputs.append(previous)
        delta = self.decoders[layer-1](torch.cat(inputs, -1), snr)
        return previous + delta.masked_fill(~active[..., None], 0)

    def encode(self, y, q, snr):
        """Encode a fixed delivery plan; each layer uses its actual donors.

        Progressive reception caps this plan at a layer boundary. It preserves
        all earlier donor sets, scales and symbols. A new allocation map is a
        new delivery plan and may legitimately require re-encoding.
        """
        validate_tiers(q, self.layer_config.layer_count)
        active = q > 0
        if y.shape != (*q.shape, self.cfg.latent_dim) or not torch.isfinite(y[active]).all():
            raise ValueError('invalid semantic latent')
        highest = int(q.max())
        if highest == 0:
            raise ValueError('channel batch must contain a retained point')
        y = y.masked_fill(~active[..., None], 0)
        proxy = torch.zeros_like(y)
        segments, scales = [], []
        for j in range(1, highest+1):
            donors = q >= j
            inputs = y if j == 1 else torch.cat((y, (y-proxy).detach()), -1)
            signal, gain = normalize_segment(self.encoders[j-1](inputs, snr), donors,
                                             self.layer_config.power_floor)
            segments.append(signal)
            scales.append(gain)
            if j < highest:
                with torch.no_grad():
                    proxy = self.advance(j, torch.cat(segments, -1), torch.stack(scales, 1), proxy, donors, snr)
        signal = torch.cat(segments, -1)
        gains = torch.stack(scales, 1)
        return signal, gains

    def receive(self, received, scales, q, xyz, snr, *, neighbor=True):
        validate_tiers(q, self.layer_config.layer_count)
        highest = int(q.max())
        if (highest < 1 or received.shape != (*q.shape, self.prefix_widths[highest]) or
                scales.shape != (len(q), highest) or not torch.isfinite(scales).all() or
                (scales <= 0).any()):
            raise ValueError('received prefix or scale layout mismatch')
        available = self.symbol_mask(q, received.shape[-1])
        if not torch.isfinite(received[available]).all():
            raise ValueError('nonfinite received symbol')
        signal = received.masked_fill(~available, 0)
        previous = signal.new_zeros(*q.shape, self.cfg.latent_dim)
        for j in range(1, highest+1):
            previous = self.advance(j, signal, scales, previous, q >= j, snr)
        core = previous
        if neighbor and bool(self.neighbor_enabled):
            pad = self.prefix_widths[-1]-signal.shape[-1]
            previous = previous + self.neighbor(previous,
                torch.nn.functional.pad(signal, (0, pad)),
                torch.nn.functional.pad(available, (0, pad)), xyz, q > 0, snr)
        return previous, core

    def symbol_mask(self, q, width):
        lengths = torch.as_tensor(self.prefix_widths, device=q.device)[q]
        return torch.arange(width, device=q.device) < lengths[..., None]

    @staticmethod
    def transmit(z, snr, *, noiseless=False, generator=None):
        # Keep all prefixes and both rate tables paired to the same noise slots.
        if noiseless:
            return z
        noise = channel(z.new_zeros(*z.shape[:-1], EIGHT_RATES[-1], 2), snr,
                        'awgn', generator).flatten(-2)
        return z + noise[..., :z.shape[-1]]

    def forward_cached(self, features, y, q, snr, *, noiseless=False, generator=None, neighbor=True):
        z, scales = self.encode(y, q, snr)
        received = self.transmit(z, snr, noiseless=noiseless, generator=generator)
        side = delivered_positions(features[..., :3], q, self.cfg)
        yh, core = self.receive(received, scales, q, side, snr, neighbor=neighbor)
        pred = self.semantic.decode(yh, q > 0, side)
        mask = self.symbol_mask(q, z.shape[-1])
        power = z.square().masked_fill(~mask, 0).sum() / (mask.sum()/2).clamp_min(1)
        return pred, yh, dict(core=core, scales=scales, received=received, transmitted_power=power)


def configure_stage(model, stage, lr):
    neighbor_stage = model.layer_config.layer_count+1
    if ((stage != 'joint' and stage not in range(1, neighbor_stage+1)) or
            not math.isfinite(lr) or lr <= 0):
        raise ValueError(f'joint or stage 1..{neighbor_stage} and positive LR required')
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    if stage == 'joint':
        modules = (model.encoders, model.decoders)
    elif stage == neighbor_stage:
        modules = (model.neighbor,)
    else:
        modules = (model.encoders[stage-1], model.decoders[stage-1])
    for module in modules:
        module.requires_grad_(True)
    model.neighbor_enabled.fill_(stage == neighbor_stage)
    model.active_stage = stage
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)


def frozen_digest(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return state_digest({k: v for k, v in model.state_dict().items() if k not in names and k != 'neighbor_enabled'})
