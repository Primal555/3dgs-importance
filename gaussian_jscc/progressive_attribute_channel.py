"""Eight independently trainable channel layers around a frozen semantic codec."""
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


@dataclass(frozen=True)
class LayerConfig:
    rates: tuple = (0, 8, 16, 24, 32, 40, 48, 56, 64)
    hidden: int = 256
    power_floor: float = .01
    neighbor_hidden: int = 64
    neighbors: int = 16

    def __post_init__(self):
        object.__setattr__(self, 'rates', tuple(self.rates))
        if (self.rates != tuple(range(0, 65, 8)) or
                any(type(x) is not int or x < 1 for x in (self.hidden, self.neighbor_hidden, self.neighbors)) or
                not math.isfinite(self.power_floor) or self.power_floor <= 0):
            raise ValueError('eight 8-symbol increments and positive channel dimensions required')

    def to_dict(self):
        return asdict(self)


def validate_tiers(q):
    if q.ndim != 2 or q.dtype != torch.long or ((q < 0) | (q > 8)).any():
        raise ValueError('expected padded int64 tiers in q0..q8')


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
        self.encoders = nn.ModuleList(ChannelMapping(d if j == 0 else 2*d, 16, c.hidden)
                                      for j in range(8))
        self.decoders = nn.ModuleList(ChannelMapping(16*(j+1)+(j+1)+(d if j else 0),
            d, c.hidden, residual=j > 0) for j in range(8))
        spatial = SimpleNamespace(latent_dim=d, rates=c.rates, channel_denoiser_mode='spatial',
            channel_denoiser_neighbors=c.neighbors, channel_denoiser_hidden=c.neighbor_hidden)
        self.neighbor = ChannelSpatialDenoiser(spatial, 8)
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
        prefix = received[..., :16*layer]
        gains = scales[:, :layer].repeat_interleave(16, dim=1)
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
        validate_tiers(q)
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
        validate_tiers(q)
        highest = int(q.max())
        if (highest < 1 or received.shape != (*q.shape, 16*highest) or
                scales.shape != (len(q), highest) or not torch.isfinite(scales).all() or
                (scales <= 0).any()):
            raise ValueError('received prefix or scale layout mismatch')
        available = torch.arange(16*highest, device=q.device) < 16*q[..., None]
        if not torch.isfinite(received[available]).all():
            raise ValueError('nonfinite received symbol')
        signal = received.masked_fill(~available, 0)
        previous = signal.new_zeros(*q.shape, self.cfg.latent_dim)
        for j in range(1, highest+1):
            previous = self.advance(j, signal, scales, previous, q >= j, snr)
        core = previous
        if neighbor and bool(self.neighbor_enabled):
            pad = 128-signal.shape[-1]
            previous = previous + self.neighbor(previous,
                torch.nn.functional.pad(signal, (0, pad)),
                torch.nn.functional.pad(available, (0, pad)), xyz, q > 0, snr)
        return previous, core

    def forward_cached(self, features, y, q, snr, *, noiseless=False, generator=None, neighbor=True):
        z, scales = self.encode(y, q, snr)
        # One fixed maximum-width draw per point makes all prefixes share noise.
        noise = channel(z.new_zeros(*q.shape, 64, 2), snr,
                        'none' if noiseless else 'awgn', generator).flatten(-2)
        received = z + noise[..., :z.shape[-1]]
        side = delivered_positions(features[..., :3], q, self.cfg)
        yh, core = self.receive(received, scales, q, side, snr, neighbor=neighbor)
        pred = self.semantic.decode(yh, q > 0, side)
        mask = torch.arange(z.shape[-1], device=q.device) < 16*q[..., None]
        power = z.square().masked_fill(~mask, 0).sum() / (mask.sum()/2).clamp_min(1)
        return pred, yh, dict(core=core, scales=scales, received=received, transmitted_power=power)


def configure_stage(model, stage, lr):
    if stage not in range(1, 10) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('stage 1..8, or 9 for mixed-neighbor training, and positive LR required')
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    modules = [model.neighbor] if stage == 9 else [model.encoders[stage-1], model.decoders[stage-1]]
    for module in modules:
        module.requires_grad_(True)
    model.neighbor_enabled.fill_(stage == 9)
    model.active_stage = stage
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)


def frozen_digest(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return state_digest({k: v for k, v in model.state_dict().items() if k not in names and k != 'neighbor_enabled'})
