"""Independent center and attribute autoencoders, clean reconstruction only.

Center sees XYZ only. Attribute decoder conditions on DECODED XYZ, never teacher
XYZ. No source-XYZ skip, side channel, shared weights or shape-scaled XYZ loss.
The clean latents are not claimed to be a communication payload.
"""
import math
import torch
from torch import nn
from .multiscale_codec import MultiScaleContext
from .split_codec import mlp
from .transformer_decoder import BlockSelfAttention


class PointEncoder(nn.Module):
    def __init__(self, cfg, input_dim, latent_dim):
        super().__init__()
        self.input = mlp(input_dim, cfg.hidden)
        self.context = MultiScaleContext(cfg, True)
        self.self_latent = mlp(cfg.hidden, cfg.hidden, latent_dim)
        self.context_latent = mlp(cfg.hidden, cfg.hidden, latent_dim)
        self.gate = nn.Parameter(torch.tensor(.1))

    def forward(self, values, xyz, active):
        values = values.masked_fill(~active[..., None], 0)
        xyz = xyz.masked_fill(~active[..., None], 0)
        h = self.input(values).masked_fill(~active[..., None], 0)
        z = self.self_latent(h) + self.gate.tanh()*self.context_latent(self.context(h, active, xyz))
        return z.masked_fill(~active[..., None], 0)


class FeatureAffine(nn.Module):
    """LayerNorm's learned affine parameters without input-dependent statistics.

    Same parameter names, shapes and deterministic initialization as LayerNorm;
    unlike LayerNorm this operation retains token feature mean and magnitude.
    It is not a normalization or an XYZ bypass.
    """
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, x):
        return x*self.weight+self.bias


class CenterDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.self_only = cfg.center_attention_scope == 'self'
        self.input = mlp(cfg.center_latent_dim, h)
        self.blocks = nn.ModuleList(BlockSelfAttention(h, cfg.attention_heads) for _ in range(cfg.decoder_depth))
        self.tap_indices = (0, (cfg.decoder_depth-1)//2, cfg.decoder_depth-1)
        readout_transform = nn.LayerNorm if cfg.center_readout_norm == 'layernorm' else FeatureAffine
        self.norms = nn.ModuleList(readout_transform(h) for _ in self.tap_indices)
        self.readout = nn.Sequential(nn.Linear(3*h, h), nn.GELU(), nn.Linear(h, 3))
        nn.init.normal_(self.readout[-1].weight, std=.02)
        nn.init.constant_(self.readout[-1].bias, .5)

    def forward(self, z, active):
        h = self.input(z.masked_fill(~active[..., None], 0)).masked_fill(~active[..., None], 0)
        taps = []
        for index, block in enumerate(self.blocks):
            h = block(h, active, self_only=self.self_only)
            if index in self.tap_indices:
                taps.append(self.norms[len(taps)](h))
        return self.readout(torch.cat(taps, -1)).masked_fill(~active[..., None], 0)


class HistoricalLightCenterDecoder(nn.Module):
    """XYZ branch of 97ef4cc MultiScaleSelfCore, adapted to clean center latent.

    MLP own-point readout + gated window/x4/x16 Context, including old slot
    encodings. This is NOT attention-free MLP, nor a reproduction of the old
    mixed-attribute JSCC system: only its receiver XYZ architecture is reused.
    """
    def __init__(self, cfg):
        super().__init__()
        self.input = mlp(cfg.center_latent_dim, cfg.hidden)
        self.context = MultiScaleContext(cfg, False)
        self.gate = nn.Parameter(torch.tensor(.1))
        self.head = nn.Linear(cfg.hidden, 3)
        self.context_head = nn.Linear(cfg.hidden, 3)
        nn.init.normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, .5)
        nn.init.normal_(self.context_head.weight, std=.02)
        nn.init.zeros_(self.context_head.bias)

    def forward(self, z, active):
        h = self.input(z.masked_fill(~active[..., None], 0)).masked_fill(~active[..., None], 0)
        pos = torch.arange(z.shape[1], device=z.device, dtype=z.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, h.shape[-1], 2, device=z.device, dtype=z.dtype)*(-math.log(10000)/h.shape[-1]))
        encoding = z.new_zeros(z.shape[1], h.shape[-1])
        encoding[:, 0::2] = torch.sin(pos*frequency)
        encoding[:, 1::2] = torch.cos(pos*frequency[:encoding[:, 1::2].shape[-1]])
        context = self.context((h+encoding[None]).masked_fill(~active[..., None], 0), active)
        return (self.head(h)+self.gate.tanh()*self.context_head(context)).masked_fill(~active[..., None], 0)


class AttributeDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.input = mlp(cfg.representation_dim-cfg.center_latent_dim, h)
        self.position = mlp(3, h)
        self.blocks = nn.ModuleList(BlockSelfAttention(h, cfg.attention_heads) for _ in range(cfg.decoder_depth))
        self.norm = nn.LayerNorm(h)
        sizes = {'opacity': 1, 'logcov': 6, 'dc': 3}
        if cfg.sh_degree:
            sizes['sh'] = 3*((cfg.sh_degree+1)**2-1)
        self.heads = nn.ModuleDict({name: nn.Linear(h, size) for name, size in sizes.items()})

    def forward(self, z, decoded_xyz, active):
        h = self.input(z.masked_fill(~active[..., None], 0))
        # In joint training this condition must NOT detach position gradients.
        h = h + self.position(decoded_xyz.masked_fill(~active[..., None], 0)*2-1)
        h = h.masked_fill(~active[..., None], 0)
        for block in self.blocks:
            h = block(h, active)
        h = self.norm(h)
        return torch.cat([head(h) for head in self.heads.values()], -1).masked_fill(~active[..., None], 0)


class CenterAttributeCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.center_encoder = PointEncoder(cfg, 3, cfg.center_latent_dim)
        self.center_decoder = CenterDecoder(cfg)
        self.attribute_encoder = PointEncoder(cfg, cfg.attr_dim, cfg.representation_dim-cfg.center_latent_dim)
        self.attribute_decoder = AttributeDecoder(cfg)
        if cfg.center_decoder_kind == 'historical_light':
            # Keep the original shared encoder/attribute initialization AND RNG
            # stream. The temporary trunk above is discarded, never optimized.
            with torch.random.fork_rng(devices=[]):
                light = HistoricalLightCenterDecoder(cfg)
            light.input.load_state_dict(self.center_decoder.input.state_dict())
            self.center_decoder = light

    def centers(self, xyz, active):
        return self.center_decoder(self.center_encoder(xyz*2-1, xyz, active), active)

    def clean(self, features, active):
        xyz = features[..., :3]
        pred_xyz = self.centers(xyz, active)
        latent = self.attribute_encoder(features[..., 3:], xyz, active)
        attributes = self.attribute_decoder(latent, pred_xyz, active)
        return torch.cat((pred_xyz, attributes), -1)

    def set_phase(self, phase):
        if phase not in ('center', 'attribute', 'joint'):
            raise ValueError('expected center, attribute or joint phase')
        for name, module in self.named_children():
            module.requires_grad_(phase == 'joint' or name.startswith('center') == (phase == 'center'))

    def module_parameters(self):
        return {name: list(module.parameters()) for name, module in self.named_children()}

    def encode(self, *args, **kwargs):
        raise ValueError('center-attribute experiment is clean-only; no trained communication adapter/payload')

    decode = encode


def center_loss(pred_xyz, source_xyz, geometry, smoothing):
    """Mean world-center smooth distance, with no covariance/scene denominator.

    Output-space world XYZ gradient <= 1/N; parameter gradients are not bounded.
    smoothing is a disclosed world-unit engineering parameter, not an axis size.
    """
    import math
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('center smoothing must be finite and positive')
    delta = (pred_xyz.double()-source_xyz.detach().double())*geometry.span.to(pred_xyz).double()
    square = delta.square().sum(-1)
    return (square/((square+smoothing*smoothing).sqrt()+smoothing)).mean().to(pred_xyz.dtype)
