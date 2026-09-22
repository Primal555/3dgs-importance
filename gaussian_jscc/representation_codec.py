"""Representation autoencoder plus independently parameterized JSCC adapters.

Only the sender sees source geometry. The clean path is an internal diagnostic,
not an over-the-air code. No source coordinate/feature skip reaches the receiver.
"""
import torch
from torch import nn
from .codec import prefix_mask
from .multiscale_codec import MultiScaleContext
from .split_codec import mlp
from .transformer_decoder import BlockSelfAttention, ReceivedTransformerDecoder


class RepresentationEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.geometry_in = mlp(9, h)
        self.appearance_in = mlp(cfg.attr_dim-6, h)
        self.geometry_context = MultiScaleContext(cfg, True)
        self.appearance_context = MultiScaleContext(cfg, True)
        self.exchange = mlp(2*h, h)
        self.self_latent = mlp(2*h, h, cfg.representation_dim)
        self.context_latent = mlp(2*h, h, cfg.representation_dim)
        self.context_gate = nn.Parameter(torch.tensor(.1))

    def forward(self, features, active):
        features = features.masked_fill(~active[..., None], 0)
        xyz = features[..., :3]
        g = self.geometry_in(torch.cat((xyz*2-1, features[..., 4:10]), -1))
        a = self.appearance_in(torch.cat((features[..., 3:4], features[..., 10:]), -1))
        g, a = (v.masked_fill(~active[..., None], 0) for v in (g, a))
        cg = self.geometry_context(g, active, xyz)
        ca = self.appearance_context(a, active, xyz)
        ca = ca + self.exchange(torch.cat((ca, cg), -1))
        y = self.self_latent(torch.cat((g, a), -1))
        y = y + self.context_gate.tanh()*self.context_latent(torch.cat((cg, ca), -1))
        # Deliberately NO channel normalization, q embedding or prefix truncation.
        return y.masked_fill(~active[..., None], 0)


class RepresentationDecoder(ReceivedTransformerDecoder):
    def __init__(self, cfg):
        super().__init__(cfg, input_dim=cfg.representation_dim)

    def forward(self, latent, active):
        return super().forward(latent, latent.new_zeros((*latent.shape[:2], self.final_norm.normalized_shape[0])), active)


class CommunicationAdapter(nn.Module):
    """Learned conditional block transform; neither tied nor analytically inverted."""
    def __init__(self, cfg, encode):
        super().__init__()
        self.cfg, self.is_encoder = cfg, encode
        channels = 2*cfg.rates[-1]
        self.input = mlp(cfg.representation_dim if encode else 2*channels, cfg.hidden)
        self.tier = nn.Embedding(4, cfg.hidden)
        self.snr = mlp(1, cfg.hidden)
        self.blocks = nn.ModuleList(BlockSelfAttention(cfg.hidden, cfg.attention_heads)
                                    for _ in range(cfg.communication_depth))
        self.output = nn.Linear(cfg.hidden, channels if encode else cfg.representation_dim)

    def forward(self, value, q, snr):
        active = q > 0
        mask = prefix_mask(q.flatten(), self.cfg.rates).reshape(*q.shape, -1)
        if not self.is_encoder:
            value = torch.cat((value.masked_fill(~mask, 0), mask.to(value)), -1)
        value = value.masked_fill(~active[..., None], 0)
        condition = self.tier(q) + self.snr(value.new_full((*q.shape, 1), float(snr)/20))
        h = (self.input(value)+condition).masked_fill(~active[..., None], 0)
        for block in self.blocks:
            h = block(h, active)
        result = self.output(h).masked_fill(~active[..., None], 0)
        if self.is_encoder:
            result = result.masked_fill(~mask, 0)
            # Same per-point soft power constraint as the research baseline.
            # Receiver is NOT supplied the source normalization factor.
            energy = result.square().sum(-1, keepdim=True)/(mask.sum(-1, keepdim=True)/2).clamp_min(1)
            result = result/(energy+self.cfg.power_floor).sqrt()
        return result


class RepresentationCommunicationCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.representation_encoder = RepresentationEncoder(cfg)
        self.representation_decoder = RepresentationDecoder(cfg)
        self.channel_encoder = CommunicationAdapter(cfg, True)
        self.channel_decoder = CommunicationAdapter(cfg, False)

    def clean(self, features, active):
        return self.representation_decoder(self.representation_encoder(features, active), active)

    def encode(self, features, xyz, q, snr):
        # xyz is retained for the transport API, never passed to the receiver.
        return self.channel_encoder(self.representation_encoder(features, q > 0), q, snr)

    def decode(self, received, q, snr):
        return self.representation_decoder(self.channel_decoder(received, q, snr), q > 0)

    def set_phase(self, phase):
        if phase not in ('representation', 'adapter', 'joint'):
            raise ValueError('unknown separated training phase')
        for name, module in self.named_children():
            is_repr = name.startswith('representation_')
            module.requires_grad_(phase == 'joint' or (is_repr == (phase == 'representation')))
            # Frozen decoder remains in autograd: input gradients must pass through.

    def module_parameters(self):
        return {name: list(module.parameters()) for name, module in self.named_children()}
