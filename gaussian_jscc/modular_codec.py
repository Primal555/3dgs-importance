"""Four-module codec. Receiver methods never access sender representation or PLY."""
import torch
from torch import nn
from torch.nn import functional as F
from .codec import pack, unpack, prefix_mask, validate_tiers, channel
from .position_delivery import delivered_positions
from .modular_core import SemanticEncoder, ChannelEncoder, ChannelDecoder, SemanticDecoder


class ModularGaussianCodec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer('attr_mean',torch.zeros(cfg.attr_dim))
        self.register_buffer('attr_std',torch.ones(cfg.attr_dim))
        self.semantic_encoder = SemanticEncoder(cfg)
        self.channel_encoder = ChannelEncoder(cfg)
        self.channel_decoder = ChannelDecoder(cfg)
        self.semantic_decoder = SemanticDecoder(cfg)

    def encode_representation(self, features, xyz, active):
        return self.semantic_encoder(features,xyz,active)

    def encode_full(self, features, xyz, retained, snr):
        if retained.shape != features.shape[:1] or retained.dtype != torch.bool:
            raise ValueError('retained must be a bool vector')
        return self.channel_encoder(self.semantic_encoder(features[None],xyz[None],retained[None]),retained[None],snr)[0]

    def encode(self, features, xyz, q, snr):
        validate_tiers(q,len(features),len(self.cfg.rates))
        return pack(self.encode_full(features,xyz,q>0,snr),q,self.cfg.rates)

    def _decode_dense(self, received, q, snr, delivered_xyz):
        active = q>0
        if delivered_xyz is None or delivered_xyz.shape != (*q.shape,3):
            raise ValueError('explicit position decoder requires matching delivered XYZ')
        if not torch.isfinite(delivered_xyz[active]).all():
            raise ValueError('nonfinite active delivered XYZ')
        side = delivered_xyz.detach().to(received).masked_fill(~active[...,None],0)
        mask = prefix_mask(q.flatten(),self.cfg.rates).reshape_as(received)
        y_hat = self.channel_decoder(received,mask,q,snr)
        return torch.cat((side,self.semantic_decoder(y_hat,active,side)),-1)

    def decode(self, symbols, q, snr, return_seed=False, delivered_xyz=None):
        validate_tiers(q,len(q),len(self.cfg.rates))
        z = unpack(symbols,q,self.cfg.rates)[None]
        side = None if delivered_xyz is None else delivered_xyz[None]
        result = self._decode_dense(z,q[None],snr,side)[0]
        return (result,result[:,:3]) if return_seed else result

    def forward(self, features, xyz, q, snr, kind='awgn', return_seed=False):
        side = delivered_positions(features[:,:3],q,self.cfg)
        return self.decode(channel(self.encode(features,xyz,q,snr),snr,kind),q,snr,return_seed,side)

    def forward_clean_batches(self, features, xyz, active):
        if active.dtype != torch.bool or active.shape != features.shape[:2]:
            raise ValueError('active must match batched features')
        side = delivered_positions(features[...,:3],active.long(),self.cfg)
        y = self.semantic_encoder(features,xyz,active)
        return torch.cat((side,self.semantic_decoder(y,active,side)),-1)

    def forward_tier_batches(self, features, xyz, choices, snr, kind='awgn', paired_noise=False):
        if features.ndim != 3 or choices.shape != (*features.shape[:2],len(self.cfg.rates)):
            raise ValueError('invalid batched hard tier choices')
        q = choices.argmax(-1)
        if choices.requires_grad or not torch.equal(choices,F.one_hot(q,len(self.cfg.rates)).to(choices)):
            raise ValueError('modular experiment supports hard actions only, not allocator gradients')
        y = self.semantic_encoder(features,xyz,q>0)
        z = self.channel_encoder(y,q>0,snr)
        mask = prefix_mask(q.flatten(),self.cfg.rates).reshape_as(z)
        if paired_noise:
            received = channel(z.reshape(-1,2),snr,kind).reshape_as(z)*mask
        else:
            noisy = channel(z[mask].reshape(-1,2),snr,kind)
            received = torch.zeros_like(z).masked_scatter(mask,noisy.flatten())
        side = delivered_positions(features[...,:3],q,self.cfg)
        result = self._decode_dense(received,q,snr,side)
        return result,result[...,:3],(q>0).to(features)

    def forward_tiers(self, features, xyz, choices, snr, kind='awgn'):
        return tuple(v[0] for v in self.forward_tier_batches(features[None],xyz[None],choices[None],snr,kind))
