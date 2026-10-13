"""Four-module codec. Receiver methods never access sender representation or PLY."""
import math
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
        if cfg.context_variant != 'baseline':
            from .pointnet_context import SpatialPointContext
            # Common modules and the next training random draw are matched across
            # variants. Both contexts get identical initialization in all arms.
            with torch.random.fork_rng(devices=[]):
                encoder_context = SpatialPointContext(cfg)
                decoder_context = SpatialPointContext(cfg)
            for name,context,enabled in (
                    ('semantic_encoder',encoder_context,cfg.context_variant in ('encoder','both')),
                    ('semantic_decoder',decoder_context,cfg.context_variant in ('decoder','both'))):
                if enabled:
                    module = getattr(self,name)
                    module.blocks = nn.ModuleList()
                    module.spatial_context = context
        self.training_snrs = ()  # Runtime checkpoint metadata; not a learned parameter.

    def set_scene_geometry(self, geometry):
        """Scene metadata, not trained weights; receiver binds from packet bbox."""
        for module in (self.semantic_encoder,self.semantic_decoder):
            if hasattr(module,'spatial_context'):
                module.spatial_context.set_scene_geometry(geometry)

    def set_training_snrs(self, values):
        values=tuple(float(v) for v in values)
        if any(not math.isfinite(v) for v in values): raise ValueError('training SNRs must be finite')
        self.training_snrs=values

    def resolve_condition_snr(self, actual_snr, override=None):
        actual=float(actual_snr)
        if not math.isfinite(actual): raise ValueError('actual SNR must be finite')
        if override is not None:
            value=float(override)
            if not math.isfinite(value): raise ValueError('condition SNR must be finite')
            return value
        if self.training_snrs:
            if any(not math.isfinite(v) for v in self.training_snrs): raise ValueError('training SNRs must be finite')
            return min(max(actual,min(self.training_snrs)),max(self.training_snrs))
        return actual

    def encode_representation(self, features, xyz, active):
        return self.semantic_encoder(features,xyz,active)

    def encode_full(self, features, xyz, retained, snr, *, condition_snr=None):
        if retained.shape != features.shape[:1] or retained.dtype != torch.bool:
            raise ValueError('retained must be a bool vector')
        condition=self.resolve_condition_snr(snr,condition_snr)
        return self.channel_encoder(self.semantic_encoder(features[None],xyz[None],retained[None]),retained[None],condition)[0]

    def encode(self, features, xyz, q, snr, *, condition_snr=None):
        validate_tiers(q,len(features),len(self.cfg.rates))
        return pack(self.encode_full(features,xyz,q>0,snr,condition_snr=condition_snr),q,self.cfg.rates)

    def _decode_dense(self, received, q, snr, delivered_xyz, *, condition_snr=None):
        mask = prefix_mask(q.flatten(),self.cfg.rates).reshape_as(received)
        y_hat = self.channel_decoder(received,mask,q,self.resolve_condition_snr(snr,condition_snr))
        return self.decode_representation(y_hat,q,delivered_xyz)

    def decode_representation(self, y_hat, q, delivered_xyz):
        """Receiver-only attribute synthesis; never reads the clean target y."""
        active = q>0
        if delivered_xyz is None or delivered_xyz.shape != (*q.shape,3):
            raise ValueError('explicit position decoder requires matching delivered XYZ')
        if not torch.isfinite(delivered_xyz[active]).all():
            raise ValueError('nonfinite active delivered XYZ')
        side = delivered_xyz.detach().to(y_hat).masked_fill(~active[...,None],0)
        return torch.cat((side,self.semantic_decoder(y_hat,active,side)),-1)

    def decode(self, symbols, q, snr, return_seed=False, delivered_xyz=None, *, condition_snr=None):
        validate_tiers(q,len(q),len(self.cfg.rates))
        z = unpack(symbols,q,self.cfg.rates)[None]
        side = None if delivered_xyz is None else delivered_xyz[None]
        result = self._decode_dense(z,q[None],snr,side,condition_snr=condition_snr)[0]
        return (result,result[:,:3]) if return_seed else result

    def forward(self, features, xyz, q, snr, kind='awgn', return_seed=False, *, tx_condition_snr=None,rx_condition_snr=None):
        side = delivered_positions(features[:,:3],q,self.cfg)
        return self.decode(channel(self.encode(features,xyz,q,snr,condition_snr=tx_condition_snr),snr,kind),q,snr,return_seed,side,
                           condition_snr=rx_condition_snr)

    def forward_clean_batches(self, features, xyz, active):
        if active.dtype != torch.bool or active.shape != features.shape[:2]:
            raise ValueError('active must match batched features')
        side = delivered_positions(features[...,:3],active.long(),self.cfg).masked_fill(~active[...,None],0)
        y = self.semantic_encoder(features,xyz,active)
        return torch.cat((side,self.semantic_decoder(y,active,side)),-1)

    def forward_channel_batches(self, features, xyz, choices, snr, kind='awgn', paired_noise=False, *, tx_condition_snr=None,rx_condition_snr=None):
        """Training boundary: clean target and differentiable received latent.

        Uses the same channel/prefix path as deployment. No semantic decoding
        is needed for latent-only channel initialization.
        """
        if features.ndim != 3 or choices.shape != (*features.shape[:2],len(self.cfg.rates)):
            raise ValueError('invalid batched hard tier choices')
        q = choices.argmax(-1)
        if choices.requires_grad or not torch.equal(choices,F.one_hot(q,len(self.cfg.rates)).to(choices)):
            raise ValueError('modular experiment supports hard actions only, not allocator gradients')
        y = self.semantic_encoder(features,xyz,q>0)
        tx_snr=self.resolve_condition_snr(snr,tx_condition_snr)
        rx_snr=self.resolve_condition_snr(snr,rx_condition_snr)
        z = self.channel_encoder(y,q>0,tx_snr)
        mask = prefix_mask(q.flatten(),self.cfg.rates).reshape_as(z)
        if paired_noise:
            received = channel(z.reshape(-1,2),snr,kind).reshape_as(z)*mask
        else:
            noisy = channel(z[mask].reshape(-1,2),snr,kind)
            received = torch.zeros_like(z).masked_scatter(mask,noisy.flatten())
        y_hat = self.channel_decoder(received,mask,q,rx_snr)
        return y,y_hat,q

    def forward_tier_batches(self, features, xyz, choices, snr, kind='awgn', paired_noise=False, *, tx_condition_snr=None,rx_condition_snr=None):
        _,y_hat,q = self.forward_channel_batches(features,xyz,choices,snr,kind,paired_noise,
                                                tx_condition_snr=tx_condition_snr,rx_condition_snr=rx_condition_snr)
        side = delivered_positions(features[...,:3],q,self.cfg)
        result = self.decode_representation(y_hat,q,side)
        return result,result[...,:3],(q>0).to(features)

    def forward_tiers(self, features, xyz, choices, snr, kind='awgn', *, tx_condition_snr=None,rx_condition_snr=None):
        return tuple(v[0] for v in self.forward_tier_batches(features[None],xyz[None],choices[None],snr,kind,
                       tx_condition_snr=tx_condition_snr,rx_condition_snr=rx_condition_snr))
