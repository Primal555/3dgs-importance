"""One payload y=[metric XYZ, attributes]; receiver never reads side XYZ."""
import torch
from torch import nn
from .modular_codec import ModularGaussianCodec
from .modular_core import SemanticEncoder, SemanticDecoder, ChannelEncoder
from .progressive_channel import ProgressiveChannelDecoder
from .pointnet_context import SpatialPointContext
from .codec import channel, unpack, validate_tiers, prefix_mask


class PayloadSemantic(nn.Module):
    def __init__(self, cfg, encoder):
        super().__init__()
        self.cfg=cfg; self.encoder=encoder
        component=cfg.attribute_config()
        self.attributes=SemanticEncoder(component) if encoder else SemanticDecoder(component)
        side='encoder' if encoder else 'decoder'
        if cfg.context_variant in (side,'both'):
            self.attributes.blocks=nn.ModuleList()
            self.attributes.spatial_context=SpatialPointContext(component)
        self.register_buffer('scene_span',torch.ones(3),persistent=False)
        self.geometry_ready=False

    @property
    def heads(self):
        return self.attributes.heads

    def set_scene_geometry(self, geometry):
        self.scene_span.copy_(geometry.span.to(self.scene_span))
        self.geometry_ready=True
        if hasattr(self.attributes,'spatial_context'):
            self.attributes.spatial_context.set_scene_geometry(geometry)

    def forward(self, values, *args):
        if not self.geometry_ready: raise ValueError('bind global scene geometry before encoding/decoding')
        if self.encoder:
            xyz,active=args
            f=values.masked_fill(~active[...,None],0)
            xyz=xyz.masked_fill(~active[...,None],0)
            metric=2*(f[...,:3]-.5)*(self.scene_span/self.scene_span.max())
            return torch.cat((metric,self.attributes(f,xyz,active)),-1).masked_fill(~active[...,None],0)
        active=args[0]
        y=values.masked_fill(~active[...,None],0)
        # No clamp: out-of-bbox estimates must keep position gradients.
        xyz=(.5+.5*y[...,:3]*(self.scene_span.max()/self.scene_span)).masked_fill(~active[...,None],0)
        attributes=self.attributes(y[...,3:],active,xyz,detach_geometry=False)
        return torch.cat((xyz,attributes),-1).masked_fill(~active[...,None],0)


class XYZPayloadCodec(ModularGaussianCodec):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg=cfg
        self.register_buffer('attr_mean',torch.zeros(cfg.attr_dim))
        self.register_buffer('attr_std',torch.ones(cfg.attr_dim))
        self.semantic_encoder=PayloadSemantic(cfg,True)
        self.semantic_decoder=PayloadSemantic(cfg,False)
        self.channel_encoder=ChannelEncoder(cfg)
        self.channel_decoder=ProgressiveChannelDecoder(cfg)
        self.training_snrs=()

    def set_scene_geometry(self, geometry):
        for module in (self.semantic_encoder,self.semantic_decoder): module.set_scene_geometry(geometry)

    def encode_representation(self, features, xyz, active):
        if (active.dtype!=torch.bool or features.shape[:-1]!=active.shape or
                features.shape[-1]!=3+self.cfg.attr_dim or xyz.shape!=(*active.shape,3)):
            raise ValueError('invalid semantic input shapes/mask')
        if not torch.isfinite(features[active]).all() or not torch.isfinite(xyz[active]).all():
            raise ValueError('nonfinite active sender input')
        return self.semantic_encoder(features,xyz,active)

    def decode_representation(self, y_hat, q, delivered_xyz=None):
        if delivered_xyz is not None: raise ValueError('external side XYZ is forbidden in XYZ payload experiment')
        if (y_hat.shape!=(*q.shape,self.cfg.latent_dim) or q.dtype not in (torch.int32,torch.int64)
                or ((q<0)|(q>=len(self.cfg.rates))).any()):
            raise ValueError('invalid payload/tier shape')
        if not torch.isfinite(y_hat[q>0]).all(): raise ValueError('nonfinite received payload')
        return self.semantic_decoder(y_hat,q>0)

    def _decode_dense(self, received, q, snr, delivered_xyz=None, *, condition_snr=None):
        if delivered_xyz is not None: raise ValueError('external side XYZ is forbidden')
        mask=prefix_mask(q.flatten(),self.cfg.rates).reshape_as(received)
        y_hat=self.channel_decoder(received,mask,q,self.resolve_condition_snr(snr,condition_snr))
        return self.decode_representation(y_hat,q)

    def decode(self, symbols, q, snr, return_seed=False, delivered_xyz=None, *, condition_snr=None):
        if delivered_xyz is not None: raise ValueError('external side XYZ is forbidden')
        validate_tiers(q,len(q),len(self.cfg.rates))
        pred=self._decode_dense(unpack(symbols,q,self.cfg.rates)[None],q[None],snr,
                                condition_snr=condition_snr)[0]
        return (pred,pred[:,:3]) if return_seed else pred

    def forward(self, features, xyz, q, snr, kind='awgn', return_seed=False, *, tx_condition_snr=None,rx_condition_snr=None):
        z=self.encode(features,xyz,q,snr,condition_snr=tx_condition_snr)
        return self.decode(channel(z,snr,kind),q,snr,return_seed,condition_snr=rx_condition_snr)

    def forward_clean_batches(self, features, xyz, active):
        y=self.encode_representation(features,xyz,active)
        return self.decode_representation(y,active.long())

    def forward_tier_batches(self, features, xyz, choices, snr, kind='awgn', paired_noise=False, **kwargs):
        _,y_hat,q=self.forward_channel_batches(features,xyz,choices,snr,kind,paired_noise,**kwargs)
        pred=self.decode_representation(y_hat,q)
        return pred,pred[...,:3],(q>0).to(features)

    def forward_st_prefix_batches(self, *args, **kwargs):
        raise ValueError('allocator training is not part of this independent codec experiment')
