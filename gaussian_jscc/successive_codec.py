"""Opt-in modular codec with an explicit reliable scale-frame boundary."""
import math
import torch
from torch.nn import functional as F

from .codec import channel,pack,unpack,prefix_mask
from .modular_codec import ModularGaussianCodec
from .position_delivery import delivered_positions
from .successive_channel import (SuccessiveChannelEncoder,SuccessiveChannelDecoder,
                                 encode_scales,decode_scales,present_scales)


class SuccessiveGaussianCodec(ModularGaussianCodec):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.channel_decoder=SuccessiveChannelDecoder(cfg)
        self.channel_encoder=SuccessiveChannelEncoder(cfg)

    def encode_channel_packet(self,y,q,snr):
        signal,scales=self.channel_encoder(y,q,snr,self.channel_decoder)
        return dict(payload=pack(signal.reshape(-1,signal.shape[-1]),q.flatten(),self.cfg.rates),
                    scales=encode_scales(scales,q))

    def decode_channel_packet(self,received,q,snr,scale_frame,delivered_xyz=None):
        signal=unpack(received,q.flatten(),self.cfg.rates).reshape(*q.shape,-1)
        scales=decode_scales(scale_frame,q,dtype=signal.dtype)
        mask=prefix_mask(q.flatten(),self.cfg.rates).reshape_as(signal)
        return self.channel_decoder(signal,mask,q,snr,scales=scales,delivered_xyz=delivered_xyz)

    def forward_channel_batches(self,features,xyz,choices,snr,kind='awgn',paired_noise=False,*,
                                tx_condition_snr=None,rx_condition_snr=None,noise_reference_symbols=None):
        if features.ndim!=3 or choices.shape!=(*features.shape[:2],4):raise ValueError('invalid original batched choices')
        q=choices.argmax(-1)
        if choices.requires_grad or not torch.equal(choices,F.one_hot(q,4).to(choices)):
            raise ValueError('successive experiment requires hard tiers')
        y=self.semantic_encoder(features,xyz,q>0)
        signal,scales=self.channel_encoder(y,q,self.resolve_condition_snr(snr,tx_condition_snr),self.channel_decoder)
        mask=prefix_mask(q.flatten(),self.cfg.rates).reshape_as(signal)
        if paired_noise:
            width=self.cfg.rates[-1] if noise_reference_symbols is None else noise_reference_symbols
            if type(width) is not int or width<self.cfg.rates[-1]:raise ValueError('noise reference shorter than payload')
            padded=F.pad(signal,(0,2*width-signal.shape[-1]))
            noisy=channel(padded.reshape(-1,2),snr,kind).reshape_as(padded)
            received=noisy[...,:signal.shape[-1]]*mask
        else:
            noisy=channel(signal[mask].reshape(-1,2),snr,kind)
            received=torch.zeros_like(signal).masked_scatter(mask,noisy.flatten())
        if not self.training:
            # Evaluation crosses the real byte boundary, not a private encoder tensor.
            scales=decode_scales(encode_scales(scales,q),q,dtype=signal.dtype)
        side=delivered_positions(features[...,:3],q,self.cfg)
        recovered=self.channel_decoder(received,mask,q,self.resolve_condition_snr(snr,rx_condition_snr),
            scales=scales,delivered_xyz=side)
        return y,recovered,q

    def scale_rate_record(self,scene,q,efficiency):
        offset=0;content=frames=0
        for ids in scene.group_ids:
            valid=ids>=0;n=int(valid.sum());tiers=torch.zeros_like(ids)
            tiers[valid]=q[offset:offset+n].to(tiers);offset+=n
            content+=2*int(present_scales(tiers).sum());frames+=8
        if offset!=len(q):raise ValueError('rate ordering differs from original scene blocks')
        return dict(scale_content_bits=8*content,scale_framing_bytes=frames,
                    scale_stream_bytes=content+frames,
                    scale_channel_uses=math.ceil(8*(content+frames)/efficiency),
                    scale_reliability='assumed reliable binary16 block scales; no FEC/packet errors simulated')

    def encode_full(self,*args,**kwargs):raise ValueError('successive codec needs encode_channel_packet with scale bytes')
    def encode(self,*args,**kwargs):raise ValueError('successive codec needs encode_channel_packet with scale bytes')
    def decode(self,*args,**kwargs):raise ValueError('successive codec needs decode_channel_packet with scale bytes')
    def forward_st_prefix_batches(self,features,xyz,q,probabilities,snr,kind='awgn'):
        """Exact hard forward with detached, paired-noise local tier proxies.

        Conditional transmitters change symbols AND block scales with q. A
        private untransmitted suffix must not be exposed to the hard receiver.
        Three uniform-positive candidate maps are evaluated only for backward
        credit assignment, holding the actual retained set fixed. Their
        differences are biased within-block/context proxies, not exact
        one-point finite differences. Codec gradients use the real map only.
        """
        if q.shape!=features.shape[:2] or probabilities.shape!=(*q.shape,4):
            raise ValueError('invalid successive allocation shapes')
        if not torch.isfinite(probabilities).all():raise ValueError('nonfinite tier probabilities')
        from .optimization import preserved_rng
        candidates=[]
        if torch.is_grad_enabled() and probabilities.requires_grad:
            for tier in (1,2,3):
                alternative=torch.where(q>0,tier,0)
                # Every candidate starts at the real draw's RNG state; none
                # consumes caller RNG or adds parameters to the codec graph.
                with torch.no_grad(),preserved_rng(features.device):
                    value=self.forward_tier_batches(features,xyz,F.one_hot(alternative,4).to(features),
                                                   snr,kind,paired_noise=True)[0]
                candidates.append(value)
        actual=self.forward_tier_batches(features,xyz,F.one_hot(q,4).to(features),
                                         snr,kind,paired_noise=True)[0]
        result=actual
        for tier,candidate in enumerate(candidates,1):
            gate=(probabilities[...,tier]-probabilities[...,tier].detach())*(q>0)
            result=result+gate[...,None]*(candidate-actual.detach())
        return result
