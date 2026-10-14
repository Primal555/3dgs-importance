"""Receiver-side latent refinement, with exactly the existing 4/12/24 prefixes.

The transmitter remains the residual channel encoder. Only received symbol
segments and earlier recovered latents enter the refinement stages: no clean
latent, source Gaussian attributes, receiver-noise feedback or extra bits.
"""
import math

import torch
from torch import nn

from .modular_core import ChannelResidualBlock


class LatentRefinementStage(nn.Module):
    def __init__(self, symbols, cfg, *, base=False):
        super().__init__()
        self.base=base
        hidden=cfg.channel_hidden
        self.input=nn.Linear(2*symbols+(0 if base else cfg.latent_dim),hidden)
        self.snr=nn.Sequential(nn.Linear(1,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.blocks=nn.ModuleList(ChannelResidualBlock(hidden) for _ in range(3))
        self.readout=nn.Sequential(nn.GELU(),nn.Linear(hidden,cfg.latent_dim))
        if not base:
            # Near-zero initial correction; small nonzero weights also allow
            # gradients into the enhancement trunk on its very first update.
            nn.init.normal_(self.readout[-1].weight,std=1e-3)
            nn.init.zeros_(self.readout[-1].bias)

    def forward(self, symbols, previous, snr):
        inputs=symbols if self.base else torch.cat((symbols,previous),-1)
        condition=self.snr(symbols.new_full((*symbols.shape[:-1],1),float(snr)/20))
        h=self.input(inputs)+condition
        for block in self.blocks:
            h=block(h,condition)
        return self.readout(h)


class ProgressiveChannelDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.stages=nn.ModuleList(LatentRefinementStage(end-start,cfg,base=index==0)
            for index,(start,end) in enumerate(zip(cfg.rates[:-1],cfg.rates[1:])))

    def decode_stages(self,received,receive_mask,q,snr):
        """Return cumulative latents; unavailable stages retain the last value.

        Lower stages never receive the final tier identity. Thus q3's first
        recovered latent is identical to q1's under the same first 4 symbols.
        Joint gradients into earlier stages are deliberately NOT detached.
        Entirely unused stages are skipped, leaving their gradients None and
        preventing Adam momentum updates on q1-only batches.
        """
        if received.shape!=(*q.shape,2*self.cfg.rates[-1]) or receive_mask.shape!=received.shape:
            raise ValueError('invalid receiver slot/mask shape')
        if q.dtype not in (torch.int32,torch.int64) or ((q<0)|(q>=len(self.cfg.rates))).any():
            raise ValueError('invalid receiver tiers')
        if not math.isfinite(float(snr)):
            raise ValueError('SNR must be finite')
        masked=received.masked_fill(~receive_mask.bool(),0)
        previous=received.new_zeros(*q.shape,self.cfg.latent_dim)
        outputs=[]
        for tier,(stage,start,end) in enumerate(zip(self.stages,self.cfg.rates[:-1],self.cfg.rates[1:]),1):
            active=q>=tier
            if active.any():
                delta=stage(masked[...,2*start:2*end][active],previous[active],snr)
                correction=torch.zeros_like(previous).masked_scatter(active[...,None].expand_as(previous),delta.flatten())
                previous=previous+correction
            outputs.append(previous)
        return torch.stack(outputs,-2)

    def forward(self,received,receive_mask,q,snr):
        return self.decode_stages(received,receive_mask,q,snr)[...,-1,:]
