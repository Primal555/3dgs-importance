"""Receiver-side latent refinement using configurable progressive prefixes.

The transmitter remains the residual channel encoder. Only received symbol
segments and earlier recovered latents enter the refinement stages: no clean
latent, source Gaussian attributes, receiver-noise feedback or extra bits.
"""
import math

import torch
from torch import nn

from .modular_core import ChannelResidualBlock


class LatentRefinementStage(nn.Module):
    def __init__(self, symbols, cfg, *, base=False, earlier_symbols=0):
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
        if cfg.channel_prefix_mode=='full_prefix' and earlier_symbols:
            self.prefix_input=nn.Linear(2*earlier_symbols,hidden,bias=False)
            nn.init.zeros_(self.prefix_input.weight)

    def forward(self, symbols, previous, snr, earlier=None):
        inputs=symbols if self.base else torch.cat((symbols,previous),-1)
        condition=self.snr(symbols.new_full((*symbols.shape[:-1],1),float(snr)/20))
        h=self.input(inputs)+condition
        if hasattr(self,'prefix_input'):
            if earlier is None or earlier.shape!=(*symbols.shape[:-1],self.prefix_input.in_features):
                raise ValueError('full-prefix stage requires matching earlier received symbols')
            h=h+self.prefix_input(earlier)
        for block in self.blocks:
            h=block(h,condition)
        return self.readout(h)


class ProgressiveChannelDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.stages=nn.ModuleList(LatentRefinementStage(end-start,cfg,base=index==0,earlier_symbols=start)
            for index,(start,end) in enumerate(zip(cfg.rates[:-1],cfg.rates[1:])))
        if cfg.channel_denoiser_mode!='none':
            from .channel_spatial import ChannelSpatialDenoiser
            tiers=(len(cfg.rates)-1,) if cfg.channel_denoiser_scope=='final' else range(1,len(cfg.rates))
            self.denoisers=nn.ModuleList(ChannelSpatialDenoiser(cfg,t) for t in tiers)
        if cfg.channel_neighbor_mode!='none':
            from .channel_spatial import ChannelNeighborRefiner
            self.neighbor_refiner=ChannelNeighborRefiner(cfg)

    def decode_stages(self,received,receive_mask,q,snr,*,delivered_xyz=None):
        """Return cumulative latents; unavailable stages retain the last value.

        Lower stages never receive the final tier identity. Thus q3's first
        recovered latent is identical to q1's under the same base prefix.
        Joint gradients into earlier stages are deliberately NOT detached.
        Final-scope cooperation is separate: these base stages stay autonomous.
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
                earlier=masked[...,:2*start][active] if hasattr(stage,'prefix_input') else None
                delta=stage(masked[...,2*start:2*end][active],previous[active],snr,earlier)
                correction=torch.zeros_like(previous).masked_scatter(active[...,None].expand_as(previous),delta.flatten())
                previous=previous+correction
                if hasattr(self,'denoisers') and self.cfg.channel_denoiser_scope=='stage':
                    previous=previous+self.denoisers[tier-1](previous,masked,receive_mask,delivered_xyz,active,snr)
            outputs.append(previous)
        return torch.stack(outputs,-2)

    def forward(self,received,receive_mask,q,snr,*,delivered_xyz=None):
        previous=self.decode_stages(received,receive_mask,q,snr,delivered_xyz=delivered_xyz)[...,-1,:]
        if hasattr(self,'denoisers') and self.cfg.channel_denoiser_scope=='final':
            # All received prefixes may help all retained points. Own unsent
            # slots remain masked; q0/padding never become donors or queries.
            previous=previous+self.denoisers[0](previous,received,receive_mask,delivered_xyz,q>0,snr)
        if hasattr(self,'neighbor_refiner'):
            previous=previous+self.neighbor_refiner(previous,received,receive_mask,delivered_xyz,q>0,snr)
        return previous


def enable_full_prefix(model,optimizer=None):
    """Function-preserving in-memory upgrade; existing tensors/Adam stay intact.

    A separate zero-initialized earlier-prefix projection is linearly equivalent
    to widening the stage input, but avoids changing old parameter shapes and
    lets new weights start with their own fresh Adam step counters.
    """
    if model.cfg.channel_variant!='progressive' or model.cfg.channel_prefix_mode!='suffix':
        raise ValueError('upgrade requires an original progressive suffix decoder')
    group=None
    if optimizer is not None:
        groups=[g for g in optimizer.param_groups if g.get('name')=='channel_decoder']
        if len(groups)!=1 or {id(p) for p in groups[0]['params']}!={id(p) for p in model.channel_decoder.parameters()}:
            raise ValueError('upgrade requires a complete named channel_decoder optimizer group')
        group=groups[0]
    names=[]
    for index,stage in enumerate(model.channel_decoder.stages[1:],1):
        earlier_symbols=model.cfg.rates[index]
        reference=stage.input.weight
        stage.prefix_input=nn.Linear(2*earlier_symbols,stage.input.out_features,bias=False).to(reference)
        nn.init.zeros_(stage.prefix_input.weight)
        names.append(f'channel_decoder.stages.{index}.prefix_input.weight')
    model.cfg.channel_prefix_mode='full_prefix'
    if group is not None:
        # Match fresh factory traversal order exactly so saved Adam state loads
        # onto the same parameter names on resume, not onto shifted tensors.
        group['params']=list(model.channel_decoder.parameters())
    return names
