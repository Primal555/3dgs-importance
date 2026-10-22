"""Conditional transmitter and scale-aware receiver; no clean y at receiver."""
import math
import struct

import numpy as np
import torch
from torch import nn

from .modular_core import ChannelResidualBlock


def validate_q(q):
    if q.ndim!=2 or q.dtype!=torch.long or ((q<0)|(q>3)).any():
        raise ValueError('expected original [blocks,points] int64 q0..q3')


def present_scales(q):
    validate_q(q)
    return torch.stack([(q>=tier).any(1) for tier in (1,2,3)],1)


def normalize_segment(raw,donors,floor):
    """Original-block donor RMS, quantized UP; receiver uses the same value.

    Binary16 is the actual forward value; STE supplies an explicit training
    surrogate for its derivative. No energy is measured at the receiver.
    """
    if (raw.ndim!=3 or raw.shape[:2]!=donors.shape or donors.dtype!=torch.bool
            or raw.shape[-1]<2 or raw.shape[-1]%2 or not math.isfinite(floor) or floor<=0):
        raise ValueError('invalid complex segment/donor mask/floor')
    raw=raw.masked_fill(~donors[...,None],0)
    if not torch.isfinite(raw).all():raise ValueError('nonfinite transmitted segment')
    energy=raw.square().sum((1,2))/(donors.sum(1).clamp_min(1)*(raw.shape[-1]//2))
    gain=(energy+floor).sqrt()
    if not torch.isfinite(gain).all() or (gain>65504).any():
        raise ValueError('block scale outside finite binary16; cannot silently clip')
    rounded=gain.detach().to(torch.float16)
    # Positive binary16 bit patterns are ordered. Increment only values rounded
    # down, including zero/subnormals; exact values and 65504 stay unchanged.
    # Some CUDA PyTorch builds do not implement nextafter for float16.
    increment=(rounded.to(gain)<gain.detach()).to(torch.int16)
    rounded=(rounded.view(torch.int16)+increment).view(torch.float16).to(gain)
    gain=gain+(rounded-gain).detach()
    gain=torch.where(donors.any(1),gain,torch.ones_like(gain))
    return raw/gain[:,None,None],gain


def encode_scales(scales,q):
    present=present_scales(q)
    if scales.shape!=present.shape:raise ValueError('scale rows do not match original blocks')
    values=scales.detach()[present].float().cpu()
    if not torch.isfinite(values).all() or (values<=0).any() or not torch.equal(values,values.half().float()):
        raise ValueError('wire scales must be finite positive binary16 values')
    return b'SCL1'+struct.pack('<I',len(values))+values.numpy().astype('<f2').tobytes()


def decode_scales(data,q,*,dtype=torch.float32):
    present=present_scales(q);count=int(present.sum())
    if (not isinstance(data,bytes) or len(data)!=8+2*count or data[:4]!=b'SCL1'
            or struct.unpack('<I',data[4:8])[0]!=count):
        raise ValueError('missing, malformed or mismatched scale frame')
    values=torch.from_numpy(np.frombuffer(data[8:],dtype='<f2').astype(np.float32).copy()).to(q.device,dtype)
    if not torch.isfinite(values).all() or (values<=0).any():raise ValueError('invalid received scales')
    result=torch.ones(present.shape,device=q.device,dtype=dtype)
    result[present]=values
    return result


class ChannelMapping(nn.Module):
    def __init__(self,inputs,outputs,hidden,*,residual=False):
        super().__init__()
        self.input=nn.Linear(inputs,hidden)
        self.snr=nn.Sequential(nn.Linear(1,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.blocks=nn.ModuleList(ChannelResidualBlock(hidden) for _ in range(3))
        self.readout=nn.Sequential(nn.GELU(),nn.Linear(hidden,outputs))
        if residual:
            nn.init.normal_(self.readout[-1].weight,std=1e-3)
            nn.init.zeros_(self.readout[-1].bias)

    def forward(self,value,snr):
        if not math.isfinite(float(snr)):raise ValueError('finite SNR required')
        condition=self.snr(value.new_full((*value.shape[:-1],1),float(snr)/20))
        hidden=self.input(value)+condition
        for block in self.blocks:hidden=block(hidden,condition)
        return self.readout(hidden)


class SuccessiveChannelEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.stages=nn.ModuleList(ChannelMapping(cfg.latent_dim*(1 if i==0 else 2),
            2*(end-start),cfg.channel_hidden) for i,(start,end) in enumerate(zip(cfg.rates[:-1],cfg.rates[1:])))

    def forward(self,y,q,snr,decoder):
        validate_q(q)
        if y.shape!=(*q.shape,self.cfg.latent_dim) or q.shape[1]>self.cfg.block_size:
            raise ValueError('conditional encoding requires original complete codec blocks')
        if not torch.isfinite(y[q>0]).all():raise ValueError('nonfinite source latent')
        original=y.masked_fill((q==0)[...,None],0)
        proxy=torch.zeros_like(original);segments=[];scales=[]
        for tier,stage in enumerate(self.stages,1):
            donors=q>=tier
            if not donors.any():
                # Keep unused parameters out of autograd: zero gradients would
                # still advance stale Adam moments after a higher-tier update.
                width=2*(self.cfg.rates[tier]-self.cfg.rates[tier-1])
                segments.append(original.new_zeros(*q.shape,width))
                scales.append(original.new_ones(len(q)))
                continue
            inputs=original if tier==1 else torch.cat((original,(original-proxy).detach()),-1)
            raw=stage(inputs,snr)
            signal,gain=normalize_segment(raw,donors,self.cfg.power_floor)
            segments.append(signal);scales.append(gain)
            if tier<3:
                # Sender simulation never contains receiver noise or spatial help.
                with torch.no_grad():
                    proxy=decoder.advance(tier,torch.cat(segments,-1),torch.stack(scales,1),proxy,donors,snr)
        return torch.cat(segments,-1),torch.stack(scales,1)


class SuccessiveChannelDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.stages=nn.ModuleList(ChannelMapping(2*end+tier+(cfg.latent_dim if tier>1 else 0),
            cfg.latent_dim,cfg.channel_hidden,residual=tier>1) for tier,end in enumerate(cfg.rates[1:],1))
        from .channel_spatial import ChannelNeighborRefiner
        self.neighbor_refiner=ChannelNeighborRefiner(cfg)
        self.register_buffer('neighbor_enabled',torch.tensor(False))

    def advance(self,tier,prefix,scales,previous,active,snr):
        end=self.cfg.rates[tier]
        if prefix.shape!=(*active.shape,2*end) or scales.shape!=(len(active),tier):
            raise ValueError('stage received prefix/scale shape mismatch')
        if not active.any():return previous
        gains=torch.cat([scales[:,i:i+1].expand(-1,2*(b-a))
                         for i,(a,b) in enumerate(zip(self.cfg.rates[:tier],self.cfg.rates[1:tier+1]))],-1)
        # Shared scale restores amplitude; log-gain also tells the MLP noise level.
        inputs=[prefix*gains[:,None,:],scales.log2()[:,None,:].expand(-1,active.shape[1],-1)]
        if tier>1:inputs.append(previous)
        delta=self.stages[tier-1](torch.cat(inputs,-1),snr).masked_fill(~active[...,None],0)
        return previous+delta

    def forward(self,received,receive_mask,q,snr,*,scales=None,delivered_xyz=None,return_stages=False):
        validate_q(q)
        if (received.shape!=(*q.shape,2*self.cfg.rates[-1]) or receive_mask.shape!=received.shape
                or scales is None or scales.shape!=(len(q),3)
                or not torch.isfinite(scales).all() or (scales<=0).any()):
            raise ValueError('receiver requires received prefixes and explicit finite positive scales')
        slots=torch.arange(received.shape[-1],device=q.device)
        available=slots<2*torch.as_tensor(self.cfg.rates,device=q.device)[q][...,None]
        if not torch.isfinite(received[available&receive_mask.bool()]).all():
            raise ValueError('nonfinite received symbol in a transmitted slot')
        signal=received.masked_fill(~(receive_mask.bool()&available),0)
        previous=torch.zeros(*q.shape,self.cfg.latent_dim,device=received.device,dtype=received.dtype)
        outputs=[]
        for tier,end in enumerate(self.cfg.rates[1:],1):
            previous=self.advance(tier,signal[...,:2*end],scales[:,:tier],previous,q>=tier,snr)
            outputs.append(previous)
        stages=torch.stack(outputs,-2);final=previous
        if bool(self.neighbor_enabled):
            final=final+self.neighbor_refiner(previous,signal,available,delivered_xyz,q>0,snr)
        return (stages,final) if return_stages else final
