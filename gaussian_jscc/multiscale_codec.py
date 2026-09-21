"""Pointwise paths plus gated multiscale context; one noisy Gaussian payload.

Sender pooling uses source geometry, but its centroids and features NEVER cross
the channel. Receiver pooling uses fixed packet slots and received features.
This is not a full Point Transformer V3 implementation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .codec import prefix_mask
from .split_codec import GeometryWindowBlock, mlp
from .learned_codec import LocalFeatureBlock
from .point_attention import GeometricPointBlock


def pool_slots(values, active, factor):
    """Masked means of contiguous slots; no learned/clean receiver topology."""
    b,n,d = values.shape
    right = (-n) % factor
    valid = F.pad(active,(0,right)).reshape(b,-1,factor)
    x = F.pad(values.masked_fill(~active[...,None],0),(0,0,0,right)).reshape(b,-1,factor,d)
    count = valid.sum(-1)
    return x.sum(-2)/count.clamp_min(1)[...,None], count>0


def masked_mean(values, active):
    return values.masked_fill(~active[...,None],0).sum(1,keepdim=True)/active.sum(1,keepdim=True).clamp_min(1)[...,None]


def zero_mean(values, active):
    return (values-masked_mean(values,active)).masked_fill(~active[...,None],0)


class MultiScaleContext(nn.Module):
    """Fine windows + pooled x4/x16 windows within each independent block.

    Pooling does not drop Gaussian outputs or change individual tier assignment.
    No attention matrix grows with the complete scene size.
    """
    factors = (4,16)

    def __init__(self,cfg,geometry):
        super().__init__()
        self.geometry = geometry
        cls = GeometryWindowBlock if geometry else LocalFeatureBlock
        if geometry and cfg.encoder_attention == 'geometric_point':
            cls = GeometricPointBlock
        self.fine = nn.ModuleList(cls(cfg,i%2==1) for i in range(cfg.depth))
        self.coarse = nn.ModuleList(cls(cfg) for _ in self.factors)
        self.merge = nn.Sequential(nn.LayerNorm(3*cfg.hidden),mlp(3*cfg.hidden,cfg.hidden))

    def forward(self,h,active,xyz=None):
        if h.shape[1] == 0:
            return h
        fine = h
        for block in self.fine:
            fine = block(fine,xyz,active) if self.geometry else block(fine,active)
        scales = [fine-h]
        for factor,block in zip(self.factors,self.coarse):
            pooled,valid = pool_slots(h,active,factor)
            if self.geometry:
                centers,_ = pool_slots(xyz,active,factor)
                pooled = block(pooled,centers,valid)
            else:
                pooled = block(pooled,valid)
            scales.append(pooled.repeat_interleave(factor,dim=1)[:,:h.shape[1]])
        return self.merge(torch.cat(scales,-1))*active[...,None]


class MultiScaleSelfCore(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.tier = nn.Embedding(4,h)
        self.snr = mlp(1,h)
        self.enc_geometry_in = mlp(9,h)
        self.enc_appearance_in = mlp(cfg.attr_dim-6,h)
        self.enc_geometry_context = MultiScaleContext(cfg,True)
        self.enc_appearance_context = MultiScaleContext(cfg,True)
        self.enc_exchange = mlp(2*h,h)
        # The own-point path bypasses ALL neighborhood mixing, not the channel.
        self.enc_self_symbols = mlp(2*h,h,2*cfg.rates[-1])
        self.enc_context_symbols = mlp(2*h,h,2*cfg.rates[-1])
        self.enc_context_gate = nn.Parameter(torch.tensor(.1))
        self.dec_geometry_in = mlp(4*cfg.rates[-1],h)
        self.dec_appearance_in = mlp(4*cfg.rates[-1],h)
        self.dec_geometry_context = MultiScaleContext(cfg,False)
        self.dec_appearance_context = MultiScaleContext(cfg,False)
        self.dec_exchange = mlp(2*h,h)
        self.dec_geometry_gate = nn.Parameter(torch.tensor(.1))
        self.dec_appearance_gate = nn.Parameter(torch.tensor(.1))
        sizes = {'xyz':3,'opacity':1,'logcov':6,'dc':3}
        if cfg.sh_degree:
            sizes['sh'] = 3*((cfg.sh_degree+1)**2-1)
        self.heads = nn.ModuleDict({key:nn.Linear(h,n) for key,n in sizes.items()})
        self.context_heads = nn.ModuleDict({key:nn.Linear(h,n) for key,n in sizes.items()})
        nn.init.normal_(self.heads['xyz'].weight,std=.02)
        nn.init.constant_(self.heads['xyz'].bias,.5)
        for head in self.context_heads.values():
            nn.init.zeros_(head.bias)
        nn.init.normal_(self.context_heads['xyz'].weight,std=.02)
        if cfg.xyz_decoder == 'block_center':
            # Added after existing layers: encoder/attribute initialization is
            # unchanged for an identical random seed. No source center is used.
            self.dec_xyz_center = mlp(h,h,3)
            nn.init.normal_(self.dec_xyz_center[-1].weight,std=.02)
            nn.init.constant_(self.dec_xyz_center[-1].bias,.5)
            # Bias would cancel under centering; do not keep dead parameters.
            self.heads['xyz'] = nn.Linear(h,3,bias=False)
            self.context_heads['xyz'] = nn.Linear(h,3,bias=False)
            nn.init.normal_(self.heads['xyz'].weight,std=.02)
            nn.init.normal_(self.context_heads['xyz'].weight,std=.02)

    def condition(self,q,snr,dtype):
        value = torch.full((*q.shape,1),float(snr)/20,device=q.device,dtype=dtype)
        return self.tier(q)+self.snr(value)

    def encode(self,features,xyz,q,snr):
        active = q>0
        features = features.masked_fill(~active[...,None],0)
        condition = self.condition(q,snr,features.dtype)
        g = (self.enc_geometry_in(torch.cat((features[...,:3]*2-1,features[...,4:10]),-1))+condition)*active[...,None]
        a = (self.enc_appearance_in(torch.cat((features[...,3:4],features[...,10:]),-1))+condition)*active[...,None]
        cg = self.enc_geometry_context(g,active,xyz)
        ca = self.enc_appearance_context(a,active,xyz)
        ca = (ca+self.enc_exchange(torch.cat((ca,cg),-1)))*active[...,None]
        z = self.enc_self_symbols(torch.cat((g,a),-1))
        z = z+self.enc_context_gate.tanh()*self.enc_context_symbols(torch.cat((cg,ca),-1))
        mask = prefix_mask(q.flatten(),self.cfg.rates).reshape_as(z)
        z = z*mask
        energy = z.square().sum(-1,keepdim=True)/(mask.sum(-1,keepdim=True)/2).clamp_min(1)
        return z/(energy+self.cfg.power_floor).sqrt()

    def decode(self,received,q,snr):
        active = q>0
        mask = prefix_mask(q.flatten(),self.cfg.rates).reshape_as(received)
        packet = torch.cat((received*mask,mask.to(received)),-1)
        condition = self.condition(q,snr,received.dtype)
        g = (self.dec_geometry_in(packet)+condition)*active[...,None]
        a = (self.dec_appearance_in(packet)+condition)*active[...,None]
        pos = torch.arange(q.shape[1],device=q.device,dtype=received.dtype)[:,None]
        freq = torch.exp(torch.arange(0,self.cfg.hidden,2,device=q.device,dtype=received.dtype)*(-math.log(10000)/self.cfg.hidden))
        encoding = received.new_zeros(q.shape[1],self.cfg.hidden)
        encoding[:,0::2] = torch.sin(pos*freq)
        encoding[:,1::2] = torch.cos(pos*freq[:encoding[:,1::2].shape[-1]])
        # Slot offsets are context-only; own-point recovery has no point ID.
        cg = self.dec_geometry_context((g+encoding[None])*active[...,None],active)
        ca = self.dec_appearance_context((a+encoding[None])*active[...,None],active)
        ca = (ca+self.dec_exchange(torch.cat((ca,cg),-1)))*active[...,None]
        result = []
        for key,head in self.heads.items():
            geometry = key in ('xyz','logcov')
            gate = self.dec_geometry_gate if geometry else self.dec_appearance_gate
            if key == 'xyz' and self.cfg.xyz_decoder == 'block_center':
                center = self.dec_xyz_center(masked_mean(g,active))
                own = zero_mean(head(g),active)
                detail = zero_mean(self.context_heads[key](cg),active)
                result.append(center+own+gate.tanh()*detail)
                continue
            result.append(head(g if geometry else a)+gate.tanh()*self.context_heads[key](cg if geometry else ca))
        return torch.cat(result,-1)*active[...,None]

    def context_diagnostics(self):
        return {key:float(getattr(self,key).detach().tanh()) for key in
                ('enc_context_gate','dec_geometry_gate','dec_appearance_gate')}
