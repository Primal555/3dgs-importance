"""Receiver-only feature-neighbor attention; no true XYZ or sender graph."""
import torch
from torch import nn
from torch.nn import functional as F
from .point_attention import gather_neighbors
from .split_codec import mlp


def feature_neighbors(features,active,count):
    """Cosine kNN within a bounded packet block, with self and q0 masking.

    Hard neighbor selection has no gradient; gathered values/relations do.
    This graph describes received features, NOT verified spatial proximity.
    """
    with torch.no_grad():
        x=F.normalize(features.detach().float().masked_fill(~active[...,None],0),dim=-1)
        distance=2-2*(x@x.transpose(1,2))
        distance.diagonal(dim1=-2,dim2=-1).fill_(-1)
        distance=distance.masked_fill(~active[:,None,:],float('inf'))
        indices=distance.topk(min(count,x.shape[1]),dim=-1,largest=False).indices
        valid=active[...,None]&gather_neighbors(active[...,None],indices).squeeze(-1)
    return indices,valid


class ReceivedPointBlock(nn.Module):
    """Encoder-style grouped relation attention over received feature kNN.

    q_i-k_j plus learned relative features affect weights AND values. Includes
    pre-norm, residual attention and FFN; no sinusoidal slot position required.
    """
    def __init__(self,cfg,shifted=False):
        super().__init__()
        self.neighbors=cfg.decoder_neighbors
        self.groups=cfg.attention_heads
        self.norm=nn.LayerNorm(cfg.hidden)
        self.qkv=nn.Linear(cfg.hidden,3*cfg.hidden)
        self.relative=mlp(cfg.hidden,cfg.hidden)
        self.relation=nn.Sequential(nn.LayerNorm(cfg.hidden),mlp(cfg.hidden,cfg.hidden,self.groups))
        self.projection=nn.Linear(cfg.hidden,cfg.hidden)
        self.ffn_norm=nn.LayerNorm(cfg.hidden)
        self.ffn=mlp(cfg.hidden,2*cfg.hidden,cfg.hidden)

    def forward(self,h,active):
        b,n,d=h.shape
        if n==0:
            return h
        x=self.norm(h.masked_fill(~active[...,None],0))
        indices,valid=feature_neighbors(x,active,self.neighbors)
        normalized=F.normalize(x,dim=-1)
        offsets=normalized[:,:,None]-gather_neighbors(normalized,indices)
        relative=self.relative(offsets.masked_fill(~valid[...,None],0))
        q,k,v=self.qkv(x).chunk(3,-1)
        scores=self.relation(q[:,:,None]-gather_neighbors(k,indices)+relative)
        safe=valid.clone();safe[...,0]|=~safe.any(-1)
        weights=scores.masked_fill(~safe[...,None],-float('inf')).softmax(dim=2)
        values=(gather_neighbors(v,indices)+relative).masked_fill(~valid[...,None],0)
        attended=(weights[...,None]*values.reshape(b,n,-1,self.groups,d//self.groups)).sum(2).reshape(b,n,d)
        h=h.masked_fill(~active[...,None],0)+self.projection(attended)*active[...,None]
        return (h+self.ffn(self.ffn_norm(h)))*active[...,None]
