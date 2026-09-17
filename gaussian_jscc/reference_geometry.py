"""V6: fixed-energy multiscale reference + independently normalized local detail.

The four reference real slots carry two phase pairs per retained row. Variable
and frequency multiplexing is determined by row order and known group length.
No reference, gain or grouping coordinates are delivered out of band. Small
groups (<24 retained rows) fall back to the v5 waveform, known at both ends.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .block_geometry import PilotBlockGeometry, BlockGeometry


class ReferenceGeometry(PilotBlockGeometry):
    def __init__(self, rates, hidden, group_size=256, bounded=True, individual=False):
        super().__init__(rates, hidden, group_size)
        self.bounded=bounded
        self.individual=individual
        self.reference_encoder = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 4))
        # Estimated reference, uncertainty, tier histogram, retained count, SNR.
        self.reference_decoder = nn.Sequential(nn.Linear(13, hidden), nn.GELU(), nn.Linear(hidden, 4))
        for head in (self.reference_encoder[-1], self.reference_decoder[-1]):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.reference_power_logit = nn.Parameter(torch.tensor(0.))

    def _grouped(self, values, q, snr, operation, output_dim):
        if not self.individual:
            return super()._grouped(values,q,snr,operation,output_dim)
        return self._stable_grouped(values,q,snr,operation,output_dim)

    def _stable_grouped(self, values, q, snr, operation, output_dim, choices=None):
        if q.ndim == 2:
            return torch.stack([self._stable_grouped(v,t,snr,operation,output_dim,
                                None if choices is None else choices[i]) for i,(v,t) in enumerate(zip(values,q))])
        # Boundaries use SOURCE rows, before q0 compaction. No transmitted IDs.
        if not len(q):return values.new_zeros((0,output_dim))
        size=self.group_size;groups=math.ceil(len(q)/size);pad=groups*size-len(q)
        t=F.pad(q,(0,pad)).reshape(groups,size)
        v=F.pad(values,(0,0,0,pad)).reshape(groups,size,-1)
        order=torch.argsort((t==0).long(),dim=-1,stable=True)
        t=t.gather(1,order);v=v.gather(1,order[...,None].expand_as(v))
        c=None
        if choices is not None:
            c=F.pad(choices,(0,0,0,pad)).reshape(groups,size,4)
            c=c.gather(1,order[...,None].expand_as(c))
        count=(t>0).sum(-1)
        out=v.new_zeros((groups,size,output_dim))+v.sum()*0
        if c is not None:out=out+c.sum()*0
        normal=torch.nonzero(count>=24,as_tuple=False).flatten()
        if len(normal):
            # All dense groups together, not a Python/GPU launch loop per group.
            y=(operation(v[normal],t[normal],snr) if c is None else
               operation(v[normal],t[normal],snr,c[normal]))
            out=out.index_copy(0,normal,y*(t[normal]>0)[...,None])
        for index in torch.nonzero((count>0)&(count<24),as_tuple=False).flatten().tolist():
            n=int(count[index])
            y=(operation(v[index:index+1,:n],t[index:index+1,:n],snr) if c is None else
               operation(v[index:index+1,:n],t[index:index+1,:n],snr,c[index:index+1,:n]))
            padded=F.pad(y,(0,0,0,size-n))
            out=out.index_copy(0,torch.tensor([index],device=q.device),padded)
        # Undo only the within-group compaction; original group IDs never move.
        out=out.scatter(1,order[...,None].expand_as(out),out)
        return out.reshape(-1,output_dim)[:len(q)]

    def enhancement_layers(self):
        for tier in (2,3):
            begin,end=2*int(self.rates[tier-1]),2*int(self.rates[tier])
            if end>begin:
                budget=(end-begin)/2
                gain=.9*math.sqrt(budget/(end-begin-1))/(1+self.correction_bound)
                yield tier,begin,end,budget,gain

    def add_enhancements(self, base, raw, choices):
        result=base
        for tier,begin,end,budget,gain in self.enhancement_layers():
            enabled=choices[...,tier:].sum(-1)
            data=raw[...,begin:end-1]*gain
            filler=(budget-data.square().sum(-1)).clamp_min(1e-8).sqrt()
            layer=torch.cat((data,filler[...,None]),-1)*enabled[...,None]
            result=result+F.pad(layer,(begin,raw.shape[-1]-end))
        return result

    def sparse_encode(self, xyz, q, snr, choices):
        # Same q1 base for everyone, INCLUDING the shared analog reference.
        # Pool that reference across this fixed group; do not send independent
        # absolute coordinates per point (too noisy at small symbol budgets).
        base_q=torch.ones_like(q)
        base=BlockGeometry.encode(self,xyz,base_q,snr)
        active=torch.ones_like(q,dtype=xyz.dtype)
        center,radius=self.reference(xyz,active);offset=(xyz-center)/radius
        shared=torch.cat((center*2-1,2*radius.log()/-math.log(self.radius_floor)+1),-1).expand(*q.shape,4)
        condition=torch.stack((torch.ones_like(q,dtype=xyz.dtype)/3,torch.full_like(q,float(snr),dtype=xyz.dtype)/20),-1)
        raw=xyz.new_zeros((*q.shape,len(self.offset_axis)))
        for axis in range(3):raw[...,self.offset_axis==axis]=offset[...,axis:axis+1]
        raw=raw+self.correction_bound*self.encoder(torch.cat((shared,offset,condition),-1)).tanh()*(self.offset_axis>=0)
        return self.add_enhancements(base,raw,choices)

    def sparse_decode(self, received, q, snr):
        active=torch.ones_like(q,dtype=received.dtype)
        _,_,gain=self.layout(torch.ones_like(q),received.dtype)
        _,mask,_=self.layout(q,received.dtype)
        mask=mask.clone();mask[...,7]=0
        data=received/gain
        precision=torch.ones_like(received)*gain.square()
        for tier,begin,end,budget,extra_gain in self.enhancement_layers():
            region=(torch.arange(data.shape[-1],device=q.device)>=begin)&(torch.arange(data.shape[-1],device=q.device)<end)
            filler=F.one_hot(torch.tensor(end-1,device=q.device),data.shape[-1]).bool()
            data=torch.where(region,received/extra_gain,data)
            mask=mask.masked_fill(filler,0.)
            precision=torch.where(region,torch.full_like(precision,extra_gain**2),precision)
        data=data*mask
        pooled=self.average(data,active)
        center=(pooled[...,:3]+1)/2
        radius=((pooled[...,3:4].clamp(-1,1)-1)*-math.log(self.radius_floor)/2).exp()
        offset=[]
        for axis in range(3):
            weight=(self.offset_axis==axis)*mask*precision
            offset.append((data*weight).sum(-1)/weight.sum(-1).clamp_min(1e-8))
        condition=torch.stack((q.to(received)/3,torch.full_like(q,float(snr),dtype=received.dtype)/20),-1)
        correction=.25*self.decoder(torch.cat((data,mask,data,condition),-1)).tanh()
        return center+radius*(torch.stack(offset,-1)+correction)

    def reference_fraction(self):
        return .25 + .5*self.reference_power_logit.sigmoid()

    @staticmethod
    def multiplex(q):
        n=q.shape[-1]
        row=torch.arange(n,device=q.device)
        variable=2*(row%2)[:,None]+torch.arange(2,device=q.device)[None]
        level=(row//2)%3
        frequencies=torch.tensor([1.,4.,16.],device=q.device)
        return variable,level,frequencies

    def _encode_groups(self, xyz, q, snr, choices=None):
        active=(q>0).to(xyz)
        if self.individual and q.shape[-1]<24:
            return self.sparse_encode(xyz,q,snr,F.one_hot(q,4).to(xyz) if choices is None else choices)
        center,radius=self.reference(xyz,active)
        ref=torch.cat((center,1+radius.log()/-math.log(self.radius_floor)),-1)
        condition=torch.cat((active.mean(-1,keepdim=True)[...,None],
                             torch.full_like(radius,float(snr)/20)),-1)
        variable,level,freq=self.multiplex(q)
        if choices is None:
            choices=F.one_hot(q,4).to(xyz)
        lengths=choices @ self.rates.to(xyz)
        fraction=self.reference_fraction()
        reference_lengths=torch.ones_like(lengths)*self.rates[1] if self.individual else lengths
        amplitude=(reference_lengths.clamp_min(1e-8)*fraction/2).sqrt()[...,None]
        # Precondition common corrections by available reference precision, not
        # an arbitrary absolute-coordinate displacement. Do not optimize this
        # scaling itself via the power allocation branch.
        correction_scale=.005
        if self.bounded:
            sigma=math.sqrt(.5*10**(-float(snr)/10))
            scales=[]
            for var in range(4):
                selected=((variable==var)&(level[:,None]==2)).to(xyz)[None]*active[...,None]
                energy=(amplitude.square()*selected).sum((-2,-1)).clamp_min(1.)
                scales.append(.5*sigma/(energy.sqrt()*math.pi*16))
            correction_scale=torch.stack(scales,-1)[:,None].detach()
        ref=ref+correction_scale*self.reference_encoder(torch.cat((ref,condition),-1)).tanh()
        phases=math.pi*freq[level][None,:,None]*(ref.squeeze(-2)[:,variable]-.5)
        reference=torch.stack((phases.cos(),phases.sin()),-1).flatten(-2)*amplitude
        # v5 data construction, but no public reference depends on its gain.
        legacy=None if self.individual else super()._encode_groups(xyz,q,snr)
        offset=(xyz-center)/radius
        shared=torch.cat((center*2-1,2*radius.log()/-math.log(self.radius_floor)+1),-1).expand(*q.shape,4)
        tier_condition=choices @ torch.arange(4,device=q.device,dtype=xyz.dtype)/3
        if self.individual:
            # Base waveform/gain cannot depend on positive-tier choices.
            tier_condition=torch.ones_like(tier_condition)/3
        conditions=torch.stack((tier_condition,torch.full_like(tier_condition,float(snr)/20)),-1)
        raw=xyz.new_zeros((*q.shape,len(self.offset_axis)))
        for axis in range(3):
            raw[...,self.offset_axis==axis]=offset[...,axis:axis+1]
        raw=raw+self.correction_bound*self.encoder(torch.cat((shared,offset,conditions),-1)).tanh()*(self.offset_axis>=0).to(xyz)
        table=(torch.arange(len(self.offset_axis),device=q.device)[None] < (2*self.rates)[:,None]).to(xyz)
        mask=choices @ table
        local_mask=mask.clone();local_mask[...,:4]=0
        selector=F.one_hot(torch.tensor(7,device=q.device),raw.shape[-1]).to(raw)
        if self.individual:
            base_end=2*int(self.rates[1])
            base_mask=(torch.arange(raw.shape[-1],device=q.device)<base_end).to(raw)
            base=(raw+self.pilot*selector)*local_mask*base_mask
            budget=active.sum(-1,keepdim=True)[...,None]*self.rates[1]*(1-fraction)
            gain=(budget/base.square().sum((-2,-1),keepdim=True).clamp_min(1e-12)).sqrt()
            result=base*gain
            result=self.add_enhancements(result,raw,choices)
            return torch.cat((reference,result[...,4:]),-1)*active[...,None]
        detail=(raw+self.pilot*selector)*local_mask
        budget=(lengths*(1-fraction)).sum(-1,keepdim=True)[...,None]
        scale=(budget/detail.square().sum((-2,-1),keepdim=True).clamp_min(1e-12)).sqrt()
        result=detail*scale
        result=torch.cat((reference,result[...,4:]),-1)*active[...,None]
        small=active.sum(-1)<24
        return torch.where(small[...,None,None],legacy,result)

    def _decode_groups(self, received,q,snr):
        if self.individual and q.shape[-1]<24:
            return self.sparse_decode(received,q,snr)
        active=(q>0).to(received)
        variable,level,freq=self.multiplex(q)
        fraction=self.reference_fraction()
        lengths=self.rates[q].to(received)
        reference_lengths=torch.ones_like(lengths)*self.rates[1] if self.individual else lengths
        amp=(reference_lengths.clamp_min(1e-8)*fraction/2).sqrt()
        pairs=received[...,:4].reshape(*q.shape,2,2)
        phase_values=[];uncertainties=[]
        sigma=math.sqrt(.5*10**(-float(snr)/10))
        for var in range(4):
            estimate=None;uncertainty=None
            for lev in range(3):
                selected=((variable==var)&(level[:,None]==lev)).to(received)[None]*active[...,None]
                pooled=(pairs*(amp[...,None]*selected)[...,None]).sum((-3,-2))
                phase=torch.atan2(pooled[...,1],pooled[...,0]+1e-12)
                candidate=.5+phase/(math.pi*freq[lev])
                if estimate is not None:
                    # Piecewise differentiable unwrapping. Integer branch is a
                    # biased/frozen decision, not claimed globally differentiable.
                    period=2/freq[lev]
                    candidate=candidate+period*torch.round((estimate-candidate)/period).detach()
                estimate=candidate
                energy=(amp.square()[...,None]*selected).sum((-2,-1)).clamp_min(1e-8)
                uncertainty=sigma/(energy.sqrt()*math.pi*freq[lev])
            phase_values.append(estimate);uncertainties.append(uncertainty)
        base=torch.stack(phase_values,-1)[:,None]
        std=torch.stack(uncertainties,-1)[:,None]
        count=active.sum(-1,keepdim=True)[...,None]
        histogram=torch.stack([(q==i).sum(-1) for i in (1,2,3)],-1).to(received)[:,None]/count.clamp_min(1)
        if self.individual:
            histogram=torch.zeros_like(histogram)
            histogram[...,0]=1  # reference recovery must not depend on positive tier mix
        inputs=torch.cat((base,std,histogram,count/256,torch.full_like(count,float(snr)/20)),-1)
        ref=base+(.5 if self.bounded else 4)*std.detach()*self.reference_decoder(inputs).tanh()
        center=ref[...,:3]
        radius=((ref[...,3:4].clamp(0,1)-1)*-math.log(self.radius_floor)).exp()
        # Local gain uncertainty affects local detail, never absolute reference.
        gain=(self.average(received[...,7:8],active)/self.pilot).clamp_min(.05)
        if self.individual:
            base_end=2*int(self.rates[1])
            budget=self.rates[1]*(1-fraction)
            lower=(budget/((base_end-5)*(1+self.correction_bound)**2+self.pilot**2)).sqrt()
            upper=budget.sqrt()/self.pilot
            gain=gain.maximum(lower).minimum(upper)
        data=received/gain
        _,mask,_=self.layout(q,received.dtype)
        data=data*mask
        if self.individual:
            selector=F.one_hot(torch.tensor(7,device=q.device),data.shape[-1]).to(data)
            # Base gain uses only unchanged base pilots. Enhancement gains are
            # fixed and their completion symbols are never used as pilots.
            data=data*(1-selector)
            precision=torch.ones_like(received)*gain.square()
            for tier,begin,end,budget,extra_gain in self.enhancement_layers():
                region=(torch.arange(data.shape[-1],device=q.device)>=begin)&(torch.arange(data.shape[-1],device=q.device)<end)
                filler=F.one_hot(torch.tensor(end-1,device=q.device),data.shape[-1]).bool()
                data=torch.where(region,received/extra_gain,data)
                data=data.masked_fill(filler,0.)
                precision=torch.where(region,torch.full_like(precision,extra_gain**2),precision)
                mask=mask.masked_fill(filler,0.)
            data=data*mask
        data=torch.cat((torch.cat((center*2-1,ref[...,3:4]*2-1),-1).expand(*q.shape,4),data[...,4:]),-1)
        offset=[]
        for axis in range(3):
            selected=(self.offset_axis==axis).to(received)*mask
            if self.individual:selected=selected*precision
            offset.append((data*selected).sum(-1)/selected.sum(-1).clamp_min(1e-8 if self.individual else 1))
        offset=torch.stack(offset,-1)
        pooled=data if self.individual else self.average(data,active)
        condition=torch.stack((q.to(received)/3,torch.full_like(q,float(snr),dtype=received.dtype)/20),-1)
        correction=self.decoder(torch.cat((data,mask,pooled.expand_as(data),condition),-1))
        if self.bounded: correction=.25*correction.tanh()
        result=center+radius*(offset+correction)
        if self.individual:
            return result
        fallback=super()._decode_groups(received,q,snr)
        return torch.where((count.squeeze(-1)<24)[...,None],fallback,result)

    def encode_choices(self,xyz,choices,snr):
        if choices.ndim==3:
            return torch.stack([self.encode_choices(x,c,snr) for x,c in zip(xyz,choices)])
        q=choices.detach().argmax(-1)
        if self.individual:
            return self._stable_grouped(xyz,q,snr,self._encode_groups,len(self.offset_axis),choices)
        keep=torch.nonzero(q>0,as_tuple=False).flatten()
        result=xyz.new_zeros((len(q),len(self.offset_axis)))
        if not len(keep): return result+choices.sum()*0
        groups=math.ceil(len(keep)/self.group_size);size=math.ceil(len(keep)/groups);pad=groups*size-len(keep)
        x=F.pad(xyz[keep],(0,0,0,pad)).reshape(groups,size,3)
        t=F.pad(q[keep],(0,pad)).reshape(groups,size)
        c=F.pad(choices[keep],(0,0,0,pad)).reshape(groups,size,4)
        y=self._encode_groups(x,t,snr,c).reshape(-1,len(self.offset_axis))[:len(keep)]
        return result.index_copy(0,keep,y)

    def supervision_scale(self,xyz,q):
        """Same compact groups as the wire. Source-only supervision, detached."""
        def scale(x,t,_):
            _,r=self.reference(x,(t>0).to(x))
            return r.expand(*t.shape,1)
        with torch.no_grad():
            return self._grouped(xyz,q,0.,scale,1).squeeze(-1).clamp_min(.001)


def reference_position_rows(pred,target,geometry,model,scale):
    """Bounded sensitivity weighting; no inverse tiny covariance gradients."""
    span=geometry.span.to(pred)
    source_scale=(target[...,4:7]*model.attr_std[1:4]+model.attr_mean[1:4]).detach().exp()
    sigma=(source_scale.amax(-1)/span.norm()).clamp_min(1e-6)
    weight=(scale.detach()/sigma).sqrt().clamp(1,4)
    delta=pred[...,:3]-target[...,:3].detach()
    local=F.smooth_l1_loss(delta/scale[...,None].detach(),torch.zeros_like(delta),beta=.01,reduction='none').mean(-1)
    world=F.smooth_l1_loss(delta*span/span.norm(),torch.zeros_like(delta),beta=.001,reduction='none').mean(-1)
    return weight*local+world
