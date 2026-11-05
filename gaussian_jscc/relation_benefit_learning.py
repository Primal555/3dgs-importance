"""Compact label replay and supervised four-state relation values."""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from comparison_baselines.side_information_accounting import side_information_cost
from .calibrated_spatial_benefit import calibrated_direction_loss
from .physical_benefit_audit_metrics import decision_report


class RelationReplay:
    """An explicit 25/75 stable/recent training mixture, with no validation labels."""
    def __init__(self, capacity=8192, anchor_capacity=2048, validation_ids=(), seed=42):
        if capacity < 2 or not 0 < anchor_capacity < capacity:
            raise ValueError('require 0 < anchor_capacity < capacity')
        self.capacity, self.anchor_capacity = capacity, anchor_capacity
        self.validation_ids = set(map(int, validation_ids))
        self.anchor = []; self.recent = []; self.visits = {}; self.total_added = 0
        self.members=set()
        self.generator = torch.Generator().manual_seed(seed)

    def add(self, records):
        for record in records:
            source = int(record['source'])
            if source in self.validation_ids:
                raise ValueError('validation label cannot enter training replay')
            values = record['values'].detach().cpu().double().clone()
            if values.ndim != 3 or values.shape[1:] != (3,2) or not torch.isfinite(values).all():
                raise ValueError('finite float64 [views,3,2] label required')
            context=int(record.get('context',0));key=(context,source)
            if context<0 or context>3:raise ValueError('invalid background context')
            if key in self.members:
                continue
            row = dict(record, source=source, context=context, values=values)
            self.total_added += 1
            self.visits.setdefault(key, 0);self.members.add(key)
            if len(self.anchor)<self.anchor_capacity:
                self.anchor.append(row)
            else:
                self.recent.append(row)
                if len(self.recent)>self.capacity-self.anchor_capacity:
                    old=self.recent.pop(0);self.members.remove((old['context'],old['source']))

    def sample(self, count):
        if count<1 or not self.anchor:
            raise ValueError('positive batch and nonempty replay required')
        # Recent-only windows do not erase the stable reference distribution.
        available=sorted({r['context'] for r in self.anchor+self.recent})
        context=available[int(torch.randint(len(available),(1,),generator=self.generator))]
        anchor=[r for r in self.anchor if r['context']==context]
        recent=[r for r in self.recent if r['context']==context]
        stable = (float(self.anchor_capacity/self.capacity) if recent and anchor else
                  1. if anchor else 0.)
        choose = torch.rand(count, generator=self.generator)<stable
        records=[]
        for use_anchor in choose.tolist():
            pool=anchor if use_anchor else recent
            row=pool[int(torch.randint(len(pool),(1,),generator=self.generator))]
            self.visits[(row['context'],row['source'])]+=1;records.append(row)
        return records

    def coverage(self):
        rows=self.anchor+self.recent
        values=torch.tensor([self.visits[(r['context'],r['source'])] for r in rows],dtype=torch.float64)
        return dict(active_groups=len(rows),stable_groups=len(self.anchor),recent_groups=len(self.recent),
            unique_labels=len(self.visits),labels_added_total=self.total_added,all_draws=sum(self.visits.values()),
            never_updated_fraction=float((values==0).double().mean()) if len(values) else 0.,
            min_visits=int(values.min()) if len(values) else 0,
            mean_visits=float(values.mean()) if len(values) else 0.,
            groups_per_context={str(c):sum(r['context']==c for r in rows) for c in range(4)})

    def state_dict(self):
        return dict(version=2,capacity=self.capacity,anchor_capacity=self.anchor_capacity,
            validation_ids=sorted(self.validation_ids),anchor=copy.deepcopy(self.anchor),
            recent=copy.deepcopy(self.recent),visits=dict(self.visits),total_added=self.total_added,
            rng=self.generator.get_state())

    def load_state_dict(self,state):
        if (state['version']!=2 or state['capacity']!=self.capacity or
            state['anchor_capacity']!=self.anchor_capacity or set(state['validation_ids'])!=self.validation_ids):
            raise ValueError('replay identity mismatch')
        rows=state['anchor']+state['recent']
        if len(rows)>self.capacity or any(r['source'] in self.validation_ids for r in rows):
            raise ValueError('invalid training/validation replay')
        self.anchor=copy.deepcopy(state['anchor']);self.recent=copy.deepcopy(state['recent'])
        self.visits=dict(state['visits']);self.total_added=int(state['total_added'])
        self.members={(r['context'],r['source']) for r in rows}
        self.generator.set_state(state['rng'].cpu())


def marginal_cost_proxy(full_cost, count, rates=(0,8,24,48)):
    if count<1 or tuple(rates)!=(0,8,24,48):
        raise ValueError('positive scene size and 0/8/24/48 required')
    measured=side_information_cost(full_cost)
    side=measured['side_channel_uses']/count
    return dict(costs=[0.]+[float(r)+side for r in rates[1:]],
        side_per_retained=side,units='complex_channel_uses',
        scope='q3 mean side-file cost per retained point; actual mixed packet measured separately',
        digital_link=measured['link'])


def choose_tiers(values, costs, *, beta, c_ref):
    if not math.isfinite(beta) or beta<0 or not math.isfinite(c_ref) or c_ref<=0:
        raise ValueError('finite beta>=0 and c_ref>0 required')
    values=torch.as_tensor(values)
    if values.ndim!=3 or values.shape[1:]!=(3,2) or not torch.isfinite(values).all():
        raise ValueError('finite original-order [N,3,2] values required')
    costs=torch.as_tensor(costs,dtype=values.dtype,device=values.device)
    if costs.shape!=(4,) or not torch.isfinite(costs).all() or (costs<0).any() or costs[0]!=0:
        raise ValueError('four explicit nonnegative costs with q0=0 required')
    distortion=torch.cat((values.new_zeros(len(values),1),values.sum(-1)),1)
    return (distortion+beta*costs/c_ref).argmin(1).cpu()


def relation_losses(prediction,truth,rates,scale,view_weights=None):
    p,t=prediction.double(),truth.to(prediction.device,dtype=torch.float64)
    scale=torch.as_tensor(scale,dtype=torch.float64,device=p.device).clamp_min(1e-8)
    weights=(torch.ones(p.shape[1],device=p.device,dtype=torch.float64) if view_weights is None
             else view_weights.to(p.device,dtype=torch.float64))
    weights=weights/weights.sum()
    common_p,common_t=p.mean(-2),t.mean(-2)
    common=F.smooth_l1_loss(common_p/scale,common_t/scale,reduction='none').mean(-1)
    relative=F.smooth_l1_loss((p-common_p[:,:,None])/scale,
                            (t-common_t[:,:,None])/scale,reduction='none').mean((-2,-1))
    common=(common*weights).sum(-1);relative=(relative*weights).sum(-1)
    mean_p=(p*weights[None,:,None,None]).sum(1)
    mean_t=(t*weights[None,:,None,None]).sum(1)
    rates=torch.as_tensor(rates,device=p.device,dtype=torch.float64)
    if rates.ndim==1:rates=rates[None].expand(len(p),-1)
    direction=calibrated_direction_loss(mean_p,mean_t,rates,scale.sum(),.02)
    return dict(common=common.mean(),relative=relative.mean(),direction=direction.mean(),
                total=(common+relative+.25*direction).mean())


class RelationLearner:
    def __init__(self,network,replay,bank,lr=1e-4,weight_decay=1e-4):
        self.network,self.replay,self.bank=network,replay,bank
        self.optimizer=torch.optim.AdamW(network.parameters(),lr=lr,weight_decay=weight_decay)
        self.scales=None;self.updates=0

    def fit_scales(self,groups):
        if self.scales is not None or not groups:
            raise ValueError('nonempty fresh training-only labels required')
        if any(g['source'] in self.replay.validation_ids for g in groups):
            raise ValueError('validation labels cannot calibrate output scales')
        target=torch.stack([g['values'].double() for g in groups])
        scales=target.square().mean((0,1,2)).sqrt()
        scales=scales.clamp_min(max(1e-8,float(scales.max())*1e-3))
        self.scales=scales
        self.network.output_scale.copy_(scales.to(self.network.output_scale))

    def train_updates(self,count,batch_groups,rates):
        if self.scales is None or count<1 or batch_groups<1:
            raise ValueError('fixed scales and positive update/batch counts required')
        stats=[];self.network.train()
        for _ in range(count):
            groups=self.replay.sample(batch_groups)
            output=self.network(self.bank,[g['source'] for g in groups],context=groups[0].get('context',0))
            truth=torch.stack([g['values'] for g in groups])
            losses=relation_losses(output['per_view'],truth,rates,self.scales,self.bank['view_weights'])
            loss=losses['total']
            if not torch.isfinite(loss):raise RuntimeError('nonfinite relation loss')
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=nn.utils.clip_grad_norm_(self.network.parameters(),10.)
            if not torch.isfinite(grad):raise RuntimeError('nonfinite relation gradient')
            self.optimizer.step();self.updates+=1
            stats.append(dict(loss=float(loss.detach()),grad_norm=float(grad),
                **{k:float(v.detach()) for k,v in losses.items() if k!='total'}))
        result={k:float(np.mean([r[k] for r in stats])) for k in stats[0]}
        result.update(updates=self.updates,grad_norm_mean=result.pop('grad_norm'),
                      grad_norm_max=max(r['grad_norm'] for r in stats),
                      clipped_fraction=sum(r['grad_norm']>10 for r in stats)/len(stats))
        return result

    @torch.inference_mode()
    def evaluate(self,groups,rates,batch_size=32):
        if not groups:raise ValueError('nonempty validation required')
        self.network.eval();pred=torch.empty(len(groups),3,2,dtype=torch.float64)
        contexts=sorted({g.get('context',0) for g in groups})
        for context in contexts:
            indices=[i for i,g in enumerate(groups) if g.get('context',0)==context]
            for start in range(0,len(indices),batch_size):
                chosen=indices[start:start+batch_size]
                pred[chosen]=self.network(self.bank,[groups[i]['source'] for i in chosen],context=context)['values'].cpu().double()
        weights=self.bank['view_weights'].double()
        weights/=weights.sum()
        truth=torch.stack([(g['values']*weights[:,None,None]).sum(0) for g in groups])
        rates=torch.as_tensor(rates,dtype=torch.float64)[None].expand(len(groups),-1)
        report=decision_report(pred,truth,rates,[1.]*len(groups))
        report['by_context']={str(c):decision_report(pred[idx],truth[idx],rates[idx],[1.]*len(idx))
            for c in contexts for idx in [[i for i,g in enumerate(groups) if g.get('context',0)==c]]}
        paired={}
        for i,g in enumerate(groups):paired.setdefault(g['source'],[]).append(i)
        complete=[idx for idx in paired.values() if len(idx)==len(contexts) and
                  len({groups[i].get('context',0) for i in idx})==len(contexts)]
        if complete:
            mean_p=torch.stack([pred[idx].mean(0) for idx in complete])
            mean_t=torch.stack([truth[idx].mean(0) for idx in complete])
            mean_r=torch.stack([rates[idx].mean(0) for idx in complete])
            report['deployment_context_mean']=decision_report(mean_p,mean_t,mean_r,[1.]*len(complete))
        return report

    def state_dict(self):
        return dict(network=self.network.state_dict(),optimizer=self.optimizer.state_dict(),
                    scales=self.scales,updates=self.updates)

    def load_state_dict(self,state):
        self.network.load_state_dict(state['network']);self.optimizer.load_state_dict(state['optimizer'])
        self.scales=None if state['scales'] is None else state['scales'].cpu().double()
        self.updates=int(state['updates'])
