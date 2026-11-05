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
    """Retain every fixed label, sample contexts then rows uniformly in O(batch)."""
    policy='full_history_context_uniform_v1'

    def __init__(self, validation_ids=(), seed=42):
        self.validation_ids = set(map(int, validation_ids))
        self.records=[];self.by_context=[[] for _ in range(4)]
        self.visits={};self.members=set();self.migration=None
        self.generator = torch.Generator().manual_seed(seed)

    def add(self, records):
        for record in records:
            source = int(record['source'])
            if source<0:raise ValueError('nonnegative original source required')
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
            self.visits.setdefault(key, 0);self.members.add(key)
            self.records.append(row);self.by_context[context].append(row)

    def sample(self, count):
        if count<1 or not self.records:
            raise ValueError('positive batch and nonempty replay required')
        available=[c for c,pool in enumerate(self.by_context) if pool]
        context=available[int(torch.randint(len(available),(1,),generator=self.generator))]
        pool=self.by_context[context]
        indices=torch.randint(len(pool),(count,),generator=self.generator)
        records=[]
        for index in indices.tolist():
            row=pool[index]
            self.visits[(row['context'],row['source'])]+=1;records.append(row)
        return records

    def coverage(self):
        rows=self.records
        values=torch.tensor([self.visits[(r['context'],r['source'])] for r in rows],dtype=torch.float64)
        return dict(active_groups=len(rows),retention_policy=self.policy,
            unique_labels=len(self.visits),labels_added_total=len(rows),all_draws=sum(self.visits.values()),
            never_updated_fraction=float((values==0).double().mean()) if len(values) else 0.,
            min_visits=int(values.min()) if len(values) else 0,
            mean_visits=float(values.mean()) if len(values) else 0.,
            max_visits=int(values.max()) if len(values) else 0,
            groups_per_context={str(c):len(pool) for c,pool in enumerate(self.by_context)})

    def state_dict(self):
        return dict(version=3,retention_policy=self.policy,validation_ids=sorted(self.validation_ids),
            records=copy.deepcopy(self.records),visits=dict(self.visits),
            migration=self.migration,rng=self.generator.get_state())

    def load_state_dict(self,state):
        if state['version'] not in (2,3) or set(state['validation_ids'])!=self.validation_ids:
            raise ValueError('replay identity mismatch')
        if state['version']==3 and state.get('retention_policy')!=self.policy:
            raise ValueError('replay retention policy mismatch')
        rows=state['records'] if state['version']==3 else state['anchor']+state['recent']
        keys={(int(r.get('context',0)),int(r['source'])) for r in rows}
        visits=dict(state['visits'])
        if state['version']==2 and set(visits)!=keys:
            raise ValueError('legacy replay has evicted historical labels; full-history migration requires all labels')
        if len(keys)!=len(rows) or set(visits)!=keys or any(type(v) is not int or v<0 for v in visits.values()):
            raise ValueError('invalid historical label identities or visits')
        temporary=RelationReplay(self.validation_ids)
        temporary.add(rows)
        self.records,self.by_context=temporary.records,temporary.by_context
        self.members=temporary.members;self.visits=visits
        self.migration=(dict(source_version=2,restored_unique_labels=len(rows),all_labels_preserved=True)
                        if state['version']==2 else state.get('migration'))
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
