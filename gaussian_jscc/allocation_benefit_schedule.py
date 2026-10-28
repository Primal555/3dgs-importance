"""Coverage-only state, fixed spatial holdout, and current-prediction action selection."""
import hashlib

import torch

from .allocation_mirror import BlockVisitScheduler


class BenefitScheduler(BlockVisitScheduler):
    def __init__(self, id_batches, *, seed=42, rounds=2):
        super().__init__(id_batches)
        self.seed,self.rounds=seed,rounds
        self.generator=torch.Generator().manual_seed(seed+2001)
        blocks=[row[row>=0] for ids in self.ids for row in ids]
        if len(blocks)<2:
            raise ValueError('benefit learning needs at least two spatial blocks')
        flags=[int.from_bytes(hashlib.sha256(f'{seed}:{i}'.encode()).digest()[:4],'little')%5==0 for i in range(len(blocks))]
        if all(flags):flags[-1]=False
        if not any(flags):flags[0]=True
        self.validation=torch.zeros(len(self.counts),dtype=torch.bool)
        for ids,flag in zip(blocks,flags):self.validation[ids]=flag
        self.positive_checks=torch.zeros(len(self.counts),3,3,dtype=torch.int32)
        self.keep_checks=torch.zeros(len(self.counts),dtype=torch.int32)
        self.locations=torch.empty(len(self.counts),3,dtype=torch.long)
        for batch,ids in enumerate(self.ids):
            rows,cols=(ids>=0).nonzero(as_tuple=True)
            self.locations[ids[rows,cols]]=torch.stack((torch.full_like(rows,batch),rows,cols),-1)

    def select_rounds(self,q,influence,probabilities,step):
        q,influence=q.cpu(),influence.detach().double().cpu()
        result=[]
        for ids in self.ids:
            valid=ids>=0;safe=ids.clamp_min(0);available=valid&(q[safe]>0)
            batch=[]
            for r in range(self.rounds):
                score=-(self.counts[safe].double()*(step+1)+self.last[safe]) if r==0 else influence[safe].abs()
                col=score.masked_fill(~available,-torch.inf).argmax(-1)
                active=available.any(-1);col=col.masked_fill(~active,-1);batch.append(col)
                available[torch.arange(len(ids))[active],col[active]]=False
            result.append(batch)
        return [[batch[r] for batch in result] for r in range(self.rounds)]

    def choose(self,sources,counts,importance,disagreement,limit,step,*,excluded=(),q=None):
        sources=sources.cpu().long();counts=counts.cpu();importance=importance.detach().cpu();disagreement=disagreement.detach().cpu()
        random=torch.randperm(len(sources),generator=self.generator)
        orders=[random[torch.argsort(counts[random],stable=True)],
                random[torch.argsort(importance[random],descending=True,stable=True)],
                random[torch.argsort(disagreement[random],descending=True,stable=True)]]
        chosen=[];used=torch.zeros(len(self.counts),dtype=torch.bool)
        if len(excluded):used[torch.as_tensor(excluded,dtype=torch.long)]=True
        for slot in range(limit):
            group=(slot+step-1)%3;order=orders[group]
            desired=None if q is None else ((slot+step)%2==0)
            eligible=~used[sources[order]]
            if desired is not None:
                typed=eligible&((q[sources[order]]>0)==desired)
                if typed.any():eligible=typed
            if not eligible.any():break
            index=int(order[eligible.long().argmax()]);used[sources[index]]=True
            chosen.append((index,('coverage','importance','disagreement')[group]))
        return chosen

    def diagnostics(self,q,step=None):
        kept=q.cpu()>0
        return dict(retained_candidate_evaluations_mean=float(self.counts[kept].double().mean()) if kept.any() else 0.,
                    finite_evaluated_rows=int(((self.positive_checks.sum((1,2))+self.keep_checks)>0).sum()))

    def state_dict(self):
        return dict(super().state_dict(),benefit=dict(version=1,seed=self.seed,rounds=self.rounds,
            validation=self.validation.clone(),positive_checks=self.positive_checks.clone(),
            keep_checks=self.keep_checks.clone(),rng=self.generator.get_state()))

    def load_state_dict(self,state):
        data=state['benefit']
        if data.get('version')!=1 or (data['seed'],data['rounds'])!=(self.seed,self.rounds):
            raise ValueError('benefit scheduler settings mismatch')
        if not torch.equal(data['validation'].cpu(),self.validation):
            raise ValueError('benefit holdout split mismatch')
        super().load_state_dict(state)
        for name in ('positive_checks','keep_checks'):
            value=data[name].cpu()
            if value.shape!=getattr(self,name).shape or (value<0).any():
                raise ValueError('invalid benefit action counts')
            getattr(self,name).copy_(value)
        self.generator.set_state(data['rng'].cpu())
