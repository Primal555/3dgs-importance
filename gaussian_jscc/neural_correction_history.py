"""Context-bound teacher residuals evaluated by the current network weights."""
import copy
import torch
from .calibrated_evidence import CalibratedEvidence


class ContextResidualBank:
    def __init__(self,half_life):
        self.half_life=half_life;self.records={};self.pending={};self.clear_predictions()
        self.learner=self.replay=None;self.views=();self.blocks={};self.visibility={}

    @property
    def keys(self):return torch.tensor(sorted({key[0] for key in self.records}),dtype=torch.long)

    def configure_views(self,count,window):pass

    def bind(self,learner,replay,reference,batch_size=64):
        self.learner,self.replay,self.reference=learner,replay,reference
        self.batch_size=max(1,min(64,int(batch_size)))

    def clear_predictions(self):self.predictions={}

    def set_context(self,views,blocks,visibility):
        self.views=tuple(views);self.blocks=blocks;self.visibility=visibility
        self.pending={};self.clear_predictions()

    def stage(self,key,record_id,step):
        source=key//9
        self.pending[key,self.views,self.blocks[source]]=dict(record_id=record_id,step=step,views=self.views,
            block=self.blocks[source],visibility=self.visibility[source].clone())

    def prune(self):
        active={e['id'] for e in self.replay.training}
        self.records={k:v for k,v in self.records.items() if v['record_id'] in active}

    def update(self,keys,values,context,step,views):
        for key in keys.tolist():
            indexed=(key,self.views,self.blocks.get(key//9))
            if indexed in self.pending:self.records[indexed]=self.pending[indexed]

    @torch.no_grad()
    def query(self,keys,context,step):
        entries={e['id']:e for e in self.replay.training} if self.replay is not None else {}
        chosen=[]
        for pos,key in enumerate(keys.flatten().tolist()):
            source=key//9;record=self.records.get((key,self.views,self.blocks.get(source)))
            if (record is not None and record['record_id'] in entries and
                    record['views']==self.views and record['block']==self.blocks.get(source) and
                    source in self.visibility):chosen.append((pos,key,record))
        missing={r['record_id']:entries[r['record_id']] for _,_,r in chosen
                 if r['record_id'] not in self.predictions}
        if missing:
            all_ids=list(missing)
            for start in range(0,len(all_ids),self.batch_size):
                ids=all_ids[start:start+self.batch_size];groups=[self.replay._read(missing[k]) for k in ids]
                outputs=self.learner.predict_group_views([[v['inputs'] for v in g['views']] for g in groups])
                for key,group,output in zip(ids,groups,outputs):
                    estimate=torch.stack(output).mean(0).cpu().double()
                    gap=estimate[group['new_tier']-1]-estimate[group['old_tier']-1]
                    residual=group['delta_values'].double().cpu()-gap
                    weight=self.reference['lambda_s']
                    residual[1]=residual[1]/weight if weight else 0.
                    self.predictions[key]=residual
        positions=[];means=[];errors=[]
        for pos,key,record in chosen:
            source=key//9;residual=self.predictions[record['record_id']]
            drift=(self.visibility[source]-record['visibility']).abs().mean()
            attenuation=torch.exp(-drift)*2**(-(step-record['step'])/self.half_life)
            positions.append(pos);means.append(residual*attenuation)
            errors.append(residual.abs()*(1-attenuation+attenuation/2**.5))
        return (torch.tensor(positions,dtype=torch.long),torch.stack(means) if means else torch.empty(0,2,dtype=torch.float64),
                torch.stack(errors) if errors else torch.empty(0,2,dtype=torch.float64))

    def state_dict(self):
        return dict(version=1,keys=self.keys,count=torch.ones(len(self.records),dtype=torch.long),
                    records=copy.deepcopy(self.records))

    def load_state_dict(self,state,count):
        if state.get('version')!=1:raise ValueError('network correction bank mismatch')
        for indexed,record in state['records'].items():
            key,views,block=indexed
            if (not isinstance(key,int) or not 0<=key<count*9 or record['step']<1 or
                    views!=record['views'] or block!=record['block'] or
                    not torch.isfinite(record['visibility']).all()):raise ValueError('invalid network correction record')
        self.records=copy.deepcopy(state['records']);self.clear_predictions()


class NetworkAwareEvidence(CalibratedEvidence):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.positive_corrections=ContextResidualBank(self.half_life)
        self.prediction_revision=torch.full_like(self.counts,-1)
        self.model_revision=0;self.reset_rows=0

    def accumulate_components(self,ids,parts,rate,step,lambda_s,**kwargs):
        ids=ids.cpu();changed=ids[self.prediction_revision[ids]!=self.model_revision]
        for name in ('mean','second','weight','positive_weight2','positive_effective_samples',
                     'positive_view_bits','positive_view_count','positive_error'):
            getattr(self,name)[changed]=0
        self.prediction_revision[ids]=self.model_revision;self.reset_rows=len(changed)
        return super().accumulate_components(ids,parts,rate,step,lambda_s,**kwargs)

    def state_dict(self):
        return dict(super().state_dict(),neural_history=dict(version=1,revision=self.prediction_revision.clone()))

    def load_state_dict(self,state):
        extra=state.get('neural_history',{})
        revision=extra.get('revision')
        if (extra.get('version')!=1 or not isinstance(revision,torch.Tensor) or
                revision.shape!=self.counts.shape or revision.dtype!=torch.long or (revision < -1).any()):
            raise ValueError('network-aware history mismatch')
        super().load_state_dict(state);self.prediction_revision=revision.cpu().clone()
