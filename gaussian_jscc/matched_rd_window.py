"""Cached common-condition predictions and exact cumulative tile interventions."""
import math

import torch

from .allocation_benefit_features import keep_rates
from .allocation_distortion import ImageDistortion
from .batched_spatial_window import BatchedSpatialWindow
from .cli import seed_all
from .optimization import preserved_rng
from .pixel_benefit_features import exact_values


class MatchedWindow(BatchedSpatialWindow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.contexts={};self.candidates={};self.active_seed=None;self.recording=False
        self.map_tensor=self.table.q;self.map_version=self.table.q._version
        self.cache_hits=0;self.cache_misses=0

    def clear_contexts(self):
        self.contexts.clear();self.candidates.clear()
        self.map_tensor=self.table.q;self.map_version=self.table.q._version

    def set_condition(self,seed): self.active_seed=int(seed)

    def prepare_context(self,cameras):
        if self.map_tensor is not self.table.q or self.map_version!=self.table.q._version:
            self.clear_contexts()
        if self.active_seed is None: raise ValueError('set a matched noise condition first')
        names=tuple(str(c.image_name) for c in cameras)
        key=(self.active_seed,names)
        if key not in self.contexts:
            if len(self.contexts)>=self.args.accept_trials:
                raise ValueError('matched context budget exceeded')
            with preserved_rng(next(self.model.parameters()).device):
                seed_all(self.active_seed)
                self.contexts[key]=super().prepare_context(cameras)
            self.cache_misses+=1
            return self.contexts[key]
        self.cache_hits+=1
        q,entries,captures,targets,components,timings=self.contexts[key]
        return q,entries,captures,targets,components,{k:0. for k in timings}

    def observe_projection(self,source,tile,changed,projected):
        if self.recording:
            row=self.candidates.setdefault((self.active_seed,source),dict(changed=changed.detach().clone(),views={}))
            row['views'][id(tile)]=projected.detach().clone()

    def _context(self,seed,cameras):
        self.set_condition(seed)
        return self.prepare_context(cameras)

    def _rates(self,entries):
        side=self.reference['side_proxy']
        return [keep_rates(q,ids>=0,self.model.cfg.rates,xyz=side['xyz_per_retained'],
                           bits=side['bits_per_use'])*self.args.beta/self.reference['c_ref']
                for _,ids,q,_,_,_ in entries]

    @torch.no_grad()
    def score(self,sources,learner,seeds,cameras):
        """Reconstruct/project once per seed; retain only shortlisted projections."""
        sources=torch.as_tensor(sources).cpu().long();self.candidates.clear()
        if not len(sources): return torch.empty(0,3,dtype=torch.float64)
        result=[];self.recording=True
        try:
            for seed in seeds:
                _,entries,captures,targets,_,_=self._context(seed,cameras)
                rates=self._rates(entries);rows={};pending=[]
                timings=dict(candidate_decode_seconds=0.,projection_seconds=0.,feature_seconds=0.,
                             teacher_seconds=0.,network_seconds=0.)
                def flush():
                    if not pending:return
                    predictions=learner.predict_group_views([r[1] for r in pending])
                    for (source,_,rate),values in zip(pending,predictions):
                        rows[source]=torch.stack(values).mean(0).sum(-1).double().cpu()+rate.cpu()
                    pending.clear()
                for source,batch,row,col,views in self.candidate_views(
                        sources.tolist(),entries,captures,targets,set(),timings):
                    pending.append((source,[v['inputs'] for v in views],rates[batch][row,col]))
                    if len(pending)>=self.candidate_batch_size:flush()
                flush()
                result.append(torch.stack([rows[int(s)] for s in sources]))
        finally:self.recording=False
        return torch.stack(result).mean(0)

    @torch.no_grad()
    def exact_costs(self,sources,seeds,cameras):
        """All four actual state labels, reusing decoded/projected candidates."""
        sources=torch.as_tensor(sources).cpu().long()
        if not len(sources):return torch.empty(0,3,dtype=torch.float64)
        result=[]
        for seed in seeds:
            _,entries,captures,targets,_,_=self._context(seed,cameras)
            rates=self._rates(entries);rows=[]
            for source in sources.tolist():
                cached=self.candidates[(seed,source)];values=[]
                for tile,target in zip(captures,targets):
                    values.append(exact_values(tile,cached['changed'],cached['views'][id(tile)],target,
                        self.reference['lambda_s'],self.reference['d_ref']).sum(-1).cpu())
                batch,row,col,_=self.positions[source]
                rows.append(torch.stack(values).mean(0)+rates[int(batch)][int(row),int(col)].cpu())
            result.append(torch.stack(rows))
        return torch.stack(result).mean(0)

    @torch.no_grad()
    def measure(self,q,seeds,cameras,meter):
        """Exact joint image effect against the shared base, including prior commits.

        Each action belongs to a different coding block. All changed blocks are
        replaced together, and every joint measurement includes the accumulated
        accepted map. No candidate decoding or neural inference occurs here.
        """
        original=self.table.q.detach().cpu();q=q.cpu().long()
        sources=(q!=original).nonzero(as_tuple=True)[0].tolist()
        costs=meter.details(q);objective=ImageDistortion(
            'mse_ssim' if self.reference['lambda_s'] else 'mse',self.reference['lambda_s'])
        distortions=[];mses=[];psnrs=[]
        for seed in seeds:
            _,_,captures,targets,components,_=self._context(seed,cameras)
            changes=[];candidates=[]
            for source in sources:
                cached=self.candidates[(seed,source)]
                changes.append(cached['changed']);candidates.append(cached)
            ids=torch.cat(changes) if changes else None
            if ids is not None and len(ids.unique())!=len(ids):
                raise ValueError('joint proposals must affect different coding blocks')
            for tile,target,base in zip(captures,targets,components):
                delta=base.new_zeros(2,dtype=torch.float64)
                if ids is not None:
                    new=torch.cat([r['views'][id(tile)][int(q[s])] for s,r in zip(sources,candidates)])
                    old=tile.projected[ids]
                    # The first returned value is state1-state0; the other
                    # states are unchanged and require no extra rendering.
                    projected=torch.stack((old,new,old,old))
                    delta=exact_values(tile,ids,projected,target,self.reference['lambda_s'],
                                       self.reference['d_ref'])[0]*self.reference['d_ref']
                mse=float(base[0]+delta[0]);distortion=float(objective.combine(base))+float(delta.sum())
                mses.append(mse);distortions.append(distortion)
                psnrs.append(-10*math.log10(max(mse,1e-30)))
        distortion=sum(distortions)/len(distortions)
        return dict(costs,source_mse=sum(mses)/len(mses),source_psnr=sum(psnrs)/len(psnrs),
            source_distortion=distortion,normalized_distortion=distortion/self.reference['d_ref'],
            normalized_cost=costs['total_channel_uses']/self.reference['c_ref'],
            tier_counts=torch.bincount(q,minlength=4).tolist(),
            rd_score=distortion/self.reference['d_ref']+
                     self.args.beta*costs['total_channel_uses']/self.reference['c_ref'])
