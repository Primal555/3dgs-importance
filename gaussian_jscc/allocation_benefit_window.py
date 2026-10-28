"""Current-network action credit and independent paired labels in the real allocation kernel."""
import torch
from torch.nn import functional as F

from .allocation_benefit_features import BenefitFeatures, keep_rates
from .allocation_benefit_replay import cpu_copy
from .budgeted_allocation_kernel import raw_batches, set_rng, timing
from .optimization import preserved_rng


class BenefitWindow:
    def __init__(self, learner, scheduler, *, finite_samples=32, chunk_size=8192):
        if finite_samples<0 or chunk_size<1:
            raise ValueError('invalid benefit window budgets')
        self.learner,self.scheduler=learner,scheduler
        self.finite_samples,self.chunk_size=finite_samples,chunk_size

    def begin(self,model,table,features,ids,cache,geometry,task,scene,q_source,
              d_ref,price,c_ref,side_proxy,snr,kind,timings):
        self.model,self.table,self.features,self.ids,self.cache=model,table,features,ids,cache
        self.task,self.scene,self.geometry=task,scene,geometry
        self.q,self.d_ref,self.price,self.c_ref=q_source,d_ref,price,c_ref
        self.side,self.snr,self.kind,self.timings=side_proxy,snr,kind,timings
        self.builder=BenefitFeatures(model,geometry,task,d_ref)
        self.conditional=table.logits.softmax(-1).detach()
        self.weights=scene.new_tensor([1.,task.distortion.lambda_s],dtype=torch.float64)/d_ref
        self.keep_components=torch.empty(len(q_source),3,2,device=scene.device,dtype=torch.float64)
        self.keep_rate=torch.empty(len(q_source),3,device=scene.device,dtype=torch.float64)
        self.keep_first=torch.empty(len(q_source),2,device=scene.device,dtype=torch.float64)
        self.entries,self.records,self.render_calls=[],[],0
        self.offsets=[];offset=0
        for batch_ids in ids:
            self.offsets.append(offset);offset+=int((batch_ids>=0).sum())

    @torch.no_grad()
    def batch(self,f,ids,entry,scene_slice,gradients):
        conditional=self.conditional[ids.clamp_min(0)]
        with timing(self.timings,'benefit_feature_seconds'):
            basis=self.builder.basis(f,ids,entry,scene_slice,gradients,conditional)
            q=entry[0];valid=ids>=0
            rates=keep_rates(q,valid,self.model.cfg.rates,xyz=self.side['xyz_per_retained'],
                             bits=self.side.get('bits_per_use',2.))*self.price/self.c_ref
            self.keep_rate[ids[valid]]=rates[valid]
            self.keep_first[ids[valid]]=basis['gate'][valid].double()
        rows,cols=valid.nonzero(as_tuple=True)
        for start in range(0,len(rows),self.chunk_size):
            r=rows[start:start+self.chunk_size];c=cols[start:start+self.chunk_size]
            rr=r.repeat_interleave(3);cc=c.repeat_interleave(3)
            targets=torch.arange(1,4,device=f.device).repeat(len(r))
            with timing(self.timings,'benefit_feature_seconds'):
                x=self.builder.make(basis,rr,cc,targets,'keep')
            with timing(self.timings,'benefit_inference_seconds'):
                value=self.learner.predict(x,'keep').reshape(len(r),3,2)
            self.keep_components[ids[r,c]]=value
        return basis

    @torch.no_grad()
    def positive(self,basis,rows,cols,new,delta,parts,position_start):
        weighted=parts.double().clone();weighted[:,1]*=self.task.distortion.lambda_s
        with timing(self.timings,'benefit_feature_seconds'):
            x=self.builder.make(basis,rows,cols,new,'positive',delta=delta,parts=weighted)
        with timing(self.timings,'benefit_inference_seconds'):
            prediction=self.learner.predict(x,'positive')
            masked=self.learner.predict(x,'positive',masked=True)
        self.entries.append(dict(features=x,prediction=prediction,masked=masked,first=weighted,
            source=basis['ids'][rows,cols].cpu(),old=basis['q'][rows,cols].cpu(),new=new.cpu(),
            position=torch.arange(len(rows),device=rows.device)+position_start))

    @torch.no_grad()
    def _basis(self,batch):
        f=self.features[batch].to(self.scene.device);ids=self.ids[batch].to(self.scene.device)
        start=self.offsets[batch];n=int((ids>=0).sum());q=self.cache[batch][0]
        gradients=[]
        for gradient in self.task.component_upstream:
            g=torch.zeros(*q.shape,self.scene.shape[-1],device=self.scene.device)
            g[ids>=0]=gradient[start:start+n];gradients.append(g)
        return self.builder.basis(f,ids,self.cache[batch],self.scene[start:start+n],gradients,
                                  self.conditional[ids.clamp_min(0)])

    @torch.no_grad()
    def _measure(self,source,new):
        batch,row,col=self.scheduler.locations[source].tolist()
        f=self.features[batch].to(self.scene.device);ids=self.ids[batch].to(self.scene.device);valid=ids>=0
        q,_,y,_,rng=self.cache[batch];old=int(q[row,col])
        if old==new:
            return torch.zeros(2,device=self.scene.device,dtype=torch.float64)
        alternative=q.clone();alternative[row,col]=new
        with preserved_rng(self.scene.device):
            set_rng(rng,self.scene.device)
            pred=self.model.forward_tier_batches(f,f[...,:3],F.one_hot(alternative,4).to(f),self.snr,self.kind,
                paired_noise=True,**({'_semantic_y':y} if old>0 and new>0 else {}))[0]
        changed=self.scene.detach().clone();raw=raw_batches(pred,self.geometry,self.model)
        start=self.offsets[batch];n=int(valid.sum())
        changed[start:start+n]=torch.cat((raw[valid],(alternative[valid]>0).to(raw)[:,None]),-1)
        self.render_calls+=1
        return self.task.delta_components(changed).detach().double()

    def _record(self,source,branch,inputs,prediction,masked,first,rates,group,new):
        return dict(source=int(source),branch=branch,step=self.side['step'],
            validation=bool(self.scheduler.validation[source]),old=int(self.q[source]),new=new,group=group,
            features=cpu_copy(inputs),prediction=prediction.detach().double().cpu().clone(),
            prediction_masked=masked.detach().double().cpu().clone(),first_order=first.detach().double().cpu().clone(),
            rates=rates.detach().double().cpu().clone(),
            conditional=self.conditional[source].detach().cpu().clone() if branch=='keep' else torch.ones(1,dtype=torch.float64),
            views=[str(c.image_name) for c in self.task.cameras],noise_seed=self.side.get('window_seed'),
            feature_protocol=self.builder.protocol.identity())

    @torch.no_grad()
    def finish(self,tier_ids,rate_values):
        device=self.scene.device;n=len(self.q);step=self.side['step']
        costs=torch.zeros(len(tier_ids),3,device=device,dtype=torch.float64)
        sources=[];old=[];new=[];positions=[];pred=[];masked=[];first=[];inputs=[]
        for entry in self.entries:
            position=entry['position'];targets=entry['new'].to(device)-1
            costs[position,targets]=entry['prediction'].sum(-1)+rate_values[position,targets]
            for key,dest in (('source',sources),('old',old),('new',new),('position',positions),
                             ('prediction',pred),('masked',masked),('first',first)):
                dest.append(entry[key])
            inputs.append(entry['features'])
        self.keep_image=(self.keep_components.sum(-1)*self.conditional).sum(-1)
        self.keep_costs=torch.zeros(n,2,device=device,dtype=torch.float64)
        self.keep_costs[:,1]=((self.keep_components.sum(-1)+self.keep_rate)*self.conditional).sum(-1)
        keep_first=(self.keep_first.sum(-1,keepdim=True)+self.keep_rate)
        keep_first=(keep_first*self.conditional).sum(-1)
        ready=bool(self.learner.network.ready)
        limit=self.finite_samples
        keep_count=min(n,(limit//3 if not sources else min(limit//3,max(1,limit//8))))
        priority=self.keep_costs[:,1] if ready else keep_first
        keep_choices=self.scheduler.choose(torch.arange(n),self.scheduler.keep_checks,priority.abs(),
            (priority-keep_first).abs(),keep_count,step,q=self.q)
        excluded=[source for source,_ in keep_choices]
        pending=[]
        # Freeze all selected q0 feature snapshots before asking for labels.
        for source,group in keep_choices:
            batch,row,col=self.scheduler.locations[source].tolist();basis=self._basis(batch)
            rr=torch.full((3,),row,device=device,dtype=torch.long);cc=torch.full_like(rr,col)
            x=self.builder.make(basis,rr,cc,torch.arange(1,4,device=device),'keep')
            record=self._record(source,'keep',x,self.keep_components[source],self.learner.predict(x,'keep',masked=True),
                self.keep_first[source].expand(3,-1),self.keep_rate[source],group,[1,2,3])
            pending.append(record)
        if sources:
            sources,old,new=torch.cat(sources),torch.cat(old),torch.cat(new)
            positions=torch.cat(positions);pred=torch.cat(pred);masked=torch.cat(masked);first=torch.cat(first)
            x={k:torch.cat([item[k] for item in inputs]) for k in inputs[0]}
            rates=rate_values[positions,new.to(device)-1]
            first_cost=first.sum(-1)+rates;pred_cost=pred.sum(-1)+rates;masked_cost=masked.sum(-1)+rates
            priority=pred_cost if ready else first_cost
            counts=self.scheduler.positive_checks[sources,old-1,new-1]
            choices=self.scheduler.choose(sources,counts,priority.abs(),
                (pred_cost-first_cost).abs()+(pred_cost-masked_cost).abs() if ready else first_cost.abs(),
                limit-3*len(keep_choices),step,excluded=excluded)
            for index,group in choices:
                source=int(sources[index]);target=int(new[index])
                record=self._record(source,'positive',{k:v[index:index+1] for k,v in x.items()},
                    pred[index:index+1],masked[index:index+1],first[index:index+1],rates[index:index+1],group,[target])
                record['position']=int(positions[index]);pending.append(record)
        for record in pending:
            source=record['source']
            if record['branch']=='keep':
                deltas=torch.stack([self._measure(source,k) for k in range(4)])
                target=(deltas[1:]-deltas[0])*self.weights
                self.scheduler.keep_checks[source]+=1
                if not record['validation']:
                    self.keep_costs[source,1]=((target.sum(-1)+self.keep_rate[source])*self.conditional[source]).sum()
                    self.keep_image[source]=(target.sum(-1)*self.conditional[source]).sum()
                record['exact_components_by_tier']=deltas.cpu()
            else:
                target=self._measure(source,record['new'][0])[None]*self.weights
                self.scheduler.positive_checks[source,record['old']-1,record['new'][0]-1]+=1
                if not record['validation']:
                    costs[record['position'],record['new'][0]-1]=target.sum()+record['rates'][0].to(device)
            record['target']=target.cpu();record['applied_to_training']=not record['validation']
            self.records.append(record)
        if self.render_calls>limit:
            raise RuntimeError('benefit exact-render budget exceeded')
        if not torch.isfinite(costs).all() or not torch.isfinite(self.keep_costs).all():
            raise RuntimeError('nonfinite learned allocation costs')
        stats=dict(benefit_keep_predicted_rows=n,benefit_exact_render_calls=self.render_calls,
            benefit_exact_positive_groups=sum(r['branch']=='positive' for r in self.records),
            benefit_exact_keep_groups=sum(r['branch']=='keep' for r in self.records),
            benefit_validation_groups=sum(r['validation'] for r in self.records),
            benefit_network_ready=ready)
        return costs,stats
