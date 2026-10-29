"""Gradient-free paired candidate reconstruction and full four-state supervision."""
import torch
from torch.nn import functional as F

from .allocation_benefit_features import attribute_groups,keep_rates
from .allocation_benefit_replay import cpu_copy
from .allocation_benefit_schedule import BenefitScheduler
from .budgeted_allocation_kernel import rng_state,set_rng,raw_batches,timing
from .optimization import preserved_rng
from .response_value_features import depth_context,patch_views
from .successive_codec import SuccessiveGaussianCodec


class ResponseScheduler(BenefitScheduler):
    def response_rounds(self,step):
        result=[]
        for ids in self.ids:
            available=ids>=0;safe=ids.clamp_min(0);batch=[]
            score=-(self.counts[safe].double()*(step+1)+self.last[safe])
            for _ in range(self.rounds):
                columns=score.masked_fill(~available,-torch.inf).argmax(-1)
                active=available.any(-1);columns=columns.masked_fill(~active,-1)
                batch.append(columns);available[torch.arange(len(ids))[active],columns[active]]=False
            result.append(batch)
        return [[batch[r] for batch in result] for r in range(self.rounds)]


@torch.no_grad()
def decode_candidates(model,f,q,columns,actual,state,geometry,snr,kind,*,semantic_y=None,stats=None):
    """Four absolute states from three full-batch, paired-noise tier shifts.

    Each row reuses its actual state and evaluates its other three tiers. Keep
    full original batch shapes: selecting rows before the channel would move
    their noise slots. Semantic caches are local to this selected-point round.
    """
    rows=torch.arange(len(q),device=q.device);active=columns>=0
    counts=stats if stats is not None else {}
    counts.setdefault('candidate_codec_calls',0);counts.setdefault('candidate_semantic_calls',0)
    if isinstance(model,SuccessiveGaussianCodec) and not model.training:
        current=q[rows,columns.clamp_min(0)]
        predictions=[actual]
        if active.any():
            with preserved_rng(f.device):
                if semantic_y is None:
                    semantic_y=model.semantic_encoder(f,f[...,:3],q>0)
                    counts['candidate_semantic_calls']+=1
                present=absent=semantic_y
                if (active&(current==0)).any():
                    keep=q>0;keep[rows[active],columns[active]]=True
                    present=model.semantic_encoder(f,f[...,:3],keep)
                    counts['candidate_semantic_calls']+=1
                if (active&(current>0)).any():
                    keep=q>0;keep[rows[active],columns[active]]=False
                    absent=model.semantic_encoder(f,f[...,:3],keep)
                    counts['candidate_semantic_calls']+=1
            for shift in (1,2,3):
                target=(current+shift)%4
                alternative=q.clone();alternative[rows[active],columns[active]]=target[active]
                y=torch.where((active&(target==0))[:,None,None],absent,present)
                with preserved_rng(f.device):
                    set_rng(state,f.device)
                    predictions.append(model.forward_tier_batches(f,f[...,:3],
                        F.one_hot(alternative,4).to(f),snr,kind,paired_noise=True,_semantic_y=y)[0])
                counts['candidate_codec_calls']+=1
        for tier in range(4):
            alternative=q.clone();alternative[rows[active],columns[active]]=tier
            relative=(tier-current)%4
            pred=actual
            for shift,value in enumerate(predictions[1:],1):
                pred=torch.where((active&(relative==shift))[:,None,None],value,pred)
            yield alternative,raw_batches(pred,geometry,model)
        return
    for tier in range(4):
        alternative=q.clone();alternative[rows[active],columns[active]]=tier
        with preserved_rng(f.device):
            set_rng(state,f.device)
            pred=model.forward_tier_batches(f,f[...,:3],F.one_hot(alternative,4).to(f),snr,kind,
                paired_noise=True)[0]
        counts['candidate_codec_calls']+=1
        unchanged=(q==alternative).all(-1)
        pred=torch.where(unchanged[:,None,None],actual,pred)
        yield alternative,raw_batches(pred,geometry,model)


@torch.no_grad()
def _neighbor_response(raw,baseline,valid,columns,attr_std):
    selected=valid.clone();active=columns>=0;rows=torch.arange(len(valid),device=valid.device)
    selected[rows[active],columns[active]]=False
    delta=((raw[...,3:]-baseline[...,3:])/attr_std).abs()*selected[...,None]
    count=selected.sum(-1).clamp_min(1)
    means=[];maxima=[]
    for group in attribute_groups(len(attr_std)):
        values=delta[...,group]
        means.append(values.sum((-1,-2))/(count*max(values.shape[-1],1)))
        maxima.append(values.amax((-1,-2)) if values.shape[-1] else count.float()*0)
    return torch.log1p(torch.stack(means+maxima,-1))


class ResponseValueWindow:
    gradient_free=True
    visible_response=False
    feature_protocol='response_patch_values_v1'

    def __init__(self,learner,scheduler,*,finite_samples=32,chunk_size=8192):
        self.learner,self.scheduler=learner,scheduler
        self.finite_samples,self.chunk_size=finite_samples,chunk_size
        self.records=[]

    @torch.no_grad()
    def step(self,model,table,features,id_batches,geometry,task,scheduler,snr,kind,*,d_ref,price,
             side_proxy,timings,rate_normalizer,**unused):
        from .rendering import render
        device=next(model.parameters()).device;n=len(table.logits);step=side_proxy['step']
        q_source=torch.empty(n,dtype=torch.long);parts=[];cache=[];offsets=[];offset=0
        for f,ids in zip(features,id_batches):
            f,ids=f.to(device),ids.to(device);valid=ids>=0
            q=table.sample(ids.clamp_min(0),snr)[0].masked_fill(~valid,0)
            state=rng_state(device)
            with timing(timings,'actual_decode_seconds'):
                semantic_y=None;semantic_args={}
                if isinstance(model,SuccessiveGaussianCodec) and not model.training:
                    semantic_y=model.semantic_encoder(f,f[...,:3],q>0)
                    semantic_args['_semantic_y']=semantic_y
                actual=model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).to(f),snr,kind,
                    paired_noise=True,**semantic_args)[0]
                raw=raw_batches(actual,geometry,model)
            q_source[ids[valid].cpu()]=q[valid].cpu()
            parts.append(torch.cat((raw[valid],(q[valid]>0).to(raw)[:,None]),-1))
            cache.append((q,actual,state,semantic_y));offsets.append(offset);offset+=int(valid.sum())
        scene=torch.cat(parts);del parts
        task.baseline=[];components=[];depth_maps=[];depth_scale=max(float(geometry.span.max()),1e-6)
        for camera in task.cameras:
            with timing(timings,'response_context_render_seconds'):
                image=render(scene[:,:-1],camera,task.degree,task.white_background,scene[:,-1])
                target=task.reference.get(camera,device);components.append(task.distortion.components(image,target))
                task.baseline.append((image.detach(),target.detach(),task.distortion.prepare(image,target)))
                depth_maps.append(depth_context(scene,camera,task.degree,render,depth_scale))
        visibility_context=[]
        if self.visible_response:
            from .visible_response_features import layer_context
            with timing(timings,'visibility_context_seconds'):
                visibility_context=[layer_context(scene,c,task.degree,render) for c in task.cameras]
        mean=torch.stack(components).mean(0)
        loss=float(task.distortion.combine(mean))
        rounds=scheduler.response_rounds(step)
        candidates=torch.cat([ids[torch.arange(len(ids))[cols>=0],cols[cols>=0]]
            for columns in rounds for ids,cols in zip(id_batches,columns)])
        # Complete groups cost at most three renders: the actual state is reused.
        order=torch.randperm(len(candidates),generator=scheduler.generator)
        order=order[torch.argsort(scheduler.keep_checks[candidates[order]],stable=True)]
        exact=set(candidates[order[:self.finite_samples//3]].tolist())
        pending=[];tier_ids=[];tier_costs=[];keep_costs=scene.new_zeros(n,2,dtype=torch.float64)
        conditional=table.logits.softmax(-1).detach();decode_stats={}
        scale=scene.new_tensor([1.,task.distortion.lambda_s],dtype=torch.float64)/d_ref
        for columns in rounds:
            for batch,(features_cpu,ids_cpu,cols_cpu,entry) in enumerate(zip(features,id_batches,columns,cache)):
                if not (cols_cpu>=0).any():continue
                f,ids,cols=features_cpu.to(device),ids_cpu.to(device),cols_cpu.to(device)
                q,actual,state,semantic_y=entry;valid=ids>=0;rows=(cols>=0).nonzero(as_tuple=True)[0];col=cols[rows]
                selected=ids[rows,col];base=raw_batches(actual,geometry,model)
                all_raw=[];neighbors=[];block_candidates=[];visible_raw=[];visible_q=[]
                with timing(timings,'candidate_decode_seconds'):
                    for alternative,raw in decode_candidates(model,f,q,cols,actual,state,geometry,snr,kind,
                            semantic_y=semantic_y,stats=decode_stats):
                        all_raw.append(raw[rows,col])
                        if self.visible_response:
                            visible_raw.append(raw[rows]);visible_q.append(alternative[rows])
                        neighbors.append(_neighbor_response(raw,base,valid&(q>0),cols,model.attr_std)[rows])
                        # Retain full changed blocks only for the finite-label subset.
                        snapshots={int(source):(raw[row,valid[row]].clone(),alternative[row,valid[row]].clone())
                            for source,row in zip(selected.cpu().tolist(),rows.cpu().tolist()) if source in exact}
                        block_candidates.append(snapshots)
                raw_candidates=torch.stack(all_raw,1)
                if self.visible_response:
                    visible_raw=torch.stack(visible_raw,1);visible_q=torch.stack(visible_q,1)
                action_tail=torch.cat((((raw_candidates[...,3:]-model.attr_mean)/model.attr_std).flatten(1),
                    torch.stack(neighbors,1).flatten(1)),1)
                rates=keep_rates(q,valid,model.cfg.rates,xyz=side_proxy['xyz_per_retained'],
                    bits=side_proxy['bits_per_use'])[rows,col]*price/rate_normalizer
                for start in range(0,len(rows),self.chunk_size):
                    stop=min(start+self.chunk_size,len(rows));r=rows[start:stop];c=col[start:stop]
                    old=q[r,c];sources=selected[start:stop]
                    selectors=f.new_zeros(len(r),12);selectors[:,2:6]=F.one_hot(old,4).to(f)
                    selectors[:,6]=1
                    with timing(timings,'response_patch_seconds'):
                        current=(base[r,c,3:]-model.attr_mean)/model.attr_std
                        point=torch.cat((f[r,c],current,F.one_hot(q[r,c],4).to(f)),1)
                        x=dict(point=point,action=torch.cat((selectors,action_tail[start:stop]),1),
                            views=patch_views(raw_candidates[start:stop],task.cameras,task.baseline,depth_maps,task.degree,depth_scale),
                            aux=f.new_zeros(len(r),15),available=f.new_zeros(len(r),1))
                    if self.visible_response:
                        from .visible_response_features import visible_block_views
                        with timing(timings,'visibility_features_seconds'):
                            extra=visible_block_views(visible_raw[start:stop],visible_q[start:stop],c,
                                task.cameras,task.baseline,visibility_context,task.degree)
                            x['views']=torch.cat((x['views'],extra),-1)
                    with timing(timings,'benefit_inference_seconds'):
                        values=self.learner.network.values(x).double()
                    costs=values[:,1:].sum(-1)+rates[start:stop]
                    keep_costs[sources,1]=(costs*conditional[sources]).sum(-1)
                    tier_ids.append(sources.cpu());tier_costs.append(costs)
                    for j,source in enumerate(sources.cpu().tolist()):
                        if source not in exact:continue
                        row=int(r[j]);source_x={k:v[j:j+1].clone() for k,v in x.items()}
                        flat_start=offsets[batch]+int(valid[:row].sum());length=int(valid[row].sum())
                        pending.append(dict(source=source,features=source_x,values=values[j].clone(),
                            blocks=[b[source] for b in block_candidates],start=flat_start,length=length,
                            old=int(old[j]),rates=rates[start+j].clone()))
        renders=0
        for item in pending:
            source=item['source'];old=item['old'];deltas=[]
            for tier,(raw,tier_q) in enumerate(item.pop('blocks')):
                if tier==old:deltas.append(scale.new_zeros(2));continue
                changed=scene.clone();start=item['start'];stop=start+item['length']
                changed[start:stop]=torch.cat((raw,(tier_q>0).to(raw)[:,None]),-1)
                with timing(timings,'exact_response_render_seconds'):
                    deltas.append(task.delta_components(changed));renders+=1
            values=(torch.stack(deltas)-deltas[0])*scale
            validation=bool(scheduler.validation[source]);x=item['features']
            common=dict(source=source,step=step,validation=validation,group='complete_four_state',
                views=[str(c.image_name) for c in task.cameras],noise_seed=side_proxy.get('window_seed'),
                feature_protocol=self.feature_protocol,applied_to_training=not validation)
            def record(branch,origin,new):
                count=len(new);inputs={k:v.expand(count,*v.shape[1:]).clone() for k,v in x.items()}
                inputs['action'][:,:12]=0
                inputs['action'][:,2:6]=F.one_hot(torch.full((count,),origin,device=device),4).to(scene)
                inputs['action'][:,6:9]=F.one_hot(torch.tensor(new,device=device)-1,3).to(scene)
                pred=item['values'][new]-(item['values'][origin] if branch=='positive' else 0)
                truth=values[new]-(values[origin] if branch=='positive' else 0)
                rate=item['rates'][torch.tensor(new,device=device)-1]
                if branch=='positive':rate=rate-item['rates'][origin-1]
                return cpu_copy(dict(common,branch=branch,old=origin,new=new,features=inputs,prediction=pred,
                    target=truth,rates=rate,conditional=conditional[source] if branch=='keep' else torch.ones(1,device=device)))
            self.records.append(record('keep',old,[1,2,3]))
            origin=old if old>0 else 1
            for target in (tier for tier in (1,2,3) if tier!=origin):
                self.records.append(record('positive',origin,[target]))
            scheduler.keep_checks[source]+=1
        if renders>self.finite_samples:raise RuntimeError('complete-group render budget exceeded')
        selected=torch.cat(tier_ids);scheduler.record(selected,step)
        probabilities=table.probabilities(torch.arange(n,device=device),snr)
        payload=scene.new_tensor(model.cfg.rates,dtype=torch.float64)
        unit=side_proxy['xyz_per_retained']+side_proxy.get('scale_per_retained',0.)
        expected=float((probabilities*payload).sum()+(1-probabilities[:,0]).sum()*unit+side_proxy['fixed_uses'])
        sampled=float(payload[q_source.to(device)].sum())+int((q_source>0).sum())*unit+side_proxy['fixed_uses']
        return dict(render_loss=loss,source_distortion=loss,source_mse=float(mean[0]),source_structure_error=float(mean[1]),
            view_mse=[float(p[0]) for p in components],views_per_step=len(task.cameras),
            keep_costs=keep_costs,tier_ids=selected,tier_costs=torch.cat(tier_costs),
            keep_image_costs=keep_costs*0,q=q_source,audit=[],expected_cost_proxy=expected,sampled_cost_proxy=sampled,
            sampled_tier_counts=torch.bincount(q_source,minlength=4).tolist(),
            benefit_exact_render_calls=renders,benefit_exact_keep_groups=len(pending),
            benefit_exact_positive_groups=2*len(pending),benefit_validation_groups=sum(r['validation'] for r in self.records),
            benefit_network_ready=bool(self.learner.network.ready),**decode_stats,
            actual_codec_calls=len(cache),first_order_input=False,render_backward_calls=0,
            **(dict(visibility_context_render_calls=sum(c['render_calls'] for c in visibility_context))
               if self.visible_response else {}),
            response_candidate_rows=len(selected),response_q0_candidate_rows=int((q_source[selected]==0).sum()),
            **scheduler.diagnostics(q_source,step))


class VisibleResponseWindow(ResponseValueWindow):
    visible_response=True
    feature_protocol='visible_response_patch_values_v1'
