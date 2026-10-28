"""Read-only paired labels for costs produced by the actual allocation kernel."""
import torch
from torch.nn import functional as F
from .structured_evidence import BoundedMirrorTable, risk_choices
from .budgeted_allocation_kernel import raw_batches, set_rng
from .optimization import preserved_rng


def restore_history(scheduler,state):
    """Restore predictive history in place; audit coverage counters may advance."""
    for name,value in state['evidence'].items():
        if isinstance(value,torch.Tensor): getattr(scheduler,name).copy_(value)
    scheduler.context_ready=state['evidence']['context_ready']
    scheduler.invalidations=state['evidence']['invalidations']
    for name,value in state['structured']['tensors'].items(): getattr(scheduler,name).copy_(value)
    scheduler.refresh_events=state['structured']['refresh_events']
    for name,value in state['calibrated']['tensors'].items(): getattr(scheduler,name).copy_(value)
    # Finite feedback is disabled in the audit kernel, so sparse banks are never updated.
    for branch in ('positive','keep'):
        bank=getattr(scheduler,branch+'_corrections')
        for name,value in state['calibrated'][branch].items():
            if not torch.equal(getattr(bank,name),value):
                raise RuntimeError('audit changed the independent correction bank')


@torch.no_grad()
def hypothetical_update(table,source,costs,old,new,keep_lr,tier_lr,gain,branch='positive',trust=None):
    """Use a one-point copy of the real bounded mirror update, including floors."""
    probe=BoundedMirrorTable(1,max_log_odds=table.max_log_odds)
    probe.keep_logits[0].copy_(table.keep_logits[source].cpu())
    probe.logits[0].copy_(table.logits[source].cpu())
    probe.keep_trust=table.keep_trust[source:source+1].cpu() if table.keep_trust is not None else None
    probe.tier_trust=table.tier_trust
    if branch=='positive':
        # The caller supplies the already-selected point's positive confidence.
        if trust is None:
            trust=table.tier_trust[source] if table.tier_trust is not None else torch.tensor(1.)
        probe.tier_trust=trust.reshape(1).cpu()
        before=float(probe.logits[0,new-1]-probe.logits[0,old-1])
        probe.update(torch.zeros(1,2),torch.tensor([0]),costs.detach().cpu()[None],keep_lr,tier_lr,gain)
        return float(probe.logits[0,new-1]-probe.logits[0,old-1])-before
    before=float(probe.keep_logits[0,1]-probe.keep_logits[0,0])
    probe.tier_trust=None
    probe.update(costs.detach().cpu()[None],torch.empty(0,dtype=torch.long),torch.empty(0,3),keep_lr,tier_lr,gain)
    return float(probe.keep_logits[0,1]-probe.keep_logits[0,0])-before


def conditional_keep_label(deltas,conditional,lambda_s,d_ref,rate):
    parts=deltas[1:]-deltas[0]
    return float(((parts[:,0]+lambda_s*parts[:,1])*conditional.to(parts)).sum()/d_ref+rate)


def assessment(predicted,actual):
    informative=abs(predicted)>1e-14 and abs(actual)>1e-14
    agrees=predicted*actual>0 if informative else None
    return dict(prediction=float(predicted),truth=float(actual),sign_correct=agrees,
        absolute_error=abs(float(predicted-actual)),regret=abs(float(actual)) if agrees is False else 0.)


class UpdateInspector:
    def __init__(self,samples,keep_lr,tier_lr):
        self.samples,self.keep_lr,self.tier_lr=samples,keep_lr,tier_lr
        self.records=[];self.render_calls=0

    @torch.no_grad()
    def __call__(self,*,model,features,id_batches,cache,geometry,task,scene,scheduler,
                 tier_ids,locations,components,rates,keep_parts,keep_tiers,table,
                 tier_costs,keep_costs,d_ref,price,c_ref,side_proxy,snr,kind):
        step=side_proxy['step'];weight=task.distortion.lambda_s
        n=len(scheduler.counts);device=scene.device
        self.records=[];self.render_calls=0
        corrected=scheduler.preview_positive(tier_ids,components,step)
        corrected_keep=scheduler.preview_keep(keep_tiers,keep_parts,step)
        raw=components.detach().double().cpu();rate=rates.detach().double().cpu()
        costs=tier_costs.detach().double().cpu();kc=keep_costs.detach().double().cpu()
        ids=tier_ids.cpu();q=scheduler.current_q
        conditional=table.logits.softmax(-1).detach().cpu()
        offsets=[];offset=0
        for batch_ids in id_batches:
            offsets.append(offset);offset+=int((batch_ids>=0).sum())

        def measure(location,new):
            batch,row,col=location
            f=features[batch].to(device);batch_ids=id_batches[batch].to(device);valid=batch_ids>=0
            old_q,_,y,_,rng=cache[batch];old=int(old_q[row,col])
            if old==new:return torch.zeros(2,dtype=torch.float64,device=device)
            alt=old_q.clone();alt[row,col]=new
            with preserved_rng(device):
                set_rng(rng,device)
                pred=model.forward_tier_batches(f,f[...,:3],F.one_hot(alt,4).to(f),snr,kind,
                    paired_noise=True,**({'_semantic_y':y} if old>0 and new>0 else {}))[0]
            changed=scene.detach().clone();raw_alt=raw_batches(pred,geometry,model)
            start=offsets[batch];count=int(valid.sum())
            changed[start:start+count]=torch.cat((raw_alt[valid],(alt[valid]>0).to(raw_alt)[:,None]),-1)
            self.render_calls+=1
            return task.delta_components(changed)

        keep_count=min(n,max(1,self.samples//16))
        positive_limit=min(len(ids),self.samples-3*keep_count)
        choices=risk_choices(ids,scheduler._combine(corrected,weight)+rate,q,scheduler,positive_limit,step)
        for index,new,group in choices:
            source=int(ids[index]);old=int(q[source]);measured=measure(locations[index],new)
            truth=float(scheduler._combine(measured,weight)/d_ref)+float(rate[index,new-1])
            predictions={
                'raw_current':float(scheduler._combine(raw[index,new-1],weight))+float(rate[index,new-1]),
                'corrected_current':float(scheduler._combine(corrected[index,new-1],weight))+float(rate[index,new-1]),
                'history_update':float(costs[index,new-1]-costs[index,old-1])}
            movement=hypothetical_update(table,source,costs[index],old,new,self.keep_lr,self.tier_lr,n,
                trust=scheduler.tier_trust[source])
            self.records.append(dict(branch='positive',source_row=source,old_tier=old,new_tier=new,
                selection_group=group,checks={k:assessment(v,truth) for k,v in predictions.items()},
                delta_log_odds=movement,update_direction_correct=assessment(-movement,truth)['sign_correct'],
                exact_components=measured.cpu().tolist(),rate_delta=float(rate[index,new-1]),
                prediction_scope='all predictions fixed before this exact label; no finite replacement'))

        rotation=torch.roll(torch.arange(n),-(step%n))
        boundary=(kc[:,1]-kc[:,0]).abs()
        risk=scheduler.error+scheduler.refresh.double()
        coverage=scheduler.keep_exact_count.sum(-1)
        orders=[rotation[torch.argsort(boundary[rotation],stable=True)],
                rotation[torch.argsort(risk[rotation],descending=True,stable=True)],
                rotation[torch.argsort(coverage[rotation],stable=True)]]
        chosen=set()
        for slot in range(keep_count):
            group=(slot+step-1)%3
            source=next(int(i) for i in orders[group] if int(i) not in chosen);chosen.add(source)
            location=None
            for batch,batch_ids in enumerate(id_batches):
                found=(batch_ids==source).nonzero()
                if len(found):location=(batch,*found[0].tolist());break
            deltas=torch.stack([measure(location,new) for new in range(4)])
            expected_payload=float(conditional[source]@torch.tensor(model.cfg.rates[1:],dtype=torch.float64))
            side=side_proxy['xyz_per_retained']+side_proxy.get('scale_per_retained',0.)
            rate_keep=price*(expected_payload+side)/c_ref
            mixture_truth=conditional_keep_label(deltas,conditional[source],weight,d_ref,rate_keep)
            history=float(kc[source,1]-kc[source,0])
            movement=hypothetical_update(table,source,kc[source],0,1,self.keep_lr,self.tier_lr,n,branch='keep')
            tier=int(keep_tiers[source])
            sampled_rate=price*(model.cfg.rates[tier]+side)/c_ref
            on_minus_off=deltas[tier]-deltas[0]
            sampled_truth=float(scheduler._combine(on_minus_off,weight)/d_ref)+sampled_rate
            use_mixture=bool(scheduler.memory and scheduler.keep_ready(step)[source])
            truth=mixture_truth if use_mixture else float(scheduler._combine(on_minus_off,weight)/d_ref)+rate_keep
            self.records.append(dict(branch='keep_mixture' if use_mixture else 'keep_sampled',source_row=source,current_tier=int(q[source]),
                selection_group=('boundary','uncertainty','coverage')[group],
                conditional=conditional[source].tolist(),exact_components_by_tier=deltas.cpu().tolist(),
                exact_conditional_delta_rd=mixture_truth,
                rate_delta=rate_keep,checks={'history_update':assessment(history,truth)},
                sampled_reference=dict(tier=tier,
                    raw_current=assessment(float(scheduler._combine(keep_parts[source].cpu(),weight))+sampled_rate,sampled_truth),
                    corrected_current=assessment(float(scheduler._combine(corrected_keep[source],weight))+sampled_rate,sampled_truth)),
                delta_log_odds=movement,update_direction_correct=assessment(-movement,truth)['sign_correct'],
                prediction_scope=('conditional keep minus drop, matched three-tier exact expectation' if use_mixture
                    else 'unready or no-memory fallback: sampled distortion plus expected rate')))
        if self.render_calls>self.samples:raise RuntimeError('audit exceeded counterfactual render budget')
