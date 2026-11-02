"""Read-only diagnosis of selected-action errors on a fixed direct-RD map."""
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import time

import numpy as np
import torch

from . import direct_rd_allocation as direct
from .allocation_distortion import ImageDistortion
from .allocation_mirror import ViewWindowScheduler
from .batched_spatial_window import BatchedSpatialWindow
from .budgeted_allocation_evaluation import ExactDeploymentCost,evaluate_map,tier_hash
from .budgeted_allocation_inputs import load_inputs
from .calibrated_spatial_benefit_allocation import batched_preflight
from .cli import device_for,seed_all
from .data import read_ply
from .direct_rd_decisions import HardTierMap,ProposalBank
from .direct_rd_diagnostic_kernel import select_actions,evaluate_actions,evaluate_other_views
from .direct_rd_diagnostic_report import summarize
from .direct_rd_diagnostic_inputs import diagnostic_scene
from .direct_rd_process import running_workers
from .model_factory import build_codec
from .modular_experiment import json_write
from .optimization import preserved_rng
from .transport import model_id


FAMILY='direct_rd_selected_action_diagnostic_v1'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda')
    for name,default in (('proposal-windows',10),('per-stratum',16),('allocation-batch',128),('sanity-actions',4)):
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--action-tie',type=float,default=1e-8)
    p.add_argument('--consistency-atol',type=float,default=1e-6)
    p.add_argument('--consistency-rtol',type=float,default=.005)
    p.add_argument('--dry-run',action='store_true')
    return p


def file_digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def average(rows):
    return {key:sum(r[key] for r in rows)/len(rows) for key in rows[0] if key not in ('seed','views','context','trial')}


@torch.no_grad()
def full_single_deltas(window,actions,cameras,objective,d_ref):
    from .rendering import render
    device=window.base_raw.device
    mask=window.table.q[window.render_sources.to(device)]>0
    baselines=[]
    for camera in cameras:
        before=render(window.base_raw[mask],camera,window.scene.degree,window.scene.white_background)
        target=window.scene.reference.get(camera,device)
        baselines.append((before,target,objective.prepare(before,target)))
    result={}
    for action in actions:
        raw,active=window.intervention_raw(action['source'],action['new_tier'])
        values=[]
        for camera,(before,target,prepared) in zip(cameras,baselines):
            after=render(raw[active.bool()],camera,window.scene.degree,window.scene.white_background)
            values.append(float(objective.combine(objective.delta(after,before,target,prepared)))/d_ref)
        result[action['source']]=sum(values)/len(values)
    return result


def main(args):
    run,out=Path(args.run).resolve(),Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):raise ValueError('diagnostic output must be separate')
    if min(args.proposal_windows,args.per_stratum,args.allocation_batch)<1 or args.sanity_actions<0:
        raise ValueError('invalid sample budget')
    if any(not np.isfinite(x) or x<0 for x in (args.action_tie,args.consistency_atol,args.consistency_rtol)):
        raise ValueError('invalid numerical tolerance')
    workers=running_workers(run)
    if workers and not args.dry_run:raise RuntimeError(f'pause source run first: {workers}')
    path=run/'checkpoints/latest/state.pt'; source_sha=file_digest(path)
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('family')!=direct.FAMILY or not state['progress']['initial_done']:
        raise ValueError('requires initialized direct RD checkpoint')
    if file_digest(path)!=source_sha:raise RuntimeError('source checkpoint changed while reading')
    original=state['plan']['settings']
    codec_root=Path(original['run']).resolve()
    if out.is_relative_to(codec_root) or codec_root.is_relative_to(out):raise ValueError('output overlaps CODEC source')
    source_args=SimpleNamespace(**dict(original,out=str(out),device=args.device))
    saved,cfg,spec,old,options,identity=load_inputs(source_args)
    if identity!=state['plan']['source']:raise ValueError('source codec identity mismatch')
    plan=dict(version=FAMILY,source_checkpoint=str(path),source_sha256=source_sha,
        source_step=state['progress']['step'],tier_sha256=tier_hash(state['q']),
        settings=vars(args),writes_source=False,
        scope='new traceable proposals on fixed checkpoint map; no historical rejected-ID replay')
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    if out.exists():raise FileExistsError('use a NEW diagnostic output')
    started=time.perf_counter();device=device_for(args.device)
    torch.set_num_threads(original['cpu_threads']);seed_all(original['seed'])
    model=build_codec(cfg).to(device);model.load_state_dict(saved['state_dict']);del saved
    model.eval().requires_grad_(False);model.set_training_snrs([10.])
    if model_id(model)!=state['codec_id']:raise ValueError('codec state differs from source experiment')
    raw,degree=read_ply(spec['ply'])
    scene=diagnostic_scene(spec,raw,degree,model,options,out,old=old,expected=state['scene_record']);del raw
    model.set_scene_geometry(scene.geometry)
    if any(state['scene_record'].get(k)!=v for k,v in scene.record.items()):raise ValueError('scene identity mismatch')
    lookup={str(c.image_name):c for c in scene.cameras}
    training=[lookup[n] for n in state['scene_record']['allocation_train_view_names']]
    calibration=[lookup[n] for n in state['scene_record']['calibration_view_names']]
    reference=copy.deepcopy(state['reference']);q=state['q'].long()
    learner=direct.FrozenBenefit(state['frozen']['inference'],device)
    network_hash=direct.tensor_digest(learner.network.state_dict())
    window_args=SimpleNamespace(**dict(state['frozen']['window_settings'],allocation_batch=args.allocation_batch,
        label_groups=0,beta=original['beta'],snr=original['snr'],cpu_threads=original['cpu_threads'],renderer_backend=device.type))
    table=HardTierMap(q,device)
    scheduler=direct.FrozenCandidateScheduler(scene.group_ids,seed=original['seed']);scheduler.load_state_dict(state['scheduler'])
    views=ViewWindowScheduler(state['scene_record']['allocation_train_view_names'],original['seed']+9001)
    views.load_state_dict(state['views'])
    bank=ProposalBank(len(q));window=BatchedSpatialWindow(model,scene,table,scheduler,reference,window_args)
    blocks=torch.empty(len(q),dtype=torch.long);offset=0
    for ids in scene.group_ids:
        r,c=(ids>=0).nonzero(as_tuple=True);blocks[ids[r,c].cpu()]=r.cpu()+offset;offset+=len(ids)
    out.mkdir(parents=True,exist_ok=True);json_write(out/'plan.json',plan)
    with preserved_rng(device):preflight=batched_preflight(model,scene,training[0],window_args,reference)
    json_write(out/'preflight.json',preflight)
    contexts=[];observations=defaultdict(list)
    for i in range(args.proposal_windows):
        step=state['progress']['step']+i+1;seed=original['seed']+3000003+step
        indices=views.next(original['window_views']);seed_all(seed)
        result=window.collect(step,[training[j] for j in indices],learner,direct.NoReplay(),False)
        bank.add(result['tier_ids'],result['tier_costs'])
        for source in result['tier_ids'].tolist():observations[source].append(i)
        contexts.append(dict(step=step,seed=seed,view_indices=list(indices),
                             view_names=[str(training[j].image_name) for j in indices]))
        print(f"proposal window {i+1}/{args.proposal_windows}: evaluated={len(result['tier_ids'])}",flush=True)
    proposal=bank.propose(q,blocks)
    actions=select_actions(bank,proposal,q,blocks,per_stratum=args.per_stratum,seed=original['seed']+713)
    if not actions:raise RuntimeError('no sampled candidate actions')
    for action in actions:action['selection_contexts']=observations[action['source']]
    static=getattr(window,'static',None);del window,result
    np.savez_compressed(out/'proposals.npz',**{k:v.numpy() for k,v in proposal.items()})
    json_write(out/'contexts.json',contexts)
    meter=ExactDeploymentCost(model,scene,state['codec_id'],original['snr'],options.net_bits_per_use)
    base_cost=meter.details(q)
    for action in actions:
        alt=q.clone();alt[action['source']]=action['new_tier'];cost=meter.details(alt)
        action['channel_use_delta']=cost['total_channel_uses']-base_cost['total_channel_uses']
        action['measured_rate_delta']=original['beta']*action['channel_use_delta']/reference['c_ref']
    collected={name:defaultdict(list) for name in ('matched','calibration_same_noise','calibration_accept')}
    for i,context in enumerate(contexts):
        chosen=[a for a in actions if i in a['selection_contexts']]
        if not chosen:continue
        rows,exact_window=evaluate_actions(model,scene,q,chosen,[training[j] for j in context['view_indices']],
            learner,reference,window_args,seed=context['seed'],static=static)
        for source,row in rows.items():
            collected['matched'][source].append(dict(row,seed=context['seed'],context=i,
                views=context['view_names']))
        shifted=evaluate_other_views(exact_window,calibration)
        for source,row in shifted.items():
            collected['calibration_same_noise'][source].append(dict(row,seed=context['seed'],context=i,
                views=[str(c.image_name) for c in calibration]))
        static=exact_window.static;del exact_window
        print(f"matched labels {i+1}/{len(contexts)}: actions={len(chosen)}",flush=True)
    objective=ImageDistortion('mse_ssim' if reference['lambda_s'] else 'mse',reference['lambda_s'])
    checks=[]
    sanity_actions=[]
    for name in ('top','middle','bottom','ordinary'):
        available=[a for a in actions if a['stratum']==name]
        if available:sanity_actions.append(available[0])
    sanity_actions=sanity_actions[:args.sanity_actions]
    for trial in range(original['accept_trials']):
        seed=original['seed']+1000+trial
        rows,exact_window=evaluate_actions(model,scene,q,actions,calibration,learner,reference,window_args,seed=seed,static=static)
        for source,row in rows.items():
            collected['calibration_accept'][source].append(dict(row,seed=seed,trial=trial,
                views=[str(c.image_name) for c in calibration]))
        if sanity_actions:
            native=full_single_deltas(exact_window,sanity_actions,calibration,objective,reference['d_ref'])
            for source,value in native.items():
                label=rows[source]['true_image_delta'];error=abs(value-label)
                tolerance=args.consistency_atol+args.consistency_rtol*abs(value)
                checks.append(dict(kind='single_local_vs_full',source=source,trial=trial,
                    full_delta=value,local_delta=label,abs_error=error,tolerance=tolerance,passed=error<=tolerance))
        static=exact_window.static;del exact_window
        print(f"calibration paired labels {trial+1}/{original['accept_trials']}",flush=True)
    if not sanity_actions:checks.append(dict(kind='single_local_vs_full',passed=False,reason='disabled'))
    for action in actions:
        source=action['source']
        action['condition_samples']={}
        for name,storage in collected.items():
            row=average(storage[source]);row['true_rd_delta']=row['true_image_delta']+action['measured_rate_delta']
            action[name]=row
            action['condition_samples'][name]=[dict(r,true_rd_delta=r['true_image_delta']+action['measured_rate_delta'])
                                                for r in storage[source]]
    eval_args=SimpleNamespace(**dict(original,trials=original['accept_trials'],lpips=False,
        allocation_image_objective=objective))
    measurements={}
    def measure(tiers):
        key=tier_hash(tiers)
        if key not in measurements:
            row=evaluate_map(model,scene,tiers,calibration,eval_args,label='diagnostic')
            cost=meter.details(tiers)
            row.update(cost)
            row['normalized_distortion']=row['source_distortion']/reference['d_ref']
            row['rd_score']=row['normalized_distortion']+original['beta']*cost['total_channel_uses']/reference['c_ref']
            measurements[key]=row
        return measurements[key]
    baseline=measure(q)
    delta=abs(baseline['rd_score']-state['calibration']['rd_score'])
    checks.append(dict(kind='source_calibration_replay',abs_error=delta,tolerance=1e-5,passed=delta<=1e-5))
    groups=[];by_source={a['source']:a for a in actions}
    proposal_lookup={int(s):(int(t),float(g)) for s,t,g in zip(proposal['ids'],proposal['tiers'],proposal['gains'])}
    def add_group(name,ids,tiers):
        if not len(ids):return
        alt=q.clone();alt[ids]=tiers;row=measure(alt)
        rd=row['rd_score']-baseline['rd_score']
        group=dict(name=name,sources=ids.tolist(),tiers=tiers.tolist(),true_rd_delta=rd,
            true_image_delta=row['normalized_distortion']-baseline['normalized_distortion'],
            channel_use_delta=row['total_channel_uses']-baseline['total_channel_uses'])
        if all(int(s) in proposal_lookup and proposal_lookup[int(s)][0]==int(t) for s,t in zip(ids,tiers)):
            group['predicted_selection_rd_delta']=-sum(proposal_lookup[int(s)][1] for s in ids)
        elif all(int(s) in by_source for s in ids):
            group['predicted_selection_rd_delta']=-sum(by_source[int(s)]['selection_predicted_gain'] for s in ids)
        if all(int(s) in by_source for s in ids):
            selected=[by_source[int(s)] for s in ids]
            summed=sum(a['calibration_accept']['true_rd_delta'] for a in selected)
            group.update(summed_single_rd_delta=summed,
                image_interaction_delta=group['true_image_delta']-sum(a['calibration_accept']['true_image_delta'] for a in selected),
                rate_nonadditivity_delta=original['beta']*group['channel_use_delta']/reference['c_ref']-sum(a['measured_rate_delta'] for a in selected))
        groups.append(group)
        print(f"group {name}: changes={len(ids)}, actual RD delta={rd:+.7g}",flush=True)
    for fraction,name in ((1.,'full_proposal'),(.5,'half_prefix'),(.25,'quarter_prefix')):
        size=int(np.ceil(len(proposal['ids'])*fraction))
        add_group(name,proposal['ids'][:size],proposal['tiers'][:size])
    for name in ('top','middle','bottom','ordinary'):
        chosen=[a for a in actions if a['stratum']==name]
        add_group(name+'_sample',torch.tensor([a['source'] for a in chosen],dtype=torch.long),
                  torch.tensor([a['new_tier'] for a in chosen],dtype=torch.long))
        if name=='top':
            for part in range(min(4,len(chosen))):
                subset=chosen[part::4]
                add_group(f'top_disjoint_{part}',torch.tensor([a['source'] for a in subset]),
                          torch.tensor([a['new_tier'] for a in subset]))
    unchanged=file_digest(path)==source_sha
    codec_ok=model_id(model)==state['codec_id']
    network_ok=direct.tensor_digest(learner.network.state_dict())==network_hash
    checks.append(dict(kind='frozen_source_and_networks',passed=unchanged and codec_ok and network_ok))
    report=summarize(actions,groups,checks,tie=args.action_tie,requested_per_stratum=args.per_stratum)
    report.update(status='complete',version=FAMILY,source_step=state['progress']['step'],
        source_unchanged=unchanged,frozen_codec_verified=codec_ok,frozen_benefit_verified=network_ok,
        source_sha256=source_sha,sampled_actions=len(actions),proposal_count=len(proposal['ids']),
        source_baseline=state['calibration']['rd_score'],recomputed_baseline=baseline['rd_score'],
        calibration_views=[str(c.image_name) for c in calibration],
        noise_seeds=[original['seed']+1000+i for i in range(original['accept_trials'])],
        elapsed_seconds=time.perf_counter()-started)
    json_write(out/'actions.json',actions);json_write(out/'groups.json',groups);json_write(out/'summary.json',report)
    flat=[]
    for action in actions:
        row={k:v for k,v in action.items() if not isinstance(v,(dict,list))}
        for section in ('matched','calibration_same_noise','calibration_accept','decomposition'):
            row.update({section+'_'+k:v for k,v in action[section].items()})
        flat.append(row)
    with (out/'actions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    with tarfile.open(out/'review.tar.gz','w:gz') as archive:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.suffix in ('.json','.csv','.npz'):archive.add(p,arcname=p.name,recursive=False)
    print(json.dumps(dict(status='complete',source_step=report['source_step'],samples=len(actions),
        consistency=report['execution_consistency_passed'],evidence=report['evidence_status'],focus=report['recommended_focus'],
        review_bundle=str(out/'review.tar.gz')),ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':main(parser().parse_args())
