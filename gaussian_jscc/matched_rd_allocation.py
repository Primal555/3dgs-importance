"""Frozen-codec allocation with matched rescoring and cached exact sub-batches."""
import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .allocation_distortion import ImageDistortion
from .budgeted_allocation_evaluation import ExactDeploymentCost,evaluate_map,tier_hash
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_state import repair,upsert_jsonl
from .calibrated_spatial_benefit_allocation import batched_preflight
from .cli import device_for,seed_all
from .data import read_ply
from .direct_rd_allocation import (FrozenBenefit,FrozenCandidateScheduler,NoReplay,
                                   source_benefit,tensor_digest,refresh_side)
from .direct_rd_decisions import HardTierMap,ProposalBank
from .matched_rd_decisions import matched_audit
from .matched_rd_window import MatchedWindow
from .model_factory import build_codec
from .modular_checkpoint import capture_rng,restore_rng
from .modular_experiment import json_write
from .optimization import preserved_rng
from .render_objective import spaced_indices
from .successive_multiscene_inputs import make_scene
from .transport import model_id

FAMILY='matched_frozen_benefit_direct_rd_v2'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','benefit-run','scene','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--benefit-checkpoint');p.add_argument('--manifest');p.add_argument('--device',default='cuda')
    p.add_argument('--beta',type=float,default=.1);p.add_argument('--snr',type=float,default=10.)
    p.add_argument('--min-rd-gain',type=float,default=1e-6)
    for name,default in (('allocation-steps',2000),('allocation-batch',128),('audit-every',10),
                         ('accept-trials',2),('calibration-views',4),('rescore-groups',512),
                         ('exact-groups',64),('audit-chunk',16),('max-local-checks',12),
                         ('full-verify-every',100),('test-views',32),('trials',4),('image-views',2),
                         ('save-every',10),('cpu-threads',4),('seed',42)):
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    for name in ('resume','dry-run'):p.add_argument('--'+name,action='store_true')
    # The common input loader reserves separate screening views only when positive.
    p.set_defaults(window_views=0)
    return p


def validate(args):
    positive=('allocation_steps','allocation_batch','audit_every','accept_trials','calibration_views',
              'rescore_groups','exact_groups','audit_chunk','max_local_checks','full_verify_every',
              'trials','image_views','save_every','cpu_threads')
    if any(getattr(args,n)<1 for n in positive):raise ValueError('positive allocation budgets required')
    if args.exact_groups>args.rescore_groups:raise ValueError('exact groups exceed rescore budget')
    if args.test_views<0 or args.snr!=10.:raise ValueError('invalid views/SNR')
    if any(not math.isfinite(v) or v<0 for v in (args.beta,args.min_rd_gain)):
        raise ValueError('invalid beta or RD threshold')
    run,benefit,out=(Path(getattr(args,k)).resolve() for k in ('run','benefit_run','out'))
    if any(out.is_relative_to(src) or src.is_relative_to(out) for src in (run,benefit)):
        raise ValueError('source and output must be separate')


def main(args,max_updates=None):
    validate(args)
    saved,cfg,spec,old,options,source=load_inputs(args)
    if tuple(cfg.rates)!=(0,8,24,48):raise ValueError('requires 0/8/24/48')
    out=Path(args.out).resolve();latest=out/'checkpoints/latest/state.pt'
    if not args.dry_run and out.exists() and not args.resume:raise FileExistsError('use NEW output or --resume')
    state=torch.load(latest,map_location='cpu',weights_only=True) if args.resume else None
    settings={k:v for k,v in vars(args).items() if k not in
              ('out','resume','dry_run','device','allocation_batch','allocation_image_objective')}
    plan=json.loads(json.dumps(dict(version=FAMILY,source=source,spec=spec,settings=settings)))
    if state is not None and (state.get('family')!=FAMILY or state['plan']!=plan):
        raise ValueError('matched RD resume source/settings mismatch')
    frozen=state['frozen'] if state is not None else source_benefit(args,saved['state_dict'],source)
    if args.dry_run:
        result=dict(plan,frozen_benefit=frozen['descriptor'],writes_output=False)
        print(json.dumps(result,indent=2));return result
    device=device_for(args.device);torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    model=build_codec(cfg).to(device);model.load_state_dict(saved['state_dict']);del saved
    model.eval().requires_grad_(False);model.set_training_snrs([10.]);codec_id=model_id(model)
    raw,degree=read_ply(spec['ply'])
    scene=make_scene(spec,raw,degree,model,options,out,old=old);del raw
    model.set_scene_geometry(scene.geometry)
    if any(frozen['scene_record'].get(k)!=v for k,v in scene.record.items()):
        raise ValueError('benefit scene geometry/grouping/camera mismatch')
    chosen=set(spaced_indices(len(scene.cameras),args.calibration_views))
    cameras=[c for i,c in enumerate(scene.cameras) if i in chosen]
    names=[str(c.image_name) for c in cameras]
    test_names={str(c.image_name) for c in scene.test_cameras}
    if test_names.intersection(names):raise ValueError('optimization/test views overlap')
    seeds=[args.seed+1000+i for i in range(args.accept_trials)]
    scene_record=dict(scene.record,calibration_view_names=names,optimization_view_names=names,
                      condition_seeds=seeds)
    if state is not None and (state['scene_record']!=scene_record or state['codec_id']!=codec_id):
        raise ValueError('matched RD resume scene/codec mismatch')
    learner=FrozenBenefit(frozen['inference'],device)
    reference=copy.deepcopy(state['reference'] if state is not None else frozen['reference'])
    args.allocation_image_objective=ImageDistortion('mse_ssim' if reference['lambda_s'] else 'mse',reference['lambda_s'])
    window_args=SimpleNamespace(**dict(frozen['window_settings'],allocation_batch=args.allocation_batch,
        label_groups=0,beta=args.beta,snr=args.snr,cpu_threads=args.cpu_threads,
        renderer_backend=device.type,accept_trials=args.accept_trials))
    q=state['q'].long() if state is not None else torch.full((len(scene.raw),),2,dtype=torch.long)
    table=HardTierMap(q,device);scheduler=FrozenCandidateScheduler(scene.group_ids,seed=args.seed)
    bank=ProposalBank(len(q));meter=ExactDeploymentCost(model,scene,codec_id,args.snr,options.net_bits_per_use)
    window=MatchedWindow(model,scene,table,scheduler,reference,window_args)
    blocks=torch.empty(len(q),dtype=torch.long);offset=0
    for ids in scene.group_ids:
        rows,cols=(ids>=0).nonzero(as_tuple=True)
        blocks[ids[rows,cols].cpu()]=rows.cpu()+offset;offset+=len(ids)
    progress=dict(step=0,initial_done=False,final_done=False,accepted_batches=0,accepted_changes=0)
    baseline=None
    if state is not None:
        progress=state['progress'];baseline=state['calibration']
        scheduler.load_state_dict(state['scheduler']);bank.load_state_dict(state['bank'])
        repair(out,progress['step']);restore_rng(state['rng'])
    out.mkdir(parents=True,exist_ok=True);json_write(out/'plan.json',plan)
    json_write(out/'training.json',dict(plan,**scene_record,objective_reference=reference,
        benefit_source=frozen['descriptor'],initial_map='uniform_q2',deployment='direct_integer_map',
        codec_frozen=True,benefit_frozen=True,
        budgets=dict(rescore=args.rescore_groups,exact=args.exact_groups,chunk=args.audit_chunk,checks=args.max_local_checks)))
    with preserved_rng(device):preflight=batched_preflight(model,scene,cameras[0],window_args,reference)
    json_write(out/'preflight.json',preflight)

    def measure(tiers,label,views,*,images=False,acceptance=False):
        eval_args=copy.copy(args)
        if acceptance:eval_args.trials=args.accept_trials;eval_args.lpips=False
        result=evaluate_map(model,scene,tiers,views,eval_args,save_images=images,
            folder=out/'evaluation'/label if not acceptance else None,label=label)
        cost=meter.details(tiers)
        return dict(result,**cost,normalized_distortion=result['source_distortion']/reference['d_ref'],
            normalized_cost=cost['total_channel_uses']/reference['c_ref'],
            rd_score=result['source_distortion']/reference['d_ref']+args.beta*cost['total_channel_uses']/reference['c_ref'],
            beta=args.beta,d_ref=reference['d_ref'],c_ref=reference['c_ref'])

    def record(label,row):upsert_jsonl(out/'metrics/validation.jsonl',dict(row,step=progress['step'],event=label))

    def save():
        checkpoint=dict(family=FAMILY,plan=plan,frozen=frozen,codec_id=codec_id,scene_record=scene_record,
            reference=reference,q=q.to(torch.uint8),progress=dict(progress),calibration=baseline,
            bank=bank.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng())
        latest.parent.mkdir(parents=True,exist_ok=True)
        temporary=latest.with_suffix('.pt.tmp');torch.save(checkpoint,temporary);temporary.replace(latest)

    if not progress['initial_done']:
        initial=measure(q,'initial',scene.test_cameras,images=True)
        record('initial',initial);record('uniform_q2',initial)
        baseline=measure(q,'acceptance',cameras,acceptance=True)
        refresh_side(reference,baseline,q,options.net_bits_per_use)
        progress['initial_done']=True;save()
    updates=0
    while progress['step']<args.allocation_steps:
        step=progress['step']+1
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter();window.set_condition(seeds[(step-1)%len(seeds)])
        result=window.collect(step,cameras,learner,NoReplay(),False)
        if not torch.equal(result['q'].cpu(),q):raise RuntimeError('candidate background differs from deployment')
        bank.add(result['tier_ids'],result['tier_costs']);progress['step']=step
        audit_seconds=0.;audit_row=None
        if step%args.audit_every==0 or step==args.allocation_steps:
            audit_start=time.perf_counter();before=baseline['rd_score']
            audited=matched_audit(window,bank,q,blocks,learner,seeds,cameras,meter,baseline,args)
            q,baseline=audited['q'],audited['measurement']
            if audited['accepted']:
                progress['accepted_batches']+=len(audited['commits'])
                progress['accepted_changes']+=audited['changed']
                table.set_map(q);refresh_side(reference,baseline,q,options.net_bits_per_use)
                window.clear_contexts()
            bank.clear();audit_seconds=time.perf_counter()-audit_start
            audit_row={k:v for k,v in audited.items() if k not in ('q','measurement')}
            audit_row.update(step=step,event='audit',before_rd=before,after_rd=baseline['rd_score'],
                total_channel_uses=baseline['total_channel_uses'],source_psnr=baseline['source_psnr'],
                tier_counts=baseline['tier_counts'],audit_seconds=audit_seconds)
            upsert_jsonl(out/'metrics/audits.jsonl',audit_row)
            print(f"matched audit {step}: rescore={audited['rescored']}, exact={audited['exact_groups']}, "
                  f"checks={len(audited['attempts'])}, changed={audited['changed']}, "
                  f"RD={before:.7f}->{baseline['rd_score']:.7f}",flush=True)
        verify_seconds=0.
        if step%args.full_verify_every==0 or step==args.allocation_steps:
            verify_start=time.perf_counter();full=measure(q,'acceptance',cameras,acceptance=True)
            error=abs(full['rd_score']-baseline['rd_score'])
            upsert_jsonl(out/'metrics/full_checks.jsonl',dict(step=step,event='native_check',
                absolute_rd_error=error,passed=error<=1e-5,local_rd=baseline['rd_score'],full_rd=full['rd_score']))
            if error>1e-5:raise RuntimeError('joint tile RD differs from full-render RD')
            # Preserve the same arithmetic state across a mid-cycle restart.
            verify_seconds=time.perf_counter()-verify_start
        if device.type=='cuda':torch.cuda.synchronize(device)
        row=dict(result['metrics'],step=step,event='update',phase='matched_direct_allocation',
            step_seconds=time.perf_counter()-started,audit_seconds=audit_seconds,full_verify_seconds=verify_seconds,
            allocation_batch=args.allocation_batch,context_cache_hits=window.cache_hits,
            context_cache_misses=window.cache_misses,benefit_updates=learner.updates,
            accepted_batches=progress['accepted_batches'],tier_counts=torch.bincount(q,minlength=4).tolist(),
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None)
        upsert_jsonl(out/'metrics/loss.jsonl',row)
        if step%args.save_every==0 or step%args.audit_every==0 or step==args.allocation_steps:save()
        if step==1 or step%10==0:
            print(f"matched {step}/{args.allocation_steps}: evaluated={row['evaluated_groups']}, "
                  f"tiers={row['tier_counts']}, sec={row['step_seconds']:.2f}",flush=True)
        updates+=1
        if max_updates is not None and updates>=max_updates:
            save();return dict(status='interrupted',step=step)
    final=measure(q,'latest',scene.test_cameras,images=True);record('latest',final)
    record('uniform_q3',measure(torch.full_like(q,3),'uniform_q3',scene.test_cameras))
    codec_ok=model_id(model)==codec_id
    network_ok=tensor_digest(learner.network.state_dict())==frozen['descriptor']['network_sha256']
    if not codec_ok or not network_ok:raise RuntimeError('frozen network changed')
    deployment=out/'deployment';deployment.mkdir(exist_ok=True)
    np.save(deployment/'latest_tiers.npy',q.numpy().astype(np.uint8),allow_pickle=False)
    progress['final_done']=True;save()
    completion=dict(status='complete',family=FAMILY,**progress,tier_sha256=tier_hash(q),
        frozen_codec_verified=codec_ok,frozen_benefit_verified=network_ok,
        benefit_source=frozen['descriptor'],calibration_rd=baseline['rd_score'])
    json_write(out/'complete.json',completion)
    from .direct_rd_artifacts import summarize
    summarize(out);print(json.dumps(completion),flush=True)
    return completion


if __name__=='__main__':main(parser().parse_args())
