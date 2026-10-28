"""Short server audit of calibrated allocation costs, with no fitting or exports."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from .allocation import scene_fingerprint
from .allocation_distortion import ComponentRenderTask,ImageDistortion
from .allocation_mirror import ViewWindowScheduler
from .allocation_update_inspection import UpdateInspector,restore_history
from .budgeted_allocation_kernel import budgeted_mask_step
from .budgeted_allocation_evaluation import ExactDeploymentCost
from .calibrated_evidence import CalibratedEvidence
from .cli import device_for,seed_all
from .data import read_ply
from .model_factory import build_codec,config_from_dict
from .modular_experiment import json_write
from .render_validation import append_json
from .structured_evidence import BoundedMirrorTable
from .successive_multiscene_inputs import make_scene
from .transport import model_id

FAMILY='allocation_update_audit_v1'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='Completed calibrated allocation run')
    p.add_argument('--out',required=True)
    p.add_argument('--checkpoint',choices=('best_rd','latest','final'),default='best_rd')
    p.add_argument('--windows',type=int,default=20)
    p.add_argument('--samples',type=int,default=32,help='Maximum counterfactual renders per window')
    p.add_argument('--seed',type=int,default=31415)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--dry-run',action='store_true')
    return p


def summarize_checks(rows):
    entry={'actions':len(rows),'stages':{}}
    stages={name for r in rows for name in r['checks']}
    for name in sorted(stages):
        checks=[r['checks'][name] for r in rows if name in r['checks']]
        informative=[c for c in checks if c['sign_correct'] is not None]
        entry['stages'][name]=dict(actions=len(checks),informative=len(informative),
            sign_errors=sum(c['sign_correct'] is False for c in informative),
            sign_error_fraction=sum(c['sign_correct'] is False for c in informative)/len(informative) if informative else None,
            mean_absolute_error=sum(c['absolute_error'] for c in checks)/len(checks),
            mean_regret=sum(c['regret'] for c in checks)/len(checks))
    return entry


def summarize(records):
    result={}
    for branch in ('positive','keep_mixture','keep_sampled'):
        rows=[r for r in records if r['branch']==branch]
        if not rows:continue
        entry=summarize_checks(rows)
        movements=[r['update_direction_correct'] for r in rows if r['update_direction_correct'] is not None]
        entry['hypothetical_update']=dict(informative=len(movements),wrong_directions=sum(x is False for x in movements),
            mean_absolute_log_odds_change=sum(abs(r['delta_log_odds']) for r in rows)/len(rows))
        result[branch]=entry
    sampled=[dict(checks={k:v for k,v in r['sampled_reference'].items()
        if k in ('raw_current','corrected_current')}) for r in records if 'sampled_reference' in r]
    if sampled:
        # These two predictors target the sampled on/off action, independently of history readiness.
        result['keep_sampled_reference']=summarize_checks(sampled)
    return result


def main(args):
    if args.windows<1 or args.samples<3 or args.cpu_threads<1:
        raise ValueError('positive windows/threads and at least three counterfactual renders required')
    run,out=Path(args.run).resolve(),Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):raise ValueError('audit output must be separate from source')
    if out.exists():raise FileExistsError('use a NEW audit output')
    complete=json.loads((run/'complete.json').read_text(encoding='utf-8'))
    if complete.get('status')!='complete':raise ValueError('completed allocation run required')
    path=run/'checkpoints'/args.checkpoint/'state.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('family')!='calibrated_structured_allocation_v2':raise ValueError('calibrated-v2 checkpoint required')
    record=saved['scene_record'];settings=saved['plan']['settings'];reference=saved['objective_reference']
    if settings['snr']!=10.:
        raise ValueError('source calibrated allocation requires fixed physical and conditioning 10dB')
    train_names=record['allocation_train_view_names'];window=settings['window_views']
    if (set(train_names)&set(record['calibration_view_names']) or set(train_names)&set(record['test_view_names'])
            or not window<=len(train_names) or settings['beta']!=reference['beta']):
        raise ValueError('source training views or objective mismatch')
    plan=dict(version=FAMILY,source_run=str(run),checkpoint=str(path),source_step=saved['progress']['step'],
        beta=reference['beta'],d_ref=reference['d_ref'],c_ref=reference['c_ref'],lambda_s=reference['lambda_s'],
        windows=args.windows,window_views=window,max_counterfactual_renders=args.samples,seed=args.seed,
        training_views=train_names,codec_frozen=True,probabilities_frozen=True,
        physical_snr=settings['snr'],conditioning_snr=10.,
        predictive_history='restore checkpoint statistics after each window; audit coverage scheduler may advance',
        rate_scope='same marginal side-cost proxy as training; no per-action packet compression',
        no_current_exact_labels_in_predictions=True)
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return plan
    device=device_for(args.device);torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    cfg=config_from_dict(saved['config']);model=build_codec(cfg).to(device)
    model.load_state_dict(saved['state_dict']);model.eval().requires_grad_(False);model.set_training_snrs([10.])
    spec=saved['plan']['spec'];raw,degree=read_ply(spec['ply'])
    if scene_fingerprint(raw)!=record['fingerprint']:raise ValueError('source PLY fingerprint mismatch')
    old={k:v for k,v in record.items() if k not in ('allocation_train_view_names','calibration_view_names')}
    options=SimpleNamespace(**{k:record[k] for k in ('blocks_per_batch','resolution','images','white_background','views_per_step')},
        net_bits_per_use=saved['plan']['source']['net_bits_per_use'],validation_views=len(record['validation_view_names']),
        test_views=len(record['test_view_names']))
    scene=make_scene(spec,raw,degree,model,options,out,old=old);del raw
    model.set_scene_geometry(scene.geometry);identity=model_id(model)
    if identity!=complete['codec_id']:raise ValueError('frozen codec identity differs from allocation run')
    lookup={str(c.image_name):c for c in scene.cameras}
    cameras=[lookup[n] for n in train_names]
    table=BoundedMirrorTable(len(scene.raw),max_log_odds=settings['max_log_odds']).to(device)
    table.load_state_dict(saved['table'])
    blocks=CalibratedEvidence(scene.group_ids,rounds=settings['candidate_rounds'],policy=settings['candidate_policy'],
        memory=settings['evidence_memory'],half_life=settings['evidence_half_life'],context_tv=settings['context_tv'],
        finite_samples=settings['finite_samples'] if settings['finite_feedback'] else 0)
    blocks.configure_views(len(cameras),window);blocks.load_state_dict(saved['blocks'])
    blocks.finite_samples=0
    views=ViewWindowScheduler(train_names,args.seed)
    inspector=UpdateInspector(args.samples,settings['keep_lr'],settings['tier_lr'])
    image_objective=ImageDistortion(settings['distortion'],reference['lambda_s'])
    if 'side_proxy' not in reference:
        # A best checkpoint selected at step zero precedes the first proxy refresh.
        cost=ExactDeploymentCost(model,scene,identity,settings['snr'],options.net_bits_per_use).details(saved['latest_q'])
        kept=int((saved['latest_q']>0).sum());bits=options.net_bits_per_use
        reference=dict(reference,side_proxy=dict(
            xyz_per_retained=cost['coordinate_channel_uses']/kept if kept else reference['q3_cost']['coordinate_channel_uses']/len(scene.raw),
            scale_per_retained=cost['scale_content_bits']/bits/kept if kept else 0.,
            fixed_uses=cost['metadata_channel_uses']+cost['scale_framing_bytes']*8/bits,bits_per_use=bits))
        plan['side_proxy_source']='measured once from saved map; step-zero checkpoint had no refresh'
    else:plan['side_proxy_source']='saved checkpoint'
    with path.open('rb') as f:plan['source_checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    plan['codec_id']=identity;json_write(out/'plan.json',plan)
    records=[];started=time.perf_counter()
    for visit in range(1,args.windows+1):
        indices=views.next(window);blocks.set_view_window(indices)
        # Give a virtual observation a fresh timestamp without changing saved history.
        step=saved['progress']['step']+visit;seed_all(args.seed+1000+visit)
        task=ComponentRenderTask([cameras[i] for i in indices],scene.reference,scene.degree,
            scene.white_background,distortion=image_objective)
        begin=time.perf_counter()
        stats=budgeted_mask_step(model,table,scene.groups,scene.group_ids,scene.geometry,task,blocks,
            settings['snr'],'awgn',d_ref=reference['d_ref'],price=reference['beta'],
            rate_normalizer=reference['c_ref'],side_proxy=dict(reference['side_proxy'],step=step),
            timings=None,audit_limit=0,inspector=inspector)
        for row in inspector.records:
            row.update(window=visit,history_step=saved['progress']['step'],virtual_step=step)
            append_json(out/'actions.jsonl',row)
        records.extend(inspector.records)
        restore_history(blocks,saved['blocks'])
        if device.type=='cuda':torch.cuda.synchronize(device)
        row=dict(window=visit,views=[train_names[i] for i in indices],
            sampled_tier_counts=stats['sampled_tier_counts'],candidate_points=len(stats['tier_ids']),
            counterfactual_renders=inspector.render_calls,seconds=time.perf_counter()-begin)
        append_json(out/'windows.jsonl',row)
        current=summarize(inspector.records)
        pos=current.get('positive',{}).get('stages',{})
        print(f'audit {visit}/{args.windows}: beta={reference["beta"]}, renders={inspector.render_calls}, '
            f'sec={row["seconds"]:.2f}, positive regret raw/history='
            f'{pos.get("raw_current",{}).get("mean_regret",0):.4g}/'
            f'{pos.get("history_update",{}).get("mean_regret",0):.4g}',flush=True)
        del stats,task
    codec_unchanged=model_id(model)==identity
    table_unchanged=all(torch.equal(v.cpu(),saved['table'][k]) for k,v in table.state_dict().items())
    if not codec_unchanged or not table_unchanged:raise RuntimeError('audit changed frozen model/probabilities')
    # restore_history has already checked the sparse banks and restored every predictive tensor.
    summary=dict(plan,groups=summarize(records),wall_seconds=time.perf_counter()-started,
        assessment_scope='selected decision-risk checks; not an unbiased scene-wide error rate',
        keep_reference='history uses matched conditional mixture when ready, otherwise sampled distortion plus expected rate; '
            'raw/corrected sampled references remain separate')
    json_write(out/'summary.json',summary)
    result=dict(status='complete',windows=args.windows,beta=reference['beta'],codec_unchanged=codec_unchanged,
        probabilities_unchanged=table_unchanged,predictive_history_unchanged=True,actions=len(records),
        wall_seconds=summary['wall_seconds'])
    json_write(out/'complete.json',result);return result


if __name__=='__main__':main(parser().parse_args())
