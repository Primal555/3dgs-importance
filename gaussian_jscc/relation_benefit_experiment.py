"""Frozen-codec four-context relation values and one RD scene selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time

import numpy as np
import torch

from .allocation_distortion import ImageDistortion,calibrate
from .budgeted_allocation_evaluation import ExactDeploymentCost,evaluate_map,tier_hash
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_state import load_checkpoint,upsert_jsonl
from .cli import device_for,seed_all
from .data import read_ply
from .dense_rd_allocation import atomic_save
from .model_factory import build_codec
from .modular_checkpoint import capture_rng,restore_rng
from .modular_experiment import json_write
from .relation_benefit_features import prepare_bank
from .relation_benefit_context import background_configurations
from .relation_benefit_learning import RelationReplay,RelationLearner,marginal_cost_proxy,choose_tiers
from .relation_benefit_model import RelationBenefitNetwork
from .relation_benefit_teacher import RelationContextTeacher
from .render_objective import spaced_indices
from .successive_multiscene_inputs import make_scene
from .transport import model_id

FAMILY='relation_benefit_context_v2'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','scene','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--manifest');p.add_argument('--device',default='cuda')
    p.add_argument('--beta',type=float,default=.1);p.add_argument('--snr',type=float,default=10.)
    for name,value in (('window-views',4),('calibration-views',4),('test-views',32),
        ('trials',4),('image-views',2),('cpu-threads',4),('seed',42),('validate-every',50),('save-every',50)):
        p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.set_defaults(budget_total_uses=None,budget_ratio_to_q3=None)
    for name,value in (('windows',700),('fit-windows',50),('label-groups',64),
        ('updates-per-window',64),('batch-groups',8),('teacher-trials',2),
        ('width',128),('attention-groups',8),('spatial-neighbors',16),('projection-neighbors',16),
        ('feature-batch',8192),('inference-batch',4096),('validation-groups',512),
        ('replay-capacity',8192),('anchor-capacity',2048),('normalizer-samples',8192)):
        p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--lambda-s',type=float)
    p.add_argument('--structure-share',type=float,default=.2)
    p.add_argument('--phase',choices=('auto','train','predict'),default='auto')
    return p


def validate(args):
    if args.beta is None or not math.isfinite(args.beta) or args.beta<0 or args.snr!=10.:
        raise ValueError('fixed beta>=0 and SNR10 required')
    if args.budget_total_uses is not None or args.budget_ratio_to_q3 is not None:
        raise ValueError('relation experiment uses fixed beta')
    positive=('windows','fit_windows','label_groups','updates_per_window','batch_groups','teacher_trials',
        'width','attention_groups','spatial_neighbors','projection_neighbors','feature_batch',
        'inference_batch','validation_groups','replay_capacity','anchor_capacity','normalizer_samples',
        'window_views','calibration_views','trials','cpu_threads','validate_every','save_every','image_views')
    if min(getattr(args,k) for k in positive)<1 or args.windows<=args.fit_windows:
        raise ValueError('positive budgets and windows>fit_windows required')
    if args.anchor_capacity>=args.replay_capacity or args.width%args.attention_groups:
        raise ValueError('anchor_capacity<replay_capacity and divisible attention width required')
    if (not math.isfinite(args.learning_rate) or args.learning_rate<=0 or
        not math.isfinite(args.weight_decay) or args.weight_decay<0 or not 0<args.structure_share<1 or
        (args.lambda_s is not None and (not math.isfinite(args.lambda_s) or args.lambda_s<0))):
        raise ValueError('invalid optimizer or distortion calibration')
    if args.phase=='predict' and not args.resume:raise ValueError('predict requires --resume')


def unified_cost(meter,q):
    legacy=meter.details(q);comparison=legacy['comparison_accounting'];streams=comparison['streams']
    return dict(payload_complex_symbols=legacy['payload_complex_symbols'],xyz_bytes=legacy['xyz_bytes'],
        metadata_bytes=legacy['metadata_bytes'],scale_stream_bytes=legacy['scale_stream_bytes'],
        coordinate_channel_uses=streams['coordinates']['total_channel_uses'],
        metadata_channel_uses=streams['metadata']['total_channel_uses'],
        scale_channel_uses=streams['scales']['total_channel_uses'],
        total_channel_uses=comparison['total_channel_uses'],comparison_accounting=comparison,
        cost_protocol='digital_side_file_v1')


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _settings(args):
    return {k:v for k,v in vars(args).items() if k not in
        ('out','resume','dry_run','device','phase','windows','allocation_image_objective')}


def _truncate(path,step):
    if path.is_file():
        lines=[s for s in path.read_text(encoding='utf-8').splitlines() if s.strip() and json.loads(s)['step']<=step]
        path.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')


def _archive(out):
    temporary=out/'review.tar.gz.tmp'
    marker=json.loads((out/'pretrain_complete.json').read_text(encoding='utf-8')) if (out/'pretrain_complete.json').exists() else {}
    complete=json.loads((out/'complete.json').read_text(encoding='utf-8')) if (out/'complete.json').exists() else {}
    current_step=marker.get('step',complete.get('step',0))
    deployment_current=complete.get('status')=='complete' and complete.get('step')==current_step
    current=f"window{current_step:06d}"
    with tarfile.open(temporary,'w:gz') as archive:
        for name in ('plan.json','training.json','preflight.json','preprocess.json','pretrain_complete.json',
                     'complete.json','review_summary.json'):
            if name in ('complete.json','review_summary.json') and not deployment_current:continue
            if (out/name).is_file():archive.add(out/name,arcname=name)
        for folder in ('metrics','evaluation','deployment'):
            if folder in ('evaluation','deployment') and not deployment_current:continue
            for path in sorted((out/folder).rglob('*')):
                if folder=='evaluation' and any(p.startswith('window') and p!=current for p in path.relative_to(out/folder).parts):
                    continue
                if path.is_file():archive.add(path,arcname=path.relative_to(out).as_posix())
    temporary.replace(out/'review.tar.gz')


def _invalidate_deployment(out,step):
    """Keep a named old generation while making active completion markers truthful."""
    names=['complete.json','review_summary.json','pretrain_complete.json','review.tar.gz',
           'checkpoints/final/state.pt','deployment/selected_tiers.npy','deployment/inference.json']
    names += [f'evaluation/{label}/metrics.json' for label in
              ('selected','uniform_q1','uniform_q2','uniform_q3')]
    for name in names:
        old=out/name
        if old.is_file():
            folder=out/f'history/window{step:06d}';folder.mkdir(parents=True,exist_ok=True)
            filename=('final_state.pt' if name.endswith('state.pt') else
                      name.split('/')[1]+'_metrics.json' if name.startswith('evaluation/') else old.name)
            destination=folder/filename
            serial=0
            while destination.exists():
                serial+=1;destination=folder/f'{old.stem}_{serial}{old.suffix}'
            old.replace(destination)


def run_loaded(args,model,scene,source_plan,*,max_windows=None,runtime_output_created=False):
    """Shared runtime entry; tests supply a real small codec and CPU rasterizer."""
    validate(args);out=Path(args.out).resolve();device=next(model.parameters()).device
    torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    model.eval().requires_grad_(False);model.set_scene_geometry(scene.geometry)
    if tuple(model.cfg.rates)!=(0,8,24,48):raise ValueError('requires source rates 0/8/24/48')
    codec_id=model_id(model)
    plan=dict(source_plan,version=FAMILY,settings=_settings(args),codec_id=codec_id,
              source_fingerprint=scene.fingerprint,codec_config=model.cfg.to_dict())
    plan=json.loads(json.dumps(plan))
    if out.exists() and not args.resume and not runtime_output_created:
        raise FileExistsError('new output or --resume required')
    latest=out/'checkpoints/latest/state.pt'
    if args.resume and not latest.is_file():raise FileNotFoundError('missing relation checkpoint')
    if args.resume and json.loads((out/'plan.json').read_text(encoding='utf-8'))!=plan:
        raise ValueError('relation source/settings mismatch')
    out.mkdir(parents=True,exist_ok=True)
    json_write(out/'plan.json',plan)
    names=lambda cameras:[str(c.image_name) for c in cameras]
    if len(scene.cameras)<args.window_views or len(scene.val_cameras)<args.calibration_views:
        raise ValueError('separate preparation and calibration cameras required')
    preparation=[scene.cameras[i] for i in spaced_indices(len(scene.cameras),args.window_views)]
    calibration=[scene.val_cameras[i] for i in spaced_indices(len(scene.val_cameras),args.calibration_views)]
    if (set(names(preparation))&set(names(calibration)) or
        set(names(preparation)+names(calibration))&set(names(scene.test_cameras))):
        raise ValueError('test, calibration and preparation views must be disjoint')
    meter=ExactDeploymentCost(model,scene,codec_id,args.snr,2.)
    q3=torch.full((len(scene.raw),),3,dtype=torch.long)
    state=load_checkpoint(latest) if args.resume else None
    if state and args.windows>state['progress']['window'] and args.phase!='predict':
        _invalidate_deployment(out,state['progress']['window'])
    if state:
        if state['plan']!=plan or args.windows<state['progress']['window']:
            raise ValueError('checkpoint identity mismatch or shrinking target')
        reference=state['reference']
    else:
        calibration_args=argparse.Namespace(**vars(args));calibration_args.lpips=False
        calibration_args.allocation_image_objective=ImageDistortion('mse_ssim',0.)
        baseline=evaluate_map(model,scene,q3,calibration,calibration_args,
                              folder=out/'calibration/q3',label='q3_reference')
        cost=unified_cost(meter,q3)
        reference=dict(**calibrate(baseline['source_objective_mse'],baseline['source_structure_error'],
                    share=args.structure_share,lambda_s=args.lambda_s),
            c_ref=cost['total_channel_uses'],q3_cost=cost,beta=args.beta,
            preparation_views=names(preparation),calibration_views=names(calibration),
            teacher_background='four_fixed_contexts_v1',value_units='delta_distortion_over_d_ref')
        reference['cost_proxy']=marginal_cost_proxy(cost,len(scene.raw))
    args.allocation_image_objective=ImageDistortion('mse_ssim',reference['lambda_s'])
    contexts=background_configurations(len(scene.raw),args.seed)
    teacher=RelationContextTeacher(model,scene,preparation,reference,contexts,snr=args.snr,trials=args.teacher_trials,
                            seed=args.seed+700001,candidate_batch=args.label_groups)
    teacher_identity=teacher.identity()
    if state and state['teacher_identity']!=teacher_identity:raise ValueError('teacher identity changed')
    if device.type=='cuda':
        print('Checking paired codec blocks and exact local rendering on CUDA...',flush=True)
        preflight=teacher.verify()
    else:
        preflight=dict(status='cpu_mathematical_execution',cuda_verified=False)
    json_write(out/'preflight.json',preflight)
    bank_path=out/'cache/bank.pt'
    if state:
        if not bank_path.is_file() or _sha(bank_path)!=state['bank_sha256']:
            raise ValueError('relation input cache missing or changed')
        bank=load_checkpoint(bank_path)
    else:
        print('Preparing immutable full-tier attributes and relation graphs...',flush=True)
        started=time.perf_counter();bank=prepare_bank(model,scene,preparation,args)
        atomic_save(bank_path,bank)
        json_write(out/'preprocess.json',dict(bank['metadata'],seconds=time.perf_counter()-started))
    bank_sha=state['bank_sha256'] if state else _sha(bank_path)
    n=len(scene.raw)
    if n<3:raise ValueError('need at least three points for disjoint labels')
    permutation=torch.randperm(n,generator=torch.Generator().manual_seed(args.seed+4021))
    val_count=min(max(1,args.validation_groups//4),max(1,n//8))
    validation_ids=permutation[:val_count];training_ids=permutation[val_count:]
    replay=RelationReplay(args.replay_capacity,args.anchor_capacity,validation_ids.tolist(),args.seed+211)
    network=RelationBenefitNetwork(bank['node'].shape[-1],bank['candidate'].shape[-1],
        bank['view_node'].shape[-1],args.width,args.attention_groups,
        context_node_dim=bank['contexts'][0]['node'].shape[-1],
        context_view_dim=bank['contexts'][0]['view_node'].shape[-1]).to(device)
    learner=RelationLearner(network,replay,bank,lr=args.learning_rate,weight_decay=args.weight_decay)
    if state:
        if not torch.equal(state['training_ids'],training_ids) or not torch.equal(state['validation_ids'],validation_ids):
            raise ValueError('point-label split changed')
        learner.load_state_dict(state['learner']);replay.load_state_dict(state['replay'])
        validation=state['validation'];progress=state['progress'];fit_labels=state['fit_labels']
        for file in ('loss.jsonl','benefit_validation.jsonl'):_truncate(out/'metrics'/file,progress['window'])
        restore_rng(state['rng'])
    else:
        network.fit_normalizers(bank,training_ids[:args.normalizer_samples])
        validation=[]
        progress=dict(window=0,label_cursor=[c*max(1,len(training_ids)//4) for c in range(4)]);fit_labels=[]
    rates=torch.tensor(reference['cost_proxy']['costs'][1:],dtype=torch.float64)*args.beta/reference['c_ref']
    json_write(out/'training.json',dict(plan,**scene.record,spec=scene.spec,objective_reference=reference,
        teacher=teacher_identity,preparation_view_names=names(preparation),
        training_label_count=len(training_ids),validation_label_ids=validation_ids.tolist(),
        network_parameters=sum(p.numel() for p in network.parameters()),bank_sha256=bank_sha,
        deployment='four fixed context graph passes; equal signed-value mean; one RD argmin',
        context_names=[c['name'] for c in contexts],context_weights=[.25]*4,
        reference_target='source_ply',test_seed=args.seed+800003))

    def save(label='latest'):
        if model_id(model)!=codec_id:raise RuntimeError('frozen codec changed')
        snapshot=dict(version=1,plan=plan,reference=reference,teacher_identity=teacher_identity,
            learner=learner.state_dict(),replay=replay.state_dict(),validation=validation,
            training_ids=training_ids,validation_ids=validation_ids,progress=dict(progress),
            fit_labels=fit_labels,bank_sha256=bank_sha,rng=capture_rng())
        atomic_save(out/f'checkpoints/{label}/state.pt',snapshot)

    if not state:save()
    if not validation:
        print(f'Preparing {4*len(validation_ids)} fixed validation label groups across four contexts...',flush=True)
        validation=[row for context in range(4) for row in teacher.collect(validation_ids.tolist(),step=0,context=context)]
        save()
    consumed=0
    if args.phase=='predict' and progress['window']<args.windows:
        raise ValueError('finish the requested training windows before prediction')
    while progress['window']<args.windows and args.phase!='predict':
        started=time.perf_counter();step=progress['window']+1
        context=(step-1)%4
        count=min(args.label_groups,len(training_ids));cursor=progress['label_cursor'][context]
        ids=training_ids[(torch.arange(count)+cursor)%len(training_ids)]
        labels=teacher.collect(ids.tolist(),step=step,context=context);replay.add(labels)
        stats=dict(updates=learner.updates,loss=None)
        if step<=args.fit_windows:fit_labels.extend(labels)
        if step==args.fit_windows:
            learner.fit_scales(fit_labels)
            fit_labels=[]
        if step>args.fit_windows:
            stats=learner.train_updates(args.updates_per_window,args.batch_groups,rates)
        cursors=list(progress['label_cursor']);cursors[context]=cursor+count
        progress=dict(window=step,label_cursor=cursors);consumed+=1
        record=dict(stats,step=step,event='training',labels=count,coverage=replay.coverage(),
                    teacher_cached_groups=teacher.cached_groups,context=context,context_name=contexts[context]['name'],
                    step_seconds=time.perf_counter()-started)
        upsert_jsonl(out/'metrics/loss.jsonl',record)
        if learner.scales is not None and (step%args.validate_every==0 or step==args.windows):
            report=learner.evaluate(validation,rates)
            upsert_jsonl(out/'metrics/benefit_validation.jsonl',dict(report,step=step,event='validation',network_updates=learner.updates,
                scope='fixed held-out points; preparation views; four equally weighted paired-noise contexts'))
            print(f"  validation {step}: positive={report['positive']['regret_mean']:.6g}, "
                  f"four-state={report['four_state_regret']:.6g}",flush=True)
        if step==1 or step%10==0:
            print(f"relation {step}/{args.windows} [{contexts[context]['name']}]: labels={count}, updates={learner.updates}, "
                  f"loss={stats['loss']}, sec={record['step_seconds']:.2f}",flush=True)
        if step%args.save_every==0 or step==args.windows or (max_windows is not None and consumed>=max_windows):save()
        if max_windows is not None and consumed>=max_windows:
            return dict(status='interrupted',step=step)
    save()
    json_write(out/'pretrain_complete.json',dict(status='pretrain_complete',step=progress['window'],
        network_updates=learner.updates,frozen_codec_verified=model_id(model)==codec_id))
    save('final')
    if args.phase=='train':
        _archive(out);return dict(status='pretrain_complete',step=progress['window'])
    # Teacher captures and noise buffers are no longer needed by deployment.
    del teacher
    if device.type=='cuda':torch.cuda.empty_cache()
    print('Predicting four context-conditioned scene values, then averaging for one RD selection...',flush=True)
    started=time.perf_counter();values=network.predict_all(bank,args.inference_batch)
    q=choose_tiers(values,reference['cost_proxy']['costs'],beta=args.beta,c_ref=reference['c_ref'])
    inference_seconds=time.perf_counter()-started
    (out/'deployment').mkdir(exist_ok=True);np.save(out/'deployment/selected_tiers.npy',q.numpy().astype(np.uint8))
    json_write(out/'deployment/inference.json',dict(seconds=inference_seconds,
        profile=network.last_inference_profile,tier_counts=torch.bincount(q,minlength=4).tolist(),
        cost_proxy=reference['cost_proxy'],q0_anchor=0.,value_units='delta_distortion_over_d_ref',
        selection='minimum predicted distortion plus explicit marginal communication cost'))
    evaluation_args=argparse.Namespace(**vars(args));evaluation_args.seed=args.seed+800003
    rows={}
    for label,tiers in [('selected',q)]+[(f'uniform_q{k}',torch.full_like(q,k)) for k in (1,2,3)]:
        folder=out/f'evaluation/window{progress["window"]:06d}/{label}'
        metrics=evaluate_map(model,scene,tiers,scene.test_cameras,evaluation_args,save_images=label=='selected',
            folder=folder,label=label)
        json_write(out/f'evaluation/{label}/metrics.json',json.loads((folder/'metrics.json').read_text(encoding='utf-8')))
        cost=unified_cost(meter,tiers)
        rows[label]=dict(metrics,**cost,step=progress['window'],beta=args.beta,d_ref=reference['d_ref'],
            c_ref=reference['c_ref'],rd_score=metrics['source_distortion']/reference['d_ref']+
                args.beta*cost['total_channel_uses']/reference['c_ref'])
    q2=rows['uniform_q2'];selected=rows['selected']
    against=dict(psnr_difference=selected['source_psnr']-q2['source_psnr'],
        cost_saving=1-selected['total_channel_uses']/q2['total_channel_uses'],
        rd_difference=selected['rd_score']-q2['rd_score'],
        meets_q2_goal=selected['source_psnr']>=q2['source_psnr'] and selected['total_channel_uses']<q2['total_channel_uses'])
    result=dict(status='complete',family=FAMILY,step=progress['window'],network_updates=learner.updates,
        codec_id=codec_id,frozen_codec_verified=model_id(model)==codec_id,
        value_units='delta_distortion_over_d_ref',tier_sha256=tier_hash(q),
        total_channel_uses=selected['total_channel_uses'],source_psnr=selected['source_psnr'],
        network_parameters=sum(p.numel() for p in network.parameters()),against_q2=against)
    json_write(out/'review_summary.json',dict(result=result,rows=rows,against_q2=against))
    json_write(out/'complete.json',result);save('final');_archive(out)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main(args,max_windows=None):
    validate(args)
    saved,cfg,spec,old,options,source=load_inputs(args)
    plan=dict(version=FAMILY,source=source,spec=spec)
    if args.dry_run:
        result=dict(plan,settings=_settings(args),windows=args.windows,writes_output=False)
        print(json.dumps(result,ensure_ascii=False,indent=2));return result
    device=device_for(args.device);seed_all(args.seed)
    model=build_codec(cfg).to(device);model.load_state_dict(saved['state_dict']);model.eval().requires_grad_(False)
    model.set_training_snrs([10.])
    raw,degree=read_ply(spec['ply'])
    # make_scene creates scene metadata below out, so protect new output first.
    out=Path(args.out).resolve()
    if out.exists() and not args.resume:raise FileExistsError('new output or --resume required')
    scene=make_scene(spec,raw,degree,model,options,out/'scene_inputs',old=old)
    return run_loaded(args,model,scene,plan,max_windows=max_windows,runtime_output_created=True)


if __name__=='__main__':main(parser().parse_args())
