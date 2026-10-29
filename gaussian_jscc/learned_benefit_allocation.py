"""Train a full-benefit evaluator, then explicitly resume scene allocation with it."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .allocation_benefit import BenefitNetwork
from .allocation_benefit_features import FeatureProtocol
from .allocation_benefit_learning import BenefitLearner
from .allocation_benefit_replay import cpu_copy
from .allocation_benefit_report import (action_log, summarize_records, validate_network,
                                       write_benefit_report, write_timing_report)
from .allocation_benefit_schedule import BenefitScheduler
from .allocation_benefit_window import BenefitWindow
from .allocation_distortion import ImageDistortion, ComponentRenderTask, calibrate
from .allocation_mirror import MirrorTierTable, ViewWindowScheduler
from .allocation_objective import FixedRateDistortion
from .allocation_reuse import AllocationTimings
from .budgeted_allocation import parser as base_parser, validate as base_validate, deploy
from .budgeted_allocation_evaluation import ExactDeploymentCost, evaluate_map, tier_hash
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_kernel import budgeted_mask_step
from .budgeted_allocation_state import save_state, restore_state, repair, upsert_jsonl
from .cli import device_for, seed_all
from .data import read_ply
from .model_factory import build_codec
from .modular_checkpoint import restore_rng
from .modular_experiment import json_write
from .render_objective import spaced_indices
from .render_validation import append_json
from .structured_evidence import BoundedMirrorTable
from .successive_multiscene_inputs import make_scene
from .transport import model_id


FAMILY = 'learned_benefit_allocation_v1'


def parser():
    p=base_parser();p.description=__doc__;p.set_defaults(audit_every=0,audit_samples=0)
    p.add_argument('--phase',choices=('pretrain','allocation'),default='pretrain')
    p.add_argument('--benefit-variant',choices=('legacy','factored','response','response_visible'),default='legacy')
    for name,default in (('pretrain-windows',200),('fit-windows',50),('benefit-steps',4),
                         ('benefit-batch-size',128),('benefit-validate-every',50),
                         ('replay-capacity',65536),('chunk-size',8192),('finite-samples',32)):
        p.add_argument('--'+name,type=int,default=default)
    for name,default in (('benefit-lr',1e-4),('mask-fraction',.25),('mask-weight',.5),
                         ('structure-share',.2),('max-log-odds',.25)):
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--distortion',choices=('mse','mse_ssim'),default='mse_ssim')
    p.add_argument('--lambda-s',type=float)
    return p


def validate(args):
    base_validate(args)
    if args.beta is None:raise ValueError('learned allocation requires fixed --beta')
    if args.phase=='allocation' and not args.resume:raise ValueError('allocation needs completed pretraining and --resume')
    if args.finite_samples<4:raise ValueError('exact budget needs at least 4 renders for both supervised branches')
    if args.replay_capacity<2:raise ValueError('replay capacity must retain at least two action groups')
    if (args.fit_windows<1 or args.pretrain_windows<=args.fit_windows or
            min(args.benefit_steps,args.benefit_validate_every,args.chunk_size,args.replay_capacity)<1 or
            args.benefit_batch_size<2 or args.finite_samples<3 or
            not math.isfinite(args.benefit_lr) or args.benefit_lr<=0 or
            not 0<=args.mask_fraction<=1 or not math.isfinite(args.mask_weight) or args.mask_weight<0 or
            not 0<args.structure_share<1 or not math.isfinite(args.max_log_odds) or args.max_log_odds<=0):
        raise ValueError('invalid learned benefit training configuration')
    if args.lambda_s is not None and (not math.isfinite(args.lambda_s) or args.lambda_s<0):
        raise ValueError('invalid structure weight')
    if args.distortion=='mse' and args.lambda_s not in (None,0.):raise ValueError('MSE requires zero structure weight')


def main(args,max_updates=None):
    response=args.benefit_variant in ('response','response_visible')
    if response:
        args.mask_fraction=0.;args.mask_weight=0.
    validate(args)
    saved,cfg,spec,old,options,source=load_inputs(args)
    if tuple(cfg.rates)!=(0,8,24,48):raise ValueError('this experiment retains 0/8/24/48 source rates')
    settings={k:v for k,v in vars(args).items() if k not in
              ('out','resume','dry_run','device','phase','allocation_image_objective')}
    for key in ('budget_total_uses','budget_ratio_to_q3','initial_price','dual_lr','ema_decay'):settings.pop(key,None)
    family=FAMILY;Network,Learner=BenefitNetwork,BenefitLearner
    Protocol,Scheduler,Window,Table=FeatureProtocol,BenefitScheduler,BenefitWindow,BoundedMirrorTable
    assess,summarize_actions,write_report=validate_network,summarize_records,write_benefit_report
    if args.benefit_variant=='factored':
        from .factored_benefit import FAMILY as family,FactoredBenefitNetwork,DecisionBenefitLearner
        Network,Learner=FactoredBenefitNetwork,DecisionBenefitLearner
    elif response:
        from .response_value import FAMILY as family,ResponseValueNetwork,ResponseValueLearner,ResponseProtocol,ResponseMirrorTable
        from .response_value_window import ResponseScheduler,ResponseValueWindow
        from .response_value_report import validate_response,summarize_response,write_response_report
        Network,Learner,Protocol=ResponseValueNetwork,ResponseValueLearner,ResponseProtocol
        Scheduler,Window=ResponseScheduler,ResponseValueWindow
        Table=ResponseMirrorTable
        assess,summarize_actions,write_report=validate_response,summarize_response,write_response_report
        if args.benefit_variant=='response_visible':
            from .visible_response import FAMILY as family,VisibleResponseNetwork,VisibleResponseLearner,VisibleResponseProtocol
            from .response_value_window import VisibleResponseWindow
            Network,Learner=VisibleResponseNetwork,VisibleResponseLearner
            Protocol,Window=VisibleResponseProtocol,VisibleResponseWindow
    else:settings.pop('benefit_variant',None)
    protocol=Protocol(len(saved['state_dict']['attr_mean']),cfg.latent_dim)
    plan=json.loads(json.dumps(dict(version=family,source=source,spec=spec,settings=settings,
        codec_config=cfg.to_dict(),feature_protocol=protocol.identity(),writes_output=False)))
    if args.dry_run:
        print(json.dumps(plan,indent=2));return plan
    out=Path(args.out).resolve();latest=out/'checkpoints/latest/state.pt'
    if out.exists() and not args.resume:raise FileExistsError('use NEW output or --resume')
    if args.resume and not latest.exists():raise FileNotFoundError('missing exact benefit checkpoint')
    device=device_for(args.device);torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    model=build_codec(cfg).to(device);model.load_state_dict(saved['state_dict']);del saved
    model.eval().requires_grad_(False);model.set_training_snrs([10.]);identity=model_id(model)
    raw,degree=read_ply(spec['ply']);scene=make_scene(spec,raw,degree,model,options,out,old=old);del raw
    model.set_scene_geometry(scene.geometry)
    chosen=set(spaced_indices(len(scene.cameras),args.calibration_views))
    calibration=[c for i,c in enumerate(scene.cameras) if i in chosen]
    training=[c for i,c in enumerate(scene.cameras) if i not in chosen]
    scene_record=dict(scene.record,allocation_train_view_names=[str(c.image_name) for c in training],
                      calibration_view_names=[str(c.image_name) for c in calibration])
    table=Table(len(scene.raw),max_log_odds=args.max_log_odds).to(device)
    blocks=Scheduler(scene.group_ids,seed=args.seed)
    views=ViewWindowScheduler(scene_record['allocation_train_view_names'],args.seed+9001)
    static=torch.empty(len(scene.raw),3+protocol.attr_dim)
    for f,ids in zip(scene.groups,scene.group_ids):static[ids[ids>=0]]=f[ids>=0]
    network=Network(protocol.point_dim,protocol.action_dim,view_dim=getattr(protocol,'view_dim',12),continuous=protocol.continuous,
                           mse_only=args.distortion=='mse').to(device)
    learner=Learner(network,static_source=static,capacity=args.replay_capacity,seed=args.seed,
        lr=args.benefit_lr,fit_windows=args.fit_windows,updates_per_window=args.benefit_steps,
        batch_size=args.benefit_batch_size,mask_fraction=args.mask_fraction,mask_weight=args.mask_weight)
    meter=ExactDeploymentCost(model,scene,identity,args.snr,options.net_bits_per_use)
    if args.resume:
        state=restore_state(latest,plan,scene_record)
        model.load_state_dict(state['state_dict']);table.load_state_dict(state['table'])
        blocks.load_state_dict(state['blocks']);views.load_state_dict(state['views']);learner.load_state_dict(state['learner'])
        reference=state['objective_reference'];progress=state['progress'];q=state['latest_q']
        repair(out,progress['step']);restore_rng(state['rng'])
        del state
    else:
        full=torch.full((len(scene.raw),),3,dtype=torch.long);full_cost=meter.details(full)
        base_args=argparse.Namespace(**vars(args));base_args.lpips=False
        base_args.allocation_image_objective=ImageDistortion(args.distortion,0.)
        baseline=evaluate_map(model,scene,full,calibration,base_args,folder=out/'evaluation/q3_reference',label='q3_reference')
        reference=dict(q3_cost=full_cost,beta=args.beta,c_ref=full_cost['total_channel_uses'],
            calibration_views=scene_record['calibration_view_names'],calibration_trials=args.trials,
            **calibrate(baseline['source_objective_mse'],baseline['source_structure_error'],
                        mode=args.distortion,share=args.structure_share,lambda_s=args.lambda_s))
        q=deploy(table,args)
        progress=dict(step=0,phase='pretrain',pretrain_windows_done=0,allocation_step=0,pretrain_complete=False,
            evaluator_selected=False,evaluator_best=None,best=None,pending=['initial'],
            initial_codec_id=identity,frozen_codec_verified=False)
        json_write(out/'plan.json',plan)
        json_write(out/'training.json',dict(plan,**scene_record,objective_reference=reference,
            source_run=str(Path(args.run).resolve()),trainable_modules=['benefit_network'],
            probability_optimizer='bounded log-domain mirror descent',attribute_statistics_frozen=True,
            allocation_training_sampling='single',deployment_draws=10,deployment_seed=args.seed,
            probability_global_gain=len(scene.raw),allocation_joint_steps=0))
    args.allocation_image_objective=ImageDistortion(args.distortion,reference['lambda_s'])
    objective=FixedRateDistortion(args.beta,reference['d_ref'],reference['c_ref'],'source_distortion')

    def objective_fields(distortion=None,cost=None):
        fields=dict(objective_mode='fixed_beta',beta=args.beta,d_ref=reference['d_ref'],c_ref=reference['c_ref'],
                    distortion_field='source_distortion',distortion_mode=args.distortion,lambda_s=reference['lambda_s'])
        if distortion is not None and cost is not None:
            fields.update(normalized_distortion=distortion/reference['d_ref'],normalized_cost=cost/reference['c_ref'],
                          rd_score=objective.score(distortion,cost))
        return fields

    def save(label='latest'):
        if model_id(model)!=identity:raise RuntimeError('frozen codec changed')
        state=save_state(out,label,model,table,None,blocks,views,progress,plan,scene_record,reference,q,learner=learner)
        json_write(out/'progress.json',dict(status='running',step=progress['step'],phase=progress['phase'],
            pretrain_windows_done=progress['pretrain_windows_done'],allocation_step=progress['allocation_step'],
            network_updates=learner.updates,pending=progress['pending'],**objective_fields()))
        return state

    def measure(label,cameras,images=False,map_q=None):
        chosen_q=q if map_q is None else map_q;started=time.perf_counter()
        values=evaluate_map(model,scene,chosen_q,cameras,args,save_images=images,
                            folder=out/'evaluation'/label,label=label)
        cost=meter.details(chosen_q)
        row=dict(values,**cost,step=progress['step'],allocation_step=progress['allocation_step'],event=label,
            evaluation_seconds=time.perf_counter()-started,**objective_fields(values['source_distortion'],cost['total_channel_uses']))
        upsert_jsonl(out/'metrics/validation.jsonl',row);return row

    def refresh_side(cost):
        kept=int((q>0).sum());fallback=reference['q3_cost']['coordinate_channel_uses']/len(q)
        reference['side_proxy']=dict(xyz_per_retained=cost['coordinate_channel_uses']/kept if kept else fallback,
            scale_per_retained=cost['scale_content_bits']/options.net_bits_per_use/kept if kept else 0.,
            fixed_uses=cost['metadata_channel_uses']+cost['scale_framing_bytes']*8/options.net_bits_per_use,
            bits_per_use=options.net_bits_per_use)

    def consider(row):
        candidate=dict(step=progress['step'],allocation_step=progress['allocation_step'],source_mse=row['source_mse'],
            source_distortion=row['source_distortion'],total_channel_uses=row['total_channel_uses'],
            rd_score=row['rd_score'],tier_sha256=row['tier_sha256'],codec_id=identity)
        if progress['best'] is None or objective.candidate_key(candidate)<objective.candidate_key(progress['best']):
            progress['best']=candidate;save('best_rd')

    def validate_benefit():
        if not bool(network.ready):return
        report=assess(learner,progress['step'])
        upsert_jsonl(out/'metrics/benefit_validation.jsonl',report);write_report(out,report)
        score=report['selection_score'];best=progress['evaluator_best']
        if progress['phase']=='pretrain' and score is not None and (best is None or score<best['score']):
            progress['evaluator_best']=dict(step=progress['step'],score=score,
                network=cpu_copy(network.state_dict()),optimizer=cpu_copy(learner.optimizer.state_dict()))
        stages=report['positive']['stages']
        if response:
            print(f'  value validation: positive regret={stages["network"]["regret_mean"]}; '
                  f'keep regret={report["keep"]["stages"]["network"]["regret_mean"]}; score={score}',flush=True)
        else:
            print(f'  benefit validation: positive regret={stages["first_order"]["regret_mean"]}'
                  f'->{stages["network"]["regret_mean"]}; both_improved={report["both_branches_lower_regret"]}',flush=True)

    def pending():
        nonlocal q
        while progress['pending']:
            event=progress['pending'][0];q=deploy(table,args)
            if event=='initial':
                refresh_side(measure('calibration_000000',calibration));measure('initial',scene.test_cameras,True)
            elif event=='allocation_initial':
                row=measure(f'calibration_{progress["step"]:06d}',calibration);refresh_side(row);consider(row)
            elif event=='control':
                started=time.perf_counter();cost=meter.details(q);refresh_side(cost)
                upsert_jsonl(out/'metrics/control.jsonl',dict(cost,step=progress['step'],event='cost_refresh',
                    allocation_step=progress['allocation_step'],cost_seconds=time.perf_counter()-started,
                    tier_counts=torch.bincount(q,minlength=4).tolist(),**objective_fields()))
            elif event=='calibration':consider(measure(f'calibration_{progress["step"]:06d}',calibration))
            elif event=='benefit_validation':validate_benefit()
            else:raise RuntimeError('unknown benefit pending event')
            progress['pending'].pop(0);save()

    if not args.resume:save('initial');save()
    if args.phase=='pretrain' and progress['phase']=='allocation':
        raise ValueError('resume allocation with --phase allocation')
    if args.phase=='allocation' and not progress['pretrain_complete']:
        raise ValueError('finish pretraining before starting allocation')
    pending()
    if args.phase=='allocation' and progress['phase']=='pretrain':
        progress['phase']='allocation';progress['pending']=['allocation_initial'];save();pending()
    updates=0
    target=args.pretrain_windows if args.phase=='pretrain' else args.allocation_steps
    counter='pretrain_windows_done' if args.phase=='pretrain' else 'allocation_step'
    while progress[counter]<target:
        step=progress['step']+1;indices=views.next(args.window_views)
        # Explicit window seed also makes the exact labels reproducible from their metadata.
        window_seed=args.seed+2000003+step;seed_all(window_seed)
        timings=AllocationTimings(device,bool(args.profile_every and (step==1 or step%args.profile_every==0)))
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        task=ComponentRenderTask([training[i] for i in indices],scene.reference,scene.degree,scene.white_background,
                                 distortion=args.allocation_image_objective)
        window=Window(learner,blocks,finite_samples=args.finite_samples,chunk_size=args.chunk_size)
        result=budgeted_mask_step(model,table,scene.groups,scene.group_ids,scene.geometry,task,blocks,args.snr,'awgn',
            d_ref=reference['d_ref'],price=args.beta,rate_normalizer=reference['c_ref'],
            side_proxy=dict(reference['side_proxy'],step=step,window_seed=window_seed),timings=timings,benefit=window)
        with timings.measure('mirror_update_seconds'):
            if args.phase=='allocation':
                table.update(result['keep_costs'],result['tier_ids'],result['tier_costs'],args.keep_lr,args.tier_lr,len(scene.raw))
        records=window.records
        for key in ('keep_costs','tier_ids','tier_costs','keep_image_costs','audit'):result.pop(key)
        sampled=result.pop('q');del window,task
        with timings.measure('benefit_training_seconds'):
            learning=learner.observe(records,step)
        if device.type=='cuda':torch.cuda.synchronize(device)
        row=dict(result,**learning,**timings.summary(),step=step,event='update',phase=args.phase,
            phase_step=progress[counter]+1,
            allocation_step=progress['allocation_step']+int(args.phase=='allocation'),
            codec_updated=False,**blocks.coverage(sampled),
            **objective_fields(result['render_loss'],result['expected_cost_proxy']),
            loss=objective.score(result['render_loss'],result['expected_cost_proxy']),
            training_view_names=[str(training[i].image_name) for i in indices],window_seed=window_seed,
            step_seconds=time.perf_counter()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None)
        append_json(out/'metrics/loss.jsonl',row)
        ready_records=[]
        for i,record in enumerate(records):
            record['network_ready']=row['benefit_network_ready']
            append_json(out/'metrics/benefit_actions.jsonl',dict(action_log(record),event=str(i)))
            if record['network_ready']:ready_records.append(record)
        if ready_records:
            upsert_jsonl(out/'metrics/benefit_prequential.jsonl',dict(summarize_actions(ready_records),step=step,event='pre_label_prediction'))
        progress['step']=step;progress[counter]+=1;progress['pending']=[]
        if step%args.benefit_validate_every==0 or progress[counter]==target:progress['pending'].append('benefit_validation')
        if args.phase=='allocation':
            if progress[counter]%args.control_every==0:progress['pending'].append('control')
            if progress[counter]%args.validate_every==0 or progress[counter]==target:progress['pending'].append('calibration')
        q=deploy(table,args)
        if progress['pending'] or step%args.save_every==0:save()
        pending()
        if step==1 or progress[counter]%10==0 or progress[counter]==target:
            print(f'{args.phase} {progress[counter]}/{target}: MSE={row["source_mse"]:.6g}, beta={args.beta:g}, '
                  f'sampled={row["sampled_tier_counts"]}, exact={row["benefit_exact_render_calls"]}, '
                  f'network_updates={learner.updates}, sec={row["step_seconds"]:.2f}',flush=True)
        updates+=1
        if max_updates is not None and updates>=max_updates:
            save();return dict(status='interrupted',step=step,phase=args.phase)
    if args.phase=='pretrain':
        best=progress['evaluator_best']
        if not progress['evaluator_selected']:
            if best is None:raise RuntimeError('pretraining produced no complete heldout evaluator assessment')
            network.load_state_dict(best['network']);learner.optimizer.load_state_dict(best['optimizer'])
            progress['evaluator_selected']=True;progress['selected_evaluator_window']=best['step']
        progress['pretrain_complete']=True;progress['frozen_codec_verified']=model_id(model)==identity
        report=assess(learner,progress['step']);write_report(out,report)
        result=dict(status='pretrain_complete',phase='pretrain',step=progress['step'],
            network_updates=learner.updates,selected_evaluator_window=progress['selected_evaluator_window'],
            allocation_started=False,frozen_codec_verified=progress['frozen_codec_verified'])
        if response:result['value_validation_score']=report['selection_score']
        else:result['both_branches_lower_regret']=report['both_branches_lower_regret']
        json_write(out/'comparison/benefit_pretrain.json',dict(result,validation=report));save()
        json_write(out/'pretrain_complete.json',result);json_write(out/'progress.json',result)
        from .budgeted_allocation_report import read_rows
        write_timing_report(out,read_rows(out/'metrics/loss.jsonl'))
        print(json.dumps(result),flush=True);return result
    selected=torch.load(out/'checkpoints/best_rd/state.pt',map_location='cpu',weights_only=True)
    selected={key:selected[key] for key in ('table','progress','latest_q')}
    chosen_q=selected['latest_q'];selected_row=measure('selected',scene.test_cameras,True,map_q=chosen_q)
    if torch.equal(q,chosen_q):
        latest_row=dict(selected_row,event='latest',evaluation_seconds=0.,reused_from='selected')
        upsert_jsonl(out/'metrics/validation.jsonl',latest_row)
        json_write(out/'evaluation/latest/alias.json',dict(metrics_and_images_from='../selected',tier_sha256=tier_hash(q)))
    else:latest_row=measure('latest',scene.test_cameras,True)
    for tier in (2,3):measure(f'uniform_q{tier}',scene.test_cameras,map_q=torch.full_like(q,tier))
    export=out/'deployment';export.mkdir(exist_ok=True)
    np.save(export/'selected_tiers.npy',chosen_q.numpy().astype(np.uint8),allow_pickle=False)
    np.save(export/'latest_tiers.npy',q.numpy().astype(np.uint8),allow_pickle=False)
    chosen_table=MirrorTierTable(len(scene.raw));chosen_table.load_state_dict(selected['table'])
    np.save(export/'selected_probabilities.npy',chosen_table.probabilities(torch.arange(len(q)),args.snr).numpy(),allow_pickle=False)
    progress['frozen_codec_verified']=model_id(model)==identity
    result=dict(status='complete',family=family,phase='allocation',step=progress['step'],allocation_step=progress['allocation_step'],
        selection='best_rd',selected_step=selected['progress']['step'],codec_id=identity,tier_sha256=tier_hash(chosen_q),
        selected_total_channel_uses=selected_row['total_channel_uses'],latest_total_channel_uses=latest_row['total_channel_uses'],
        selected_source_psnr=selected_row['source_psnr'],selection_rd_score=selected['progress']['best']['rd_score'],
        selection_rd_scope='calibration views only; final rd_score uses test views',
        frozen_codec_verified=progress['frozen_codec_verified'],
        **objective_fields(selected_row['source_distortion'],selected_row['total_channel_uses']))
    json_write(export/'selection.json',result);save('final');save()
    from .budgeted_allocation_report import summarize,read_rows
    summarize(out,result);write_timing_report(out,read_rows(out/'metrics/loss.jsonl'))
    json_write(out/'complete.json',result);json_write(out/'progress.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':main(parser().parse_args())
