"""Frozen-codec allocation with an explicit budget OR a fixed RD coefficient."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .allocation import deployment_tiers
from .allocation_budget import BudgetController
from .allocation_objective import FixedRateDistortion
from .allocation_mirror import MirrorTierTable, BlockVisitScheduler, ViewWindowScheduler
from .allocation_reuse import AllocationTimings
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_kernel import budgeted_mask_step
from .budgeted_allocation_evaluation import ExactDeploymentCost, evaluate_map, tier_hash
from .budgeted_allocation_state import save_state, restore_state, repair, upsert_jsonl
from .cli import seed_all, device_for
from .data import read_ply
from .model_factory import build_codec
from .modular_checkpoint import restore_rng
from .modular_experiment import json_write
from .optimization import preserved_rng
from .render_objective import MaskedMultiViewRenderTask, spaced_indices
from .render_validation import append_json
from .successive_multiscene_inputs import make_scene
from .transport import model_id

FAMILY = 'budgeted_allocation_v1'
RD_FAMILY = 'fixed_beta_allocation_v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run','scene','out'): p.add_argument('--'+name, required=True)
    p.add_argument('--manifest'); p.add_argument('--device', default='cuda')
    budget = p.add_mutually_exclusive_group(required=True)
    budget.add_argument('--budget-total-uses', type=int)
    budget.add_argument('--budget-ratio-to-q3', type=float)
    budget.add_argument('--beta', type=float, help='Fixed RD coefficient; no budget or adaptive price')
    for name, default in (('allocation-steps',2000),('window-views',4),('calibration-views',4),
                         ('control-every',25),('validate-every',250),('save-every',100),('profile-every',10),
                         ('audit-every',250),('audit-samples',4),('test-views',32),('trials',4),
                         ('image-views',2),('cpu-threads',4),('seed',42)):
        p.add_argument('--'+name, type=int, default=default)
    for name, default in (('initial-price',.01),('dual-lr',.05),('ema-decay',.9),
                         ('keep-lr',.1),('tier-lr',.1),('snr',10.)):
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--dry-run', action='store_true')
    return p


def validate(args):
    required = ('allocation_steps','window_views','calibration_views','control_every','validate_every',
                'save_every','trials','image_views','cpu_threads')
    if min(getattr(args,k) for k in required) < 1 or min(args.audit_every,args.audit_samples,args.profile_every,args.test_views) < 0:
        raise ValueError('invalid update/window/evaluation budgets')
    if args.snr != 10.: raise ValueError('this controlled experiment retains fixed 10dB')
    beta = getattr(args, 'beta', None)
    if sum(v is not None for v in (args.budget_total_uses, args.budget_ratio_to_q3, beta)) != 1:
        raise ValueError('supply exactly one explicit budget or beta')
    for value in (args.budget_total_uses, args.budget_ratio_to_q3):
        if value is not None and (not math.isfinite(value) or value <= 0): raise ValueError('invalid explicit budget')
    if beta is not None:
        FixedRateDistortion(beta, 1., 1.)
        if (args.initial_price, args.dual_lr, args.ema_decay) != (.01, .05, .9):
            raise ValueError('adaptive price options do not apply to fixed beta')
    else:
        BudgetController(1, args.initial_price, args.dual_lr, args.ema_decay)
    if any(not math.isfinite(v) or v <= 0 for v in (args.keep_lr,args.tier_lr)):
        raise ValueError('invalid mirror learning rates')


@torch.no_grad()
def deploy(table, args):
    return deployment_tiers(table.probabilities(torch.arange(len(table.logits),device=table.logits.device),args.snr), args.seed)


def main(args, max_updates=None):
    validate(args)
    fixed_rd = getattr(args, 'beta', None) is not None
    evidence = getattr(args, 'evidence_experiment', False)
    structured = getattr(args, 'structured_evidence', False)
    if evidence:
        from .evidence_allocation import validate as validate_evidence
        validate_evidence(args)
    family = 'evidence_allocation_v1' if evidence else (RD_FAMILY if fixed_rd else FAMILY)
    if structured:
        from .structured_evidence_allocation import validate as validate_structured, FAMILY as structured_family
        validate_structured(args); family = structured_family
    saved, cfg, spec, old, options, source = load_inputs(args)
    settings = {k:v for k,v in vars(args).items() if k not in ('out','resume','dry_run','device','allocation_image_objective')}
    # Keep old budget checkpoint plan identity byte-for-byte compatible.
    if not fixed_rd: settings.pop('beta', None)
    else:
        for key in ('budget_total_uses','budget_ratio_to_q3','initial_price','dual_lr','ema_decay'):
            settings.pop(key, None)
    plan = json.loads(json.dumps(dict(version=family, source=source, spec=spec, settings=settings,
                                     codec_config=cfg.to_dict(), writes_output=False)))
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return plan
    out = Path(args.out).resolve(); latest = out/'checkpoints/latest/state.pt'
    if out.exists() and not args.resume: raise FileExistsError('use NEW output or --resume')
    if args.resume and not latest.exists(): raise FileNotFoundError('missing exact allocation checkpoint')
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    model = build_codec(cfg).to(device); model.load_state_dict(saved['state_dict']); del saved
    model.eval().requires_grad_(False); model.set_training_snrs([10.]); identity = model_id(model)
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, options, out, old=old); del raw
    model.set_scene_geometry(scene.geometry)
    chosen = set(spaced_indices(len(scene.cameras), args.calibration_views))
    calibration = [c for i,c in enumerate(scene.cameras) if i in chosen]
    training = [c for i,c in enumerate(scene.cameras) if i not in chosen]
    scene_record = dict(scene.record, allocation_train_view_names=[str(c.image_name) for c in training],
                         calibration_view_names=[str(c.image_name) for c in calibration])
    if structured:
        from .structured_evidence import BoundedMirrorTable
        table = BoundedMirrorTable(len(scene.raw), max_log_odds=args.max_log_odds).to(device)
    else:
        table = MirrorTierTable(len(scene.raw)).to(device)
    if evidence:
        from .allocation_evidence import EvidenceScheduler
        if structured:
            from .calibrated_evidence import CalibratedEvidence as EvidenceScheduler
        blocks = EvidenceScheduler(scene.group_ids,rounds=args.candidate_rounds,policy=args.candidate_policy,
            memory=args.evidence_memory,half_life=args.evidence_half_life,context_tv=args.context_tv,
            finite_samples=args.finite_samples if args.finite_feedback else 0)
        if structured: blocks.configure_views(len(training),args.window_views)
    else:
        blocks = BlockVisitScheduler(scene.group_ids)
    views = ViewWindowScheduler(scene_record['allocation_train_view_names'], args.seed+9001)
    meter = ExactDeploymentCost(model,scene,identity,args.snr,options.net_bits_per_use)
    if args.resume:
        state = restore_state(latest, plan, scene_record)
        model.load_state_dict(state['state_dict']); table.load_state_dict(state['table'])
        blocks.load_state_dict(state['blocks']); views.load_state_dict(state['views'])
        if fixed_rd:
            controller = None
        else:
            controller = BudgetController(state['controller']['budget']); controller.load_state_dict(state['controller'])
        reference = state['objective_reference' if fixed_rd else 'budget_reference']
        progress, q = state['progress'], state['latest_q']
        repair(out, progress['step']); restore_rng(state['rng'])
    else:
        full = torch.full((len(scene.raw),), 3, dtype=torch.long)
        full_cost = meter.details(full)
        if not fixed_rd:
            budget = args.budget_total_uses if args.budget_total_uses is not None else math.floor(args.budget_ratio_to_q3*full_cost['total_channel_uses'])
            if budget < 1: raise ValueError('resolved budget is less than one channel use')
        base_args = argparse.Namespace(**vars(args)); base_args.lpips = False
        if structured:
            from .allocation_distortion import ImageDistortion
            base_args.allocation_image_objective = ImageDistortion(args.distortion, 0.)
        baseline = evaluate_map(model,scene,full,calibration,base_args,folder=out/'evaluation/q3_reference',label='q3_reference')
        reference = dict(q3_cost=full_cost, d_ref_raw=baseline['source_mse'],
                         d_ref=max(baseline['source_mse'],1e-12), d_ref_floor=1e-12)
        if structured:
            from .allocation_distortion import calibrate
            reference.update(calibrate(baseline['source_objective_mse'], baseline['source_structure_error'],
                mode=args.distortion, share=args.structure_share, lambda_s=args.lambda_s))
            reference['calibration_views'] = scene_record['calibration_view_names']
            reference['calibration_trials'] = args.trials
        if fixed_rd:
            reference.update(beta=args.beta, c_ref=full_cost['total_channel_uses'])
            controller = None
        else:
            reference['budget'] = int(budget)
            controller = BudgetController(budget,args.initial_price,args.dual_lr,args.ema_decay)
        q = deploy(table,args)
        progress = dict(step=0, pending=['initial'], best=None, lowest=None, initial_q=q.clone(),
                        initial_codec_id=identity, frozen_codec_verified=False)
        json_write(out/'plan.json',plan)
        json_write(out/'training.json',dict(plan, **scene_record, source_run=str(Path(args.run).resolve()),
            **{'objective_reference' if fixed_rd else 'budget_reference': reference},
            trainable_modules=[], attribute_statistics_frozen=True,
            optimizer='log-domain mirror descent; fixed N_source global gain for image and rate action costs',
            probability_global_gain=len(scene.raw), distortion=args.distortion if structured else 'source PLY RGB MSE',
            allocation_training_sampling='single',deployment_draws=10,deployment_seed=args.seed,
            allocation_steps_scope='one update per multi-view window',allocation_joint_steps=0))

    distortion_key = 'source_distortion' if structured else 'source_mse'
    if structured:
        from .allocation_distortion import ImageDistortion
        args.allocation_image_objective = ImageDistortion(args.distortion, reference['lambda_s'])
    objective = FixedRateDistortion(args.beta,reference['d_ref'],reference['c_ref'],distortion_key) if fixed_rd else None
    rate_normalizer = reference['c_ref'] if fixed_rd else reference['budget']

    def objective_fields(distortion=None, cost=None):
        if fixed_rd:
            fields = dict(objective_mode='fixed_beta', beta=objective.beta,
                          d_ref=objective.d_ref, c_ref=objective.c_ref)
            if structured:
                fields.update(distortion_field=distortion_key, distortion_mode=args.distortion, lambda_s=reference['lambda_s'])
            if distortion is not None and cost is not None:
                fields.update(normalized_distortion=distortion/objective.d_ref,
                              normalized_cost=cost/objective.c_ref, rd_score=objective.score(distortion,cost))
            return fields
        return dict(budget=reference['budget'],price=controller.price,residual_ema=controller.ema)

    def save(label='latest', map_q=None):
        if model_id(model) != identity: raise RuntimeError('frozen codec or attribute statistics changed')
        result = save_state(out,label,model,table,controller,blocks,views,progress,plan,scene_record,reference,
                            q if map_q is None else map_q)
        json_write(out/'progress.json',dict(status='running',step=progress['step'],phase='allocation',
            **objective_fields(), pending=progress['pending'],
            **{'best_rd' if fixed_rd else 'best_feasible': progress['best']}))
        return result

    def proxy(cost, deployed):
        kept = int((deployed>0).sum())
        fallback = reference['q3_cost']['coordinate_channel_uses']/len(scene.raw)
        return dict(xyz_per_retained=cost['coordinate_channel_uses']/kept if kept else fallback,
            scale_per_retained=cost['scale_content_bits']/options.net_bits_per_use/kept if kept else 0.,
            fixed_uses=cost['metadata_channel_uses']+cost['scale_framing_bytes']*8/options.net_bits_per_use,
            bits_per_use=options.net_bits_per_use)

    def measure(label, cameras, images=False, map_q=None):
        chosen_q = q if map_q is None else map_q
        started = time.perf_counter()
        result = evaluate_map(model,scene,chosen_q,cameras,args,save_images=images,
                              folder=out/'evaluation'/label,label=label)
        cost = meter.details(chosen_q)
        row = dict(result, **cost, step=progress['step'],event=label,
                   **objective_fields(result[distortion_key],cost['total_channel_uses']),
                   evaluation_seconds=time.perf_counter()-started)
        if not fixed_rd: row['budget_feasible'] = cost['total_channel_uses']<=reference['budget']
        upsert_jsonl(out/'metrics/validation.jsonl',row)
        return row

    def consider(row):
        candidate = dict(step=progress['step'], source_mse=row['source_mse'],
            total_channel_uses=row['total_channel_uses'], tier_sha256=row['tier_sha256'], codec_id=identity)
        if fixed_rd:
            if structured: candidate['source_distortion'] = row['source_distortion']
            candidate['rd_score'] = objective.score(row[distortion_key],row['total_channel_uses'])
            if progress['best'] is None or objective.candidate_key(candidate) < objective.candidate_key(progress['best']):
                progress['best'] = candidate; save('best_rd')
        elif row['budget_feasible']:
            if progress['best'] is None or row['source_mse'] < progress['best']['source_mse']:
                progress['best'] = candidate; save('best_feasible')
        else:
            previous = progress['lowest']
            if previous is None or (candidate['total_channel_uses'],candidate['source_mse']) < (previous['total_channel_uses'],previous['source_mse']):
                progress['lowest'] = candidate; save('lowest_violation')

    def pending():
        nonlocal q
        while progress['pending']:
            event = progress['pending'][0]
            q = deploy(table,args)
            if event == 'initial':
                row = measure('calibration_000000',calibration); consider(row)
                reference['side_proxy'] = proxy(row,q)
                measure('initial',scene.test_cameras,True)
            elif event == 'control':
                started = time.perf_counter(); cost = meter.details(q)
                reference['side_proxy'] = proxy(cost,q)
                control = objective_fields() if fixed_rd else controller.observe(cost['total_channel_uses'])
                upsert_jsonl(out/'metrics/control.jsonl',dict(control,**cost,step=progress['step'],
                    event='cost_refresh' if fixed_rd else 'control',
                    tier_counts=torch.bincount(q,minlength=4).tolist(),cost_seconds=time.perf_counter()-started))
            elif event == 'calibration':
                row = measure(f'calibration_{progress["step"]:06d}',calibration); consider(row)
            else: raise RuntimeError('unknown pending allocation event')
            progress['pending'].pop(0); save()

    if not args.resume: save('initial'); save()
    pending()
    updates = 0
    while progress['step'] < args.allocation_steps:
        step = progress['step']+1
        indices = views.next(args.window_views)
        if structured: blocks.set_view_window(indices)
        timings = AllocationTimings(device, bool(args.profile_every and (step==1 or step%args.profile_every==0)))
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        task_type = MaskedMultiViewRenderTask
        if evidence:
            from .allocation_finite import PairedMaskedRenderTask
            task_type = PairedMaskedRenderTask
        if structured:
            from .allocation_distortion import ComponentRenderTask
            task_type = ComponentRenderTask
        task = task_type([training[i] for i in indices],scene.reference,scene.degree,scene.white_background,
                         **({'distortion':args.allocation_image_objective} if structured else {}))
        audit_count = args.audit_samples if args.audit_every and step%args.audit_every==0 else 0
        result = budgeted_mask_step(model,table,scene.groups,scene.group_ids,scene.geometry,task,blocks,
            args.snr,'awgn',d_ref=reference['d_ref'],price=args.beta if fixed_rd else controller.price,
            rate_normalizer=rate_normalizer,
            side_proxy=dict(reference['side_proxy'],step=step),timings=timings,audit_limit=audit_count)
        with timings.measure('mirror_update_seconds'):
            table.update(result.pop('keep_costs'),result.pop('tier_ids'),result.pop('tier_costs'),
                         args.keep_lr,args.tier_lr,len(scene.raw))
        result.pop('keep_image_costs'); sampled_q = result.pop('q'); audits = result.pop('audit')
        timing_row = timings.summary()
        if structured: result.update(table.update_stats)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        score = (objective.score(result['render_loss'],result['expected_cost_proxy']) if fixed_rd else
                 result['render_loss']/reference['d_ref']+controller.price*(result['expected_cost_proxy']/reference['budget']-1))
        row = dict(result, **timing_row, step=step,event='update',
            **objective_fields(result['render_loss'],result['expected_cost_proxy']),
            loss=score,
            training_view_names=[str(training[i].image_name) for i in indices],codec_updated=False,
            step_seconds=time.perf_counter()-started,**blocks.coverage(sampled_q),
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None)
        row['d_ref'] = reference['d_ref']
        if evidence:
            row['candidate_evaluations_this_step'] = int(blocks.last.eq(step).sum())
            row['candidate_evaluations_per_second'] = row['candidate_evaluations_this_step']/row['step_seconds']
            with torch.no_grad():
                confidence = table.logits.softmax(-1).amax(-1)
                retained = sampled_q.to(confidence.device)>0
                row['retained_confidence_below_04'] = float((confidence[retained]<.4).double().mean()) if retained.any() else None
        if fixed_rd:
            row['rd_score_scope'] = 'single-draw training distortion + expected side-cost proxy; not deployment score'
        append_json(out/'metrics/loss.jsonl',row)
        for audit in audits:
            audit_row = dict(audit,step=step,event=f'{audit["source_row"]}_{audit["new_tier"]}')
            if evidence: append_json(out/'metrics/audit.jsonl',audit_row)
            else: upsert_jsonl(out/'metrics/audit.jsonl',audit_row)
        progress['step'] = step; progress['pending'] = []
        if step%args.control_every==0: progress['pending'].append('control')
        if step%args.validate_every==0 or step==args.allocation_steps: progress['pending'].append('calibration')
        if progress['pending'] or step%args.save_every==0: q=deploy(table,args); save()
        pending()
        if step==1 or step%10==0 or step==args.allocation_steps:
            coefficient = f'beta={args.beta:.5g}' if fixed_rd else f'price={controller.price:.5g}'
            mse = row['source_mse'] if structured else row['render_loss']
            print(f'allocation {step}/{args.allocation_steps}: MSE={mse:.6g}, {coefficient}, '
                  f'sampled={row["sampled_tier_counts"]}, covered={row["evaluated_rows"]}/{len(scene.raw)}, '
                  f'sec={row["step_seconds"]:.2f}',flush=True)
            if evidence:
                print(f'  evidence: candidates={row["candidate_evaluations_this_step"]}, '
                      f'per_sec={row["candidate_evaluations_per_second"]:.1f}, '
                      f'retained_visits={row["retained_candidate_evaluations_mean"]:.2f}, '
                      f'finite={row["finite_corrections"]}, '
                      f'sign_errors={row["finite_sign_disagreements"]}/{row["finite_nonzero_pairs"]}',flush=True)
            if structured:
                print(f'  mixed: D={row["source_distortion"]:.6g}, S={row["source_structure_error"]:.6g}, '
                      f'lambda_s={reference["lambda_s"]:.6g}, history_weight={row["retained_effective_evidence_mean"]:.3f}',flush=True)
                print(f'  calibrated: ESS={row["positive_mean_effective_samples"]:.2f}, '
                      f'trust={row["positive_mean_trust"]:.3f}, '
                      f'reused={row["positive_correction_reused_actions"]}/{row["keep_correction_reused_actions"]}, '
                      f'sign_errors={row.get("calibrated_sign_disagreements",0)}/{row.get("calibrated_nonzero_pairs",0)}, '
                      f'regret={row.get("finite_decision_regret_mean",0):.3g}->{row.get("calibrated_decision_regret_mean",0):.3g}',flush=True)
        updates += 1
        if max_updates is not None and updates >= max_updates:
            q=deploy(table,args); save(); return dict(status='interrupted',step=step)
    q=deploy(table,args)
    chosen_label = 'best_rd' if fixed_rd else ('best_feasible' if progress['best'] is not None else 'lowest_violation')
    selected = torch.load(out/'checkpoints'/chosen_label/'state.pt',map_location='cpu',weights_only=True)
    chosen_q = selected['latest_q']
    selected_row = measure('selected',scene.test_cameras,True,map_q=chosen_q)
    if torch.equal(chosen_q,q):
        latest_row = dict(selected_row,event='latest',evaluation_seconds=0.,reused_from='selected')
        upsert_jsonl(out/'metrics/validation.jsonl',latest_row)
        json_write(out/'evaluation/latest/alias.json',dict(metrics_and_images_from='../selected',
            codec_id=identity,tier_sha256=tier_hash(q),reason='identical fixed map, codec, cameras and paired noise'))
    else:
        latest_row=measure('latest',scene.test_cameras,True)
    if evidence and args.uniform_controls:
        for tier in (2,3):
            measure(f'uniform_q{tier}',scene.test_cameras,map_q=torch.full_like(q,tier))
    export = out/'deployment'; export.mkdir(exist_ok=True)
    np.save(export/'latest_tiers.npy',q.numpy().astype(np.uint8),allow_pickle=False)
    np.save(export/'selected_tiers.npy',chosen_q.numpy().astype(np.uint8),allow_pickle=False)
    selected_table = MirrorTierTable(len(scene.raw)); selected_table.load_state_dict(selected['table'])
    np.save(export/'selected_probabilities.npy',selected_table.probabilities(torch.arange(len(scene.raw)),args.snr).numpy(),allow_pickle=False)
    progress['frozen_codec_verified'] = model_id(model)==identity
    status = 'complete' if fixed_rd or progress['best'] is not None else 'budget_unmet'
    result = dict(status=status,step=progress['step'],
        **objective_fields(selected_row[distortion_key],selected_row['total_channel_uses']),
        selection=chosen_label,selected_step=selected['progress']['step'],selected_total_channel_uses=selected_row['total_channel_uses'],
        selected_source_psnr=selected_row['source_psnr'],latest_total_channel_uses=latest_row['total_channel_uses'],
        codec_id=identity,tier_sha256=tier_hash(chosen_q),frozen_codec_verified=progress['frozen_codec_verified'])
    if not fixed_rd: result['budget_feasible'] = progress['best'] is not None
    else:
        result['selection_rd_score'] = selected['progress']['best']['rd_score']
        result['selection_rd_scope'] = 'calibration views only; final rd_score uses test views'
    json_write(export/'selection.json',result); save('final'); save()
    from .budgeted_allocation_report import summarize
    summarize(out,result=result)
    if evidence:
        from .allocation_evidence import summarize_evidence
        summarize_evidence(out,selected,args.seed)
    if structured:
        from .structured_allocation_report import summarize_structured
        summarize_structured(out,selected)
    json_write(out/'complete.json',result)
    json_write(out/'progress.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__': main(parser().parse_args())
