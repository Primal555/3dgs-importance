"""Matched 200-update student continuations for exact keep/drop feedback."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace
import numpy as np
import torch
from . import student_policy_experiment as entry
from .student_policy_state import FAMILY, cpu_copy
from .student_policy_features import atomic_torch, digest, VERSION as FEATURE_VERSION
from .student_policy_model import StudentConfig, StudentPolicy, hard_tiers, teacher_loss
from .student_policy_soft import load_soft_teacher, soft_teacher_loss
from .student_policy_inputs import load_teacher, split_student_cameras
from .student_policy_graph import graph_to
from .student_policy_feedback import CurrentPolicy, actual_tier_keep_cost
from .student_keep_feedback import KeepResidualFeedback
from .student_digital_cost import DigitalDeploymentCost, digital_side_proxy, PROTOCOL
from .student_rd_audit_tools import split_action_maps, transition_stats, paired_evaluate, take_step
from .modular_checkpoint import capture_rng, restore_rng
from .modular_experiment import json_write
from .transport import model_id
from .budgeted_allocation_evaluation import evaluate_map, tier_hash
from .budgeted_allocation_state import upsert_jsonl, read_log

VERSION = 'student_keep_residual_controls_v1'
ARMS = ('replacement', 'weighted_residual')


def parser(arms=ARMS, default_arm='both'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', choices=('best', 'latest'), default='best')
    p.add_argument('--arm', choices=('both', *arms), default=default_arm)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--keep-draws', type=int, default=64)
    p.add_argument('--uniform-share', type=float, default=.5)
    p.add_argument('--holdout-draws', type=int, default=16)
    p.add_argument('--diagnostic-every', type=int, default=25)
    p.add_argument('--evaluate-every', type=int, default=50)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    return p


def load_context(cli, state, out):
    """Load the exact source feature/teacher/camera contract once for both arms."""
    plan = state['plan']; reference = plan['reference']; source = Path(cli.source).resolve()
    settings = plan['settings']
    args = entry.parser().parse_args(['--run', settings['run'], '--scene', settings['scene'], '--out', str(out)])
    vars(args).update(settings); args.device = cli.device; args.out = str(out); args.resume = False
    entry.validate(args)
    if args.cost_protocol != PROTOCOL: raise ValueError('source must use actual digital-file accounting')
    device = entry.device_for(args.device); torch.set_num_threads(args.cpu_threads)
    saved, cfg, spec, old, options, source_info = entry.load_student_source(args)
    model = entry.build_codec(cfg).to(device); model.load_state_dict(saved['state_dict']); del saved
    model.eval().requires_grad_(False); model.set_training_snrs([10.]); codec = model_id(model)
    if codec != plan['codec_id']: raise ValueError('source codec changed')
    raw, degree = entry.read_ply(spec['ply'])
    scene = entry.make_scene(spec, raw, degree, model, options, out, old=old); del raw
    model.set_scene_geometry(scene.geometry)
    if scene.record != plan['scene'] or source_info != plan['source']: raise ValueError('source provenance changed')
    teacher_q, loaded_ref, teacher = load_teacher(args.teacher_run, scene, codec, args.beta, cfg.to_dict(),
        seed=args.seed, snr=args.snr, trials=args.trials)
    teacher_soft = None
    if args.distill_target == 'soft_deployment':
        teacher_soft, soft_record = load_soft_teacher(args.teacher_run, teacher_q, out,
            seed=args.seed, required_cap=args.required_teacher_tier_cap)
        teacher['soft_target'] = soft_record; teacher_soft = teacher_soft.to(device)
    teacher['historical_rd_reference'] = copy.deepcopy(loaded_ref)
    meter = DigitalDeploymentCost(model, scene, codec, args.snr, options.net_bits_per_use)
    full_cost = meter.details(torch.full((len(scene.raw),), 3, dtype=torch.long))
    loaded_ref.update(c_ref=full_cost['total_channel_uses'], q3_cost=full_cost, cost_protocol=PROTOCOL)
    if teacher != plan['teacher'] or loaded_ref != reference: raise ValueError('teacher or RD reference changed')
    boundary = dict(train_names=plan['train_view_names'], calibration_names=plan['calibration_view_names'],
                    test_names=plan['test_view_names'])
    training, _, preparation = split_student_cameras(scene, boundary, args.preparation_views)
    if [str(c.image_name) for c in preparation] != plan['preparation_view_names']:
        raise ValueError('source preparation views changed')
    features = torch.load(source/'preparation/features.pt', map_location='cpu', weights_only=True)
    expected = dict(version=FEATURE_VERSION, identity=dict(codec_id=codec, scene=scene.record, source=source_info),
        camera_names=[str(c.image_name) for c in preparation], fine_points=args.fine_points,
        coarse_points=args.coarse_points, neighbors=args.neighbors, visibility_reduction='visible_welford_float64_v1')
    if features['protocol'] != expected or features['signature'] != digest(expected) or features['signature'] != plan['feature_signature']:
        raise ValueError('source feature cache changed')
    if set(plan['test_view_names']) & set(plan['train_view_names']): raise ValueError('test/train overlap')
    distortion = entry.ImageDistortion('mse_ssim', reference['lambda_s'])
    args.allocation_image_objective = distortion
    graph = graph_to(features['graph'], device)
    controls = entry.controls_for(args, reference, len(scene.raw), device)
    execution = entry.FastFirstOrderExecution(args.candidate_batch); execution.verify(model, scene, args.snr, out)
    return SimpleNamespace(args=args, model=model, cfg=cfg, scene=scene, codec=codec, meter=meter,
        training=training, features=features, graph=graph, controls=controls, distortion=distortion,
        teacher_q=teacher_q, teacher_soft=teacher_soft, execution=execution, device=device, reference=reference)


def heldout_update_summary(holdout, before, after):
    if not holdout: return dict(count=0)
    ids = torch.tensor([r['source_row'] for r in holdout])
    truth = torch.tensor([r['exact_keep_cost'] for r in holdout], dtype=torch.float64)
    approx = torch.tensor([r['approximate_keep_cost'] for r in holdout], dtype=torch.float64)
    dp = after[0].double().softmax(-1)[ids, 1]-before[0].double().softmax(-1)[ids, 1]
    dq = hard_tiers(*after)[ids].gt(0).double()-hard_tiers(*before)[ids].gt(0).double()
    population = holdout[0]['population']
    def estimate(values):
        return dict(population_sum_estimate=float(values.mean()*population),
                    standard_error=float(values.std(unbiased=True)*population/len(values)**.5) if len(values)>1 else None)
    return dict(count=len(ids), population=population,
        scope='independent uniform actions outside current correction sample; training views; no label applied to this update',
        prediction_bias=float((approx-truth).mean()), prediction_rmse=float((approx-truth).square().mean().sqrt()),
        sign_error_fraction=float(((approx<0)!=(truth<0)).double().mean()),
        soft_update_exact=estimate(dp*truth), soft_update_approximate=estimate(dp*approx),
        hard_keep_update_exact=estimate(dq*truth), hard_changed_sample_count=int((dq!=0).sum()), records=holdout)


def losses_for(heads, keep_costs, tier_ids, tier_costs, inverse, sampler):
    keep, tier = heads
    signal = (keep_costs[:, 1]-keep_costs[:, 0]).detach().to(keep)
    correction = sampler.last_correction
    ids = correction['ids'].to(keep.device)
    residual = (correction['exact']-correction['approximate']).to(keep)
    if sampler.mode == 'weighted_residual': residual = residual*correction['weights'].to(keep)
    dense_signal = signal.clone(); dense_signal[ids] -= residual
    probability = keep.softmax(-1)[:, 1]
    positive = tier[tier_ids.to(tier.device)].softmax(-1)
    return dict(keep_dense=(probability*dense_signal).sum(), keep_residual=(probability[ids]*residual).sum(),
        positive=(positive*tier_costs.detach().to(tier)*inverse.to(tier)[:, None]).sum())


def run_arm(cli, pinned, ctx, root, mode, experiment_plan, max_updates=None, *,
            sampler_factory=None, loss_builder=None, step_function=None, version=VERSION):
    out = root/mode; out.mkdir(exist_ok=True)
    args, scene, ref = ctx.args, ctx.scene, ctx.reference
    student = StudentPolicy(StudentConfig(**pinned['config'])).to(ctx.device)
    optimizer = entry.optimizer_for(student, args, 'rd')
    sampler = (sampler_factory or KeepResidualFeedback)(scene.group_ids, args.candidate_rounds, args.finite_samples, args.seed+1103,
        mode=mode, keep_draws=cli.keep_draws, uniform_share=cli.uniform_share, holdout_draws=cli.holdout_draws)
    views = entry.ViewWindowScheduler(pinned['plan']['train_view_names'], args.seed+9001)
    checkpoint = out/'checkpoints/latest/student.pt'
    current = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists() else pinned
    if checkpoint.exists() and current['experiment_plan'] != experiment_plan: raise ValueError('resume plan changed')
    student.load_state_dict(current['student']); optimizer.load_state_dict(copy.deepcopy(current['optimizer']))
    sampler.load_state_dict(current['sampler']); views.load_state_dict(current['views'])
    completed = current.get('continuation_step', 0)
    side = copy.deepcopy(current.get('side_proxy', pinned['progress'].get('side_proxy')))
    if not checkpoint.exists() and 'control' in pinned['progress'].get('pending', []): side = None
    restore_rng(current['rng'])
    def forward():
        f = ctx.features
        return student(f['attributes'], f['latent'], f['visibility'], ctx.graph, ctx.controls)
    @torch.no_grad()
    def heads():
        student.eval(); return tuple(h.detach().cpu() for h in forward())
    def sync():
        if ctx.device.type == 'cuda': torch.cuda.synchronize(ctx.device)
    def save(q):
        atomic_torch(checkpoint, dict(family=FAMILY, config=student.cfg.to_dict(), student=cpu_copy(student.state_dict()),
            optimizer=cpu_copy(optimizer.state_dict()), sampler=sampler.state_dict(), views=views.state_dict(),
            rng=capture_rng(), continuation_step=completed, side_proxy=side, experiment_plan=experiment_plan,
            plan=pinned['plan'], progress=dict(pinned['progress'], rd_step=pinned['progress']['rd_step']+completed,
                step=pinned['progress']['step']+completed), evaluated_tiers=q.to(torch.uint8),
            continuation_family=version, continuation_arm=mode))
    def measure(q, step):
        label = f'test_{step:06d}'
        # Both arms have the same initial map; reuse its four-trial test rendering.
        folder = root/'evaluation/initial' if step == 0 else out/'evaluation'/label
        result = evaluate_map(ctx.model, scene, q, scene.test_cameras, args,
            folder=folder, label=label)
        cost = ctx.meter.details(q)
        row = dict(result, **cost, event=label, step=step,
            rd_score=result['source_distortion']/ref['d_ref']+args.beta*cost['total_channel_uses']/ref['c_ref'],
            scope='original test cameras; reporting only; fixed endpoint, no selection/early stopping')
        upsert_jsonl(out/'metrics/validation.jsonl', row)
        print(f'{mode} test {step}: source PSNR={row["source_psnr"]:.4f}, photo PSNR={row["photo_psnr"]:.4f}, '
              f'cost={row["total_channel_uses"]/1e6:.5f}M, RD={row["rd_score"]:.7g}', flush=True)
        return row
    if not checkpoint.exists():
        save(hard_tiers(*heads()))
    if completed % cli.evaluate_every == 0 or completed == cli.steps:
        evaluated = torch.load(checkpoint, map_location='cpu', weights_only=True)['evaluated_tiers'].long()
        measure(evaluated, completed)
    updates = 0
    for local_step in range(completed+1, cli.steps+1):
        sync(); started = time.perf_counter()
        before = heads(); q = hard_tiers(*before)
        global_step = pinned['progress']['rd_step']+local_step
        if side is None:
            side = digital_side_proxy(ctx.meter.details(q), int((q>0).sum()), ref['q3_cost'], len(q))
        indices = views.next(args.window_views); cameras = [ctx.training[i] for i in indices]
        diagnostic = local_step == 1 or local_step % cli.diagnostic_every == 0 or local_step == cli.steps
        sampler.check_this_step = diagnostic
        paired_rng = capture_rng()
        task = entry.ComponentRenderTask(cameras, scene.reference, scene.degree, scene.white_background, distortion=ctx.distortion)
        adapter = CurrentPolicy(*[h.to(ctx.device) for h in before])
        timings = entry.AllocationTimings(ctx.device, True)
        feedback = entry.budgeted_mask_step(ctx.model, adapter, scene.groups, scene.group_ids, scene.geometry,
            task, sampler, args.snr, 'awgn', d_ref=ref['d_ref'], price=args.beta, rate_normalizer=ref['c_ref'],
            side_proxy=dict(side, step=global_step), timings=timings, execution=ctx.execution)
        if not torch.equal(feedback['q'], q): raise RuntimeError('hard deployment/training map mismatch')
        keep_costs = actual_tier_keep_cost(feedback, adapter, ctx.cfg.rates, args.beta, ref['c_ref'])
        ids, costs = feedback['tier_ids'], feedback['tier_costs']; inverse = sampler.inverse_inclusion[ids]
        student.train(); h = forward()
        losses = (loss_builder or losses_for)(h, keep_costs, ids, costs, inverse, sampler)
        imitation = (soft_teacher_loss(*h, ctx.teacher_soft) if ctx.teacher_soft is not None else teacher_loss(*h, ctx.teacher_q))[0]
        weight = entry.teacher_weight_at(args, global_step); losses['teacher'] = weight*imitation
        update_record = None
        if step_function is not None:
            update_record = step_function(student, optimizer, losses, args.grad_clip,
                before=before, heads=heads, diagnostic=diagnostic)
            gradient_record = update_record if diagnostic else None
            norm = update_record['grad_norm_before_clip']
        elif diagnostic:
            gradient_record = take_step(student, optimizer, losses, args.grad_clip)
            norm = gradient_record['grad_norm_before_clip']
        else:
            optimizer.zero_grad(set_to_none=True); sum(losses.values()).backward()
            norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip, error_if_nonfinite=True))
            optimizer.step(); gradient_record = None
        after = heads(); new_q = hard_tiers(*after)
        diagnostics = None
        if diagnostic:
            maps = split_action_maps(before, after, ids)
            evaluations = {}; cache = {}
            for name in ('baseline', 'full', 'keep_only', 'positive_only'):
                key = tier_hash(maps[name])
                if key not in cache:
                    cache[key] = paired_evaluate(ctx.model, scene, maps[name], dict(training=cameras), args.snr,
                        paired_rng, ctx.distortion, ctx.meter, ref)
                evaluations[name] = cache[key]
            baseline = evaluations['baseline']['evaluation']['training']
            if abs(baseline['distortion']-feedback['source_distortion']) > max(1e-8, abs(feedback['source_distortion'])*1e-5):
                raise RuntimeError('paired diagnostic baseline differs from feedback')
            diagnostics = dict(event='diagnostic', step=local_step, gradients=gradient_record,
                transitions=transition_stats(before, after, ids), maps=evaluations,
                heldout=heldout_update_summary(sampler.holdout, before, after),
                delta_rd={name: row['evaluation']['training']['rd']-baseline['rd'] for name, row in evaluations.items()},
                scope='same training window/noise; consequences of keep/positive hard-map changes after shared update')
            upsert_jsonl(out/'metrics/diagnostics.jsonl', diagnostics)
        sync(); completed = local_step
        row = dict(event='update', step=local_step, source_rd_step=global_step, mode=mode,
            source_distortion=feedback['source_distortion'], source_mse=feedback['source_mse'],
            teacher_weight=weight, grad_norm=norm, losses={k: float(v.detach()) for k,v in losses.items()},
            tier_counts=torch.bincount(new_q, minlength=4).tolist(),
            camera_names=[str(c.image_name) for c in cameras], **timings.summary(),
            correction={k: feedback[k] for k in ('keep_draws', 'keep_unique_renders', 'holdout_unique_renders',
                'keep_exact_seconds', 'keep_residual_rmse', 'residual_weight_max', 'residual_weight_mean',
                'residual_sum_estimate', 'residual_sum_standard_error', 'keep_sampling_min',
                'finite_positive_corrections', 'keep_batch_execution_verified')},
            cost_proxy_scope=side['proxy_scope'], elapsed_seconds=time.perf_counter()-started)
        if hasattr(sampler, 'extra_stats'): row['shared_feedback'] = sampler.extra_stats
        if update_record is not None: row['output_step'] = update_record['output_step']
        upsert_jsonl(out/'metrics/loss.jsonl', row)
        if diagnostic:
            json_write(out/'checks'/f'{local_step:06d}.json', dict(actions=feedback['audit'], diagnostics=diagnostics))
        if global_step % args.control_every == 0:
            actual = ctx.meter.details(new_q)
            side = digital_side_proxy(actual, int((new_q>0).sum()), ref['q3_cost'], len(new_q))
        # Save every update; the student and optimizer are small and reuse the source feature cache.
        save(new_q)
        if local_step % cli.evaluate_every == 0 or local_step == cli.steps: measure(new_q, local_step)
        if local_step == 1 or local_step % 10 == 0:
            print(f'{mode} {local_step}/{cli.steps}: D={row["source_distortion"]:.7g}, '
                  f'keep labels={row["correction"]["keep_unique_renders"]}/{cli.keep_draws}, '
                  f'grad={norm:.4g}, sec={row["elapsed_seconds"]:.2f}' +
                  (f', paired keep/full RD change={diagnostics["delta_rd"]["keep_only"]:+.5g}/{diagnostics["delta_rd"]["full"]:+.5g}'
                   if diagnostics else ''), flush=True)
            if update_record is not None:
                s = sampler.extra_stats; u = update_record['output_step']
                print(f'  shared: full groups={s["full_group_unique_points"]}/{s["full_group_draws"]}, '
                      f'alternative renders={s["shared_teacher"]["alternative_renders"]}, '
                      f'positive errors={s["positive_sign_errors"]}/{s["positive_checked_actions"]}; '
                      f'wrong-action cost={s["positive_actions"]["mean_wrong_action_cost"]}; '
                      f'parameter scale={u["applied_scale"]:.4f}, '
                      f'keep/tier KL={u["accepted"]["keep"]["mean"]:.3g}/{u["accepted"]["tier"]["mean"]:.3g}', flush=True)
        del h, losses, imitation, feedback, task, keep_costs, costs, adapter
        updates += 1
        if max_updates is not None and updates >= max_updates and completed < cli.steps:
            return dict(status='interrupted', step=completed)
    final_q = torch.load(checkpoint, map_location='cpu', weights_only=True)['evaluated_tiers'].long()
    (out/'deployment').mkdir(exist_ok=True)
    np.save(out/'deployment/latest_tiers.npy', final_q.numpy().astype('uint8'), allow_pickle=False)
    validation = read_log(out/'metrics/validation.jsonl')[0]
    diagnostic_rows = read_log(out/'metrics/diagnostics.jsonl')[0]
    complete = dict(status='complete', family=version, mode=mode, steps=completed,
        source_rd_step=pinned['progress']['rd_step'], final_rd_step=pinned['progress']['rd_step']+completed,
        tier_sha256=tier_hash(final_q), frozen_codec_verified=model_id(ctx.model)==ctx.codec,
        cost_protocol=PROTOCOL, side_link_simulated=False, selection='fixed final update')
    if not complete['frozen_codec_verified']: raise RuntimeError('codec changed')
    json_write(out/'complete.json', complete)
    json_write(out/'review_summary.json', dict(complete=complete, test_curve=validation, diagnostics=diagnostic_rows))
    return complete


def main(cli, max_updates=None, *, version=VERSION, arms=ARMS, plan_extra=None,
         sampler_factory=None, loss_builder=None, step_function=None):
    source, out = Path(cli.source).resolve(), Path(cli.out).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('use a separate output')
    if min(cli.steps, cli.keep_draws, cli.diagnostic_every, cli.evaluate_every) < 1 or cli.holdout_draws < 1 or not 0 < cli.uniform_share <= 1:
        raise ValueError('invalid continuation budgets')
    if out.exists() and not cli.resume: raise FileExistsError('use a new output or --resume')
    out.mkdir(parents=True, exist_ok=True)
    pinned_path = out/'source_student.pt'
    if not pinned_path.exists():
        blob = (source/'checkpoints'/cli.checkpoint/'student.pt').read_bytes()
        pinned = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
        if pinned.get('family') != FAMILY or pinned['progress']['phase'] != 'rd':
            raise ValueError('source must contain an RD student and its Adam state')
        temporary = pinned_path.with_suffix('.tmp'); temporary.write_bytes(blob); temporary.replace(pinned_path)
    else:
        pinned = torch.load(pinned_path, map_location='cpu', weights_only=True)
    with pinned_path.open('rb') as f: identity = hashlib.file_digest(f, 'sha256').hexdigest()
    plan = dict(version=version, source=str(source), source_checkpoint=cli.checkpoint, source_sha256=identity,
        source_rd_step=pinned['progress']['rd_step'], arms=list(arms),
        steps=cli.steps, keep_draws=cli.keep_draws, uniform_share=cli.uniform_share,
        holdout_draws=cli.holdout_draws, diagnostic_every=cli.diagnostic_every, evaluate_every=cli.evaluate_every,
        source_settings=pinned['plan']['settings'], reference=pinned['plan']['reference'],
        test_view_names=pinned['plan']['test_view_names'],
        evaluation='original test cameras only; reporting at fixed steps; no selection or early stopping',
        training='shared student; unchanged positive/teacher loss and Adam; paired current-context keep labels',
        gradient_target='dense single-keep-switch sum conditional on fixed current context; marginal-rate proxy shared by both arms')
    if plan_extra is not None: plan.update(plan_extra)
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text(encoding='utf8')) != plan:
        raise ValueError('continuation suite plan changed')
    json_write(out/'plan.json', plan)
    ctx = load_context(cli, pinned, out)
    print(f'Source RD step {plan["source_rd_step"]}; {cli.steps} updates/arm; '
          f'{len(ctx.scene.test_cameras)} original test views; test metrics do not choose checkpoints.', flush=True)
    for mode in (arms if cli.arm == 'both' else (cli.arm,)):
        if (out/mode/'complete.json').is_file(): continue
        result = run_arm(cli, pinned, ctx, out, mode, plan, max_updates=max_updates,
            sampler_factory=sampler_factory, loss_builder=loss_builder, step_function=step_function, version=version)
        if result['status'] != 'complete': return result
    reports = {mode: json.loads((out/mode/'review_summary.json').read_text(encoding='utf8'))
               for mode in arms if (out/mode/'complete.json').is_file()}
    summary = dict(plan=plan, arms=reports)
    json_write(out/'review_summary.json', summary)
    complete = dict(status='complete' if len(reports)==len(arms) else 'arm_complete', arms=list(reports), steps_per_arm=cli.steps)
    json_write(out/'complete.json', complete)
    temporary = out/'review.tar.gz.tmp'
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in sorted(out.rglob('*')):
            relative = path.relative_to(out)
            if path.is_file() and (path.suffix == '.json' or path.suffix == '.jsonl' or path.suffix == '.npy'):
                archive.add(path, arcname=relative.as_posix(), recursive=False)
    temporary.replace(out/'review.tar.gz')
    print(json.dumps(complete), flush=True)
    return complete


if __name__ == '__main__': main(parser().parse_args())
