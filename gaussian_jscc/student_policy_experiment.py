"""Source-conditioned one-forward policy: action distillation then render RD."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .allocation_mirror import ViewWindowScheduler
from .allocation_objective import FixedRateDistortion
from .allocation_reuse import AllocationTimings
from .allocation_distortion import ImageDistortion, ComponentRenderTask, calibrate
from .budgeted_allocation_evaluation import ExactDeploymentCost, evaluate_map, tier_hash
from .budgeted_allocation_kernel import budgeted_mask_step
from .budgeted_allocation_state import repair, upsert_jsonl
from .cli import device_for, seed_all
from .data import read_ply
from .fast_first_order_execution import FastFirstOrderExecution
from .model_factory import build_codec
from .modular_experiment import json_write
from .optimization import preserved_rng
from .render_objective import spaced_indices
from .render_validation import append_json
from .student_policy_features import prepare_features, digest
from .student_policy_feedback import CurrentPolicy, UniformFeedback, policy_surrogate, actual_tier_keep_cost
from .student_policy_graph import graph_to
from .student_policy_inputs import load_student_source, load_teacher, split_student_cameras, json_read
from .student_policy_model import StudentConfig, StudentPolicy, hard_tiers, teacher_loss, teacher_metrics
from .student_policy_soft import load_soft_teacher, soft_teacher_loss
from .student_policy_report import finish_student_report
from .student_policy_state import FAMILY, save_student, read_student, restore_student, recoverable_selected_map
from .successive_multiscene_inputs import make_scene
from .transport import model_id


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'scene', 'out'): p.add_argument('--'+name, required=True)
    p.add_argument('--teacher-run'); p.add_argument('--student-checkpoint'); p.add_argument('--manifest')
    p.add_argument('--distill-target', choices=('hard', 'soft_deployment'), default='hard')
    p.add_argument('--required-teacher-tier-cap', type=float)
    p.add_argument('--required-teacher-window-views', type=int)
    p.add_argument('--cost-protocol', choices=('legacy', 'digital_side_file_v1'), default='legacy')
    p.add_argument('--teacher-weight-schedule', choices=('linear_decay', 'constant'), default='linear_decay',
                   help='RD imitation weight schedule; soft-policy launcher uses constant')
    p.add_argument('--phase', choices=('train', 'predict'), default='train')
    p.add_argument('--device', default='cuda'); p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    for name, default in (('beta', .5), ('snr', 10.), ('distill-lr', 1e-4), ('rd-lr', 1e-5),
                          ('weight-decay', .01), ('teacher-weight', .1), ('grad-clip', 1.)):
        p.add_argument('--'+name, type=float, default=default)
    for name, default in (('distill-epochs', 1000), ('rd-steps', 1000), ('preparation-views', 8),
        ('window-views', 4), ('calibration-views', 4), ('test-views', 32), ('trials', 4), ('image-views', 2),
        ('width', 128), ('coarse-width', 256), ('blocks', 2), ('chunk-size', 16384),
        ('fine-points', 16), ('coarse-points', 4), ('neighbors', 16), ('candidate-batch', 128),
        ('candidate-rounds', 2), ('finite-samples', 32), ('validate-every', 100), ('save-every', 25),
        ('control-every', 25), ('profile-every', 10), ('cpu-threads', 4), ('seed', 42)):
        p.add_argument('--'+name, type=int, default=default)
    return p


def validate(args):
    if args.snr != 10. or not math.isfinite(args.beta) or args.beta < 0: raise ValueError('fixed 10dB and nonnegative beta required')
    positive = ('distill_epochs', 'rd_steps', 'preparation_views', 'window_views', 'calibration_views', 'test_views',
        'trials', 'image_views', 'width', 'coarse_width', 'blocks', 'chunk_size', 'fine_points', 'coarse_points',
        'neighbors', 'candidate_batch', 'candidate_rounds', 'validate_every', 'save_every', 'control_every', 'cpu_threads')
    if min(getattr(args, k) for k in positive) < 1 or min(args.finite_samples, args.profile_every) < 0:
        raise ValueError('invalid student execution/training budgets')
    if args.candidate_rounds > 8: raise ValueError('at most eight distinct candidates per block')
    for k in ('distill_lr', 'rd_lr', 'grad_clip'):
        if not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0: raise ValueError('invalid optimizer value: '+k)
    if any(not math.isfinite(x) or x < 0 for x in (args.weight_decay, args.teacher_weight)):
        raise ValueError('invalid auxiliary loss/decay')
    if args.phase == 'train' and not args.teacher_run: raise ValueError('--teacher-run required for distillation')
    if args.phase == 'predict' and not args.student_checkpoint: raise ValueError('--student-checkpoint required for prediction')
    if args.required_teacher_tier_cap is not None and (not math.isfinite(args.required_teacher_tier_cap) or
            args.required_teacher_tier_cap <= 0 or args.distill_target != 'soft_deployment'):
        raise ValueError('teacher cap check requires soft_deployment and a positive finite cap')
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out): raise ValueError('codec source and student output must be separate')
    if args.teacher_run:
        teacher = Path(args.teacher_run).resolve()
        if out.is_relative_to(teacher) or teacher.is_relative_to(out): raise ValueError('teacher and student output must be separate')


def optimizer_for(student, args, phase):
    return torch.optim.AdamW(student.parameters(), lr=args.distill_lr if phase == 'distill' else args.rd_lr,
                             weight_decay=args.weight_decay)


def teacher_weight_at(args, step):
    if args.teacher_weight_schedule == 'constant': return args.teacher_weight
    return args.teacher_weight*max(0., 1.-(step-1)/max(args.rd_steps/2, 1.))


def controls_for(args, reference, count, device):
    return torch.tensor([args.snr/10., args.beta, math.log(reference['d_ref']),
                         math.log(reference['c_ref']/count)], device=device, dtype=torch.float32)


def reusable_reference(scene, trained_plan):
    keys = ('fingerprint', 'source_gaussians', 'sh_degree', 'geometry', 'camera_fingerprints',
            'resolution', 'images', 'white_background', 'blocks_per_batch')
    return (all(scene.record.get(k) == trained_plan['scene'].get(k) for k in keys) and
            scene.net_bits_per_use == trained_plan['source']['net_bits_per_use'])


def prepare_prediction_reference(model, scene, cameras, args, meter, out, *, clock=None):
    clock = time.perf_counter if clock is None else clock
    started = clock()
    q3 = torch.full((len(scene.raw),), 3, dtype=torch.long)
    ref_args = argparse.Namespace(**vars(args)); ref_args.lpips = False
    ref_args.allocation_image_objective = ImageDistortion('mse_ssim', 0.)
    base = evaluate_map(model, scene, q3, cameras, ref_args, folder=out/'evaluation/q3_reference', label='q3_reference')
    reference = calibrate(base['source_objective_mse'], base['source_structure_error'],
                          mode='mse_ssim', share=.2, lambda_s=None)
    reference.update(c_ref=meter.details(q3)['total_channel_uses'], beta=args.beta)
    return reference, clock()-started


def allocation_timing(feature_seconds, reference_seconds, forward_seconds):
    return dict(feature_preparation_seconds=feature_seconds, reference_preparation_seconds=reference_seconds,
        preparation_seconds_this_run=feature_seconds+reference_seconds,
        student_forward_seconds=forward_seconds,
        allocation_seconds=feature_seconds+reference_seconds+forward_seconds,
        timing_scope='source feature preparation + q3 reference preparation + student inference; delivery evaluation separate')


def main(args, max_updates=None):
    request_started = time.perf_counter()
    validate(args)
    saved, cfg, spec, old, options, source = load_student_source(args)
    if tuple(cfg.rates) != (0, 8, 24, 48) or cfg.channel_variant != 'successive':
        raise ValueError('student requires the fixed successive 0/8/24/48 codec')
    if args.dry_run:
        result = dict(version=FAMILY, settings=vars(args), source=source, codec_config=cfg.to_dict())
        print(json.dumps(result, indent=2)); return result
    out = Path(args.out).resolve(); latest = out/'checkpoints/latest/student.pt'
    if out.exists() and not args.resume: raise FileExistsError('use a new student output or --resume')
    resume_training = args.resume and latest.is_file()
    if args.resume and args.phase == 'train' and not resume_training:
        loss_log = out/'metrics/loss.jsonl'
        if (not (out/'preparation').is_dir() or (loss_log.is_file() and loss_log.stat().st_size) or
                (out/'complete.json').is_file()):
            raise FileNotFoundError('missing student checkpoint; only interrupted source preparation can restart without it')
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    model = build_codec(cfg).to(device); model.load_state_dict(saved['state_dict']); del saved
    model.eval().requires_grad_(False); model.set_training_snrs([10.]); codec_id = model_id(model)
    raw, degree = read_ply(spec['ply']); source_fingerprint = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    scene = make_scene(spec, raw, degree, model, options, out, old=old); del raw
    model.set_scene_geometry(scene.geometry)
    (out/'metrics').mkdir(parents=True, exist_ok=True)
    teacher_q = None; teacher_soft = None; prediction_state = None
    if args.phase == 'train':
        teacher_q, reference, teacher = load_teacher(args.teacher_run, scene, codec_id, args.beta,
            cfg.to_dict(), seed=args.seed, snr=args.snr, trials=args.trials)
        if args.required_teacher_window_views is not None:
            actual_views = json_read(Path(args.teacher_run)/'plan.json')['settings']['window_views']
            if actual_views != args.required_teacher_window_views:
                raise ValueError(f'teacher window views: expected {args.required_teacher_window_views}, got {actual_views}')
        if args.distill_target == 'soft_deployment':
            teacher_soft, soft_record = load_soft_teacher(args.teacher_run, teacher_q, out,
                seed=args.seed, required_cap=args.required_teacher_tier_cap)
            teacher['soft_target'] = soft_record
            teacher_soft = teacher_soft.to(device)
            print(f'Teacher: {teacher["path"]}; selected step={soft_record["selected_step"]}; '
                  f'positive cap={soft_record["teacher_tier_cap"]}; exact ten-draw soft targets; '
                  f'probability SHA256={soft_record["probability_sha256"]}', flush=True)
        training, calibration, preparation = split_student_cameras(
            scene, teacher, args.preparation_views, args.calibration_views)
    else:
        prediction_state = read_student(args.student_checkpoint)
        protocol = prediction_state['plan']
        if args.cost_protocol != protocol['settings'].get('cost_protocol', 'legacy'):
            raise ValueError('prediction cost protocol differs from trained student')
        if protocol['codec_id'] != codec_id or protocol['settings']['beta'] != args.beta or protocol['settings']['snr'] != args.snr:
            raise ValueError('prediction must retain the trained codec/beta/SNR')
        for key in ('fine_points', 'coarse_points', 'neighbors', 'preparation_views'):
            if getattr(args, key) != protocol['settings'][key]:
                raise ValueError('prediction source-feature protocol changed: '+key)
        teacher = None
        if reusable_reference(scene, protocol):
            boundary = dict(train_names=protocol['train_view_names'], calibration_names=protocol['calibration_view_names'],
                            test_names=protocol['test_view_names'])
            training, calibration, preparation = split_student_cameras(scene, boundary, args.preparation_views)
            if [str(c.image_name) for c in preparation] != protocol['preparation_view_names']:
                raise ValueError('prediction preparation cameras differ from student training protocol')
            reference = copy.deepcopy(protocol['reference'])
        else:
            indices = set(spaced_indices(len(scene.cameras), args.calibration_views))
            calibration = [c for i, c in enumerate(scene.cameras) if i in indices]
            training = [c for i, c in enumerate(scene.cameras) if i not in indices]
            preparation = [training[i] for i in spaced_indices(len(training), min(args.preparation_views, len(training))) ]
            reference = None
    if len(training) < args.window_views: raise ValueError('insufficient independent training/preparation cameras')
    if len(preparation) != args.preparation_views:
        raise ValueError('insufficient training cameras for requested preparation views')
    digital_cost = args.cost_protocol == 'digital_side_file_v1'
    if digital_cost:
        from .student_digital_cost import DigitalDeploymentCost, digital_side_proxy
    meter_type = DigitalDeploymentCost if digital_cost else ExactDeploymentCost
    meter = meter_type(model, scene, codec_id, args.snr, options.net_bits_per_use)
    reference_seconds = 0.
    if reference is None:
        reference, reference_seconds = prepare_prediction_reference(model, scene, calibration, args, meter, out)
    if args.phase == 'train':
        full_cost = meter.details(torch.full((len(scene.raw),), 3, dtype=torch.long))
        if digital_cost:
            teacher['historical_rd_reference'] = copy.deepcopy(reference)
            reference.update(c_ref=full_cost['total_channel_uses'], q3_cost=full_cost,
                             cost_protocol=args.cost_protocol)
        elif full_cost['total_channel_uses'] != reference['c_ref']:
            raise ValueError('teacher complete cost accounting differs')
    print(f'Student cameras: preparation={len(preparation)}, RD per step={args.window_views}, '
          f'training pool={len(training)}, calibration={len(calibration)}, test={len(scene.test_cameras)}; '
          f'cost={args.cost_protocol}, C_ref={reference["c_ref"]}', flush=True)
    args.allocation_image_objective = ImageDistortion('mse_ssim', reference['lambda_s'])
    identity = dict(codec_id=codec_id, scene=scene.record, source=source)
    feature_start = time.perf_counter(); cache_hit = (out/'preparation/features.pt').is_file()
    features = prepare_features(model, scene, preparation, identity, out, fine_points=args.fine_points,
                                coarse_points=args.coarse_points, neighbors=args.neighbors, streaming=digital_cost)
    preparation_seconds = time.perf_counter()-feature_start
    network_cfg = StudentConfig(features['attributes'].shape[1], features['latent'].shape[1], features['visibility'].shape[1],
        width=args.width, coarse_width=args.coarse_width, blocks=args.blocks, chunk_size=args.chunk_size)
    if prediction_state is not None: network_cfg = StudentConfig(**prediction_state['config'])
    student = StudentPolicy(network_cfg).to(device)
    if prediction_state is None: student.fit_normalizers(features['attributes'], features['latent'], features['visibility'])
    else: student.load_state_dict(prediction_state['student'])
    graph = graph_to(features['graph'], device); controls = controls_for(args, reference, len(scene.raw), device)
    settings = {k: v for k, v in vars(args).items() if k not in ('out', 'resume', 'dry_run', 'device', 'allocation_image_objective')}
    # Keep existing hard-target checkpoints resumable with their original plan.
    if args.distill_target == 'hard': settings.pop('distill_target')
    if args.required_teacher_tier_cap is None: settings.pop('required_teacher_tier_cap')
    if args.required_teacher_window_views is None: settings.pop('required_teacher_window_views')
    if args.cost_protocol == 'legacy': settings.pop('cost_protocol')
    if args.teacher_weight_schedule == 'linear_decay': settings.pop('teacher_weight_schedule')
    plan = dict(version=FAMILY, settings=settings, source=source, scene=scene.record, codec_id=codec_id,
        teacher=teacher, reference=reference, feature_signature=features['signature'], config=network_cfg.to_dict(),
        train_view_names=[str(c.image_name) for c in training], calibration_view_names=[str(c.image_name) for c in calibration],
        preparation_view_names=[str(c.image_name) for c in preparation], test_view_names=[str(c.image_name) for c in scene.test_cameras],
        deployment_rule='keep logit >= drop logit, then positive argmax; one forward',
        feedback='current paired context; dense keep and uniform inverse-inclusion positive')
    if not (out/'plan.json').is_file(): json_write(out/'plan.json', plan)
    elif json_read(out/'plan.json') != plan: raise ValueError('student run plan changed')
    if args.phase == 'train':
        print(f'RD teacher constraint: schedule={args.teacher_weight_schedule}; weight={args.teacher_weight:g}', flush=True)
    objective = FixedRateDistortion(args.beta, reference['d_ref'], reference['c_ref'], 'source_distortion')
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    def forward():
        return student(features['attributes'], features['latent'], features['visibility'], graph, controls)
    def imitation_loss(keep, tier):
        if teacher_soft is not None: return soft_teacher_loss(keep, tier, teacher_soft)
        return teacher_loss(keep, tier, teacher_q)
    @torch.no_grad()
    def infer():
        student.eval(); sync(); started = time.perf_counter()
        heads = forward(); q = hard_tiers(*heads).cpu(); sync()
        return q, time.perf_counter()-started
    progress = dict(phase='distill', step=0, distill_epoch=0, rd_step=0, best=None, pending=[], optimizer_resets=0)
    def measure(label, q, cameras, images=False):
        started = time.perf_counter()
        result = evaluate_map(model, scene, q, cameras, args, save_images=images,
                              folder=out/'evaluation'/label, label=label)
        cost = meter.details(q)
        row = dict(result, **cost, event=label, step=progress['step'],
            rd_score=objective.score(result['source_distortion'], cost['total_channel_uses']),
            d_ref=reference['d_ref'], c_ref=reference['c_ref'], beta=args.beta,
            evaluation_seconds=time.perf_counter()-started)
        upsert_jsonl(out/'metrics/validation.jsonl', row)
        print(f'{label}: PSNR={row["source_psnr"]:.4f}, cost={cost["total_channel_uses"]/1e6:.5f}M, RD={row["rd_score"]:.7g}', flush=True)
        return row
    if args.phase == 'predict':
        q, seconds = infer()
        startup_to_policy_seconds = time.perf_counter()-request_started
        selected_row = measure('selected', q, scene.test_cameras, True)
        for tier in (2, 3): measure(f'uniform_q{tier}', torch.full_like(q, tier), scene.test_cameras)
        export = out/'deployment'; export.mkdir(exist_ok=True)
        np.save(export/'selected_tiers.npy', q.numpy().astype('uint8'), allow_pickle=False)
        complete = dict(status='complete', family=FAMILY, phase='predict', optimizer_updates=0,
            codec_id=codec_id, tier_sha256=tier_hash(q), rd_score=selected_row['rd_score'],
            **allocation_timing(preparation_seconds, reference_seconds, seconds), preparation_cache_hit=cache_hit,
            predict_startup_to_policy_seconds=startup_to_policy_seconds,
            fixed_student_weights=True, fixed_feature_normalizers=True)
        json_write(out/'complete.json', complete); json_write(export/'selection.json', complete)
        finish_student_report(out, complete); print(json.dumps(complete), flush=True); return complete
    sampler = UniformFeedback(scene.group_ids, args.candidate_rounds, args.finite_samples, args.seed+1103)
    views = ViewWindowScheduler(plan['train_view_names'], args.seed+9001)
    if resume_training:
        checkpoint_state = read_student(latest); progress = copy.deepcopy(checkpoint_state['progress'])
        optimizer = optimizer_for(student, args, progress['phase'])
        progress = restore_student(checkpoint_state, student, optimizer, sampler, views, plan)
        repair(out, progress['step'])
    else: optimizer = optimizer_for(student, args, 'distill')
    execution = FastFirstOrderExecution(args.candidate_batch)
    execution.verify(model, scene, args.snr, out)
    json_write(out/'training.json', dict(plan, trainable_modules=['student'],
        parameters=sum(p.numel() for p in student.parameters()), frozen_codec=True,
        preparation_seconds_this_run=preparation_seconds, preparation_cache_hit=cache_hit,
        optimizer='AdamW; new optimizer at distill/RD transition; no per-point probability optimizer'))
    def save(label='latest', evaluated_tiers=None):
        if model_id(model) != codec_id: raise RuntimeError('frozen codec changed')
        state = save_student(out, label, student, optimizer, progress, plan, sampler, views,
                             evaluated_tiers=evaluated_tiers)
        json_write(out/'progress.json', dict(status='running', **progress)); return state
    def consider(row, q):
        if progress['best'] is None or row['rd_score'] < progress['best']['rd_score']:
            progress['best'] = dict(step=progress['step'], rd_score=row['rd_score'], tier_sha256=tier_hash(q))
            save('best', evaluated_tiers=q)
    def refresh_proxy(q):
        cost = meter.details(q); retained = int((q > 0).sum())
        if digital_cost:
            fallback = meter.details(torch.full_like(q,3)) if not retained else cost
            progress['side_proxy'] = digital_side_proxy(cost, retained, fallback, len(q))
            return
        progress['side_proxy'] = dict(xyz_per_retained=cost['coordinate_channel_uses']/retained if retained else reference['q3_cost']['coordinate_channel_uses']/len(q),
            scale_per_retained=cost['scale_content_bits']/options.net_bits_per_use/retained if retained else 0.,
            fixed_uses=cost['metadata_channel_uses']+cost['scale_framing_bytes']*8/options.net_bits_per_use,
            bits_per_use=options.net_bits_per_use)
    def pending():
        while progress['pending']:
            event = progress['pending'][0]; q, _ = infer()
            if event == 'distilled':
                row = measure(f'calibration_{progress["step"]:06d}', q, calibration); consider(row, q)
                measure('initial', q, scene.test_cameras, True); refresh_proxy(q)
            elif event in ('calibration', 'distill_calibration'):
                row = measure(f'calibration_{progress["step"]:06d}', q, calibration); consider(row, q)
            elif event == 'control': refresh_proxy(q)
            else: raise RuntimeError('unknown pending student event')
            progress['pending'].pop(0); save()
    if not resume_training: save()
    pending(); updates = 0
    while progress['distill_epoch'] < args.distill_epochs:
        started = time.perf_counter(); student.train(); optimizer.zero_grad(set_to_none=True)
        keep, tier = forward(); loss, stats = imitation_loss(keep, tier)
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step(); sync()
        progress['distill_epoch'] += 1; progress['step'] += 1
        q = hard_tiers(keep.detach(), tier.detach()).cpu()
        stats.update(teacher_metrics(keep.detach(), tier.detach(), teacher_q))
        row = dict(stats, step=progress['step'], event='update', phase='distill', epoch=progress['distill_epoch'],
            loss=float(loss.detach()), grad_norm=float(norm), seconds=time.perf_counter()-started,
            samples=len(q), full_point_pass=True, codec_updated=False)
        append_json(out/'metrics/loss.jsonl', row)
        if progress['distill_epoch']%args.validate_every == 0 and progress['distill_epoch'] < args.distill_epochs:
            progress['pending'] = ['distill_calibration']
        save()
        if progress['distill_epoch'] == 1 or progress['distill_epoch']%5 == 0 or progress['distill_epoch'] == args.distill_epochs:
            by_tier = '/'.join('NA' if value is None else f'{value:.1%}' for value in stats['action_accuracy_by_teacher_tier'])
            positive = stats['teacher_positive_tier_accuracy']
            positive_text = 'NA' if positive is None else f'{positive:.2%}'
            print(f'distill {progress["distill_epoch"]}/{args.distill_epochs}: loss={row["loss"]:.6g}, keep={stats["teacher_keep_accuracy"]:.2%}, action={stats["teacher_action_accuracy"]:.2%}, positive={positive_text}, q0/q1/q2/q3={by_tier}, sec={row["seconds"]:.2f}', flush=True)
            if teacher_soft is not None:
                print(f'  soft teacher: keep_KL={stats["teacher_keep_kl"]:.6g}, tier_KL={stats["teacher_tier_kl"]:.6g}, '
                      f'expected_action_agreement={stats["teacher_expected_action_agreement"]:.2%}; '
                      'keep/action/positive above compare the exported sampled map', flush=True)
        del keep, tier, loss
        pending()
        updates += 1
        if max_updates is not None and updates >= max_updates:
            save(); return dict(status='interrupted', step=progress['step'])
    if progress['phase'] == 'distill':
        progress['phase'] = 'rd'; progress['optimizer_resets'] += 1
        optimizer = optimizer_for(student, args, 'rd'); progress['pending'] = ['distilled']
        save('distilled'); save(); pending()
    while progress['rd_step'] < args.rd_steps:
        started = time.perf_counter(); step = progress['rd_step']+1
        student.eval()
        with torch.no_grad(): keep, tier = forward()
        adapter = CurrentPolicy(keep, tier); q = hard_tiers(keep, tier).cpu(); del keep, tier
        indices = views.next(args.window_views)
        timings = AllocationTimings(device, bool(args.profile_every and (step == 1 or step%args.profile_every == 0)))
        task = ComponentRenderTask([training[i] for i in indices], scene.reference, scene.degree,
                                    scene.white_background, distortion=args.allocation_image_objective)
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        feedback = budgeted_mask_step(model, adapter, scene.groups, scene.group_ids, scene.geometry, task, sampler,
            args.snr, 'awgn', d_ref=reference['d_ref'], price=args.beta, rate_normalizer=reference['c_ref'],
            side_proxy=dict(progress['side_proxy'], step=step), timings=timings, execution=execution)
        if not torch.equal(feedback['q'], q): raise RuntimeError('training hard map differs from deployment policy')
        keep_costs = actual_tier_keep_cost(feedback, adapter, cfg.rates, args.beta, reference['c_ref'])
        ids, costs = feedback.pop('tier_ids'), feedback.pop('tier_costs')
        inverse = sampler.inverse_inclusion[ids.cpu()]
        student.train(); optimizer.zero_grad(set_to_none=True)
        keep, tier = forward()
        surrogate, stats = policy_surrogate(keep, tier, keep_costs, ids, costs, inverse)
        imitation, imitation_stats = imitation_loss(keep, tier)
        teacher_weight = teacher_weight_at(args, step)
        loss = surrogate+teacher_weight*imitation
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step(); sync()
        progress['rd_step'] = step; progress['step'] += 1
        audits = feedback.pop('audit')
        for key in ('keep_costs', 'keep_image_costs', 'q'): feedback.pop(key)
        row = dict(feedback, **timings.summary(), **stats, **imitation_stats, step=progress['step'],
            event='update', phase='rd', rd_step=step, loss=float(loss.detach()), teacher_weight=teacher_weight,
            teacher_weight_schedule=args.teacher_weight_schedule,
            grad_norm=float(norm), step_seconds=time.perf_counter()-started,
            training_view_names=[str(training[i].image_name) for i in indices],
            objective_scope='hard current render distortion; marginal-rate/VJP training surrogate',
            hard_rd_proxy=feedback['source_distortion']/reference['d_ref']+args.beta*feedback['sampled_cost_proxy']/reference['c_ref'],
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None)
        if digital_cost:
            row.update(cost_protocol=args.cost_protocol, cost_proxy_scope=progress['side_proxy']['proxy_scope'])
        append_json(out/'metrics/loss.jsonl', row)
        for a in audits: append_json(out/'metrics/audit.jsonl', dict(a, step=progress['step'], event=f'{a["source_row"]}_{a["new_tier"]}'))
        progress['pending'] = []
        if step%args.control_every == 0: progress['pending'].append('control')
        if step%args.validate_every == 0 or step == args.rd_steps: progress['pending'].append('calibration')
        if progress['pending'] or step%args.save_every == 0: save()
        pending()
        if step == 1 or step%10 == 0 or step == args.rd_steps:
            print(f'student RD {step}/{args.rd_steps}: D={row["source_distortion"]:.6g}, tiers={torch.bincount(q, minlength=4).tolist()}, sampled={len(ids)}, sec={row["step_seconds"]:.2f}, teacher_weight={teacher_weight:.4g}', flush=True)
        del feedback, task, adapter, keep, tier, loss, surrogate, imitation, keep_costs, costs
        updates += 1
        if max_updates is not None and updates >= max_updates:
            save(); return dict(status='interrupted', step=progress['step'])
    latest_q, seconds = infer(); save()
    selected_state = read_student(out/'checkpoints/best/student.pt')
    student.load_state_dict(selected_state['student']); inferred_q, selected_seconds = infer()
    selected_q, selection_check = recoverable_selected_map(selected_state, inferred_q, progress['best'],
        out/'checkpoints/best/student.pt', out/'checkpoints/recovered_selection.pt')
    if not selection_check['matches_original_evaluation']:
        recovered = measure('recovered_best_calibration', selected_q, calibration)
        selection_check.update(recovered_calibration_rd=recovered['rd_score'],
            calibration_rd_change=recovered['rd_score']-progress['best']['rd_score'])
        print('Legacy best map reconstructed; calibration RD '
              f'{progress["best"]["rd_score"]:.7g}->{recovered["rd_score"]:.7g}; '
              'see selection_reproducibility.json', flush=True)
    json_write(out/'evaluation/selection_reproducibility.json',selection_check)
    selected_row = measure('selected', selected_q, scene.test_cameras, True)
    measure('latest', latest_q, scene.test_cameras)
    measure('teacher', teacher_q, scene.test_cameras)
    for tier in (2, 3): measure(f'uniform_q{tier}', torch.full_like(selected_q, tier), scene.test_cameras)
    export = out/'deployment'; export.mkdir(exist_ok=True)
    np.save(export/'selected_tiers.npy', selected_q.numpy().astype('uint8'), allow_pickle=False)
    np.save(export/'latest_tiers.npy', latest_q.numpy().astype('uint8'), allow_pickle=False)
    original = torch.empty_like(scene.raw); original[scene.order] = scene.raw
    complete = dict(status='complete', family=FAMILY, phase='train', step=progress['step'],
        distill_epochs=progress['distill_epoch'], rd_updates=progress['rd_step'], optimizer_resets=progress['optimizer_resets'],
        teacher_weight=args.teacher_weight, teacher_weight_schedule=args.teacher_weight_schedule,
        selected_step=progress['best']['step'], codec_id=codec_id, tier_sha256=tier_hash(selected_q),
        selection_reproducibility=selection_check,
        rd_score=selected_row['rd_score'], selected_source_psnr=selected_row['source_psnr'],
        selected_total_channel_uses=selected_row['total_channel_uses'], beta=args.beta,
        frozen_codec_verified=model_id(model) == codec_id,
        frozen_attributes_verified=hashlib.sha256(original.numpy().tobytes()).hexdigest() == source_fingerprint,
        latest_forward_seconds=seconds, **allocation_timing(preparation_seconds, reference_seconds, selected_seconds),
        preparation_cache_hit=cache_hit,
        deployment_rule=plan['deployment_rule'])
    if digital_cost:
        complete.update(cost_protocol=args.cost_protocol, c_ref=reference['c_ref'],
                        side_link_simulated=False)
    json_write(out/'complete.json', complete); json_write(out/'progress.json', complete)
    json_write(export/'selection.json', complete); finish_student_report(out, complete)
    print(json.dumps(complete), flush=True)
    return complete


if __name__ == '__main__': main(parser().parse_args())
