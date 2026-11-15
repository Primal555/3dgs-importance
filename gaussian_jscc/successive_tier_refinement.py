"""Sequential q1, q2, q3 refinement with a fixed semantic codec and lower stages."""
import copy
import json
from pathlib import Path
import statistics
import time

import torch

from . import successive_q3_refinement as base
from .channel_normalization_metrics import cache_teacher
from .cli import device_for, seed_all
from .data import read_ply
from .model_factory import build_codec, config_from_dict
from .modular_checkpoint import capture_rng, restore_rng, save_training_checkpoint, validate_scene
from .modular_experiment import json_write, repair_logs
from .multisnr_evaluation import actual_delivery, evaluate_matrix, read
from .multisnr_training import cached_render_step
from .optimization import clip_codec_gradients, preserved_rng
from .perceptual_metrics import LPIPSEvaluator
from .render_objective import MultiViewRenderTask
from .render_validation import append_json
from .refinement_views import RefinementViewCycle, make_refinement_scene, refinement_split
from .successive_multiscene_inputs import make_scene
from .successive_multisnr import write_review
from .transport import model_id

FAMILY = 'successive_tier_refinement_multiview_v2'
ENDPOINTS = ('initial', 'after_q1', 'after_q2', 'final')
LAYOUTS = ('1', '2', '3', 'mixed')


def prefixes(tier):
    if tier not in (1, 2, 3):
        raise ValueError('expected tier 1, 2 or 3')
    return (f'channel_encoder.stages.{tier-1}.', f'channel_decoder.stages.{tier-1}.')


ALL_STAGES = tuple(p for tier in (1, 2, 3) for p in prefixes(tier))


def parser():
    p = base.parser()
    p.description = __doc__
    p.set_defaults(steps_per_scene=150)
    p.add_argument('--views-per-step', type=int, default=32)
    p.add_argument('--validation-views', type=int, default=16)
    return p


def configure_optimizer(model, tier, lr):
    prefixes(tier)
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    modules = (model.channel_encoder.stages[tier-1], model.channel_decoder.stages[tier-1])
    groups = []
    for name, module in zip(prefixes(tier), modules):
        module.requires_grad_(True)
        groups.append(dict(params=list(module.parameters()), lr=lr, name=name.rstrip('.')))
    return torch.optim.Adam(groups)


def inputs(args):
    requested_views = dict(views_per_step=args.views_per_step, validation_views=args.validation_views)
    saved, metadata, cfg, specs, heldouts, plan = base.inputs(args)
    for key, value in requested_views.items():
        setattr(args, key, value)
    splits = {s['name']: refinement_split(metadata['scene_records'][s['name']],
                                        args.validation_views, args.views_per_step) for s in specs}
    args.eval_layouts = list(LAYOUTS)
    args.delivery_layouts = list(LAYOUTS)
    per_stage = len(specs)*args.steps_per_scene
    plan.update(version=FAMILY, total_updates=3*per_stage,
        stage_updates={f'q{q}': per_stage for q in (1, 2, 3)}, stage_order=[1, 2, 3],
        trainable_prefixes=list(ALL_STAGES),
        frozen_state_sha256=base.frozen_digest(saved['state_dict'], ALL_STAGES),
        objective='source PLY RGB MSE + latent_weight * latent MSE; current uniform tier, 10 dB AWGN',
        optimizer_initialization='new Adam at each tier; only current sender/receiver stage; shared across scenes',
        stage_scope={f'q{q}': list(prefixes(q)) for q in (1, 2, 3)},
        endpoint_evaluation='initial, after_q1, after_q2, final; all positive tiers and fixed mixed map',
        lower_tier_checks='uniform q1 after q2; uniform q1/q2 after q3, paired views/noise',
        camera_splits=splits,
        view_sampling='independent scene/stage shuffled cycles; distinct views per update; exact resume',
        view_loss_reduction='mean over cameras observing the same noisy decoded scene',
        validation_protocol=dict(trials=args.trials, source_validation_retained=True,
            added_views='held out from this refinement; previously used in source codec training',
            selection=False, paired_seeds=[args.seed+1000+i for i in range(args.trials)]),
        actual_delivery_scope=dict(scenes=args.delivery_scenes, layouts=list(LAYOUTS), snrs=[10.], trials=args.trials))
    plan['settings'].update(requested_views, eval_layouts=list(LAYOUTS), delivery_layouts=list(LAYOUTS))
    return saved, metadata, cfg, specs, heldouts, json.loads(json.dumps(plan))


def monitor_metrics(rows, args, scene, tier):
    names = [str(c.image_name) for c in scene.val_cameras]
    if (len(rows) != args.trials or {r['trial'] for r in rows} != set(range(args.trials)) or
            any(r['path'] != 'awgn' or r['layout'] != str(tier) or r['snr_db'] != 10. or
                r['scope'] != 'validation' or r['seed'] != args.seed+1000+r['trial'] or
                [v['view'] for v in r['views']] != names for r in rows)):
        raise RuntimeError('validation trials, views or paired noise changed')
    result = dict(trials=len(rows), trial_seeds=sorted(r['seed'] for r in rows),
                  validation_view_names=names, validation_views=len(names))
    for key in base.METRICS:
        if key in rows[0]:
            values = [r[key] for r in rows]
            result[key] = statistics.mean(values)
            result[key+'_std_trials'] = statistics.pstdev(values)
    for key in ('latent_mse', 'latent_nmse'):
        values = [r['diagnostics'][key] for r in rows]
        result[key] = statistics.mean(values)
        result[key+'_std_trials'] = statistics.pstdev(values)
    return result


def summarize(out, plan):
    rows, paired, checks = [], {}, []
    for index, endpoint in enumerate(ENDPOINTS):
        for spec in plan['selected_scenes']+plan['heldout_scene']:
            records = read(out/'evaluation'/endpoint/spec['name']/'summary.json')['rows']
            for r in records:
                key = (spec['name'], r['path'], r['layout'], r['trial'])
                if index:
                    before = paired[ENDPOINTS[index-1], key]
                    if (r['seed'] != before['seed'] or r['snr_db'] != before['snr_db'] or
                            [v['view'] for v in r['views']] != [v['view'] for v in before['views']]):
                        raise RuntimeError('stage evaluation cameras/noise changed')
                    if r['total_channel_uses'] != before['total_channel_uses']:
                        raise RuntimeError('fixed-budget stage accounting changed')
                    if r['layout'] != 'mixed' and int(r['layout']) < index:
                        errors = {k: abs(r[k]-before[k]) for k in base.METRICS if k in r}
                        errors.update({k: abs(r['diagnostics'][k]-before['diagnostics'][k])
                                       for k in ('latent_mse', 'latent_nmse')})
                        passed = all(abs(r[k]-before[k]) <= 1e-8+1e-6*abs(before[k])
                                     for k in base.METRICS if k in r)
                        passed &= all(abs(r['diagnostics'][k]-before['diagnostics'][k]) <=
                                      1e-10+1e-6*abs(before['diagnostics'][k])
                                      for k in ('latent_mse', 'latent_nmse'))
                        checks.append(dict(endpoint=endpoint, scene=spec['name'], path=r['path'],
                                           layout=r['layout'], trial=r['trial'], passed=passed,
                                           absolute_metric_changes=errors))
                paired[endpoint, key] = r
            for path in ('finite_noiseless', 'awgn'):
                for layout in LAYOUTS:
                    selected = [r for r in records if r['path'] == path and r['layout'] == layout]
                    row = dict(endpoint=endpoint, scene=spec['name'], role=spec['role'], path=path,
                               layout=layout, trials=len(selected), total_channel_uses=selected[0]['total_channel_uses'])
                    for k in base.METRICS:
                        if k in selected[0]:
                            row[k] = statistics.mean(r[k] for r in selected)
                            row[k+'_std_trials'] = statistics.pstdev(r[k] for r in selected)
                    for k in ('latent_mse', 'latent_nmse', 'latent_target_power'):
                        row[k] = statistics.mean(r['diagnostics'][k] for r in selected)
                    rows.append(row)
    lookup = {(r['endpoint'], r['scene'], r['path'], r['layout']): r for r in rows}
    for r in rows:
        index = ENDPOINTS.index(r['endpoint'])
        if index:
            for name, endpoint in [('change_from_initial', 'initial'), ('change_from_previous_stage', ENDPOINTS[index-1])]:
                before = lookup[endpoint, r['scene'], r['path'], r['layout']]
                r[name] = {k: r[k]-before[k] for k in base.METRICS+('latent_mse', 'latent_nmse') if k in r}
    report = dict(status='complete' if all(c['passed'] for c in checks) else 'lower_tier_invariance_failed',
        rows=rows, lower_tiers_unchanged=all(c['passed'] for c in checks), lower_tier_checks=checks,
        selection='fixed_stage_endpoints', cost_protocol='digital_side_file_v1',
        quality_conditioning='reliable-side assumption; actual delivery reported separately',
        actual_delivery={p.parent.name: read(p)['results'] for p in (out/'actual_delivery').glob('*/summary.json')})
    json_write(out/'review_summary.json', report)
    if not report['lower_tiers_unchanged']:
        raise RuntimeError('lower-tier invariance failed; inspect review_summary.json')
    return report


def main(args, max_updates=None):
    saved, metadata, cfg, specs, heldouts, plan = inputs(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return plan
    out = Path(args.out).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError('use a new OUT or --resume')
    if args.resume and (not (out/'checkpoints/latest/codec.pt').is_file() or read(out/'plan.json') != plan):
        raise ValueError('resume requires the same source, settings and latest checkpoint')
    if args.resume and (out/'complete.json').exists():
        write_review(out); return read(out/'complete.json')
    device = device_for(args.device)
    torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    if args.resume:
        saved = torch.load(out/'checkpoints/latest/codec.pt', map_location='cpu', weights_only=True)
        progress = saved['training']['progress']
        if progress['training_family'] != FAMILY or saved['config'] != cfg.to_dict():
            raise ValueError('checkpoint experiment/config mismatch')
    model = build_codec(config_from_dict(saved['config'])).to(device)
    model.load_state_dict(saved['state_dict']); model.set_training_snrs([10.])
    if not args.resume:
        progress = dict(training_family=FAMILY, step=0, phase='channel_render', phase_step=0,
            stage=1, stage_step=0, channel_trained=True, train_snrs=[10.], phase_complete=False,
            evaluated=[], completed_stages=[], next_scene=0, monitor_done_step=-1,
            view_schedulers={}, view_coverage={f'q{q}': {} for q in (1, 2, 3)},
            scene_completed={f'q{q}': {s['name']: 0 for s in specs} for q in (1, 2, 3)},
            stage_frozen_sha256=base.frozen_digest(model.state_dict(), prefixes(1)))
        json_write(out/'plan.json', plan)
    optimizer = configure_optimizer(model, progress['stage'], args.lr)
    if args.resume:
        optimizer.load_state_dict(saved['training']['optimizer'])
        rng = saved['training']['rng']; old_records = saved['training']['scene_record']
        repair_logs(out, progress['step'])
    else:
        rng = capture_rng(); old_records = None
    del saved

    def verify_frozen():
        if base.frozen_digest(model.state_dict(), ALL_STAGES) != plan['frozen_state_sha256']:
            raise RuntimeError('semantic codec, neighbor refinement or statistics changed')
        if base.frozen_digest(model.state_dict(), prefixes(progress['stage'])) != progress['stage_frozen_sha256']:
            raise RuntimeError('inactive channel stage changed')

    verify_frozen()
    scenes = []
    for spec in specs:
        raw, degree = read_ply(spec['ply'])
        scenes.append(make_refinement_scene(spec, raw, degree, model, args, out,
            metadata['scene_records'][spec['name']], plan['camera_splits'][spec['name']]))
        del raw
    records = {s.name: s.record for s in scenes}
    if len({s.fingerprint for s in scenes}) != len(scenes):
        raise ValueError('training PLY aliases')
    if old_records is not None: validate_scene(old_records, records)
    else: json_write(out/'training.json', dict(plan, scene_records=records))
    for s in scenes: json_write(s.out/'scene.json', dict(s.record, **s.spec))
    (out/'metrics').mkdir(parents=True, exist_ok=True)
    restore_rng(rng)
    perceptual, targets = None, {}

    def view_cycles():
        return {s.name: RefinementViewCycle(records[s.name]['train_view_names'], args.seed,
                s.name, progress['stage'], progress['view_schedulers'].get(s.name)) for s in scenes}

    cycles = view_cycles()
    for s in scenes:
        if cycles[s.name].updates != progress['scene_completed'][f'q{progress["stage"]}'][s.name]:
            raise ValueError('view scheduler and scene update count disagree')
        print(f'{s.name} cameras: training={len(s.cameras)}, per step={args.views_per_step}, '
              f'validation={len(s.val_cameras)} x {args.trials} trials, test={len(s.test_cameras)}.', flush=True)

    def save(label='latest'):
        verify_frozen()
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt', model, optimizer,
                                 progress, capture_rng(), records)
        json_write(out/'progress.json', dict(progress, status='running'))

    def prepare_perceptual():
        nonlocal perceptual
        if args.lpips and perceptual is None:
            with preserved_rng(device): perceptual = LPIPSEvaluator(device)

    def evaluate(label):
        if label in progress['evaluated']: return
        prepare_perceptual(); save()
        with preserved_rng(device):
            for s in scenes: evaluate_matrix(model, s, args, out, label, perceptual=perceptual)
            for spec in heldouts:
                raw, degree = read_ply(spec['ply'])
                s = make_scene(spec, raw, degree, model, args, out)
                if s.fingerprint in {x.fingerprint for x in scenes}:
                    raise ValueError('heldout/training PLY alias')
                path = s.out/'scene.json'; record = dict(s.record, **s.spec)
                if path.exists(): validate_scene(read(path), record)
                else: json_write(path, record)
                evaluate_matrix(model, s, args, out, label, perceptual=perceptual)
                del raw, s
        progress['evaluated'].append(label); save()

    def monitor():
        visits = progress['stage_step']//len(scenes)
        if (progress['next_scene'] != 0 or progress['monitor_done_step'] == progress['step'] or
                (visits % args.validate_every and visits != args.steps_per_scene)):
            return
        options = copy.copy(args)
        options.eval_layouts = [str(progress['stage'])]; options.eval_paths = ['awgn']
        label = f'monitor_q{progress["stage"]}_{visits:06d}'
        save()
        log = out/'metrics/validation.jsonl'
        logged = {(r['step'], r['scene']) for line in log.read_text(encoding='utf8').splitlines()
                  if line.strip() and (r := json.loads(line))} if log.exists() else set()
        with preserved_rng(device):
            for s in scenes:
                rows = evaluate_matrix(model, s, options, out, label, scope='validation')
                row = monitor_metrics(rows, args, s, progress['stage'])
                if (progress['step'], s.name) not in logged:
                    append_json(log, dict(step=progress['step'], stage=progress['stage'], scene_step=visits,
                        scene=s.name, **row,
                        scope='validation', selection=False))
                    print(f'validation q{progress["stage"]} {s.name} {visits}/{args.steps_per_scene}: '
                          f'source PSNR={row["source_psnr"]:.4f} +/- {row["source_psnr_std_trials"]:.4f}, '
                          f'photo PSNR={row["photo_psnr"]:.4f} +/- {row["photo_psnr_std_trials"]:.4f}; '
                          f'{row["validation_views"]} views x {row["trials"]} trials.', flush=True)
        progress['monitor_done_step'] = progress['step']; save()

    if not args.resume: save('initial'); save()
    if not progress.get('digital_preflight_done'):
        from comparison_baselines.run import _execute
        _execute(args.digital_python, 'comparison_baselines.digital_link',
                 ['self-test', '--device', args.digital_device, '--snr-db', 30], out/'metrics/digital_preflight.log')
        progress['digital_preflight_done'] = True; save()
    evaluate('initial'); monitor()
    print(f'Sequential q1 -> q2 -> q3: {args.steps_per_scene} updates/scene/STAGE; '
          f'{plan["total_updates"]} total; {args.views_per_step} rotating views/update; '
          f'10dB, LR={args.lr:g}.', flush=True)
    updated = 0
    while True:
        tier = progress['stage']
        if progress['stage_step'] == plan['stage_updates'][f'q{tier}']:
            monitor(); evaluate(ENDPOINTS[tier])
            if tier not in progress['completed_stages']: progress['completed_stages'].append(tier)
            if tier == 3: progress['phase_complete'] = True
            save('final' if tier == 3 else f'after_q{tier}'); save()
            if tier == 3: break
            verify_frozen()
            progress.update(stage=tier+1, stage_step=0, next_scene=0, view_schedulers={},
                stage_frozen_sha256=base.frozen_digest(model.state_dict(), prefixes(tier+1)))
            cycles = view_cycles()
            optimizer = configure_optimizer(model, tier+1, args.lr)
            save()
            print(f'Begin q{tier+1}: lower stages fixed; new Adam for stage {tier+1}.', flush=True)
            continue
        s = scenes[progress['next_scene']]
        step = progress['step']+1; visit = progress['scene_completed'][f'q{tier}'][s.name]+1
        if s.name not in targets:
            print(f'Caching fixed semantic latent: {s.name}', flush=True)
            with preserved_rng(device): targets[s.name] = [b['y'] for b in cache_teacher(model, s)]
        indices = cycles[s.name].next(args.views_per_step)
        task = MultiViewRenderTask([s.cameras[i] for i in indices], s.reference, s.degree, s.white_background)
        optimizer.zero_grad(set_to_none=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss, stats = cached_render_step(model, base.CachedQ3Batches(s, targets[s.name], tier),
                                         s.geometry, 10., task, args.latent_weight)
        norm, gradient_stats = clip_codec_gradients(model, mode='none')
        optimizer.step(); stats.update(gradient_stats, grad_norm=float(norm))
        if device.type == 'cuda': torch.cuda.synchronize(device)
        seconds = time.perf_counter()-started
        progress.update(step=step, phase_step=step, stage_step=progress['stage_step']+1,
                        next_scene=(progress['next_scene']+1)%len(scenes))
        progress['scene_completed'][f'q{tier}'][s.name] = visit
        progress['view_schedulers'][s.name] = cycles[s.name].state_dict()
        coverage = cycles[s.name].coverage()
        progress['view_coverage'][f'q{tier}'][s.name] = coverage
        append_json(out/'metrics/loss.jsonl', dict(stats, step=step, stage=tier, scene=s.name, scene_step=visit,
            layout=str(tier), snr_db=10., loss=float(loss), lr=args.lr, training_view_indices=indices,
            training_view_names=[str(s.cameras[i].image_name) for i in indices], view_coverage=coverage,
            step_seconds=seconds, weighted_latent_loss=args.latent_weight*stats['latent_mse']))
        if visit == 1 or visit % 10 == 0 or visit == args.steps_per_scene:
            print(f'q{tier} {s.name} {visit}/{args.steps_per_scene} (global {step}/{plan["total_updates"]}): '
                  f'render MSE={stats["render_loss"]:.7g}, latent MSE={stats["latent_mse"]:.7g}, '
                  f'views={len(indices)}, covered={coverage["covered"]}/{coverage["pool"]}, '
                  f'mean visits={coverage["mean_visits"]:.2f}, sec={seconds:.2f}', flush=True)
        if progress['next_scene'] == 0 and (visit % args.save_every == 0 or visit == args.steps_per_scene): save()
        monitor(); updated += 1
        if max_updates is not None and updated >= max_updates:
            save(); return dict(progress, status='interrupted')
    prepare_perceptual()
    for name in args.delivery_scenes:
        s = next((s for s in scenes if s.name == name), None)
        if s is None:
            spec = next(s for s in heldouts if s['name'] == name)
            raw, degree = read_ply(spec['ply']); s = make_scene(spec, raw, degree, model, args, out)
            del raw
        actual_delivery(model, s, args, out, perceptual=perceptual)
    report = summarize(out, plan)
    result = dict(status='complete', version=FAMILY, step=progress['step'], scene_completed=progress['scene_completed'],
        view_coverage=progress['view_coverage'],
        completed_stages=progress['completed_stages'], frozen_state_verified=True,
        lower_tiers_unchanged=report['lower_tiers_unchanged'], selection='fixed_stage_endpoints',
        heldout_optimizer_updates=0, actual_side_delivery_completed=True, codec_id=model_id(model))
    json_write(out/'complete.json', result); json_write(out/'progress.json', dict(progress, status='complete'))
    write_review(out); print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__': main(parser().parse_args())
