"""Refine the last successive channel stage at 10 dB with fixed q1/q2 paths."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

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
from .successive_multiscene_inputs import make_scene
from .successive_multisnr import write_review
from .successive_transfer import load_source
from .transport import model_id

FAMILY = 'successive_q3_refinement_v1'
TRAINABLE = ('channel_encoder.stages.2.', 'channel_decoder.stages.2.')
METRICS = ('source_mse', 'photo_mse', 'source_psnr', 'photo_psnr', 'source_ssim',
           'photo_ssim', 'source_lpips', 'photo_lpips')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--scenes', nargs='+', help='default: original nine training scenes')
    p.add_argument('--heldout-scenes', nargs='*', default=['room'])
    for name, value in [('steps-per-scene', 250), ('validate-every', 50), ('save-every', 25),
                        ('trials', 4), ('cpu-threads', 4), ('seed', 42)]:
        p.add_argument('--'+name, type=int, default=value)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--latent-weight', type=float, default=1.)
    p.add_argument('--device', default='cuda')
    p.add_argument('--digital-python', required=True)
    p.add_argument('--digital-device', default='cuda')
    p.add_argument('--delivery-scenes', nargs='+', default=['truck'])
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def frozen_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith(TRAINABLE):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def configure_optimizer(model, lr):
    if (model.cfg.channel_variant != 'successive' or tuple(model.cfg.rates) != (0, 8, 24, 48)
            or not math.isfinite(lr) or lr <= 0):
        raise ValueError('q3 refinement requires successive 8/24/48 and a positive finite LR')
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    groups = []
    for prefix, module in zip(TRAINABLE, (model.channel_encoder.stages[2], model.channel_decoder.stages[2])):
        module.requires_grad_(True)
        groups.append(dict(params=list(module.parameters()), lr=lr, name=prefix.rstrip('.')))
    return torch.optim.Adam(groups)


def inputs(args):
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('use an output separate from the source run')
    saved, metadata, cfg, identity = load_source(run)
    if saved['training']['progress'].get('train_snrs') != [10.]:
        raise ValueError('initialize from the completed 10 dB multi-scene codec')
    lookup = {s['name']: s for s in metadata['selected_scenes']}
    names = args.scenes or list(lookup)
    held = metadata.get('heldout_scene', [])
    if isinstance(held, dict):
        held = [held]
    held_lookup = {s['name']: s for s in held}
    for selected, available in ((names, lookup), (args.heldout_scenes, held_lookup)):
        if len(set(selected)) != len(selected) or not set(selected) <= set(available):
            raise ValueError('scene names must be unique members of the original split')
    if set(args.heldout_scenes) & set(identity['training_scene_history']):
        raise ValueError('heldout scene appears in source training history')
    specs = [dict(lookup[n], prior='none') for n in names]
    heldouts = [dict(held_lookup[n], prior='none') for n in args.heldout_scenes]
    if not set(args.delivery_scenes) <= set(names+args.heldout_scenes):
        raise ValueError('actual delivery scenes must be evaluated by this experiment')
    if len(set(args.delivery_scenes)) != len(args.delivery_scenes):
        raise ValueError('duplicate delivery scene')
    for spec in specs+heldouts:
        for key in ('ply', 'source'):
            if not Path(spec[key]).exists():
                raise FileNotFoundError(f'{spec["name"]}: {spec[key]}')
    if not Path(args.digital_python).is_file():
        raise FileNotFoundError(args.digital_python)
    if min(args.steps_per_scene, args.validate_every, args.save_every, args.trials,
           args.cpu_threads) < 1 or any(not math.isfinite(x) or x <= 0 for x in (args.lr, args.latent_weight)):
        raise ValueError('positive budgets and finite positive LR/latent weight required')
    for key in ('blocks_per_batch', 'views_per_step', 'resolution', 'images', 'white_background',
                'validation_views', 'test_views'):
        setattr(args, key, metadata['settings'][key])
    args.net_bits_per_use = 2.  # Legacy scene constructor; reporting uses actual digital file framing.
    args.snr_range = [10., 10.]
    args.eval_snrs = [10.]
    args.eval_layouts = ['2', '3']
    args.eval_paths = ['finite_noiseless', 'awgn']
    args.delivery_layouts = ['2', '3']
    args.side_link_policy = 'fixed_16qam_v1'
    settings = {k: v for k, v in vars(args).items()
                if k not in ('run', 'out', 'resume', 'dry_run', 'net_bits_per_use')}
    plan = dict(version=FAMILY, **identity, source_run=str(run), settings=settings,
        selected_scenes=specs, heldout_scene=heldouts, codec_config=cfg.to_dict(),
        total_updates=len(specs)*args.steps_per_scene, trainable_prefixes=list(TRAINABLE),
        frozen_state_sha256=frozen_digest(saved['state_dict']),
        objective='source PLY RGB MSE + latent_weight * latent MSE; uniform q3, 10 dB AWGN',
        optimizer_initialization='new Adam for q3 sender/receiver only; shared across scenes',
        selection='fixed final update; validation/test never select weights',
        cost_protocol='digital_side_file_v1', allocation_training=False,
        cache='immutable semantic y in original Morton blocks; fresh AWGN on every update',
        actual_delivery_scope=dict(scenes=args.delivery_scenes, layouts=['2', '3'], snrs=[10.], trials=args.trials))
    return saved, metadata, cfg, specs, heldouts, json.loads(json.dumps(plan))


class CachedQ3Batches:
    """Cache y only; construct one packed CPU batch at a time in both replay passes."""
    def __init__(self, scene, targets):
        if len(scene.groups) != len(targets):
            raise ValueError('latent cache does not match original batches')
        self.scene, self.targets = scene, targets

    def __len__(self):
        return len(self.targets)

    def __iter__(self):
        for features, ids, y in zip(self.scene.groups, self.scene.group_ids, self.targets):
            if features.shape[:2] != y.shape[:2] or features.shape[:2] != ids.shape:
                raise ValueError('cached block layout changed')
            yield torch.cat((features, y), -1), (ids >= 0).long()*3


def step_optimizer(model, optimizer, before):
    norm, stats = clip_codec_gradients(model, mode='none')
    optimizer.step()
    stats['q3_parameter_updates'] = {}
    for prefix in TRAINABLE:
        values = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        delta = torch.stack([(p.detach()-before[n]).square().sum() for n, p in values]).sum().sqrt()
        stats['q3_parameter_updates'][prefix.rstrip('.')] = float(delta)
    stats['grad_norm'] = float(norm)
    return stats


def summarize(out, plan):
    rows, matched, checks = [], {}, []
    for endpoint in ('initial', 'final'):
        for spec in plan['selected_scenes']+plan['heldout_scene']:
            name = spec['name']
            source = read(out/'evaluation'/endpoint/name/'summary.json')['rows']
            for r in source:
                key = (name, r['path'], r['layout'], r['trial'])
                if endpoint == 'initial':
                    matched[key] = r
                else:
                    base = matched[key]
                    if (r['seed'] != base['seed'] or r['snr_db'] != base['snr_db'] or
                            [v['view'] for v in r['views']] != [v['view'] for v in base['views']]):
                        raise RuntimeError('endpoint cameras/noise do not match')
                    if r['total_channel_uses'] != base['total_channel_uses']:
                        raise RuntimeError('fixed-budget endpoint accounting changed')
                    if r['layout'] == '2':
                        differences = {k: abs(r[k]-base[k]) for k in METRICS if k in r}
                        ok = all(math.isclose(r[k], base[k], rel_tol=1e-6, abs_tol=1e-8) for k in differences)
                        ok &= all(math.isclose(r['diagnostics'][k], base['diagnostics'][k],
                                              rel_tol=1e-6, abs_tol=1e-10)
                                  for k in ('latent_mse', 'latent_nmse'))
                        checks.append(dict(scene=name, path=r['path'], trial=r['trial'],
                                           passed=ok, absolute_metric_changes=differences))
            for path in ('finite_noiseless', 'awgn'):
                for layout in ('2', '3'):
                    selected = [r for r in source if r['path'] == path and r['layout'] == layout]
                    row = dict(endpoint=endpoint, scene=name, role=spec['role'], path=path, layout=layout,
                        trials=len(selected), total_channel_uses=selected[0]['total_channel_uses'])
                    for key in METRICS:
                        if key in selected[0]:
                            row[key] = statistics.mean(r[key] for r in selected)
                            row[key+'_std_trials'] = statistics.pstdev(r[key] for r in selected)
                    for key in ('latent_mse', 'latent_nmse', 'latent_target_power'):
                        row[key] = statistics.mean(r['diagnostics'][key] for r in selected)
                    rows.append(row)
    lookup = {(r['endpoint'], r['scene'], r['path'], r['layout']): r for r in rows}
    changes = []
    for spec in plan['selected_scenes']+plan['heldout_scene']:
        for path in ('awgn', 'finite_noiseless'):
            a, b = [lookup[e, spec['name'], path, '3'] for e in ('initial', 'final')]
            q2 = lookup['initial', spec['name'], path, '2']
            extra = b['total_channel_uses']-q2['total_channel_uses']
            change = dict(scene=spec['name'], role=spec['role'], path=path,
                q2_source_psnr=q2['source_psnr'], q3_initial_source_psnr=a['source_psnr'],
                q3_final_source_psnr=b['source_psnr'], extra_complex_uses_over_q2=extra,
                change={k: b[k]-a[k] for k in METRICS+('latent_mse', 'latent_nmse') if k in b},
                initial_incremental_mse_per_million_uses=(q2['source_mse']-a['source_mse'])/extra*1e6,
                final_incremental_mse_per_million_uses=(q2['source_mse']-b['source_mse'])/extra*1e6)
            changes.append(change)
            if path == 'awgn':
                print(f'Final {spec["name"]}: q2={q2["source_psnr"]:.4f}, '
                      f'q3={a["source_psnr"]:.4f}->{b["source_psnr"]:.4f}, '
                      f'photo change={change["change"]["photo_psnr"]:+.4f} dB', flush=True)
    report = dict(status='complete' if all(c['passed'] for c in checks) else 'q2_invariance_failed',
        rows=rows, q3_changes=changes, q2_unchanged=all(c['passed'] for c in checks), q2_checks=checks,
        cost_protocol='digital_side_file_v1', selection='fixed_final',
        quality_conditioning='reliable-side assumption; actual delivery reported separately',
        actual_delivery={p.parent.name: read(p)['results'] for p in (out/'actual_delivery').glob('*/summary.json')})
    json_write(out/'review_summary.json', report)
    if not report['q2_unchanged']:
        raise RuntimeError('q2 endpoint check failed; inspect review_summary.json')
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
        # Recreate a review archive if interruption occurred during final packaging.
        write_review(out)
        return read(out/'complete.json')
    device = device_for(args.device)
    torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    if args.resume:
        saved = torch.load(out/'checkpoints/latest/codec.pt', map_location='cpu', weights_only=True)
        progress = saved['training']['progress']
        if progress['training_family'] != FAMILY or saved['config'] != cfg.to_dict():
            raise ValueError('checkpoint belongs to a different experiment')
    model = build_codec(config_from_dict(saved['config'])).to(device)
    model.load_state_dict(saved['state_dict']); model.set_training_snrs([10.])
    optimizer = configure_optimizer(model, args.lr)
    if args.resume:
        optimizer.load_state_dict(saved['training']['optimizer'])
        rng = saved['training']['rng']; old_records = saved['training']['scene_record']
        repair_logs(out, progress['step'])
    else:
        progress = dict(training_family=FAMILY, step=0, phase='channel_render', phase_step=0,
            successive_joint=True, channel_trained=True, q3_only=True, train_snrs=[10.],
            phase_complete=False, initial_done=False, final_done=False, next_scene=0,
            scene_completed={s['name']: 0 for s in specs}, monitor_done_step=-1)
        rng = capture_rng(); old_records = None
        json_write(out/'plan.json', plan)
    del saved
    if frozen_digest(model.state_dict()) != plan['frozen_state_sha256']:
        raise RuntimeError('frozen q1/q2, semantic, neighbor or statistics state changed')
    scenes = []
    for spec in specs:
        raw, degree = read_ply(spec['ply'])
        scenes.append(make_scene(spec, raw, degree, model, args, out, old=metadata['scene_records'][spec['name']]))
        del raw
    records = {s.name: s.record for s in scenes}
    if len({s.fingerprint for s in scenes}) != len(scenes):
        raise ValueError('training PLY aliases')
    if old_records is not None:
        validate_scene(old_records, records)
    else:
        json_write(out/'training.json', dict(plan, scene_records=records))
    for s in scenes:
        json_write(s.out/'scene.json', dict(s.record, **s.spec))
    (out/'metrics').mkdir(parents=True, exist_ok=True)
    restore_rng(rng)
    perceptual = None
    targets = {}

    def save(label='latest'):
        if frozen_digest(model.state_dict()) != plan['frozen_state_sha256']:
            raise RuntimeError('frozen state changed')
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt', model, optimizer,
                                 progress, capture_rng(), records)
        json_write(out/'progress.json', dict(progress, status='running'))

    def prepare_perceptual():
        nonlocal perceptual
        if args.lpips and perceptual is None:
            with preserved_rng(device):
                perceptual = LPIPSEvaluator(device)

    def evaluate(label):
        prepare_perceptual()
        save()
        with preserved_rng(device):
            for s in scenes:
                evaluate_matrix(model, s, args, out, label, perceptual=perceptual)
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
        progress[label+'_done'] = True; save()

    def monitor():
        visits = progress['step']//len(scenes)
        if (progress['next_scene'] != 0 or progress['monitor_done_step'] == progress['step']
                or (visits % args.validate_every and progress['step'] != plan['total_updates'])):
            return
        options = copy.copy(args); options.trials = 1
        options.eval_layouts = ['3']; options.eval_paths = ['awgn']
        label = f'monitor_{visits:06d}'
        save()
        log = out/'metrics/validation.jsonl'
        logged = {(r['step'], r['scene']) for line in log.read_text(encoding='utf8').splitlines()
                  if line.strip() and (r := json.loads(line))} if log.exists() else set()
        with preserved_rng(device):
            for s in scenes:
                result = evaluate_matrix(model, s, options, out, label, scope='validation')
                row = result[0]
                if (progress['step'], s.name) not in logged:
                    append_json(log, dict(step=progress['step'], scene_step=visits,
                        scene=s.name, source_psnr=row['source_psnr'], photo_psnr=row['photo_psnr'],
                        source_mse=row['source_mse'], latent_mse=row['diagnostics']['latent_mse'],
                        scope='validation', selection=False))
        progress['monitor_done_step'] = progress['step']; save()

    if not args.resume:
        save('initial'); save()
    if not progress.get('digital_preflight_done'):
        from comparison_baselines.run import _execute
        _execute(args.digital_python, 'comparison_baselines.digital_link',
                 ['self-test', '--device', args.digital_device, '--snr-db', 30], out/'metrics/digital_preflight.log')
        progress['digital_preflight_done'] = True; save()
    if not progress['initial_done']:
        evaluate('initial')
    monitor()
    print(f'Q3-only: {args.steps_per_scene} updates/scene, {plan["total_updates"]} total, '
          f'10 dB, LR={args.lr:g}, latent weight={args.latent_weight:g}; fixed final endpoint', flush=True)
    updated = 0
    while progress['step'] < plan['total_updates']:
        index = progress['next_scene']; s = scenes[index]
        step = progress['step']+1; visit = progress['scene_completed'][s.name]+1
        if s.name not in targets:
            print(f'Caching frozen semantic features: {s.name}', flush=True)
            with preserved_rng(device):
                targets[s.name] = [b['y'] for b in cache_teacher(model, s)]
        indices = random.Random(args.seed+400009*step).sample(range(len(s.cameras)), args.views_per_step)
        task = MultiViewRenderTask([s.cameras[i] for i in indices], s.reference, s.degree, s.white_background)
        optimizer.zero_grad(set_to_none=True)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if device.type == 'cuda': torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss, stats = cached_render_step(model, CachedQ3Batches(s, targets[s.name]),
                                         s.geometry, 10., task, args.latent_weight)
        stats.update(step_optimizer(model, optimizer, before))
        if device.type == 'cuda': torch.cuda.synchronize(device)
        seconds = time.perf_counter()-started
        progress.update(step=step, phase_step=step, next_scene=(index+1)%len(scenes),
                        phase_complete=step == plan['total_updates'])
        progress['scene_completed'][s.name] = visit
        append_json(out/'metrics/loss.jsonl', dict(stats, step=step, scene=s.name, scene_step=visit,
            layout='3', snr_db=10., loss=float(loss), lr=args.lr, training_view_indices=indices,
            step_seconds=seconds, weighted_latent_loss=args.latent_weight*stats['latent_mse']))
        if visit == 1 or visit % 10 == 0:
            print(f'q3 {s.name} {visit}/{args.steps_per_scene} (global {step}/{plan["total_updates"]}): '
                  f'render MSE={stats["render_loss"]:.7g}, latent MSE={stats["latent_mse"]:.7g}, '
                  f'grad={stats["grad_norm"]:.5g}, sec={seconds:.2f}', flush=True)
        if progress['next_scene'] == 0 and (visit % args.save_every == 0 or progress['phase_complete']):
            save()
        monitor()
        updated += 1
        if max_updates is not None and updated >= max_updates:
            save(); return dict(progress, status='interrupted')
    if not progress['final_done']:
        evaluate('final')
    save('final'); save()
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
        frozen_state_verified=True, q2_unchanged=report['q2_unchanged'], selection='fixed_final',
        heldout_optimizer_updates=0, actual_side_delivery_completed=True, codec_id=model_id(model))
    json_write(out/'complete.json', result)
    json_write(out/'progress.json', dict(progress, status='complete'))
    write_review(out)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
