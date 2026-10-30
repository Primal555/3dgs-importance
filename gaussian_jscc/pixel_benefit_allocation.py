"""Independent pixel-benefit pretraining and frozen-codec allocation."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .allocation_distortion import ImageDistortion, calibrate
from .allocation_mirror import ViewWindowScheduler
from .budgeted_allocation import parser as base_parser, validate as base_validate, deploy
from .budgeted_allocation_evaluation import ExactDeploymentCost, evaluate_map, tier_hash
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_state import repair, upsert_jsonl
from .cli import device_for, seed_all
from .data import read_ply
from .model_factory import build_codec
from .modular_checkpoint import capture_rng, restore_rng
from .modular_experiment import json_write
from .optimization import preserved_rng
from .pixel_benefit_learning import PixelLearner
from .pixel_benefit_replay import PixelReplay
from .pixel_benefit_window import PixelWindow
from .render_objective import spaced_indices
from .render_validation import append_json
from .response_value import ResponseMirrorTable
from .response_value_window import ResponseScheduler
from .successive_multiscene_inputs import make_scene
from .transport import model_id

FAMILY = 'pixel_benefit_allocation_v1'


def _atomic_manifest(path, manifest):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, allow_nan=False)
        stream.flush()
    temporary.replace(path)


def _read_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(f'checkpoint replay manifest missing: {path}')
    with path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    if (not isinstance(manifest, dict) or manifest.get('version') != 1 or
        not isinstance(manifest.get('root'), str) or not isinstance(manifest.get('store_id'), str) or
        not isinstance(manifest.get('training'), list) or not isinstance(manifest.get('validation'), list) or
        any(not isinstance(v, str) for k in ('training', 'validation') for v in manifest[k])):
        raise ValueError(f'invalid checkpoint replay manifest: {path}')
    return manifest


def save_replay_checkpoint(path, state, replay):
    """Atomically replace a checkpoint while protecting both generations of shards."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = path.parent.parent
    for checkpoint in checkpoint_root.glob('*/state.pt'):
        _read_manifest(checkpoint.parent/'replay.json')
    manifest_path = path.parent/'replay.json'
    new = replay.manifest()
    expected = state['replay']
    if (expected['root'] != new['root'] or expected['store_id'] != new['store_id'] or
        any([entry['path'] for entry in expected[name]] != new[name] for name in ('training', 'validation'))):
        raise ValueError('checkpoint replay state and manifest mismatch')
    if manifest_path.exists():
        old = _read_manifest(manifest_path)
        if any(old[key] != new[key] for key in ('version', 'root', 'store_id')):
            raise ValueError('checkpoint replay manifest identity mismatch')
        union = dict(new, **{name: list(dict.fromkeys(old[name]+new[name]))
                             for name in ('training', 'validation')})
    else:
        union = new
    _atomic_manifest(manifest_path, union)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
    _atomic_manifest(manifest_path, new)
    manifests = [_read_manifest(checkpoint.parent/'replay.json')
                 for checkpoint in checkpoint_root.glob('*/state.pt')]
    return replay.garbage_collect(manifests)


def parser():
    p = base_parser()
    p.description = __doc__
    p.set_defaults(audit_every=0, audit_samples=0)
    p.add_argument('--phase', choices=('pretrain', 'allocation'), default='pretrain')
    for name, default in (('pretrain-windows', 200), ('fit-windows', 50), ('label-groups', 64),
                          ('net-updates', 16), ('batch-groups', 4), ('tile-batch', 8),
                          ('replay-capacity', 8192), ('validation-capacity', 512), ('hidden', 32),
                          ('candidate-batch', 16), ('benefit-validate-every', 50)):
        p.add_argument('--'+name, type=int, default=default)
    for name, default in (('benefit-lr', 1e-4), ('structure-share', .2), ('max-log-odds', .25)):
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--distortion', choices=('mse', 'mse_ssim'), default='mse_ssim')
    p.add_argument('--lambda-s', type=float)
    p.add_argument('--renderer-backend', choices=('cuda', 'cpu'), default='cuda')
    return p


def validate(args):
    base_validate(args)
    if args.beta is None:
        raise ValueError('pixel allocation requires --beta')
    if args.phase == 'allocation' and not args.resume:
        raise ValueError('allocation requires completed pretraining and --resume')
    if (args.fit_windows < 1 or args.pretrain_windows <= args.fit_windows or
        min(args.label_groups, args.net_updates, args.batch_groups, args.tile_batch,
            args.replay_capacity, args.validation_capacity, args.hidden, args.candidate_batch,
            args.benefit_validate_every) < 1 or
        not math.isfinite(args.benefit_lr) or args.benefit_lr <= 0 or
        not 0 < args.structure_share < 1 or args.max_log_odds <= 0):
        raise ValueError('invalid pixel training configuration')
    if args.lambda_s is not None and (not math.isfinite(args.lambda_s) or args.lambda_s < 0):
        raise ValueError('invalid lambda_s')
    if args.distortion == 'mse' and args.lambda_s not in (None, 0.):
        raise ValueError('MSE requires zero structure weight')


def preflight(model, scene, camera, args, reference):
    from .pixel_benefit_preflight import verify_renderer
    return verify_renderer(model, scene, args, reference, camera=camera)


def main(args, max_updates=None, *, family=None, learner_factory=None, window_factory=None, preflight_fn=None,
         replay_factory=None):
    family = family or FAMILY
    window_factory = window_factory or PixelWindow
    preflight_fn = preflight_fn or preflight
    validate(args)
    saved, cfg, spec, old, options, source = load_inputs(args)
    if tuple(cfg.rates) != (0, 8, 24, 48):
        raise ValueError('requires 0/8/24/48 source rates')
    settings = {k: v for k, v in vars(args).items() if k not in
                ('out', 'resume', 'dry_run', 'device', 'phase', 'allocation_image_objective')}
    for key in ('budget_total_uses', 'budget_ratio_to_q3', 'initial_price', 'dual_lr', 'ema_decay'):
        settings.pop(key, None)
    plan = json.loads(json.dumps(dict(version=family, source=source, spec=spec, settings=settings,
                                     codec_config=cfg.to_dict(), writes_output=False)))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    out = Path(args.out).resolve()
    latest = out/'checkpoints/latest/state.pt'
    if out.exists() and not args.resume:
        raise FileExistsError('use NEW output or --resume')
    if args.resume and not latest.exists():
        raise FileNotFoundError('missing exact pixel checkpoint')
    device = device_for(args.device)
    if device.type != args.renderer_backend:
        raise ValueError('renderer backend must match device')
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    model = build_codec(cfg).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval().requires_grad_(False)
    model.set_training_snrs([10.])
    identity = model_id(model)
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, options, out, old=old)
    model.set_scene_geometry(scene.geometry)
    chosen = set(spaced_indices(len(scene.cameras), args.calibration_views))
    calibration = [c for i, c in enumerate(scene.cameras) if i in chosen]
    training = [c for i, c in enumerate(scene.cameras) if i not in chosen]
    scene_record = dict(scene.record, allocation_train_view_names=[str(c.image_name) for c in training],
                        calibration_view_names=[str(c.image_name) for c in calibration])
    table = ResponseMirrorTable(len(scene.raw), max_log_odds=args.max_log_odds).to(device)
    blocks = ResponseScheduler(scene.group_ids, seed=args.seed)
    views = ViewWindowScheduler(scene_record['allocation_train_view_names'], args.seed+9001)
    replay = (replay_factory or PixelReplay)(out/'replay', seed=args.seed+123, max_groups=args.replay_capacity,
                                            validation_groups=args.validation_capacity)
    learner = (learner_factory(replay, device, args) if learner_factory else
               PixelLearner(replay, device, hidden=args.hidden, lr=args.benefit_lr, tile_batch=args.tile_batch))
    meter = ExactDeploymentCost(model, scene, identity, args.snr, options.net_bits_per_use)
    if args.resume:
        state = torch.load(latest, map_location='cpu', weights_only=True)
        if state['plan'] != plan or state['scene_record'] != scene_record or state['family'] != family:
            raise ValueError('exact resume source/settings/scene mismatch')
        model.load_state_dict(state['state_dict']); table.load_state_dict(state['table'])
        blocks.load_state_dict(state['blocks']); views.load_state_dict(state['views'])
        replay.load_state_dict(state['replay']); learner.load_state_dict(state['learner'])
        reference, progress, q = state['objective_reference'], state['progress'], state['latest_q']
        window = window_factory(model, scene, table, blocks, reference, args)
        window.load_state_dict(state['window'])
        repair(out, progress['step']); restore_rng(state['rng'])
    else:
        full = torch.full((len(scene.raw),), 3, dtype=torch.long)
        full_cost = meter.details(full)
        calibration_args = argparse.Namespace(**vars(args)); calibration_args.lpips = False
        calibration_args.allocation_image_objective = ImageDistortion(args.distortion, 0.)
        baseline = evaluate_map(model, scene, full, calibration, calibration_args,
                                folder=out/'evaluation/q3_reference', label='q3_reference')
        reference = dict(q3_cost=full_cost, beta=args.beta, c_ref=full_cost['total_channel_uses'],
            calibration_views=scene_record['calibration_view_names'], calibration_trials=args.trials,
            **calibrate(baseline['source_objective_mse'], baseline['source_structure_error'],
                        mode=args.distortion, share=args.structure_share, lambda_s=args.lambda_s))
        q = deploy(table, args)
        progress = dict(step=0, phase='pretrain', pretrain_windows_done=0, allocation_step=0,
                        pretrain_complete=False, pending=['initial'])
        window = window_factory(model, scene, table, blocks, reference, args)
        json_write(out/'plan.json', plan)
        json_write(out/'training.json', dict(plan, **scene_record, objective_reference=reference,
                                            deployment_draws=10, deployment_seed=args.seed))
    with preserved_rng(device):
        preflight_result = preflight_fn(model, scene, training[0], args, reference)
    json_write(out/'preflight.json', preflight_result)
    args.allocation_image_objective = ImageDistortion(args.distortion, reference['lambda_s'])

    gc_stats = {'removed': 0}
    def save(label='latest'):
        if model_id(model) != identity:
            raise RuntimeError('frozen codec changed')
        state = dict(version=1, family=family, state_dict=model.state_dict(), table=table.state_dict(),
            blocks=blocks.state_dict(), views=views.state_dict(), learner=learner.state_dict(),
            replay=replay.state_dict(), window=window.state_dict(), progress=dict(progress), plan=plan,
            scene_record=scene_record, objective_reference=reference, latest_q=q.detach().cpu(), rng=capture_rng())
        path = out/'checkpoints'/label/'state.pt'
        gc_stats['removed'] += save_replay_checkpoint(path, state, replay)

    def measure(label, cameras, images=False, map_q=None):
        selected = q if map_q is None else map_q
        row = evaluate_map(model, scene, selected, cameras, args, save_images=images,
                           folder=out/'evaluation'/label, label=label)
        cost = meter.details(selected)
        distortion = row['source_distortion']
        channel_uses = cost['total_channel_uses']
        upsert_jsonl(out/'metrics/validation.jsonl', dict(row, **cost, step=progress['step'], event=label,
            beta=args.beta, d_ref=reference['d_ref'], c_ref=reference['c_ref'], lambda_s=reference['lambda_s'],
            normalized_distortion=distortion/reference['d_ref'], normalized_cost=channel_uses/reference['c_ref'],
            rd_score=distortion/reference['d_ref']+args.beta*channel_uses/reference['c_ref']))
        return cost

    def refresh_side(cost):
        kept = int((q > 0).sum())
        reference['side_proxy'] = dict(xyz_per_retained=cost['coordinate_channel_uses']/kept if kept else
            reference['q3_cost']['coordinate_channel_uses']/len(q),
            scale_per_retained=cost['scale_content_bits']/options.net_bits_per_use/kept if kept else 0.,
            fixed_uses=cost['metadata_channel_uses']+cost['scale_framing_bytes']*8/options.net_bits_per_use,
            bits_per_use=options.net_bits_per_use)

    def pending():
        while progress['pending']:
            event = progress['pending'][0]
            if event == 'initial':
                refresh_side(measure('calibration_000000', calibration)); measure('initial', scene.test_cameras, True)
            elif event == 'validation':
                report = learner.validate()
                upsert_jsonl(out/'metrics/benefit_validation.jsonl', dict(report, step=progress['step'], event=event))
                print(f"value validation step={progress['step']} groups={report['validation_groups']}", flush=True)
            elif event == 'calibration':
                measure(f'calibration_{progress["step"]:06d}', calibration)
            elif event == 'control':
                refresh_side(meter.details(q))
            else:
                raise ValueError('unknown pending event')
            progress['pending'].pop(0)
            save()

    if not args.resume:
        save('initial'); save()
    if args.phase == 'pretrain' and progress['phase'] == 'allocation':
        raise ValueError('resume allocation with --phase allocation')
    if args.phase == 'allocation' and not progress['pretrain_complete']:
        raise ValueError('finish pretraining before allocation')
    pending()
    if args.phase == 'allocation' and progress['phase'] == 'pretrain':
        progress['phase'] = 'allocation'; save()
    target = args.pretrain_windows if args.phase == 'pretrain' else args.allocation_steps
    counter = 'pretrain_windows_done' if args.phase == 'pretrain' else 'allocation_step'
    updates = 0
    while progress[counter] < target:
        step = progress['step']+1
        camera_indices = views.next(args.window_views)
        seed_all(args.seed+2000003+step)
        started = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        result = window.collect(step, [training[i] for i in camera_indices], learner, replay,
                                pretrain=args.phase == 'pretrain')
        if args.phase == 'allocation':
            table.update(result['keep_costs'], result['tier_ids'], result['tier_costs'],
                         args.keep_lr, args.tier_lr, len(scene.raw))
        progress['step'] = step; progress[counter] += 1
        if args.phase == 'pretrain' and progress[counter] == args.fit_windows:
            learner.fit_scales(); replay.freeze_validation()
        elif learner.scales is not None:
            learning = learner.train_updates(args.net_updates, args.batch_groups)
            result['metrics'].update(learning)
        q = deploy(table, args)
        probabilities = table.probabilities(torch.arange(len(q), device=device), args.snr)
        payload = probabilities.new_tensor(model.cfg.rates)
        side = reference['side_proxy']
        expected_cost = ((probabilities * payload).sum() + (1-probabilities[:, 0]).sum() *
                         (side['xyz_per_retained']+side['scale_per_retained']) + side['fixed_uses'])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        row = dict(result['metrics'], step=step, event='update', phase=args.phase,
                   step_seconds=time.perf_counter()-started, beta=args.beta, d_ref=reference['d_ref'],
                   c_ref=reference['c_ref'], lambda_s=reference['lambda_s'],
                   sampled_tier_counts=torch.bincount(result['q'].cpu(), minlength=4).tolist(),
                   expected_cost_proxy=float(expected_cost), network_updates=learner.updates,
                   replay_live_bytes=sum(entry['size'] for entry in replay.training + replay.validation),
                   replay_training_groups=len(replay.training), replay_gc_removed=gc_stats['removed'],
                   peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None)
        (out/'metrics').mkdir(parents=True, exist_ok=True)
        append_json(out/'metrics/loss.jsonl', row)
        progress['pending'] = []
        if learner.scales is not None and (progress[counter] % args.benefit_validate_every == 0 or progress[counter] == target):
            progress['pending'].append('validation')
        if args.phase == 'allocation' and progress[counter] % args.control_every == 0:
            progress['pending'].append('control')
        if args.phase == 'allocation' and (progress[counter] % args.validate_every == 0 or progress[counter] == target):
            progress['pending'].append('calibration')
        if progress['pending'] or step % args.save_every == 0:
            save()
        pending()
        if progress[counter] == 1 or progress[counter] % 10 == 0 or progress[counter] == target:
            print(f"{args.phase} {progress[counter]}/{target}: MSE={row['source_mse']:.6g}, "
                  f"groups={row.get('labeled_groups', 0)}, tiles={row.get('evaluated_tiles', 0)}, "
                  f"teacher={row.get('teacher_seconds', 0):.2f}s, updates={learner.updates}, "
                  f"time={row['step_seconds']:.2f}s, peak_mib={row['peak_allocated_mib']}", flush=True)
            if 'replay_mean_visits' in row:
                print(f"  replay: min={row['replay_min_visits']}, mean={row['replay_mean_visits']:.2f}, "
                      f"p10={row['replay_p10_visits']:.1f}, unused={row['replay_never_updated_fraction']:.2%}, "
                      f"draws={row['replay_all_group_draws']}, pair_loss={row['pair_loss']:.6g}", flush=True)
        updates += 1
        if max_updates is not None and updates >= max_updates:
            save()
            return dict(status='interrupted', step=step, phase=args.phase)
    if args.phase == 'pretrain':
        progress['pretrain_complete'] = True; save()
        result = dict(status='pretrain_complete', step=progress['step'], network_updates=learner.updates,
                      frozen_codec_verified=model_id(model) == identity)
        json_write(out/'pretrain_complete.json', result)
        print(json.dumps(result), flush=True)
        return result
    measure('latest', scene.test_cameras, True)
    for tier in (2, 3):
        measure(f'uniform_q{tier}', scene.test_cameras, map_q=torch.full_like(q, tier))
    export = out/'deployment'; export.mkdir(exist_ok=True)
    np.save(export/'latest_tiers.npy', q.cpu().numpy().astype(np.uint8), allow_pickle=False)
    probabilities = table.probabilities(torch.arange(len(q), device=device), args.snr).detach().cpu().numpy()
    np.save(export/'latest_probabilities.npy', probabilities, allow_pickle=False)
    result = dict(status='complete', family=family, phase='allocation', step=progress['step'],
                  allocation_step=progress['allocation_step'], network_updates=learner.updates,
                  tier_sha256=tier_hash(q), frozen_codec_verified=model_id(model) == identity)
    save('final'); save(); json_write(out/'complete.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
