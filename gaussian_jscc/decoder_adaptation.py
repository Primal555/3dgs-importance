"""Frozen-channel, noisy-feature adaptation of ONLY the semantic decoder.

Independent Truck fork: never updates the original channel endpoint, never
trains an allocator, and never adds the constant latent error to the objective.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .cli import device_for, seed_all
from .data import read_ply, to_raw
from .modular_checkpoint import (capture_rng, restore_rng, restore_training_checkpoint,
                                 save_training_checkpoint, validate_scene)
from .modular_evaluation import evaluate_paths
from .modular_experiment import json_write, make_scene, repair_logs
from .modular_snr_controls import control_inputs
from .modular_training import configure_phase, finish_update, latent_diagnostics, latent_reconstruction_loss
from .optimization import preserved_rng
from .perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask
from .position_delivery import delivered_positions
from .power_followup import joint_inputs
from .power_followup_evaluation import bind_test_cameras
from .render_validation import append_json
from .training import full_scene_step
from .transport import model_id


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='channel-only run, or stability suite root (selects block continuation)')
    p.add_argument('--out', required=True, help='NEW independent output directory')
    p.add_argument('--manifest', help='relocated Truck paths; original PLY identity and splits must match')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--probe-every', type=int, default=50, help='same-input post-update check; first16 always checked')
    p.add_argument('--test-views', type=int, default=32)
    p.add_argument('--render-trials', type=int, default=4)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--resume', action='store_true', help='exact resume in the same output and with identical settings')
    p.add_argument('--dry-run', action='store_true')
    return p


def frozen_digest(state):
    """Include buffers, power statistics and attribute normalization, not just weights."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not name.startswith('semantic_decoder.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_plan(args):
    if (min(args.steps, args.validate_every, args.save_every, args.probe_every, args.test_views,
            args.render_trials, args.cpu_threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError('positive training/evaluation budgets and finite positive learning rate required')
    root = run = Path(args.run).resolve()
    if not (run/'training.json').exists(): run = run/'block_continuation/block'
    out = Path(args.out).resolve()
    if any(out.is_relative_to(source) or source.is_relative_to(out) for source in (root, run)):
        raise ValueError('output must be separate from the source channel run')
    path = run/'checkpoints/end_channel_bootstrap/codec.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = json.loads((run/'training.json').read_text(encoding='utf-8'))
    progress = saved['training']['progress']; config = saved['config']
    if (saved.get('version') != 4 or config.get('architecture') != 'modular_jscc_v1'
            or config.get('channel_variant') != 'progressive' or tuple(config['rates']) != (0, 8, 24, 48)
            or config.get('position_delivery') != 'quantized' or config.get('position_bits') != 16
            or progress['phase'] != 'channel_bootstrap' or not progress.get('phase_complete')
            or not progress.get('freeze_semantic') or progress['completed'].get('channel_render', 0)
            or progress['step'] <= 0 or progress['step'] != progress['completed']['channel_bootstrap']):
        raise ValueError('initializer must be a completed frozen-semantic channel-only 8/24/48 endpoint with reliable16bit XYZ')
    if json.loads(json.dumps(config)) != metadata['codec_config']:
        raise ValueError('source configuration differs from training metadata')
    validate_scene(saved['training']['scene_record'], metadata['scene'])
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(version=1, source_run=str(run), checkpoint=str(path), source_sha256=digest,
        source_step=progress['step'], frozen_sha256=frozen_digest(saved['state_dict']),
        settings={k: getattr(args, k) for k in ('steps', 'lr', 'validate_every', 'save_every', 'probe_every',
            'test_views', 'render_trials', 'lpips', 'seed', 'device', 'cpu_threads')},
        manifest_override=str(Path(args.manifest).resolve()) if args.manifest else None,
        trainable_modules=['semantic_decoder'], frozen_modules=['semantic_encoder', 'channel_encoder', 'channel_decoder'],
        objective='source PLY RGB MSE + (1-SSIM)/3; latent MSE diagnostic only',
        training='10dB AWGN; q1/q2/q3/mixed rotation; fresh noise per update; no allocator/no gradient clipping',
        initialization='all weights retained; NEW semantic decoder Adam; no source channel Adam is used')


def set_adaptation_mode(model):
    # requires_grad(False) alone does not freeze running buffers or dropout.
    model.eval()
    model.semantic_decoder.train()


def adaptation_batch(model, features, q):
    with torch.no_grad():
        y, y_hat, _ = model.forward_channel_batches(features, features[..., :3],
            F.one_hot(q, len(model.cfg.rates)).to(features), 10., 'awgn')
        _, metrics = latent_reconstruction_loss(y, y_hat, q, len(model.cfg.rates))
    pred = model.decode_representation(y_hat.detach(), q,
                                      delivered_positions(features[..., :3], q, model.cfg))
    return pred, metrics


def render_adaptation(model, batches, geometry, task):
    model.set_scene_geometry(geometry)
    def forward(features, q):
        pred, metrics = adaptation_batch(model, features, q)
        raw = to_raw(pred[q > 0], geometry, model)
        return raw, raw.sum()*0, metrics
    loss, stats = full_scene_step(model, batches, geometry, 10., 'awgn', task,
                                 attr_weight=0, mode='replay', profile=True, batch_forward=forward)
    metrics = {key[:-5]: stats.pop(key) for key in list(stats)
               if key.startswith('latent_') and key.endswith('_loss')}
    stats.update(latent_diagnostics(metrics), weighted_latent_loss=0.,
                 objective='source_ply_rgb_mse_ssim_decoder_only', **task.stats)
    return loss, stats


@torch.no_grad()
def measure_same_input(model, scene, batches, indices, seed):
    device = next(model.parameters()).device
    rows = []; metrics = {}; total = 0
    with preserved_rng(device):
        seed_all(seed)
        for features, q in batches:
            features, q = features.to(device), q.to(device)
            pred, terms = adaptation_batch(model, features, q)
            raw = to_raw(pred[q > 0], scene.geometry, model)
            rows.append(raw); total += len(raw)
            for key, value in terms.items(): metrics[key] = metrics.get(key, 0)+value.detach()*len(raw)
        task = PerceptualMultiViewRenderTask([scene.cameras[i] for i in indices], scene.reference,
            scene.degree, scene.white_background, PerceptualImageObjective('mse_ssim', 1/3, device))
        loss = task(torch.cat(rows))
    return dict(loss=float(loss), **task.stats,
                **latent_diagnostics({k: v/total for k, v in metrics.items()}))


def evaluate_saved(model, scene, out, args, step, *, label, scope, images):
    """Idempotent evaluation; caller saves a pending cursor before entering."""
    folder = out/'evaluation'/label
    cameras = scene.test_cameras if scope == 'test' else scene.val_cameras
    identity = dict(step=step, label=label, scope=scope, images=images,
        view_names=[str(c.image_name) for c in cameras], model_id=model_id(model),
        render_trials=args.render_trials, seed=args.seed, lpips=args.lpips)
    marker = folder/'complete.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != identity:
            raise ValueError('evaluation identity changed: '+label)
        return
    folder.mkdir(parents=True, exist_ok=True)
    history = folder/'metrics/validation.jsonl'
    if history.exists(): history.write_text('', encoding='utf-8')
    original = scene.val_cameras; modes = {m: m.training for m in model.modules()}
    try:
        with preserved_rng(next(model.parameters()).device):
            model.eval(); scene.val_cameras = cameras
            evaluate_paths(model, scene, step=step, phase=label, snrs=[10.], trials=args.render_trials,
                seed=args.seed, save_images=images, out=folder, condition_policy='matched',
                color_metrics=True, lpips_metrics=args.lpips, save_noiseless_images=True)
    finally:
        scene.val_cameras = original
        for module, mode in modes.items(): module.training = mode
    if model_id(model) != identity['model_id']: raise RuntimeError('evaluation changed model state')
    json_write(marker, identity)


def summarize(out):
    from .channel_audit import write_csv
    rows = []
    for label in ('initial', 'final'):
        data = [json.loads(x) for x in (out/'evaluation'/label/'metrics/validation.jsonl').read_text(encoding='utf-8').splitlines()]
        for record in data:
            row = {k: record[k] for k in ('path', 'layout', 'source_psnr', 'photo_psnr', 'source_ssim',
                   'photo_ssim', 'latent_mse', 'latent_nmse', 'views', 'trials')}
            row.update({k: record[k] for k in ('source_lpips', 'photo_lpips') if k in record})
            row['endpoint'] = label
            row['source_psnr_change_db'] = None
            if label == 'final':
                previous = next(r for r in rows if r['endpoint'] == 'initial' and r['path'] == row['path'] and r['layout'] == row['layout'])
                row['source_psnr_change_db'] = row['source_psnr']-previous['source_psnr']
                if row['latent_mse'] != previous['latent_mse']:
                    raise RuntimeError('fixed channel latent error changed on paired evaluation')
            rows.append(row)
    json_write(out/'comparison/summary.json', dict(rows=rows,
        clean='diagnostic of the adapted decoder on original y, not a required upper bound',
        mixed='fixed random positive tiers, NOT learned allocation',
        protocol='same test cameras and paired noise; only semantic decoder changes'))
    write_csv(out/'comparison/endpoints.csv', rows)


def main(args, max_updates=None):
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return plan
    out = Path(args.out).resolve(); resumed = out.exists()
    if resumed:
        if not args.resume: raise FileExistsError('use NEW output or --resume')
        if json.loads((out/'plan.json').read_text(encoding='utf-8')) != plan: raise ValueError('resume settings/source differ')
        if (out/'complete.json').exists(): return json.loads((out/'complete.json').read_text(encoding='utf-8'))
    else:
        if args.resume: raise FileNotFoundError('resume requires an existing output')
        out.mkdir(parents=True, exist_ok=False); json_write(out/'plan.json', plan)
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads)
    latest = out/'checkpoints/latest/codec.pt'
    model, optimizer, progress, rng, record = restore_training_checkpoint(latest if latest.exists() else plan['checkpoint'], device)
    fresh = not latest.exists()
    if fresh:
        optimizer = configure_phase(model, 'channel_render', {'semantic_decoder': args.lr}, semantic_decoder_only=True)
        progress = copy.deepcopy(progress)
        progress.update(phase='channel_render', phase_step=0, phase_complete=False, freeze_semantic=False,
            semantic_decoder_only=True, channel_denoiser_only=False, channel_neighbor_only=False,
            module_lrs={'semantic_decoder': args.lr}, initial_done=False, final_done=False,
            source_step=plan['source_step'], train_snrs=[10.])
        progress['phase_budgets']['channel_render'] = args.steps
        progress['training_objective'] = dict(version='decoder_adaptation_v1',
            render_image_objective={'kind': 'mse_ssim', 'aux_weight': 1/3}, channel_render_latent_weight=0.)
    else:
        if not progress.get('semantic_decoder_only'): raise ValueError('resume is not a decoder-only checkpoint')
        repair_logs(out, progress['step'])
    if frozen_digest(model.state_dict()) != plan['frozen_sha256']: raise RuntimeError('frozen source state mismatch')
    metadata = json.loads((Path(plan['source_run'])/'training.json').read_text(encoding='utf-8'))
    original, manifest, spec, old, _ = control_inputs(metadata, args.manifest)
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, original, old, out); del raw
    validate_scene(record, scene.record); scene.channel_trained = True
    test_names = bind_test_cameras(scene, spec, args.test_views)
    model.set_training_snrs([10.]); set_adaptation_mode(model); restore_rng(rng)
    training = dict(source_run=plan['source_run'], manifest=str(manifest), selected_scenes=[spec],
        codec_config=model.cfg.to_dict(), scene=scene.record, test_view_names=test_names,
        decoder_adaptation=plan, training_objective=progress['training_objective'],
        fixed_xyz=True, render_backward='replay', lpips_metrics=args.lpips)
    if (out/'training.json').exists():
        if json.loads((out/'training.json').read_text(encoding='utf-8')) != json.loads(json.dumps(training)):
            raise ValueError('scene/test split changed during resume')
    else: json_write(out/'training.json', training)

    def save(label='latest'):
        if frozen_digest(model.state_dict()) != plan['frozen_sha256']:
            raise RuntimeError('frozen channel/encoder/buffers changed')
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt', model, optimizer, progress, capture_rng(), scene.record)

    def finish_evaluation():
        pending = progress['pending_evaluation']
        evaluate_saved(model, scene, out, args, progress['step'], **pending)
        progress.pop('pending_evaluation'); save()

    def evaluate(label, scope='test', images=False):
        progress['pending_evaluation'] = dict(label=label, scope=scope, images=images)
        save(); finish_evaluation()

    if fresh: save('initial'); save()
    if progress.get('pending_evaluation'): finish_evaluation()
    if not progress['initial_done']:
        evaluate('initial', images=True); progress['initial_done'] = True; save()
    print(f'Decoder-only: source={plan["source_step"]}, steps={args.steps}, lr={args.lr}; '
          'semantic encoder + BOTH channel modules FROZEN; latent MSE diagnostic only', flush=True)
    updates = 0
    while progress['completed']['channel_render'] < args.steps:
        local = progress['completed']['channel_render']+1; step = plan['source_step']+local
        set_adaptation_mode(model); optimizer.zero_grad(set_to_none=True)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        batches, indices, layout = joint_inputs(scene, step, local, args)
        task = PerceptualMultiViewRenderTask([scene.cameras[i] for i in indices], scene.reference,
            scene.degree, scene.white_background, PerceptualImageObjective('mse_ssim', 1/3, device))
        if device.type == 'cuda': torch.cuda.synchronize(device)
        started = time.perf_counter(); noise_seed = args.seed+500009*step
        with preserved_rng(device):
            seed_all(noise_seed)
            loss, stats = render_adaptation(model, batches, scene.geometry, task)
        stats.update(finish_update(model, optimizer, before))
        if local <= 16 or local % args.probe_every == 0:
            after = measure_same_input(model, scene, batches, indices, noise_seed)
            stats['same_input'] = dict(loss_before=float(loss), loss_after=after['loss'],
                image_mse_before=stats['image_mse'], image_mse_after=after['image_mse'],
                latent_mse_before=stats['latent_mse'], latent_mse_after=after['latent_mse'])
        if device.type == 'cuda': torch.cuda.synchronize(device)
        progress.update(step=step, phase_step=local, phase_complete=local == args.steps)
        progress['completed']['channel_render'] = local
        append_json(out/'metrics/loss.jsonl', dict(stats, step=step, phase='decoder_adaptation',
            phase_step=local, layout=layout, training_view_indices=indices, loss=float(loss), lr=args.lr,
            snr=10., step_seconds=time.perf_counter()-started))
        if local == 1 or local % 10 == 0:
            print(f'decoder {local}/{args.steps}: q={layout}, loss={float(loss):.6g}, '
                  f'grad={stats["grad_norm"]:.5g}, y_mse={stats["latent_mse"]:.6g} (monitor only)', flush=True)
        if local % args.validate_every == 0 and not progress['phase_complete']:
            evaluate(f'adapt_{local:06d}', scope='validation')
        if local % args.save_every == 0 or local % args.validate_every == 0 or progress['phase_complete']: save()
        updates += 1
        if max_updates is not None and updates >= max_updates:
            save(); return dict(status='interrupted', step=step)
    if not progress['final_done']:
        evaluate('final', images=True); progress['final_done'] = True
    summarize(out); save('final'); save()
    result = dict(status='complete', step=progress['step'], adaptation_updates=args.steps,
        source_step=plan['source_step'], frozen_state_unchanged=True, trainable_modules=['semantic_decoder'])
    json_write(out/'complete.json', result)
    return result


if __name__ == '__main__': main(parser().parse_args())
