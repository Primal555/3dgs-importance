"""Clean-attribute diagnosis, +4000 channel recovery and +1000 joint finetuning."""
import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from .channel_normalization_controls import semantic_digest
from .channel_normalization_metrics import cache_teacher
from .cli import device_for, seed_all
from .codec import channel, prefix_mask
from .data import read_ply
from .modular_checkpoint import capture_rng, restore_rng, restore_training_checkpoint, save_training_checkpoint, validate_scene
from .modular_experiment import json_write, make_scene, repair_logs
from .modular_snr_controls import control_inputs
from .modular_training import configure_phase, latent_reconstruction_loss, render_step, objective_settings
from .optimization import preserved_rng, clip_codec_gradients
from .perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask
from .power_followup_evaluation import attribute_ablation, bind_test_cameras, evaluate_endpoint
from .power_followup_report import summarize
from .render_validation import append_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='completed power-controls suite, containing point_awgn10 and block_awgn10')
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', help='optional relocated paths, with identical Truck PLY and splits')
    p.add_argument('--device', default='cuda')
    p.add_argument('--channel-steps', type=int, default=4000)
    p.add_argument('--joint-steps', type=int, default=1000)
    p.add_argument('--channel-lr', type=float, default=1e-4)
    p.add_argument('--semantic-decoder-lr', type=float, default=1e-5)
    p.add_argument('--validate-every', type=int, default=1000)
    p.add_argument('--joint-validate-every', type=int, default=250)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--test-views', type=int, default=32)
    p.add_argument('--render-trials', type=int, default=4)
    p.add_argument('--diagnostic-blocks', type=int, default=32)
    p.add_argument('--response-blocks', type=int, default=8)
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--modes', nargs='+', choices=('point', 'block'), default=['point', 'block'])
    p.add_argument('--mode', choices=('point', 'block'), help=argparse.SUPPRESS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--resume-suite', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def encoder_digest(state):
    selected = {n: v for n, v in state.items() if n.startswith('semantic_encoder.') or n in ('attr_mean', 'attr_std')}
    return semantic_digest(selected)


def joint_optimizer(model, previous, channel_lr, decoder_lr):
    """Preserve real channel Adam step/moments; introduce only semantic decoder state."""
    if {g['name'] for g in previous.param_groups} != {'channel_encoder', 'channel_decoder'}:
        raise ValueError('joint transition requires a complete channel optimizer')
    lrs = dict(channel_encoder=channel_lr, channel_decoder=channel_lr, semantic_decoder=decoder_lr)
    optimizer = configure_phase(model, 'channel_render', lrs)
    old = {g['name']: g for g in previous.param_groups}
    for group in optimizer.param_groups:
        if group['name'] in old:
            source = old[group['name']]
            if [id(p) for p in group['params']] != [id(p) for p in source['params']]:
                raise ValueError('channel parameter order differs at joint transition')
            for key, value in source.items():
                if key not in ('params', 'lr', 'name'): group[key] = value
    for parameter, state in previous.state.items(): optimizer.state[parameter] = state
    return optimizer


def build_plan(args):
    if (min(args.channel_steps, args.validate_every, args.joint_validate_every,
            args.save_every, args.test_views, args.diagnostic_blocks, args.response_blocks, args.cpu_threads) < 1
            or args.joint_steps < 0 or args.render_trials < 2 or args.trials < 2
            or any(not math.isfinite(v) or v <= 0 for v in (args.channel_lr, args.semantic_decoder_lr))):
        raise ValueError('invalid follow-up training/evaluation budget')
    if len(set(args.modes)) != len(args.modes) or (args.mode and args.mode not in args.modes):
        raise ValueError('duplicate modes or worker outside selected suite')
    root = Path(args.run).resolve(); out = Path(args.out).resolve()
    if out.is_relative_to(root) or root.is_relative_to(out): raise ValueError('output must be separate from source suite')
    sources = {}
    for mode in args.modes:
        run = root/(mode+'_awgn10'); checkpoint = run/'checkpoints/final/codec.pt'
        metadata = json.loads((run/'training.json').read_text(encoding='utf-8'))
        complete = json.loads((run/'complete.json').read_text(encoding='utf-8'))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        cfg = saved['config']; training = saved['training']; progress = training['progress']
        if (complete.get('status') != 'complete' or saved.get('version') != 4
                or cfg.get('architecture') != 'modular_jscc_v1' or cfg.get('channel_variant') != 'progressive'
                or cfg.get('channel_power_mode', 'point') != mode or tuple(cfg['rates']) != (0, 8, 24, 48)
                or progress['phase'] != 'channel_bootstrap' or not progress.get('phase_complete')
                or not progress.get('freeze_semantic') or progress['completed'].get('channel_render', 0)
                or progress['step'] != progress['completed']['channel_bootstrap']
                or not training['optimizer'] or not training['optimizer']['state']
                or json.loads(json.dumps(cfg)) != metadata['codec_config']):
            raise ValueError('source must be completed frozen-semantic point/block AWGN channel recovery')
        if metadata.get('channel_kind') != 'awgn': raise ValueError('source must be noise-trained')
        validate_scene(training['scene_record'], metadata['scene'])
        with checkpoint.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        sources[mode] = dict(run=str(run), checkpoint=str(checkpoint), sha256=digest, step=progress['step'],
            scene=metadata['scene'], config=metadata['codec_config'], semantic_sha256=semantic_digest(saved['state_dict']),
            encoder_sha256=encoder_digest(saved['state_dict']))
    if len({s['semantic_sha256'] for s in sources.values()}) != 1 or len({s['step'] for s in sources.values()}) != 1:
        raise ValueError('point/block must share semantic teacher and source update budget')
    normalized = []
    for s in sources.values():
        cfg = dict(s['config']); cfg.pop('channel_power_mode', None)
        normalized.append(dict(config=cfg, scene=s['scene']))
    if any(s != normalized[0] for s in normalized[1:]): raise ValueError('unmatched point/block scene or architecture')
    settings = {k: getattr(args, k) for k in ('channel_steps', 'joint_steps', 'channel_lr', 'semantic_decoder_lr',
        'validate_every', 'joint_validate_every', 'save_every', 'test_views', 'render_trials',
        'diagnostic_blocks', 'response_blocks', 'trials', 'lpips', 'seed', 'cpu_threads')}
    common = ['--run', str(root), '--out', str(out), '--device', args.device, '--modes', *args.modes]
    for key, value in settings.items():
        if key != 'lpips': common += ['--'+key.replace('_', '-'), str(value)]
    common += ['--lpips' if args.lpips else '--no-lpips']
    if args.manifest: common += ['--manifest', str(Path(args.manifest).resolve())]
    return dict(version=1, sources=sources, settings=settings,
        manifest_override=str(Path(args.manifest).resolve()) if args.manifest else None,
        stages=['clean_attribute_replacements', 'channel_continuation', 'joint_finetuning'],
        initialization='inherit complete weights and channel Adam; keep original y teacher; new semantic decoder Adam at joint boundary',
        test_protocol='same actual test cameras and paired 10dB noise at initial/channel_end/final; original validation cameras tracked separately',
        jobs=[dict(mode=mode, command=[sys.executable, '-u', '-m', 'gaussian_jscc.power_followup', *common, '--mode', mode]) for mode in args.modes])


def checkpoint(folder, label, model, optimizer, progress, scene_record):
    save_training_checkpoint(folder/'checkpoints'/label/'codec.pt', model, optimizer,
                             progress, capture_rng(), scene_record)


def initial_progress(parent, args):
    p = copy.deepcopy(parent); start = p['step']
    p.update(phase_complete=False, initial_done=False, ablation_done=False, channel_endpoint_done=False,
             final_done=False, source_step=start, train_snrs=[10.], channel_trained=True)
    p.pop('status', None)
    p['phase_budgets'] = dict(semantic_bootstrap=0, semantic_render=0,
                             channel_bootstrap=start+args.channel_steps, channel_render=args.joint_steps)
    p['module_lrs'] = dict(channel_encoder=args.channel_lr, channel_decoder=args.channel_lr,
                           semantic_decoder=args.semantic_decoder_lr)
    p['training_objective'] = dict(objective_settings(), render_image_objective=dict(
        version='perceptual_rgb_v1', kind='mse_ssim', aux_weight=1/3))
    return p


def channel_update(model, optimizer, cache, step, args):
    device = next(model.parameters()).device
    index = random.Random(args.seed+100003*step).randrange(len(cache)); batch = cache[index]
    y, active, xyz = [batch[k].to(device) for k in ('y', 'active', 'xyz')]
    tier = (step-1) % 3+1; q = active.long()*tier
    z = model.channel_encoder(y, active, 10.)
    mask = prefix_mask(q.flatten(), model.cfg.rates).reshape_as(z)
    generator = torch.Generator(device=device).manual_seed(args.seed+200003*step)
    received = channel(z.reshape(-1, 2), 10., 'awgn', generator).reshape_as(z)*mask
    decoded = model.channel_decoder(received, mask, q, 10., delivered_xyz=xyz)
    loss, _ = latent_reconstruction_loss(y, decoded, q, len(model.cfg.rates))
    if not torch.isfinite(loss): raise RuntimeError('nonfinite channel loss')
    loss.backward()
    return loss.detach(), dict(latent_mse=float(loss.detach()), layout=str(tier), batch_index=index,
                               objective='original_y_mse')


def joint_inputs(scene, step, phase_step, args):
    layout = ('1', '2', '3', 'mixed')[(phase_step-1) % 4]
    qs = []
    for index, ids in enumerate(scene.group_ids):
        values = (torch.randint(1, 4, ids.shape, generator=torch.Generator().manual_seed(args.seed+300007*step+index))
                  if layout == 'mixed' else torch.full_like(ids, int(layout)))
        qs.append(values.masked_fill(ids < 0, 0))
    indices = random.Random(args.seed+400009*step).sample(range(len(scene.cameras)), scene.views_per_step)
    return list(zip(scene.groups, qs)), indices, layout


def joint_update(model, scene, step, phase_step, args, *, task_factory=None, replay_observer=None):
    device = next(model.parameters()).device
    batches, indices, layout = joint_inputs(scene, step, phase_step, args)
    objective = PerceptualImageObjective('mse_ssim', 1/3, device)
    factory = task_factory or PerceptualMultiViewRenderTask
    task = factory([scene.cameras[i] for i in indices], scene.reference,
                   scene.degree, scene.white_background, objective)
    # Keep full-scene replay noise paired between modes without consuming the
    # continuation's RNG stream a second time during replay.
    with preserved_rng(device):
        seed_all(args.seed+500009*step)
        loss, stats = render_step(model, batches, scene.geometry,
                                  10., 'awgn', task, 'channel_render', mode='replay', channel_latent_weight=1.,
                                  replay_observer=replay_observer)
    stats.update(task.stats, layout=layout, training_view_indices=indices)
    return loss, stats


def run_worker(args, mode, max_updates=None):
    plan = build_plan(args)
    if mode not in args.modes: raise ValueError('worker outside selected modes')
    source = plan['sources'][mode]; folder = Path(args.out).resolve()/mode
    control = dict(plan=plan, mode=mode)
    resumed = folder.exists()
    if resumed:
        if not args.resume_suite: raise FileExistsError('use new output or --resume-suite')
        if json.loads((folder/'control.json').read_text(encoding='utf-8')) != control: raise ValueError('resume settings differ')
        if (folder/'complete.json').exists(): return json.loads((folder/'complete.json').read_text())
    else:
        folder.mkdir(parents=True, exist_ok=False); json_write(folder/'control.json', control)
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    latest = folder/'checkpoints/latest/codec.pt'
    path = latest if latest.exists() else Path(source['checkpoint'])
    model, optimizer, progress, rng, scene_record = restore_training_checkpoint(path, device)
    restore_rng(rng)
    if path == Path(source['checkpoint']):
        progress = initial_progress(progress, args)
        for group in optimizer.param_groups: group['lr'] = args.channel_lr
        checkpoint(folder, 'initial', model, optimizer, progress, scene_record)
        checkpoint(folder, 'latest', model, optimizer, progress, scene_record)
    else:
        repair_logs(folder, progress['step'])
    metadata = json.loads((Path(source['run'])/'training.json').read_text(encoding='utf-8'))
    original, manifest, spec, old, _ = control_inputs(metadata, args.manifest)
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, original, old, folder); del raw
    validate_scene(scene_record, scene.record)
    test_names = bind_test_cameras(scene, spec, args.test_views)
    view_path = folder/'test_views.json'
    if view_path.exists() and json.loads(view_path.read_text()) != test_names: raise ValueError('test camera set changed on resume')
    json_write(view_path, test_names)
    record = dict(source_run=source['run'], manifest=str(manifest), selected_scenes=[spec], scene=scene.record,
        codec_config=model.cfg.to_dict(), source_step=source['step'], phase_budgets=progress['phase_budgets'],
        module_lrs=progress['module_lrs'], train_snrs=[10.], seed=args.seed, test_view_names=test_names,
        training_objective=progress['training_objective'], render_backward='replay', lpips_metrics=args.lpips,
        continuation=control, fixed_xyz=True, semantic_encoder_frozen=True)
    if not (folder/'training.json').exists(): json_write(folder/'training.json', record)
    # Teacher caching occurs in eval mode but cannot change checkpointed RNG.
    with preserved_rng(device): cache = cache_teacher(model, scene)
    restore_rng(rng)
    print(f'{mode}: source update {source["step"]}; +{args.channel_steps} channel +{args.joint_steps} joint; '
          f'{len(test_names)} actual test views; {args.render_trials} paired noise trials', flush=True)
    def save(label='latest'):
        checkpoint(folder, label, model, optimizer, progress, scene.record)
    def finish_evaluation():
        pending = progress['pending_evaluation']
        evaluate_endpoint(model, scene, cache, folder, args, pending['step'], pending['label'],
                          scope=pending['scope'], images=pending['images'])
        progress.pop('pending_evaluation')
        save()
    def evaluate_saved(label, scope='test', images=False):
        # Persist the exact weights/Adam and an evaluation cursor BEFORE the
        # expensive render. A failed render resumes here, not by replaying updates.
        progress['pending_evaluation'] = dict(step=progress['step'], label=label, scope=scope, images=images)
        save()
        finish_evaluation()
    if progress.get('pending_evaluation'):
        finish_evaluation()
    if not progress['initial_done']:
        evaluate_saved('initial', images=True)
        progress['initial_done'] = True; save()
    if not progress['ablation_done']:
        attribute_ablation(model, scene, folder, args)
        progress['ablation_done'] = True; save()
    updates = 0
    while progress['completed']['channel_bootstrap'] < progress['phase_budgets']['channel_bootstrap']:
        step = progress['step']+1
        model.train(); model.semantic_encoder.eval(); model.semantic_decoder.eval()
        optimizer.zero_grad(set_to_none=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss, stats = channel_update(model, optimizer, cache, step, args)
        norm, gradient = clip_codec_gradients(model, mode='none'); optimizer.step()
        if device.type == 'cuda': torch.cuda.synchronize(device)
        progress['step'] = progress['phase_step'] = step
        progress['completed']['channel_bootstrap'] = step
        progress['phase_complete'] = step == progress['phase_budgets']['channel_bootstrap']
        append_json(folder/'metrics/loss.jsonl', dict(stats, **gradient, step=step, phase='channel_bootstrap',
            phase_step=step, loss=float(loss), grad_norm=float(norm), lr=args.channel_lr, step_seconds=time.perf_counter()-started))
        if step % 50 == 0 or step == source['step']+1:
            print(f'{mode} channel {step}/{progress["phase_budgets"]["channel_bootstrap"]}: loss={float(loss):.6g}', flush=True)
        if step % args.validate_every == 0 and not progress['phase_complete']:
            evaluate_saved(f'channel_{step:06d}', scope='validation')
        if step % args.save_every == 0 or step % args.validate_every == 0 or progress['phase_complete']: save()
        updates += 1
        if max_updates is not None and updates >= max_updates:
            save(); return dict(status='interrupted', step=step)
    if not progress['channel_endpoint_done']:
        if semantic_digest(model.state_dict()) != source['semantic_sha256']:
            raise RuntimeError('frozen semantic teacher changed during channel recovery')
        evaluate_saved('channel_end')
        progress['channel_endpoint_done'] = True; save('end_channel_bootstrap'); save()
    if args.joint_steps and progress['phase'] != 'channel_render':
        optimizer = joint_optimizer(model, optimizer, args.channel_lr, args.semantic_decoder_lr)
        progress.update(phase='channel_render', phase_step=0, phase_complete=False, freeze_semantic=False)
        save('joint_initial'); save()
    while progress['completed']['channel_render'] < args.joint_steps:
        step = progress['step']+1; local_step = progress['completed']['channel_render']+1
        model.train(); model.semantic_encoder.eval(); optimizer.zero_grad(set_to_none=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss, stats = joint_update(model, scene, step, local_step, args)
        norm, gradient = clip_codec_gradients(model, mode='none'); optimizer.step()
        if device.type == 'cuda': torch.cuda.synchronize(device)
        progress.update(step=step, phase_step=local_step, phase_complete=local_step == args.joint_steps)
        progress['completed']['channel_render'] = local_step
        append_json(folder/'metrics/loss.jsonl', dict(stats, **gradient, step=step, phase='channel_render',
            phase_step=local_step, loss=float(loss), grad_norm=float(norm), step_seconds=time.perf_counter()-started,
            module_lrs={g['name']: g['lr'] for g in optimizer.param_groups}))
        if local_step == 1 or local_step % 10 == 0:
            print(f'{mode} joint {local_step}/{args.joint_steps}: loss={float(loss):.6g}, '
                  f'image_mse={stats["image_mse"]:.6g}, y_mse={stats["latent_mse"]:.6g}', flush=True)
        if local_step % args.joint_validate_every == 0 and not progress['phase_complete']:
            evaluate_saved(f'joint_{local_step:06d}', scope='validation')
        if local_step % args.save_every == 0 or local_step % args.joint_validate_every == 0 or progress['phase_complete']: save()
        updates += 1
        if max_updates is not None and updates >= max_updates:
            save(); return dict(status='interrupted', step=step)
    if not progress['final_done']:
        evaluate_saved('final', images=True)
        progress['final_done'] = True
    if encoder_digest(model.state_dict()) != source['encoder_sha256']: raise RuntimeError('frozen y encoder/statistics changed')
    save('final'); save()
    result = dict(status='complete', step=progress['step'], source_step=source['step'],
                  completed=progress['completed'], channel_adam_inherited=True, semantic_encoder_unchanged=True)
    json_write(folder/'complete.json', result)
    return result


def main(args):
    plan = build_plan(args)
    if args.dry_run: print(json.dumps(plan, indent=2)); return plan
    if args.mode: return run_worker(args, args.mode)
    out = Path(args.out).resolve()
    if out.exists():
        if not args.resume_suite: raise FileExistsError('use NEW output or --resume-suite')
        if json.loads((out/'plan.json').read_text()) != plan: raise ValueError('resume plan differs')
    else:
        out.mkdir(parents=True, exist_ok=False); json_write(out/'plan.json', plan)
    for job in plan['jobs']:
        mode = job['mode']
        if (out/mode/'complete.json').exists(): continue
        command = job['command']+(['--resume-suite'] if args.resume_suite else [])
        print(f'Start {mode}; worker progress: {out/(mode+".log")}', flush=True)
        with (out/(mode+'.log')).open('a', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            json_write(out/'status.json', dict(status='failed', mode=mode, returncode=result.returncode))
            raise RuntimeError(f'{mode} failed; inspect {out/(mode+".log")}')
        summarize(out, args.modes)
    result = summarize(out, args.modes)
    json_write(out/'complete.json', dict(status='complete', modes=args.modes))
    return result


if __name__ == '__main__': main(parser().parse_args())
