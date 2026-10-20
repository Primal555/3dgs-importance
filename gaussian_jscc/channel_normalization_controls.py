"""Three power normalizers x noiseless/10 dB, with one frozen Truck teacher."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from .channel_normalization_metrics import cache_teacher, calibrate_and_measure, diagnose_normalization
from .cli import device_for, seed_all
from .codec import channel, prefix_mask
from .data import read_ply
from .modular_checkpoint import (capture_rng, restore_rng, restore_training_checkpoint,
                                 save_training_checkpoint, validate_scene)
from .modular_config import ModularCodecConfig
from .modular_codec import ModularGaussianCodec
from .modular_evaluation import evaluate_paths
from .modular_experiment import json_write, make_scene, repair_logs
from .modular_snr_controls import control_inputs
from .modular_training import configure_phase, latent_reconstruction_loss
from .optimization import preserved_rng
from .render_validation import append_json
from .channel_audit import write_csv

MODES = ('point', 'statistical', 'block')
ARMS = {f'{mode}_{condition}': (mode, kind) for mode in MODES
        for condition, kind in (('noiseless', 'none'), ('awgn10', 'awgn'))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='modular Truck run; only semantic weights are copied')
    p.add_argument('--checkpoint', help='default RUN/checkpoints/final/codec.pt; semantic teacher only')
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', help='optional relocated data paths; scene identity/splits must match')
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=4000, help='real optimizer updates PER ARM')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=.01, help='statistical power EMA; eval calibration uses full population')
    p.add_argument('--validate-every', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--diagnostic-blocks', type=int, default=32)
    p.add_argument('--response-blocks', type=int, default=8)
    p.add_argument('--trials', type=int, default=8, help='paired latent noise trials')
    p.add_argument('--render-trials', type=int, default=2)
    p.add_argument('--validation-views', type=int, default=16, help='intermediate fixed heldout subset; final uses all original validation cameras')
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=list(ARMS))
    p.add_argument('--arm', choices=tuple(ARMS), help=argparse.SUPPRESS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--resume-suite', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='CPU metadata/checkpoint preflight; no output or scene load')
    return p


def semantic_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith(('semantic_encoder.', 'semantic_decoder.')) or name in ('attr_mean', 'attr_std'):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def channel_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(('channel_encoder.', 'channel_decoder.')) and '.normalizer.' not in name:
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fresh_model(saved, mode, seed, momentum, device):
    cfg = dict(saved['config'], channel_power_mode=mode, channel_power_momentum=momentum)
    # Norm buffers consume no random numbers. All channel PARAMETERS are shared
    # initial draws, including zero-output compensation modules in the source config.
    with preserved_rng(torch.device('cpu')):
        torch.manual_seed(seed)
        model = ModularGaussianCodec(ModularCodecConfig.from_dict(cfg))
    semantic = {n: v for n, v in saved['state_dict'].items()
                if n.startswith(('semantic_encoder.', 'semantic_decoder.')) or n in ('attr_mean', 'attr_std')}
    model.load_state_dict(semantic, strict=False)
    model.requires_grad_(False)
    model.set_training_snrs([10.])
    return model.to(device)


def build_plan(args):
    if (min(args.steps, args.validate_every, args.save_every, args.diagnostic_blocks,
            args.response_blocks, args.render_trials, args.validation_views, args.cpu_threads) < 1 or args.trials < 2
            or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.momentum) or not 0 < args.momentum <= 1):
        raise ValueError('invalid normalization experiment settings')
    if len(args.arms) != len(set(args.arms)) or (args.arm and args.arm not in args.arms):
        raise ValueError('duplicate arms or worker outside selected suite')
    run = Path(args.run).resolve(); out = Path(args.out).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run/'checkpoints/final/codec.pt'
    if out.is_relative_to(run) or run.is_relative_to(out) or checkpoint.is_relative_to(out):
        raise ValueError('output must be independent of source run/checkpoint')
    metadata = json.loads((run/'training.json').read_text(encoding='utf-8'))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['config']
    if (saved.get('version') != 4 or cfg.get('architecture') != 'modular_jscc_v1'
            or cfg.get('channel_variant') != 'progressive' or tuple(cfg['rates']) != (0, 8, 24, 48)
            or json.loads(json.dumps(cfg)) != metadata['codec_config']):
        raise ValueError('requires a matching modular progressive 8/24/48 teacher checkpoint/config')
    validate_scene(saved['training']['scene_record'], metadata['scene'])
    with checkpoint.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    settings = {key: getattr(args, key) for key in ('steps', 'lr', 'momentum', 'validate_every', 'save_every',
        'diagnostic_blocks', 'response_blocks', 'trials', 'render_trials', 'validation_views', 'lpips', 'seed', 'cpu_threads')}
    common = ['--run', str(run), '--checkpoint', str(checkpoint), '--out', str(out), '--device', args.device,
              '--arms', *args.arms]
    for key, value in settings.items():
        if key != 'lpips': common += ['--'+key.replace('_', '-'), str(value)]
    common += ['--lpips' if args.lpips else '--no-lpips']
    if args.manifest: common += ['--manifest', str(Path(args.manifest).resolve())]
    return dict(version=1, source_run=str(run), checkpoint=str(checkpoint), source_checkpoint_sha256=digest,
        teacher_semantic_sha256=semantic_digest(saved['state_dict']), rates=[0, 8, 24, 48], settings=settings,
        manifest=str(Path(args.manifest or metadata['manifest']).resolve()),
        initialization='same freshly seeded channel weights and fresh Adam; only semantic modules/statistics copied',
        objective='original teacher-y MSE only; semantics frozen; no image/color loss or allocator',
        layouts=['1', '2', '3'], noise_variance_per_real=.05, condition_snr=10.,
        calibration='statistical: full frozen teacher population initially and at each validation; detached EMA between validations; frozen eval',
        jobs=[dict(arm=arm, mode=ARMS[arm][0], channel_kind=ARMS[arm][1],
                   command=[sys.executable, '-u', '-m', 'gaussian_jscc.channel_normalization_controls', *common, '--arm', arm])
              for arm in args.arms])


def evaluate(model, scene, cache, folder, args, step, phase, *, calibrate):
    """Validation is RNG-isolated; statistical calibration is checkpointed TX state."""
    all_cameras = scene.val_cameras
    if phase not in ('initial', 'final') and len(all_cameras) > args.validation_views:
        indices = torch.linspace(0, len(all_cameras)-1, args.validation_views).round().long().tolist()
        scene.val_cameras = [all_cameras[i] for i in indices]
    try:
        with preserved_rng(next(model.parameters()).device):
            model.eval()
            power = calibrate_and_measure(model, cache, calibrate=calibrate)
            append_json(folder/'metrics/power.jsonl', dict(step=step, phase=phase, **power))
            report = diagnose_normalization(model, scene, cache, folder/'diagnostics'/phase,
                blocks=args.diagnostic_blocks, response_blocks=args.response_blocks, trials=args.trials, seed=args.seed)
            evaluate_paths(model, scene, step=step, phase=phase, snrs=[10.], trials=args.render_trials,
                seed=args.seed, save_images=phase in ('initial', 'final'), out=folder,
                condition_policy='matched', color_metrics=True, lpips_metrics=args.lpips, save_noiseless_images=True)
    finally:
        scene.val_cameras = all_cameras
    return dict(power=power, sample_sha256=report['sample_sha256'])


def save_checkpoint(folder, label, model, optimizer, step, args, scene_record, initial_done):
    progress = dict(step=step, phase='channel_bootstrap', phase_step=step, phase_complete=step == args.steps,
        completed=dict(semantic_bootstrap=0, semantic_render=0, channel_bootstrap=step, channel_render=0),
        phase_budgets=dict(semantic_bootstrap=0, semantic_render=0, channel_bootstrap=args.steps, channel_render=0),
        module_lrs={'channel_encoder': args.lr, 'channel_decoder': args.lr},
        freeze_semantic=True, channel_trained=step > 0, train_snrs=[10.], initial_done=initial_done)
    save_training_checkpoint(folder/'checkpoints'/label/'codec.pt', model, optimizer, progress,
                             capture_rng(), scene_record)


def run_arm(args, arm, max_updates=None):
    plan = build_plan(args)
    if arm not in args.arms: raise ValueError('arm not in selected suite')
    folder = Path(args.out).resolve()/arm; mode, kind = ARMS[arm]
    settings = dict(plan=plan, arm=arm)
    if folder.exists() and not args.resume_suite: raise FileExistsError('use new output or --resume-suite')
    resumed = folder.exists()
    if resumed and json.loads((folder/'control.json').read_text(encoding='utf-8')) != settings:
        raise ValueError('resume settings differ')
    if resumed and (folder/'complete.json').exists(): return json.loads((folder/'complete.json').read_text())
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    metadata = json.loads((Path(plan['source_run'])/'training.json').read_text(encoding='utf-8'))
    saved = torch.load(plan['checkpoint'], map_location='cpu', weights_only=True)
    if resumed and (folder/'checkpoints/latest/codec.pt').exists():
        model, optimizer, progress, rng, scene_record = restore_training_checkpoint(folder/'checkpoints/latest/codec.pt', device)
        step = progress['step']; initial_done = progress.get('initial_done', False)
        repair_logs(folder, step)
        path = folder/'metrics/power.jsonl'
        if path.exists():
            kept = [s for s in path.read_text().splitlines() if json.loads(s)['step'] <= step]
            path.write_text('\n'.join(kept)+'\n', encoding='utf-8')
    else:
        if resumed and read_rows(folder/'metrics/loss.jsonl'):
            raise ValueError('training log exists but resumable checkpoint is missing')
        if not resumed:
            folder.mkdir(parents=True, exist_ok=False); json_write(folder/'control.json', settings)
        model = fresh_model(saved, mode, args.seed, args.momentum, device)
        optimizer = configure_phase(model, 'channel_bootstrap', {'channel_encoder': args.lr, 'channel_decoder': args.lr}, freeze_semantic=True)
        step = 0; initial_done = False; scene_record = metadata['scene']
        json_write(folder/'initialization.json', dict(channel_parameters_sha256=channel_digest(model),
            teacher_semantic_sha256=semantic_digest(model.state_dict()), optimizer_state_entries=len(optimizer.state)))
        # Persist before camera loading, caching, LPIPS loading or rendering.
        # An initialization failure must be resumable without deleting the arm.
        save_checkpoint(folder, 'latest', model, optimizer, 0, args, scene_record, False)
        rng = capture_rng()
    source, manifest, spec, old, _ = control_inputs(metadata, args.manifest)
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, source, old, folder); del raw
    validate_scene(scene_record, scene.record)
    record = dict(source_run=plan['source_run'], manifest=str(manifest), codec_config=model.cfg.to_dict(),
        scene=scene.record, selected_scenes=[spec], freeze_semantic=True, train_snrs=[10.],
        normalization_control=dict(arm=arm, settings=plan['settings'], teacher_checkpoint=plan['checkpoint']),
        channel_kind=kind, phase_budgets={'channel_bootstrap': args.steps}, lpips_metrics=args.lpips)
    if not resumed: json_write(folder/'training.json', record)
    cache = cache_teacher(model, scene)
    print(f'{arm}: {sum(int(b["active"].sum()) for b in cache)} cached teacher points; '
          f'{args.steps} channel updates; original block size={model.cfg.block_size}', flush=True)
    if rng is not None: restore_rng(rng)
    if not initial_done:
        # A failed initial evaluation may have published only some metrics.
        # Replay replaces the endpoint instead of duplicating those rows.
        for name in ('validation', 'power'):
            path = folder/'metrics'/f'{name}.jsonl'
            if path.exists():
                kept = [s for s in path.read_text(encoding='utf-8').splitlines() if json.loads(s)['phase'] != 'initial']
                path.write_text(''.join(s+'\n' for s in kept), encoding='utf-8')
        evaluate(model, scene, cache, folder, args, 0, 'initial', calibrate=True)
        initial_done = True
        save_checkpoint(folder, 'initial', model, optimizer, 0, args, scene.record, initial_done)
        save_checkpoint(folder, 'latest', model, optimizer, 0, args, scene.record, initial_done)
    updates = 0; started = time.perf_counter()
    while step < args.steps:
        step += 1; model.train(); model.semantic_encoder.eval(); model.semantic_decoder.eval()
        local = random.Random(args.seed+100003*step); batch_index = local.randrange(len(cache))
        batch = cache[batch_index]
        y, active, xyz = [batch[k].to(device) for k in ('y', 'active', 'xyz')]
        tier = (step-1) % 3+1; q = active.long()*tier
        generator = torch.Generator(device=device).manual_seed(args.seed+200003*step)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        tick = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        z = model.channel_encoder(y, active, 10.)
        mask = prefix_mask(q.flatten(), model.cfg.rates).reshape_as(z)
        rx = channel(z.reshape(-1, 2), 10., kind, generator).reshape_as(z)*mask
        yh = model.channel_decoder(rx, mask, q, 10., delivered_xyz=xyz)
        loss, _ = latent_reconstruction_loss(y, yh, q, len(model.cfg.rates))
        if not torch.isfinite(loss): raise RuntimeError('nonfinite latent loss; optimizer not updated')
        loss.backward()
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad = torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=True)
        optimizer.step()
        loss_value = float(loss.detach())
        if device.type == 'cuda': torch.cuda.synchronize(device)
        seconds = time.perf_counter()-tick
        append_json(folder/'metrics/loss.jsonl', dict(step=step, phase='channel_bootstrap', phase_step=step,
            layout=str(tier), batch_index=batch_index, loss=loss_value, latent_mse=loss_value, grad_norm=float(grad),
            lr=args.lr, snr=10., channel_kind=kind, power_mode=mode, step_seconds=seconds,
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
        if step == 1 or step % 50 == 0:
            print(f'{arm} {step}/{args.steps}: q{tier}, y_MSE={loss_value:.6g}, grad={float(grad):.5g}, sec={seconds:.3f}', flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            evaluate(model, scene, cache, folder, args, step, 'final' if step == args.steps else f'step_{step:06d}', calibrate=True)
            if semantic_digest(model.state_dict()) != plan['teacher_semantic_sha256']:
                raise RuntimeError('frozen semantic teacher changed')
        if step % args.save_every == 0 or step % args.validate_every == 0 or step == args.steps:
            save_checkpoint(folder, 'latest', model, optimizer, step, args, scene.record, initial_done)
        updates += 1
        if max_updates is not None and updates >= max_updates and step < args.steps:
            save_checkpoint(folder, 'latest', model, optimizer, step, args, scene.record, initial_done)
            return dict(status='interrupted', step=step)
    save_checkpoint(folder, 'final', model, optimizer, step, args, scene.record, initial_done)
    result = dict(status='complete', updates=step, semantic_weights_unchanged=True,
                  elapsed_training_seconds=time.perf_counter()-started, arm=arm)
    json_write(folder/'complete.json', result)
    return result


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def summarize(out, arms):
    out = Path(out); rows = []; initializations = []; sample_hashes = set()
    for arm in arms:
        folder = out/arm
        if not (folder/'complete.json').exists(): continue
        initializations.append(json.loads((folder/'initialization.json').read_text()))
        report = json.loads((folder/'diagnostics/final/latent_errors.json').read_text())
        sample_hashes.add(report['sample_sha256'])
        power = [r for r in read_rows(folder/'metrics/power.jsonl') if r['phase'] == 'final'][-1]
        validation = [r for r in read_rows(folder/'metrics/validation.jsonl') if r['phase'] == 'final']
        for tier in report['tiers']:
            t = tier['tier']; row = dict(arm=arm, tier=t, symbols=tier['symbols'], **tier['moments']['summary'],
                mean_complex_power=power['prefixes'][t-1]['mean_complex_power'],
                within_average_power_budget=power['within_average_power_budget'])
            for path, key in (('clean', 'clean'), ('finite_noiseless', 'noiseless'), ('awgn', 'awgn10')):
                selected = [r for r in validation if r['path'] == path and r['layout'] == ('clean' if path == 'clean' else str(t))]
                if selected:
                    for metric in ('source_psnr', 'source_ssim', 'source_lpips', 'photo_psnr', 'photo_ssim', 'photo_lpips'):
                        row[key+'_'+metric] = selected[-1].get(metric)
            rows.append(row)
    if not rows: return dict(status='partial', rows=[])
    if (len({r['channel_parameters_sha256'] for r in initializations}) != 1 or
            len({r['teacher_semantic_sha256'] for r in initializations}) != 1 or len(sample_hashes) != 1):
        raise RuntimeError('unpaired initialization, teacher or diagnostic sample')
    for tier in (1, 2, 3):
        selected = [r for r in rows if r['tier'] == tier]
        powers = [r['mean_complex_power'] for r in selected]
        matched = max(powers)/max(min(powers), 1e-30) <= 1.05
        for row in selected:
            row['average_power_matched_within_5pct'] = matched
            row['eligible_for_power_matched_quality_comparison'] = matched and row['within_average_power_budget']
    comparison = out/'comparison'; comparison.mkdir(exist_ok=True)
    write_csv(comparison/'summary.csv', rows)
    result = dict(status='complete' if len(initializations) == len(arms) else 'partial', rows=rows,
        matched_initial_channel_weights=True, matched_teacher=True, sample_sha256=next(iter(sample_hashes)),
        interpretation='rank quality only when power-budget and power-match flags pass; no free per-point/block gain side information')
    json_write(comparison/'summary.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for arm in arms:
        history = read_rows(out/arm/'metrics/validation.jsonl')
        for col, tier in enumerate(('1', '2', '3')):
            for path, style in (('finite_noiseless', '--'), ('awgn', '-')):
                series = [r for r in history if r['path'] == path and r['layout'] == tier]
                series = sorted({r['step']: r for r in series}.values(), key=lambda r: r['step'])
                for ax, metric in zip(axes[:, col], ('source_psnr', 'latent_nmse')):
                    if series: ax.plot([r['step'] for r in series], [r[metric] for r in series], style,
                                       label=arm+'/'+('finite' if path == 'finite_noiseless' else '10dB'))
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]; ax.set(title=f'q{col+1}', xlabel='Channel optimizer updates',
                                       ylabel='Source PLY PSNR (dB)' if row == 0 else 'Latent NMSE')
            ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=6)
            if row: ax.set_yscale('log')
    fig.savefig(comparison/'curves.png', dpi=150); plt.close(fig)
    return result


def main(args):
    plan = build_plan(args)
    if args.dry_run: print(json.dumps(plan, indent=2)); return plan
    if args.arm: return run_arm(args, args.arm)
    out = Path(args.out).resolve()
    if out.exists():
        if not args.resume_suite: raise FileExistsError('use NEW output or --resume-suite')
        if json.loads((out/'plan.json').read_text(encoding='utf-8')) != plan: raise ValueError('resume suite plan differs')
    else:
        out.mkdir(parents=True, exist_ok=False); json_write(out/'plan.json', plan)
    for job in plan['jobs']:
        arm = job['arm']
        if (out/arm/'complete.json').exists(): continue
        command = list(job['command'])
        if args.resume_suite: command += ['--resume-suite']
        print(f'Start {arm}; progress log: {out/(arm+".log")}', flush=True)
        with (out/(arm+'.log')).open('a', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            json_write(out/'status.json', dict(status='failed', arm=arm, returncode=result.returncode))
            raise RuntimeError(f'{arm} failed; inspect {out/(arm+".log")}')
        summarize(out, args.arms)
    result = summarize(out, args.arms)
    json_write(out/'complete.json', dict(status='complete', arms=args.arms))
    print(f'Completed suite: {out}/comparison/summary.csv', flush=True)
    return result


if __name__ == '__main__': main(parser().parse_args())
