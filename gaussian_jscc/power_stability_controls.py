"""Finish the missing block continuation, then reproduce short joint instability.

Never continues the failed joint/latest weights. Both probes fork the frozen
channel endpoint, preserving its Adam state; plain and instrumented runs start
independently. Probe failure does not prevent another arm/group from running.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from .channel_normalization_controls import semantic_digest
from .cli import device_for
from .data import read_ply
from .joint_stability_audit import audit_joint_step
from .modular_checkpoint import capture_rng, restore_rng, restore_training_checkpoint, save_training_checkpoint, validate_scene
from .modular_experiment import json_write, make_scene
from .modular_snr_controls import control_inputs
from .power_followup import build_plan as followup_plan, encoder_digest, joint_optimizer, parser as followup_parser
from .render_validation import append_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--power-run', required=True, help='original six-arm normalization suite')
    p.add_argument('--followup', required=True, help='partial suite containing point/end_channel_bootstrap')
    p.add_argument('--out', required=True, help='NEW output; no source modifications')
    p.add_argument('--manifest')
    p.add_argument('--channel-steps', type=int, default=4000)
    p.add_argument('--audit-steps', type=int, default=16)
    p.add_argument('--top-k', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume-suite', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--mode', choices=('point', 'block'), help=argparse.SUPPRESS)
    p.add_argument('--arm', choices=('plain', 'instrumented'), help=argparse.SUPPRESS)
    return p


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def block_options(args, endpoint):
    values = ['--run', str(Path(args.power_run).resolve()), '--out', str(Path(args.out).resolve()/'block_continuation'),
        '--modes', 'block', '--channel-steps', str(args.channel_steps), '--joint-steps', '0',
        '--test-views', str(len(endpoint['view_names'])), '--render-trials', str(endpoint['render_trials']),
        '--seed', str(args.seed), '--device', args.device, '--cpu-threads', str(args.cpu_threads),
        '--lpips' if args.lpips else '--no-lpips']
    if args.manifest: values += ['--manifest', str(Path(args.manifest).resolve())]
    return values


def read_endpoint(path, mode, target_step):
    checkpoint = Path(path)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    progress = saved['training']['progress']
    if (saved.get('version') != 4 or saved['config'].get('channel_power_mode', 'point') != mode
            or progress['phase'] != 'channel_bootstrap' or not progress.get('phase_complete')
            or not progress.get('freeze_semantic') or progress['completed'].get('channel_render', 0)
            or progress['step'] != target_step or not saved['training']['optimizer']['state']):
        raise ValueError('probe requires completed frozen-semantic channel endpoint with Adam, never joint/latest')
    return saved


def build_plan(args):
    if min(args.channel_steps, args.audit_steps, args.top_k, args.cpu_threads) < 1:
        raise ValueError('positive continuation/probe budgets required')
    out = Path(args.out).resolve(); parent = Path(args.followup).resolve(); power = Path(args.power_run).resolve()
    for source in (parent, power):
        if out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('NEW output must be separate from sources')
    endpoint = json.loads((parent/'point/evaluation/channel_end/complete.json').read_text(encoding='utf-8'))
    if endpoint['scope'] != 'test' or endpoint['seed'] != args.seed or endpoint['lpips'] != args.lpips:
        raise ValueError('seed/LPIPS must match the saved point test endpoint')
    options = block_options(args, endpoint)
    block = followup_plan(followup_parser().parse_args(options))
    target = block['sources']['block']['step']+args.channel_steps
    point_path = parent/'point/checkpoints/end_channel_bootstrap/codec.pt'
    saved = read_endpoint(point_path, 'point', target)
    if endpoint['step'] != target: raise ValueError('point test endpoint step mismatch')
    cfg = json.loads(json.dumps(saved['config'])); cfg.pop('channel_power_mode', None)
    other = dict(block['sources']['block']['config']); other.pop('channel_power_mode', None)
    if (cfg != other or semantic_digest(saved['state_dict']) != block['sources']['block']['semantic_sha256']
            or saved['training']['scene_record'] != block['sources']['block']['scene']):
        raise ValueError('point and block must share scene, semantic teacher and channel architecture')
    return dict(version=1, point_checkpoint=str(point_path), point_sha256=digest(point_path),
        point_run=str(parent/'point'), block_source=block['sources']['block'], target_channel_step=target,
        block_options=options, seed=args.seed, audit_steps=args.audit_steps, top_k=args.top_k,
        test_view_names=endpoint['view_names'], render_trials=endpoint['render_trials'], lpips=args.lpips,
        protocol='block +4000 channel only; each mode independent plain/instrumented short joint probes; unchanged loss/LRs/no clipping',
        restart='completed probes skipped; an interrupted short probe restarts from its immutable channel endpoint')


def source_paths(args, mode, plan):
    if mode == 'point': return Path(plan['point_checkpoint']), Path(plan['point_run'])
    run = Path(args.out).resolve()/'block_continuation/block'
    return run/'checkpoints/end_channel_bootstrap/codec.pt', run


def audit_worker(args, mode):
    plan = build_plan(args); checkpoint, run = source_paths(args, mode, plan)
    saved = read_endpoint(checkpoint, mode, plan['target_channel_step'])
    if semantic_digest(saved['state_dict']) != plan['block_source']['semantic_sha256']:
        raise ValueError('probe teacher mismatch')
    source_sha = digest(checkpoint); folder = Path(args.out).resolve()/'audit'/mode
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads)
    results = {}
    for arm in ((args.arm,) if args.arm else ('plain', 'instrumented')):
        child = folder/arm
        control = dict(plan=plan, mode=mode, arm=arm, source_sha256=source_sha)
        if child.exists():
            if not args.resume_suite: raise FileExistsError('use new output or --resume-suite')
            if json.loads((child/'control.json').read_text(encoding='utf-8')) != control: raise ValueError('probe resume settings differ')
            if (child/'complete.json').exists():
                previous_result = json.loads((child/'complete.json').read_text(encoding='utf-8'))
                if previous_result['status'] == 'complete':
                    results[arm] = previous_result; continue
            if (child/'error.json').exists(): (child/'error.json').unlink()
            # Re-running at most16 diagnostic updates is explicit, not exact
            # continuation and never adds another budget to source training.
            (child/'steps.jsonl').write_text('', encoding='utf-8')
        else:
            child.mkdir(parents=True, exist_ok=False); json_write(child/'control.json', control)
        model, old, progress, rng, record = restore_training_checkpoint(checkpoint, device)
        original_digest = encoder_digest(model.state_dict())
        metadata = json.loads((run/'training.json').read_text(encoding='utf-8'))
        original, _, spec, previous, _ = control_inputs(metadata, args.manifest)
        raw, degree = read_ply(spec['ply'])
        scene = make_scene(spec, raw, degree, model, original, previous, child); del raw
        validate_scene(record, scene.record)
        optimizer = joint_optimizer(model, old, 1e-4, 1e-5); restore_rng(rng)
        progress = copy.deepcopy(progress)
        progress.update(phase='channel_render', phase_step=0, phase_complete=False, freeze_semantic=False)
        progress['phase_budgets']['channel_render'] = args.audit_steps
        progress['module_lrs'].update(channel_encoder=1e-4, channel_decoder=1e-4, semantic_decoder=1e-5)
        save_training_checkpoint(child/'checkpoints/initial/codec.pt', model, optimizer, progress, capture_rng(), record)
        save_training_checkpoint(child/'checkpoints/latest/codec.pt', model, optimizer, progress, capture_rng(), record)
        status = 'complete'; completed = 0
        try:
            for local_step in range(1, args.audit_steps+1):
                started = time.perf_counter(); step = plan['target_channel_step']+local_step
                row = audit_joint_step(model, optimizer, scene, step, local_step, args, instrumented=arm == 'instrumented')
                row.update(mode=mode, arm=arm, step_seconds=time.perf_counter()-started)
                append_json(child/'steps.jsonl', row)
                completed = local_step; progress.update(step=step, phase_step=local_step, phase_complete=local_step == args.audit_steps)
                progress['completed']['channel_render'] = local_step
                save_training_checkpoint(child/'checkpoints/latest/codec.pt', model, optimizer, progress, capture_rng(), record)
                after = row['after']['loss'] if row['after'] is not None else None
                print(f'{mode}/{arm} {local_step}/{args.audit_steps}: grad={row["grad_norm"]:.6g}; '
                      f'loss={row["before"]["loss"]:.6g}; same-input post={after}', flush=True)
                if row['after'] is None: status = 'post_update_error'; break
        except (RuntimeError, ValueError) as exc:
            status = 'probe_error'; json_write(child/'error.json', dict(error=str(exc), next_phase_step=completed+1))
            print(f'{mode}/{arm}: {exc}; preserve evidence and continue next arm', flush=True)
        if encoder_digest(model.state_dict()) != original_digest: raise RuntimeError('probe changed frozen semantic encoder')
        if digest(checkpoint) != source_sha: raise RuntimeError('probe modified source checkpoint')
        results[arm] = dict(status=status, updates=completed, source_step=plan['target_channel_step'],
            source_sha256=source_sha, semantic_encoder_unchanged=True, channel_adam_inherited=True)
        json_write(child/'complete.json', results[arm])
        del model, optimizer, old, scene
        if device.type == 'cuda': torch.cuda.empty_cache()
    return results


def paired_endpoints(args, plan):
    from .channel_audit import write_csv
    rows = []; identities = []
    for mode in ('point', 'block'):
        _, run = source_paths(args, mode, plan)
        root = run/'evaluation/channel_end'
        identity = json.loads((root/'complete.json').read_text(encoding='utf-8')); identities.append(identity)
        power = json.loads((root/'power.json').read_text(encoding='utf-8'))
        for line in (root/'metrics/validation.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r['path'] not in ('awgn', 'finite_noiseless') or r['layout'] not in ('1', '2', '3'): continue
            rows.append(dict(mode=mode, path=r['path'], tier=int(r['layout']), symbols=(8, 24, 48)[int(r['layout'])-1],
                views=r['views'], trials=r['trials'], step=r['step'],
                mean_complex_power=power['prefixes'][int(r['layout'])-1]['mean_complex_power'],
                **{k: r.get(k) for k in ('source_psnr', 'source_ssim', 'source_lpips', 'photo_psnr', 'photo_ssim', 'photo_lpips', 'latent_nmse')}))
    for key in ('step', 'scope', 'view_names', 'render_trials', 'seed', 'lpips'):
        if identities[0][key] != identities[1][key]: raise ValueError('unpaired channel endpoint: '+key)
    out = Path(args.out)/'comparison'; out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'channel_endpoints.csv', rows)
    json_write(out/'channel_endpoints.json', dict(rows=rows, test_view_names=identities[0]['view_names'],
        caveat='same trained steps/cameras/noise; actual mean power recorded, not silently rescaled'))
    return rows


def summarize_probes(args):
    from .channel_audit import write_csv
    rows = []; summaries = []
    for mode in ('point', 'block'):
        for arm in ('plain', 'instrumented'):
            root = Path(args.out)/'audit'/mode/arm
            if (root/'complete.json').exists():
                done = json.loads((root/'complete.json').read_text(encoding='utf-8'))
                summaries.append(dict(mode=mode, arm=arm, **done))
            if not (root/'steps.jsonl').exists(): continue
            for line in (root/'steps.jsonl').read_text(encoding='utf-8').splitlines():
                r = json.loads(line)
                rows.append(dict(mode=mode, arm=arm, phase_step=r['phase_step'], grad_norm=r['grad_norm'],
                    loss_before=r['before']['loss'], loss_after=r['after']['loss'] if r['after'] else None,
                    latent_before=r['before']['latent_mse'], latent_after=r['after']['latent_mse'] if r['after'] else None,
                    raw_attribute_grad=r['scene_gradient_norms']['attributes'],
                    decomposition_relative_error=r.get('decomposition', {}).get('gradient_sum_relative_error')))
    out = Path(args.out)/'comparison'; out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'joint_probes.csv', rows); json_write(out/'joint_probes.json', dict(rows=rows, arms=summaries,
        interpretation='plain/instrumented share immutable source and deterministic input schedule; CUDA atomic backward may still vary; no clipping or protective optimizer changes'))
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        for mode in ('point', 'block'):
            for arm in ('plain', 'instrumented'):
                selected = [r for r in rows if r['mode'] == mode and r['arm'] == arm]
                if not selected: continue
                for ax, key in zip(axes, ('grad_norm', 'latent_before')):
                    ax.plot([r['phase_step'] for r in selected], [max(r[key], 1e-30) for r in selected],
                            '-o' if arm == 'plain' else '--x', label=mode+'/'+arm)
        for ax, title in zip(axes, ('Joint parameter gradient norm', 'Original-y MSE before update')):
            ax.set(title=title, xlabel='Joint update', yscale='log'); ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.savefig(out/'joint_probes.png', dpi=150); plt.close(fig)
    return summaries


def main(args):
    plan = build_plan(args)
    if args.dry_run: print(json.dumps(plan, indent=2)); return plan
    if args.mode: return audit_worker(args, args.mode)
    out = Path(args.out).resolve()
    if out.exists():
        if not args.resume_suite: raise FileExistsError('use new output or --resume-suite')
        if json.loads((out/'plan.json').read_text(encoding='utf-8')) != plan: raise ValueError('suite settings changed')
    else:
        out.mkdir(parents=True, exist_ok=False); json_write(out/'plan.json', plan)
    jobs = [('block_continuation', [sys.executable, '-u', '-m', 'gaussian_jscc.power_followup', *plan['block_options']])]
    for mode in ('point', 'block'):
        command = [sys.executable, '-u', '-m', 'gaussian_jscc.power_stability_controls',
            '--power-run', str(Path(args.power_run).resolve()), '--followup', str(Path(args.followup).resolve()),
            '--out', str(out), '--channel-steps', str(args.channel_steps), '--audit-steps', str(args.audit_steps),
            '--top-k', str(args.top_k), '--seed', str(args.seed), '--device', args.device,
            '--cpu-threads', str(args.cpu_threads), '--mode', mode, '--lpips' if args.lpips else '--no-lpips']
        if args.manifest: command += ['--manifest', str(Path(args.manifest).resolve())]
        for arm in ('plain', 'instrumented'):
            # Separate CUDA processes also isolate a poisoned device context,
            # not merely Python exceptions inside an optimizer loop.
            jobs.append((mode+'_'+arm, command+['--arm', arm]))
    failures = []
    for name, command in jobs:
        if name.startswith('block_') and name != 'block_continuation' and not source_paths(args, 'block', plan)[0].exists():
            failures.append(dict(job=name, error='block continuation unavailable')); continue
        print(f'Start {name}: {out/(name+".log")}', flush=True)
        with (out/(name+'.log')).open('a', encoding='utf-8') as log:
            result = subprocess.run(command+(['--resume-suite'] if args.resume_suite else []), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode: failures.append(dict(job=name, returncode=result.returncode))
        if name == 'block_continuation' and result.returncode == 0:
            try:
                paired_endpoints(args, plan)
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                failures.append(dict(job='channel_comparison', error=str(exc)))
                print(f'Channel comparison failed: {exc}; continue independent probes', flush=True)
    summaries = summarize_probes(args)
    partial = bool(failures or len(summaries) != 4 or any(r['status'] != 'complete' for r in summaries))
    record = dict(status='partial' if partial else 'complete', failures=failures, probes=summaries)
    json_write(out/'status.json', record)
    if not partial: json_write(out/'complete.json', record)
    return record


if __name__ == '__main__': main(parser().parse_args())
