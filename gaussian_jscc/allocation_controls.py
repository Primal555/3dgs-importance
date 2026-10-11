"""Frozen-codec q0 controls: draw rules, training sampling, and independent beta sweep."""
import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='completed multiscene run with final and end_render bundles')
    p.add_argument('--out', required=True, help='new common output directory')
    p.add_argument('--manifest', help='override the manifest recorded in the source run')
    p.add_argument('--scenes', nargs='+', help='default: all training scenes selected by the source run')
    p.add_argument('--allocation-steps', type=int, default=1000, help='updates PER SCENE, for each fresh allocator')
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--test-views', type=int, default=0, help='0: all held-out test cameras; positive: labelled quick subset')
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--image-views', type=int, default=2)
    p.add_argument('--seed', type=int, help='default: original training seed')
    p.add_argument('--deployment-seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true', help='preflight and print commands without creating outputs')
    return p


def build_plan(args, python=None):
    """Build fail-fast commands; beta=.01/deployment is trained only once."""
    python = python or sys.executable
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    training = json.loads((run/'training.json').read_text(encoding='utf-8'))
    manifest_path = Path(args.manifest or training['manifest']).resolve()
    contents = json.loads(manifest_path.read_text(encoding='utf-8'))
    root = (manifest_path.parent/contents.get('root', '.')).resolve()
    available = {s['name']: s for s in contents['scenes'] if s['role'] == 'train'}
    scenes = args.scenes or [s['name'] for s in training['selected_scenes'] if s['role'] == 'train']
    if not scenes or len(set(scenes)) != len(scenes) or not set(scenes).issubset(available):
        raise ValueError('select distinct training scenes in the source manifest')
    if (min(args.allocation_steps, args.validate_every, args.trials, args.image_views) < 1 or
            args.test_views < 0):
        raise ValueError('invalid step/trial/view count')
    checkpoints = run/'checkpoints'
    for path in [checkpoints/'final/codec.pt', checkpoints/'end_render/codec.pt',
                 *(checkpoints/'final'/f'{name}.pt' for name in scenes),
                 *(root/available[name]['ply'] for name in scenes),
                 *(root/available[name]['source'] for name in scenes)]:
        if not path.exists():
            raise FileNotFoundError(f'preflight: missing {path}')
    seed = training.get('seed', 42) if args.seed is None else args.seed
    snr = training.get('snr', 10.)
    channel = training.get('channel', 'awgn')
    net_bits = training.get('net_bits_per_use', 2.)
    if not math.isfinite(snr) or not math.isfinite(net_bits) or net_bits <= 0:
        raise ValueError('invalid recorded SNR or digital efficiency')
    evaluation_base = [python, '-u', '-m', 'gaussian_jscc.multiscene_experiments',
        '--manifest', str(manifest_path), '--role', 'train', '--scenes', *scenes,
        '--device', args.device, '--snrs', str(snr), '--channel', channel,
        '--seed', str(seed), '--deployment-seed', str(args.deployment_seed),
        '--trials', str(args.trials), '--test-views', str(args.test_views),
        '--net-bits-per-use', str(net_bits), '--output-profile', 'compact',
        '--image-views', str(args.image_views), '--image-layouts', 'mask',
        '--require-allocations', '--shuffle-seeds']
    jobs = [{'name': 'draw_rules', 'kind': 'evaluate', 'section': 'draw_rules',
        'command': evaluation_base+['--checkpoint', str(checkpoints/'final/codec.pt'),
             '--allocations-dir', str(checkpoints/'final'), '--out', str(out/'01_draw_rules'),
             '--draw-counts', '10', '5', '3', '--image-layouts', 'mask', 'draws_5', 'draws_3']}]
    controls = [('deployment', .01, out/'02_sampling/deployment_beta0p01'),
                ('single', .01, out/'02_sampling/single_beta0p01'),
                ('deployment', .003, out/'03_beta/deployment_beta0p003'),
                ('deployment', .001, out/'03_beta/deployment_beta0p001')]
    for sampling, beta, dest in controls:
        name = dest.name
        command = [python, '-u', '-m', 'gaussian_jscc.multiscene', '--allocation-only',
            '--manifest', str(manifest_path), '--train-scenes', *scenes,
            '--checkpoint', str(checkpoints/'end_render/codec.pt'), '--out', str(dest),
            '--bootstrap-steps', '0', '--render-steps', '0', '--joint-steps', '0',
            '--allocation-steps', str(args.allocation_steps), '--allocation-sampling', sampling,
            '--beta', str(beta), '--seed', str(seed), '--device', args.device,
            '--validate-every', str(args.validate_every), '--save-every', str(args.validate_every),
            '--output-profile', 'compact', '--image-policy', 'none']
        # Copy all other optimization/batching controls, never architecture/statistics.
        for key in ('blocks_per_batch', 'views_per_step', 'validation_views', 'validation_trials',
                    'mask_lr', 'keep_lr', 'mask_adam_eps', 'rate_chunk_size'):
            if key in training:
                command.extend(['--'+key.replace('_', '-'), str(training[key])])
        jobs.append({'name': name, 'kind': 'fit', 'sampling': sampling, 'beta': beta,
                     'command': command})
        jobs.append({'name': name+'_test', 'kind': 'evaluate', 'sampling': sampling, 'beta': beta,
            'command': evaluation_base+['--checkpoint', str(dest/'checkpoints/final/codec.pt'),
                                       '--out', str(dest/'test')]})
    return {'source_run': str(run), 'manifest': str(manifest_path), 'scenes': scenes,
        'seed': seed, 'allocation_steps_per_scene': args.allocation_steps,
        'frozen_checkpoint': str(checkpoints/'end_render/codec.pt'),
        'sampling_baseline_reused_for_beta': str(controls[0][2]/'test'),
        'sampling_comparison_note': 'single vs deployment changes BOTH keep and positive-tier sampling/rate gradients; not a q0-only causal intervention',
        'draw_comparison_note': 'same final probabilities; fewer draws affect deletion AND positive-tier votes',
        'jobs': jobs}


def summarize(out, plan):
    """Keep evidence and metrics separate from beta-dependent total objectives."""
    out = Path(out)
    rows = []
    controls_identity = None
    for job in plan['jobs']:
        command = job['command']
        dest = Path(command[command.index('--out')+1])
        if job['kind'] == 'fit':
            complete = json.loads((dest/'complete.json').read_text(encoding='utf-8'))
            if complete.get('frozen_codec_verified') is not True:
                raise RuntimeError(f'{dest}: missing frozen codec verification')
            identity = complete['codec_id']
            if controls_identity is not None and identity != controls_identity:
                raise RuntimeError('controls did not use the same frozen codec')
            controls_identity = identity
            continue
        for line in (dest/'results.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['layout'] != 'mask' and not row['layout'].startswith('draws_'):
                continue
            row.update(control=job['name'], control_sampling=job.get('sampling'),
                       control_beta=job.get('beta'), evaluation_directory=str(dest))
            rows.append(row)
    with (out/'comparison.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    plot_controls(out, rows, plan)
    return rows


def plot_controls(out, rows, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    charts = Path(out)/'charts'
    charts.mkdir()
    for scene in plan['scenes']:
        data = [r for r in rows if r['scene'] == scene]
        fig, axes = plt.subplots(2, 3, figsize=(17, 8), layout='constrained',
                                 gridspec_kw={'width_ratios': [1, 1, 1.5]})
        for i, is_draw_study in enumerate((True, False)):
            group = [r for r in data if (r['control'] == 'draw_rules') == is_draw_study]
            title = 'FINAL codec: draw rules' if is_draw_study else 'END_RENDER codec: frozen allocator fits'
            for ax, metric in zip(axes[i, :2], ('source_psnr', 'photo_psnr')):
                for row in group:
                    name = row['layout'] if is_draw_study else row['control'].removesuffix('_test')
                    ax.scatter(row['total_channel_uses']/1e6, row[metric], label=name)
                ax.set(xlabel='Total complex channel uses (millions)', ylabel=metric+' (dB)', title=title)
                ax.grid(alpha=.2)
            labels = [r['layout'] if is_draw_study else r['control'].removesuffix('_test') for r in group]
            left = [0.]*len(group)
            for q in range(4):
                shares = [r['tier_counts'][q]/r['source_gaussians'] for r in group]
                axes[i, 2].barh(labels, shares, left=left, label=f'q{q}')
                left = [a+b for a,b in zip(left, shares)]
            axes[i, 2].set(xlim=(0, 1), xlabel='Actual deployed Gaussian share', title=title)
            axes[i, 2].legend(fontsize=8)
            axes[i, 0].legend(fontsize=7, loc='best')
        fig.suptitle(scene+': separate studies; do not rank across the two codec initializers')
        fig.savefig(charts/f'{scene}_controls.png', dpi=160)
        plt.close(fig)
        # q0 learning, not just the ten-draw export, is visible over time.
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
        for job in plan['jobs']:
            if job['kind'] != 'fit':
                continue
            dest = Path(job['command'][job['command'].index('--out')+1])
            history = [json.loads(line) for line in
                (dest/'scenes'/scene/'allocation_history.jsonl').read_text(encoding='utf-8').splitlines()]
            updates = [h['step']/len(plan['scenes']) for h in history]
            label = job['sampling']+f', beta={job["beta"]:g}'
            for ax, key in zip(axes, ('mean_q0_probability', 'expected_q0_after_ten_draws', 'hard_payload_symbols')):
                ax.plot(updates, [h[key] for h in history], label=label)
                ax.set(xlabel='Allocator updates per scene', ylabel=key)
                ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.savefig(charts/f'{scene}_q0_history.png', dpi=160)
        plt.close(fig)


def main(args):
    plan = build_plan(args)
    for job in plan['jobs']:
        print(job['name']+': '+shlex.join(job['command']), flush=True)
    if args.dry_run:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    progress = []
    for job in plan['jobs']:
        print('\nStarting '+job['name'], flush=True)
        subprocess.run(job['command'], check=True)
        # Refuse to start another control if a frozen fit changed the codec.
        if job['kind'] == 'fit':
            command = job['command']
            complete = json.loads((Path(command[command.index('--out')+1])/'complete.json').read_text(encoding='utf-8'))
            if complete.get('frozen_codec_verified') is not True:
                raise RuntimeError('frozen-codec verification failed')
        progress.append(job['name'])
        (out/'progress.json').write_text(json.dumps({'completed_jobs': progress}, indent=2), encoding='utf-8')
    rows = summarize(out, plan)
    (out/'complete.json').write_text(json.dumps({'completed_jobs': progress, 'comparison_rows': len(rows),
        'codec_retrained': False, 'joint_training_run': False}, indent=2), encoding='utf-8')
    print(f'Controls complete: {out}', flush=True)


if __name__ == '__main__':
    main(parser().parse_args())
