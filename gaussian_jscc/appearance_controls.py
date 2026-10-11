"""Two controlled appearance studies on existing training/evaluation paths.

Color: same END_RENDER initializer, DC+SH heads versus full attribute codec.
Directions: paired random initializers, random-sphere versus training-camera
local responses; identical downstream render budgets. No allocator is fitted.
"""
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
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True, help='new directory; never overwrite prior results')
    p.add_argument('--manifest', help='override source manifest after migration')
    p.add_argument('--scenes', nargs='+', help='default: Truck if available, otherwise first trained scene')
    p.add_argument('--suite', choices=['color','directions','all'], default='all')
    p.add_argument('--color-steps', type=int, default=2000)
    p.add_argument('--bootstrap-steps', type=int, default=2000)
    p.add_argument('--render-steps', type=int, default=1000)
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--test-views', type=int, default=0)
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--image-views', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--color-rois', help='optional PREDECLARED scene/view/region boxes; same file for all groups')
    p.add_argument('--skip-oracle', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    return p


def build_plan(args, python=None):
    python = python or sys.executable
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    source = json.loads((run/'training.json').read_text(encoding='utf-8'))
    manifest = Path(args.manifest or source['manifest']).resolve()
    from .multiscene import manifest as read_manifest
    specs = read_manifest(manifest)
    trained = {s['name'] for s in source['selected_scenes'] if s['role'] == 'train'}
    available = {s['name']: s for s in specs if s['role'] == 'train' and s['name'] in trained}
    names = args.scenes or (['truck'] if 'truck' in available else list(available)[:1])
    if not names or len(set(names)) != len(names) or not set(names).issubset(available):
        raise ValueError('select distinct scenes present in BOTH source run and training manifest')
    if (min(args.color_steps, args.bootstrap_steps, args.render_steps, args.validate_every,
            args.trials, args.image_views) < 1 or args.test_views < 0 or
            not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError('invalid budgets/views/trials/learning rate')
    from .appearance_metrics import load_regions
    rois = load_regions(args.color_rois)
    initializer = run/'checkpoints/end_render/codec.pt'
    checks = [Path(available[n][key]) for n in names for key in ('ply','source')]
    if args.suite in ('all','color'):
        checks.append(initializer)
    checks.extend(run/'scenes'/n/'scene.json' for n in names)
    for path in checks:
        if not path.exists():
            raise FileNotFoundError(f'preflight: missing {path}')
    if out.exists():
        raise FileExistsError(f'use a new common output directory: {out}')
    from .data import read_ply
    from .allocation import scene_fingerprint
    identities = {}
    for name in names:
        raw, degree = read_ply(available[name]['ply'])
        old = json.loads((run/'scenes'/name/'scene.json').read_text(encoding='utf-8'))
        actual = scene_fingerprint(raw)
        if old.get('fingerprint') != actual or old.get('source_gaussians') != len(raw):
            raise ValueError(f'preflight: {name} PLY differs from source scene')
        if not old.get('train_view_names') or not old.get('validation_view_names'):
            raise ValueError(f'preflight: {name} missing recorded camera split')
        identities[name] = {'fingerprint': actual, 'count': len(raw), 'sh_degree': degree}
        del raw
    seed, snr = source.get('seed', 42), source.get('snr', 10.)
    channel = source.get('channel', 'awgn')
    rates = source.get('codec_config', {}).get('rates', source.get('rates', [0,4,12,24]))
    if len(rates) != 4 or not math.isfinite(snr):
        raise ValueError('these controls expect three positive prefixes and finite SNR')
    common = [python, '-u', '-m', 'gaussian_jscc.multiscene', '--manifest', str(manifest),
        '--train-scenes', *names, '--device', args.device, '--allocation-steps', '0', '--joint-steps', '0',
        '--seed', str(seed), '--snr', str(snr), '--channel', channel, '--rates', *map(str, rates),
        '--lr', str(args.lr), '--render-lr', str(args.lr), '--paired-training-rng', '--color-metrics',
        '--scene-reference-run', str(run), '--net-bits-per-use', str(source.get('net_bits_per_use', 2.)),
        '--validate-every', str(args.validate_every), '--save-every', str(args.validate_every),
        '--output-profile', 'compact', '--image-policy', 'endpoints', '--image-views', str(args.image_views)]
    for key in ('blocks_per_batch','block_size','decoder_window','hidden','depth','local_response_views',
                'views_per_step','validation_views','validation_trials','resolution','images','cpu_threads'):
        if key in source:
            common.extend(['--'+key.replace('_','-'), str(source[key])])
    if source.get('white_background'):
        common.append('--white-background')
    if args.color_rois:
        common += ['--color-rois', str(Path(args.color_rois).resolve())]
    evaluate = [python, '-u', '-m', 'gaussian_jscc.multiscene_experiments', '--manifest', str(manifest),
        '--role', 'train', '--scenes', *names, '--device', args.device, '--snrs', str(snr), '--channel', channel,
        '--seed', str(seed), '--trials', str(args.trials), '--test-views', str(args.test_views),
        '--net-bits-per-use', str(source.get('net_bits_per_use', 2.)),
        '--uniform-only', '--color-metrics', '--shuffle-seeds', '--output-profile', 'compact',
        '--image-layouts', '1', '2', '3', '--image-views', str(args.image_views)]
    if args.color_rois:
        evaluate += ['--color-rois', str(Path(args.color_rois).resolve())]
    jobs = []
    if args.suite in ('all','color'):
        if not args.skip_oracle:
            for name in names:
                spec = available[name]
                jobs.append({'name': name+'_color_oracle', 'kind': 'diagnose', 'section': 'color',
                    'command': [python, '-u', '-m', 'gaussian_jscc.color_ablation', '--training', str(run),
                        '--scene', name, '--checkpoint', str(initializer), '--ply', spec['ply'],
                        '--source', spec['source'], '--tiers', '2', '--trials', str(args.trials),
                        '--image-views', str(args.image_views), '--device', args.device,
                        '--out', str(out/'color_oracle'/name)]})
        jobs.append({'name': 'color_initial', 'kind': 'evaluate', 'section': 'color',
            'command': evaluate+['--checkpoint', str(initializer), '--out', str(out/'color_initial_test')]})
        for scope in ('color_heads', 'all'):
            dest = out/'color'/scope
            jobs.append({'name': 'color_'+scope, 'kind': 'fit', 'section': 'color',
                'command': common+['--render-only', '--checkpoint', str(initializer), '--trainable', scope,
                    '--bootstrap-steps', '0', '--render-steps', str(args.color_steps),
                    '--uniform-tier', '2', '--drop', '0', '--out', str(dest)]})
            jobs.append({'name': 'color_'+scope+'_test', 'kind': 'evaluate', 'section': 'color',
                'command': evaluate+['--checkpoint', str(dest/'checkpoints/final/codec.pt'), '--out', str(dest/'test')]})
    if args.suite in ('all','directions'):
        for directions in ('random', 'camera'):
            dest = out/'directions'/directions
            jobs.append({'name': 'directions_'+directions, 'kind': 'fit', 'section': 'directions',
                'command': common+['--bootstrap-directions', directions, '--trainable', 'all',
                    '--bootstrap-steps', str(args.bootstrap_steps), '--render-steps', str(args.render_steps),
                    '--drop', str(source.get('drop', .05)), '--out', str(dest)]})
            jobs.append({'name': 'directions_'+directions+'_test', 'kind': 'evaluate', 'section': 'directions',
                'command': evaluate+['--checkpoint', str(dest/'checkpoints/final/codec.pt'), '--out', str(dest/'test')]})
    return {'source_run': str(run), 'manifest': str(manifest), 'scenes': names, 'suite': args.suite,
        'initializer': str(initializer), 'seed': seed, 'rates': rates, 'snr': snr, 'channel': channel,
        'color_regions': rois, 'scene_identities': identities, 'jobs': jobs, 'no_allocator_training': True,
        'scope': 'color: scene adaptation of shared checkpoint; directions: paired single/subset-scene random starts',
        'comparison_boundary': 'compare WITHIN each study, not across different initializers/stage budgets'}


def summarize(out, plan):
    rows, history = [], []
    for job in plan['jobs']:
        c = job['command']; dest = Path(c[c.index('--out')+1])
        if job['kind'] == 'evaluate':
            for line in (dest/'results.jsonl').read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                rows.append(dict(row, control=job['name'], section=job['section']))
        if job['kind'] == 'fit':
            for name in plan['scenes']:
                for line in (dest/'scenes'/name/'validation.jsonl').read_text(encoding='utf-8').splitlines():
                    h = json.loads(line)
                    history.append(dict(h, control=job['name'], section=job['section'], scene=name,
                                        per_scene_updates=h['step']/len(plan['scenes'])))
    out = Path(out)
    (out/'comparison.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        columns = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    charts = out/'charts'; charts.mkdir()
    for section in ('color','directions'):
        for scene in plan['scenes']:
            data = [h for h in history if h['section'] == section and h['scene'] == scene]
            if not data:
                continue
            final = [r for r in rows if r['section'] == section and r['scene'] == scene]
            fig, axes = plt.subplots(1, 3, figsize=(13,4), layout='constrained')
            for label in dict.fromkeys(r['control'] for r in final):
                branch = sorted((r for r in final if r['control']==label), key=lambda r:int(r['layout']))
                for ax, metric in zip(axes, ('source_psnr','photo_psnr','source_color_rg_error_rmse')):
                    ax.plot([int(r['layout']) for r in branch], [r[metric] for r in branch], marker='o', label=label)
                    ax.set(xlabel='Prefix tier', ylabel=metric, xticks=[1,2,3]); ax.grid(alpha=.2)
            axes[0].legend(fontsize=7)
            fig.suptitle(f'{scene}: {section}, final heldout test cameras')
            fig.savefig(charts/f'{scene}_{section}_final_test.png', dpi=150); plt.close(fig)
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
            metrics = ['source_psnr','photo_psnr','source_color_rg_error_rmse',
                       'source_color_r_bias','source_color_g_bias','source_color_b_bias']
            for label in dict.fromkeys(h['control'] for h in data):
                branch = [h for h in data if h['control'] == label]
                for q in ('1','2','3'):
                    entries = [(h, next(e for e in h['layouts'] if e['layout'] == q)) for h in branch]
                    for ax, metric in zip(axes.flat, metrics):
                        ax.plot([h['per_scene_updates'] for h,e in entries], [e[metric] for h,e in entries],
                                label=f'{label}, q{q}', linewidth=1)
                        ax.set(xlabel='Actual updates per scene', ylabel=metric)
                        ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=7)
            fig.suptitle(f'{scene}: {section} paired controls; validation, not test selection')
            fig.savefig(charts/f'{scene}_{section}.png', dpi=150); plt.close(fig)
            fig, ax = plt.subplots(figsize=(9,4), layout='constrained')
            for job in plan['jobs']:
                if job['kind'] != 'fit' or job['section'] != section:
                    continue
                dest = Path(job['command'][job['command'].index('--out')+1])
                losses = [json.loads(line) for line in (dest/'scenes'/scene/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
                for phase in ('bootstrap','render'):
                    selected = [r for r in losses if r['phase'] == phase]
                    if selected:
                        offset = int(job['command'][job['command'].index('--bootstrap-steps')+1]) if phase=='render' else 0
                        ax.plot([offset+r['scene_step'] for r in selected], [r['loss'] for r in selected],
                                label=job['name']+' '+phase, alpha=.7)
            ax.set(xlabel='Actual updates per scene', ylabel='Phase objective (different phases are not comparable)')
            ax.legend(fontsize=8); ax.grid(alpha=.2)
            fig.savefig(charts/f'{scene}_{section}_loss.png', dpi=150); plt.close(fig)
    return rows


def main(args):
    plan = build_plan(args)
    for job in plan['jobs']:
        print(job['name']+': '+shlex.join(job['command']), flush=True)
    if args.dry_run:
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    completed, initializers = [], {}
    for job in plan['jobs']:
        print('Starting '+job['name'], flush=True)
        subprocess.run(job['command'], check=True)
        if job['kind'] == 'fit':
            c = job['command']; dest = Path(c[c.index('--out')+1])
            check = json.loads((dest/'complete.json').read_text(encoding='utf-8'))
            if not check.get('frozen_parameters_verified') or any(
                    phases.get('allocation',0) or phases.get('joint',0) for phases in check['scene_visits'].values()):
                raise RuntimeError('freeze/no-allocation audit failed')
            record = json.loads((dest/'training.json').read_text(encoding='utf-8'))
            identity = record.get('initializer_codec_id') if job['section']=='color' else record['random_initializer_codec_id']
            previous = initializers.setdefault(job['section'], identity)
            if identity != previous:
                raise RuntimeError('paired branches did not share the same initializer/statistics')
        completed.append(job['name'])
        (out/'progress.json').write_text(json.dumps({'completed_jobs': completed}, indent=2), encoding='utf-8')
    rows = summarize(out, plan)
    (out/'complete.json').write_text(json.dumps({'completed_jobs': completed, 'comparison_rows': len(rows),
        'paired_initializers_verified': initializers, 'allocator_trained': False}, indent=2), encoding='utf-8')
    print(f'Appearance controls complete: {out}', flush=True)


if __name__ == '__main__':
    main(parser().parse_args())
