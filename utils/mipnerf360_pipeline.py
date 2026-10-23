"""Prune nine Mip-NeRF 360 assets, then expand the completed shared codec."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from gaussian_jscc.modular_experiment import json_write
from gaussian_jscc.successive_transfer import load_source, NEW_TRAIN, NEW_HELDOUT, OLD_TRAIN
from utils.maskgaussian_batch import build_plan, run_plan, REPO
from utils.prune_initialization import ply_info


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='completed shared successive multi-scene run')
    p.add_argument('--out', required=True)
    p.add_argument('--raw-manifest', default=str(REPO / 'configs/mipnerf360_raw.json'))
    for name, value in (('prune-steps', 5000), ('prune-resolution', 2), ('prune-validate-every', 1000),
                        ('steps-per-scene', 1250), ('validate-every', 250), ('save-every', 100),
                        ('test-views', 32), ('render-trials', 4)):
        p.add_argument('--' + name, type=int, default=value)
    p.add_argument('--lambda-mask', type=float, default=.1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--latent-weight', type=float, default=1.)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def training_command(args):
    command = [sys.executable, '-u', '-m', 'gaussian_jscc.successive_transfer',
               '--run', str(Path(args.run).resolve()), '--manifest', str(Path(args.out).resolve() / 'dataset.json'),
               '--out', str(Path(args.out).resolve() / 'training'), '--scenes', *(OLD_TRAIN + NEW_TRAIN),
               '--heldout-scenes', *NEW_HELDOUT]
    for key in ('steps_per_scene', 'lr', 'latent_weight', 'validate_every', 'save_every', 'test_views', 'render_trials'):
        command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
    command.append('--lpips' if args.lpips else '--no-lpips')
    return command


def run_training(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def asset_identity(scene):
    """Once per invocation, not per update: verify PLY, camera files and photos."""
    source = Path(scene['source'])
    files = sorted(p for p in (source / 'images').rglob('*') if p.is_file())
    files += sorted(p for p in (source / 'sparse/0').glob('*')
                    if p.stem in ('cameras', 'images') and p.suffix in ('.bin', '.txt'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode('utf-8') + b'\0')
        digest.update(bytes.fromhex(file_sha256(path)))
    return dict(input_sha256=file_sha256(scene['input_ply']), source_assets_sha256=digest.hexdigest())


def completed_pruning(out, preflight):
    """Reuse measured, fully exported scenes; never resume a partial PLY as Adam state."""
    expected = {s['name']: s for s in preflight['scenes']}
    completed = {}
    passes = sorted((out / 'preprocessing').glob('pass_*'))
    for directory in passes:
        plan_path, summary_path = directory / 'plan.json', directory / 'summary.json'
        if not plan_path.exists() or not summary_path.exists():
            continue
        plan, summary = read_json(plan_path), read_json(summary_path)
        by_name = {s['name']: s for s in plan['scenes']}
        for record in summary.get('scenes', []):
            name = record['name']; scene = by_name[name]; original = expected[name]
            for key in ('source', 'input_ply', 'input_info', 'input_sha256', 'source_assets_sha256'):
                if scene[key] != original[key]:
                    raise ValueError(name + ': preprocessing inputs changed')
            info = ply_info(scene['output_ply'], validate_values=True)
            if (info['points'] != record['output_points'] or info['bytes'] != record['output_bytes']
                    or file_sha256(scene['output_ply']) != record.get('output_sha256')):
                raise ValueError(name + ': completed pruning export changed')
            completed[name] = dict(name=name, role='train' if name in NEW_TRAIN else 'heldout',
                                   source=scene['source'], ply=scene['output_ply'])
    return completed, len(passes)


def main(args):
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('source and output must be separate')
    if (min(args.prune_steps, args.prune_resolution, args.prune_validate_every, args.steps_per_scene,
            args.validate_every, args.save_every, args.render_trials) <= 0 or args.test_views < 0
            or any(not math.isfinite(x) or x <= 0 for x in (args.lr, args.latent_weight))):
        raise ValueError('invalid training/preprocessing budget')
    saved, metadata, cfg, source = load_source(run)
    del saved
    previous = {s['name']: s for s in metadata['selected_scenes']}
    if not set(OLD_TRAIN) <= previous.keys():
        raise ValueError('source must include the original Train/Truck/Playroom scenes')
    old = [previous[name] for name in OLD_TRAIN]
    for scene in old:
        for key in ('source', 'ply'):
            if not Path(scene[key]).exists():
                raise FileNotFoundError(f'{scene["name"]}: missing old {key}: {scene[key]}')
    if set(NEW_HELDOUT) & set(source.get('training_scene_history', ())):
        raise ValueError('heldout scene already present in codec training history')
    # Read-only validation of ALL nine assets, including images, before any GPU subprocess.
    preflight = build_plan(args.raw_manifest, out / 'preprocessing/_preflight',
        scenes=NEW_TRAIN + NEW_HELDOUT, steps=args.prune_steps, lambda_mask=args.lambda_mask,
        resolution=args.prune_resolution, validate_every=args.prune_validate_every)
    for scene in preflight['scenes']:
        if scene['input_info']['sh_degree'] != cfg.sh_degree:
            raise ValueError(scene['name'] + ': SH degree differs from source codec')
        print('Verify input content: ' + scene['name'], flush=True)
        scene.update(asset_identity(scene))
    identity = dict(version='successive_mip360_pipeline_v1', source=source, old_scenes=old,
                    settings={k: v for k, v in vars(args).items() if k not in ('resume', 'dry_run')},
                    preprocessing=preflight, training_scenes=list(OLD_TRAIN + NEW_TRAIN),
                    heldout_scenes=list(NEW_HELDOUT), total_updates=9 * args.steps_per_scene,
                    training_command=training_command(args))
    if args.dry_run:
        print(json.dumps(identity, indent=2)); return identity
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not gpu or gpu == '-1':
        raise ValueError('set CUDA_VISIBLE_DEVICES explicitly to an available GPU')
    if args.resume:
        if read_json(out / 'pipeline.json') != identity:
            raise ValueError('resume source, paths or settings changed')
    else:
        out.mkdir(parents=True, exist_ok=False)
        json_write(out / 'pipeline.json', identity)
    start = time.perf_counter(); stage = 'preprocessing'
    try:
        completed, pass_number = completed_pruning(out, preflight)
        pending = [name for name in NEW_TRAIN + NEW_HELDOUT if name not in completed]
        json_write(out / 'progress.json', dict(status='running', stage=stage, reused_scenes=list(completed)))
        if pending:
            plan = build_plan(args.raw_manifest, out / 'preprocessing' / f'pass_{pass_number:03d}',
                scenes=pending, steps=args.prune_steps, lambda_mask=args.lambda_mask,
                resolution=args.prune_resolution, validate_every=args.prune_validate_every)
            identities = {s['name']: s for s in preflight['scenes']}
            for scene in plan['scenes']:
                for key in ('input_sha256', 'source_assets_sha256'):
                    scene[key] = identities[scene['name']][key]
            print(f'Preprocessing: {len(pending)} remaining; {len(completed)} completed scenes reused', flush=True)
            run_plan(plan, gpu)
            completed, _ = completed_pruning(out, preflight)
        if set(completed) != set(NEW_TRAIN + NEW_HELDOUT):
            raise RuntimeError('incomplete pruning exports')
        dataset = dict(root='.', scenes=old + [completed[n] for n in NEW_TRAIN + NEW_HELDOUT])
        manifest = out / 'dataset.json'
        if manifest.exists() and read_json(manifest) != dataset:
            raise ValueError('generated dataset changed')
        json_write(manifest, dataset)
        stage = 'training'; training = out / 'training'; command = training_command(args)
        json_write(out / 'progress.json', dict(status='running', stage=stage,
                   preprocessing_seconds=time.perf_counter() - start))
        done = training / 'complete.json'
        if not done.exists():
            if training.exists():
                if not (training / 'checkpoints/latest/codec.pt').exists():
                    raise RuntimeError('training initialization interrupted before first checkpoint; choose a new training output')
                command.append('--resume')
            print('Shared codec: 9 training scenes, 3 heldouts; ' + str(identity['total_updates']) + ' actual updates', flush=True)
            run_training(command, cwd=str(REPO), env=dict(os.environ, PYTHONUNBUFFERED='1'))
        training_result = read_json(done)
        if training_result.get('status') != 'complete' or training_result.get('step') != identity['total_updates']:
            raise RuntimeError('codec subprocess did not finish the requested budget')
        result = dict(status='complete', training=str(training), total_updates=identity['total_updates'],
                      heldout_scenes=list(NEW_HELDOUT), elapsed_seconds=time.perf_counter() - start)
        json_write(out / 'complete.json', result); json_write(out / 'progress.json', result)
        return result
    except BaseException as exc:
        json_write(out / 'progress.json', dict(status='failed', stage=stage, error=str(exc),
                   elapsed_seconds=time.perf_counter() - start))
        raise


if __name__ == '__main__':
    main(parser().parse_args())
