"""Sequential, explicit-GPU MaskGaussian post-training from pretrained PLYs."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from utils.prune_initialization import ply_info


REPO = Path(__file__).resolve().parents[1]


def check_cameras(source):
    """Reuse COLMAP's CPU parser without importing scene's CUDA package."""
    sparse = source/'sparse'/'0'
    if sparse.is_dir():
        spec = importlib.util.spec_from_file_location('_prune_colmap_loader', REPO/'scene/colmap_loader.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        if all((sparse/f'{key}.bin').is_file() for key in ('images','cameras')):
            cameras = loader.read_intrinsics_binary(sparse/'cameras.bin')
            images = loader.read_extrinsics_binary(sparse/'images.bin')
        elif all((sparse/f'{key}.txt').is_file() for key in ('images','cameras')):
            cameras = loader.read_intrinsics_text(sparse/'cameras.txt')
            images = loader.read_extrinsics_text(sparse/'images.txt')
        else:
            raise FileNotFoundError(f'{source}: need sparse/0/cameras and images (.bin or .txt)')
        if not any((sparse/f'points3D.{extension}').is_file() and (sparse/f'points3D.{extension}').stat().st_size
                   for extension in ('ply','bin','txt')):
            raise FileNotFoundError(f'{source}: Scene requires sparse/0/points3D (.ply/.bin/.txt)')
        if len(images) < 2 or not cameras:
            raise ValueError(f'{source}: need nonempty train and held-out cameras')
        for image in images.values():
            if image.camera_id not in cameras:
                raise ValueError(f'{source}: unknown camera id {image.camera_id}')
            camera = cameras[image.camera_id]
            if camera.model not in ('PINHOLE','SIMPLE_PINHOLE'):
                raise ValueError(f'{source}: use undistorted PINHOLE/SIMPLE_PINHOLE cameras')
            photograph = source/'images'/Path(image.name).name
            if not photograph.is_file() or photograph.stat().st_size == 0:
                raise FileNotFoundError(f'{source}: missing/empty referenced photograph {photograph}')
        return
    # Blender's reader requires both files when --eval is selected.
    for split in ('train','test'):
        transform = source/f'transforms_{split}.json'
        frames = json.loads(transform.read_text(encoding='utf-8'))['frames']
        if not frames:
            raise ValueError(f'{source}: empty {split} cameras')
        for frame in frames:
            photograph = source/(frame['file_path']+'.png')
            if not photograph.is_file() or photograph.stat().st_size == 0:
                raise FileNotFoundError(f'{source}: missing/empty referenced photograph {photograph}')


def build_plan(manifest, out, scenes=None, steps=5000, start_iteration=30000,
               lambda_mask=.1, resolution=2, validate_every=1000):
    scenes = list(scenes if scenes is not None else ['train','truck','playroom'])
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError('select unique scene names')
    if steps <= 0 or start_iteration < 0 or validate_every <= 0 or resolution <= 0:
        raise ValueError('steps, validation interval and resolution must be positive')
    if not math.isfinite(lambda_mask) or lambda_mask < 0:
        raise ValueError('lambda_mask must be finite and nonnegative')
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists; choose a NEW directory: {out}')
    manifest = Path(manifest).resolve()
    config = json.loads(manifest.read_text(encoding='utf-8'))
    root = (manifest.parent/config.get('root', '.')).resolve()
    entries = {}
    for entry in config['scenes']:
        name = entry['name']
        if name in entries or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('manifest scene names must be unique, filesystem-safe identifiers')
        entries[name] = entry
    unknown = set(scenes)-entries.keys()
    if unknown:
        raise ValueError(f'Unknown scenes: {sorted(unknown)}')
    end = start_iteration+steps
    validations = sorted({start_iteration, end, *range(start_iteration+validate_every, end, validate_every)})
    plan = {'repo':str(REPO), 'out':str(out), 'manifest':str(manifest),
            'steps':steps, 'start_iteration':start_iteration, 'final_iteration':end,
            'lambda_mask':lambda_mask, 'resolution':resolution, 'scenes':[]}
    # Preflight EVERY scene before any process or output is created.
    for name in scenes:
        item = entries[name]
        if item.get('role') != 'train':
            raise ValueError(f'{name}: only training scenes are selected by this preprocessing workflow')
        source = (root/item['source']).resolve()
        ply = (root/item['ply']).resolve()
        if not ply.is_file():
            raise FileNotFoundError(f'{name}: missing pretrained PLY: {ply}')
        check_cameras(source)
        info = ply_info(ply, validate_values=True)
        scene_out = out/name
        training = [sys.executable, str(REPO/'prune_finetune.py'), '-s', str(source), '-m', str(scene_out),
                    '--start_pointcloud', str(ply), '--ply_iteration', str(start_iteration),
                    '--iterations', str(end), '--position_lr_max_steps', str(end),
                    '--lambda_mask', str(lambda_mask), '--resolution', str(resolution),
                    '--sh_degree', str(info['sh_degree']), '--data_device', 'cpu', '--eval', '--disable_gui',
                    '--test_iterations', *map(str, validations), '--save_iterations', str(end)]
        commands = [training,
                    [sys.executable, str(REPO/'render.py'), '-m', str(scene_out), '--iteration', str(end), '--skip_train'],
                    [sys.executable, str(REPO/'metrics.py'), '-m', str(scene_out)]]
        plan['scenes'].append({'name':name, 'source':str(source), 'input_ply':str(ply),
            'output_dir':str(scene_out), 'output_ply':str(scene_out/'point_cloud'/f'iteration_{end}'/'point_cloud.ply'),
            'input_info':info, 'commands':commands})
    return plan


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run_plan(plan, gpu):
    if not gpu or not str(gpu).strip() or str(gpu).strip() == '-1':
        raise ValueError('Set CUDA_VISIBLE_DEVICES explicitly to an available GPU')
    out = Path(plan['out'])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'plan.json', plan)
    result = {'status':'running', 'gpu':str(gpu), 'steps_per_scene':plan['steps'], 'scenes':[]}
    write_json(out/'summary.json', result)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    try:
        for scene in plan['scenes']:
            print(f'\n=== {scene["name"]}: {scene["input_info"]["points"]:,} Gaussians ===', flush=True)
            scene_out = Path(scene['output_dir'])
            with (out/f'{scene["name"]}.log').open('w', encoding='utf-8') as log:
                for phase, command in zip(('prune','render','metrics'), scene['commands']):
                    print(f'{scene["name"]} {phase}; progress log: {log.name}', flush=True)
                    log.write(f'\n### {phase}: {shlex.join(command)}\n')
                    log.flush()
                    subprocess.run(command, cwd=plan['repo'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            output_info = ply_info(scene['output_ply'], validate_values=True)
            if output_info['sh_degree'] != scene['input_info']['sh_degree']:
                raise ValueError(f'{scene["name"]}: export SH degree changed')
            # metrics.py catches its own exceptions: exit=0 is NOT proof of success.
            metrics = json.loads((scene_out/'results.json').read_text())[f'ours_{plan["final_iteration"]}']
            for key in ('PSNR','SSIM','LPIPS'):
                if key not in metrics or not math.isfinite(metrics[key]):
                    raise ValueError(f'{scene["name"]}: missing/nonfinite {key}')
            baseline_path = scene_out/'baseline_metrics.json'
            baseline = json.loads(baseline_path.read_text())
            if baseline.get('views', 0) <= 0 or any(not math.isfinite(baseline[key]) for key in ('PSNR','SSIM')):
                raise ValueError(f'{scene["name"]}: invalid baseline quality metrics')
            record = {'name':scene['name'], 'input_points':scene['input_info']['points'],
                      'output_points':output_info['points'], 'retained_fraction':output_info['points']/scene['input_info']['points'],
                      'input_bytes':scene['input_info']['bytes'], 'output_bytes':output_info['bytes'],
                      'metrics_reference':'held-out photos', 'metrics':metrics, 'baseline_metrics':baseline}
            record['psnr_change_db'] = metrics['PSNR']-baseline['PSNR']
            record['ssim_change'] = metrics['SSIM']-baseline['SSIM']
            result['scenes'].append(record)
            write_json(out/'summary.json', result)
            print(f'{scene["name"]} DONE: {record["input_points"]:,} -> {record["output_points"]:,}; PSNR={metrics["PSNR"]:.3f}', flush=True)
        manifest_path = out/'multiscene_pruned.json'
        write_json(manifest_path, {'root':'.', 'scenes':[
            {'name':s['name'], 'role':'train', 'source':s['source'], 'ply':s['output_ply']}
            for s in plan['scenes']]})
        result.update(status='complete', manifest=str(manifest_path))
        write_json(out/'summary.json', result)
        return result
    except BaseException as exc:
        result.update(status='failed', error=str(exc), failed_scene=scene['name'])
        write_json(out/'summary.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(REPO/'configs/multiscene_tandt_db.json'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--scenes', nargs='+', default=['train','truck','playroom'])
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--start-iteration', type=int, default=30000)
    parser.add_argument('--lambda-mask', type=float, default=.1)
    parser.add_argument('--resolution', type=int, default=2)
    parser.add_argument('--validate-every', type=int, default=1000)
    parser.add_argument('--dry-run', action='store_true', help='check inputs and print commands, no CUDA or writes')
    args = parser.parse_args()
    plan = build_plan(args.manifest, args.out, args.scenes, args.steps, args.start_iteration,
                      args.lambda_mask, args.resolution, args.validate_every)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('set CUDA_VISIBLE_DEVICES explicitly (e.g. CUDA_VISIBLE_DEVICES=1)')
    print(json.dumps(run_plan(plan, os.environ['CUDA_VISIBLE_DEVICES']), indent=2))


if __name__ == '__main__':
    main()
