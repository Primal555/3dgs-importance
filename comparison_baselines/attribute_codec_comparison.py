"""Evaluate three attribute codecs with one fixed point set and received XYZ."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from comparison_baselines.attribute_codec_worker import FCGS_RATES, UPSTREAM
from comparison_baselines.run import _execute
from gaussian_jscc.allocation import scene_fingerprint
from gaussian_jscc.cli import device_for
from gaussian_jscc.data import read_ply
from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_experiment import json_write
from gaussian_jscc.modular_multiscene_state import camera_fingerprint
from gaussian_jscc.multiscene import Scene, manifest
from gaussian_jscc.perceptual_metrics import LPIPSEvaluator, image_metric_protocol
from gaussian_jscc.position_delivery import delivered_positions
from gaussian_jscc.render_objective import image_metrics, spaced_indices
from gaussian_jscc.rendering import RenderReference, load_cameras, render
from gaussian_jscc.semantic_attribute_codec import GROUPS, PRIVATE_ONLY, SemanticAttributeCodec
from gaussian_jscc.semantic_attribute_experiment import FAMILY, read
from gaussian_jscc.semantic_attribute_metrics import attribute_metrics, decode_scene, save_image


VERSION = 'fixed_xyz_attribute_comparison_v1'
METHODS = ('semantic', 'fcgs', 'c3dgs')
SEMANTIC_XYZ_INPUT = 'shared_16bit_grid'


def camera_selection(cameras, names=None, excluded=()):
    by_name = {str(c.image_name): c for c in cameras}
    if len(by_name) != len(cameras):
        raise ValueError('duplicate camera names in the dataset split')
    if names is None:
        names = [name for name in by_name if name not in excluded]
    if not names or len(set(names)) != len(names) or set(names).intersection(excluded):
        raise ValueError('camera split is empty, duplicated or overlaps held-out views')
    missing = set(names).difference(by_name)
    if missing:
        raise ValueError(f'camera identity mismatch: {sorted(missing)}')
    return [by_name[name] for name in names]


def variants(reference, recovered):
    if (recovered.shape != reference.shape or not torch.isfinite(recovered).all()
            or not torch.equal(recovered[:, :3], reference[:, :3])
            or (recovered[:, 7:11].norm(dim=-1) < 1e-8).any()):
        raise ValueError('codec output must preserve shared XYZ and contain a valid row for each Gaussian')
    yield 'all_attributes', recovered
    boundary = 3 + GROUPS['geometry'][1]
    for name, span in (('geometry_opacity_only', slice(3, boundary)), ('color_only', slice(boundary, None))):
        hybrid = reference.clone()
        hybrid[:, span] = recovered[:, span]
        yield name, hybrid


@torch.no_grad()
def evaluate(raw, cameras, reference, label, perceptual, device, images, image_views):
    raw = raw.to(device)
    rows = []
    panel_indices = set(spaced_indices(len(cameras), min(image_views, len(cameras)))) if image_views else set()
    for i, camera in enumerate(cameras):
        target = reference.get(camera, device)
        photo = camera.original_image[:3].to(device)
        recovered = target if label == 'xyz_only' else render(raw, camera, 3, reference.white_background)
        if not torch.isfinite(recovered).all():
            raise ValueError(f'nonfinite render for {label}, camera {camera.image_name}')
        row = {'view': str(camera.image_name)}
        for prefix, ref in (('same_xyz', target), ('photo', photo)):
            scores = image_metrics(recovered, ref)
            if perceptual is not None:
                scores['lpips'] = perceptual(recovered, ref)
            row.update({prefix + '_' + key: value for key, value in scores.items()})
        if i in panel_indices:
            save_image(images / f'{i:03d}_{label}.png', recovered)
            if label == 'xyz_only':
                save_image(images / f'{i:03d}_photo.png', photo)
        rows.append(row)
    means = {k: sum(row[k] for row in rows) / len(rows) for k in rows[0] if k != 'view'}
    return dict(metrics=means, views=rows)


def store_camera_parameters(path, cameras):
    scalars = ('image_width', 'image_height', 'FoVx', 'FoVy')
    tensors = ('world_view_transform', 'full_proj_transform', 'camera_center')
    rows = []
    for camera in cameras:
        row = {key: float(getattr(camera, key)) for key in scalars}
        row.update({key: getattr(camera, key).detach().cpu().tolist() for key in tensors})
        rows.append(dict(row, image_name=str(camera.image_name)))
    json_write(path, rows)


def compare_scene(spec, record, model, settings, methods, plan, perceptual, device, out):
    name = spec['name']
    raw, degree = read_ply(spec['ply'])
    if degree != 3 or scene_fingerprint(raw) != record['fingerprint']:
        raise ValueError(f'{name}: source PLY differs from the semantic experiment')
    scene_args = SimpleNamespace(**settings, net_bits_per_use=2.)
    scene = Scene(dict(spec, prior='none'), raw, degree, model, scene_args, out,
                  with_mask=False, cameras=False)
    del raw
    if scene.geometry.to_dict() != record['geometry']:
        raise ValueError(f'{name}: scene geometry does not match the semantic checkpoint')
    camera_args = (spec['source'], record['resolution'], record['white_background'], record['images'])
    test = camera_selection(load_cameras(*camera_args, split='test'), record['test_view_names'])
    if camera_fingerprint(test) != record['camera_fingerprints']['test']:
        raise ValueError(f'{name}: test camera data differs from the semantic experiment')
    common_unit = delivered_positions(scene.geometry.normalize(scene.raw[:, :3]),
                                      torch.ones(len(scene.raw), dtype=torch.bool), model.cfg)
    common_xyz = scene.geometry.denormalize(common_unit)
    source = scene.raw
    reference = RenderReference(torch.cat((common_xyz, source[:, 3:]), dim=1), 3, record['white_background'])
    baseline_path = scene.out / 'references.json'
    if baseline_path.is_file():
        baselines = read(baseline_path)
    else:
        baselines = {
            'xyz_only': evaluate(reference.raw, test, reference, 'xyz_only', perceptual,
                                 device, scene.out / 'images', plan['image_views']),
            'source': evaluate(source, test, reference, 'source', perceptual,
                               device, scene.out / 'images', plan['image_views']),
        }
        json_write(baseline_path, baselines)
    baseline_photo = baselines['xyz_only']['metrics']['photo_psnr']

    for method in methods:
        folder = scene.out / method
        report = folder / 'metrics.json'
        if report.is_file() and (method != 'semantic' or read(report).get('xyz_input') == SEMANTIC_XYZ_INPUT):
            continue
        print(f'{name}/{method}: {len(source)} Gaussians, {len(test)} fixed test views', flush=True)
        folder.mkdir(exist_ok=True)
        payload = None
        if method == 'semantic':
            recovered = decode_scene(model, scene, received_unit=common_unit)
        else:
            with tempfile.TemporaryDirectory(prefix='.attribute_work_', dir=out) as tmp:
                worker = Path(tmp)
                np.save(worker / 'input.npy', reference.raw.numpy(), allow_pickle=False)
                cfg = plan['external'][method]
                arguments = [method, '--repo', cfg['repo'], '--input', worker / 'input.npy',
                             '--out', worker, '--seed', plan['seed']]
                if method == 'fcgs':
                    arguments.extend(['--fcgs-rate', plan['fcgs_rate']])
                else:
                    if record['white_background']:
                        raise ValueError('the pinned C3DGS sensitivity renderer uses a black background')
                    excluded = record['test_view_names'] + record['validation_view_names']
                    train = camera_selection(load_cameras(*camera_args, split='train'),
                                             record['train_view_names'] or None, excluded)
                    if record['train_view_names'] and camera_fingerprint(train) != record['camera_fingerprints']['train']:
                        raise ValueError(f'{name}: sensitivity camera data mismatch')
                    store_camera_parameters(worker / 'cameras.json', train)
                    del train
                    arguments.extend(['--cameras', worker / 'cameras.json'])
                gc.collect()
                torch.cuda.empty_cache()
                _execute(cfg['python'], 'comparison_baselines.attribute_codec_worker', arguments, folder / 'codec.log')
                recovered = torch.from_numpy(np.load(worker / 'decoded.npy', allow_pickle=False))
                payload = read(worker / 'payload.json')
        results = {}
        for variant, rows in variants(reference.raw, recovered):
            scored = evaluate(rows, test, reference, variant, perceptual, device,
                              folder / 'images', plan['image_views'])
            psnr = scored['metrics']['photo_psnr']
            scored['photo_psnr_loss_vs_xyz_only'] = baseline_photo - psnr
            results[variant] = scored
            print(f'  {variant}: same XYZ PSNR={scored["metrics"]["same_xyz_psnr"]:.4f}, '
                  f'photo PSNR={psnr:.4f}, loss={baseline_photo - psnr:+.4f} dB', flush=True)
            del rows
        measurements = attribute_metrics(model, scene, recovered, test)
        row = dict(method=method, scene=name, recovered_gaussians=len(recovered),
                   attribute_metrics=measurements, payload=payload, variants=results)
        if method == 'semantic':
            row['xyz_input'] = SEMANTIC_XYZ_INPUT
        json_write(report, row)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='completed private128 semantic experiment')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', help='replacement data paths checked against the saved fingerprints')
    p.add_argument('--scenes', nargs='+', help='default: the two training scenes and the held-out scene')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--fcgs-repo', type=Path, default=Path('external/attribute_codecs/FCGS'))
    p.add_argument('--c3dgs-repo', type=Path, default=Path('external/attribute_codecs/C3DGS'))
    p.add_argument('--fcgs-python', type=Path)
    p.add_argument('--c3dgs-python', type=Path)
    p.add_argument('--fcgs-rate', type=float, choices=FCGS_RATES, default=1e-4)
    p.add_argument('--image-views', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    return p


def main(args):
    run, out = args.run.resolve(), args.out.resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('use an output directory separate from the semantic run')
    if args.image_views < 0 or not 0 <= args.seed < 2**32 or len(set(args.methods)) != len(args.methods):
        raise ValueError('invalid image count, seed or repeated methods')
    arm = run / PRIVATE_ONLY
    checkpoint = arm / 'checkpoints/final.pt'
    training_plan = read(run / 'plan.json')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (saved['family'] != FAMILY or saved['arm'] != PRIVATE_ONLY or saved['plan'] != training_plan
            or saved['progress']['step'] != training_plan['budgets']['updates_per_arm']
            or read(arm / 'complete.json')['status'] != 'complete'):
        raise ValueError('expected the completed private128 semantic checkpoint')
    specs = {s['name']: s for s in training_plan['selected_scenes'] + [training_plan['heldout_scene']]}
    records = saved['scene_records']
    names = args.scenes or list(specs)
    if len(set(names)) != len(names) or not set(names) <= set(specs):
        raise ValueError(f'select distinct known scenes: {list(specs)}')
    overrides = {s['name']: s for s in manifest(args.manifest)} if args.manifest else {}
    selected = [dict(specs[n], **{key: overrides[n][key] for key in ('source', 'ply')})
                if n in overrides else specs[n] for n in names]
    for s in selected:
        for key in ('source', 'ply'):
            if not Path(s[key]).exists():
                raise FileNotFoundError(f'{s["name"]}: {s[key]}')
    external = {}
    for method in args.methods:
        if method == 'semantic':
            continue
        interpreter = getattr(args, method + '_python')
        repo = getattr(args, method + '_repo')
        if interpreter is None or not interpreter.is_file() or not (repo / '.git').exists():
            raise FileNotFoundError(f'{method}: run scripts/setup_attribute_codec_comparison.sh and set its Python path')
        # Keep the environment symlink instead of resolving to the base interpreter.
        external[method] = dict(python=str(interpreter.absolute()), repo=str(repo.resolve()),
                                revision=UPSTREAM[method][1])
    with checkpoint.open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    plan = dict(version=VERSION, semantic_checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                scenes=selected, records={key: records[key] for key in names},
                methods=list(args.methods), external=external, seed=args.seed, fcgs_rate=args.fcgs_rate,
                image_views=args.image_views, metric_protocol=image_metric_protocol(args.lpips),
                scope='fixed XYZ, fixed Gaussian rows, no AWGN or per-scene attribute optimization',
                cost_scope='actual attribute payload bytes for external codecs; no channel-use estimate')
    if out.exists():
        if not args.resume or read(out / 'plan.json') != plan:
            raise ValueError('resume requires the existing output and the same plan')
    else:
        if args.resume:
            raise FileNotFoundError('resume requires the existing output directory')
        out.mkdir(parents=True)
        json_write(out / 'plan.json', plan)
    device = device_for('cuda')
    cfg = ModularCodecConfig.from_dict(saved['config'])
    if cfg.position_delivery != 'quantized' or cfg.position_bits != 16:
        raise ValueError('expected the original 16-bit received coordinate convention')
    model = SemanticAttributeCodec(PRIVATE_ONLY, cfg)
    model.load_state_dict(saved['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    del saved
    perceptual = LPIPSEvaluator(device) if args.lpips else None
    for spec in selected:
        compare_scene(spec, records[spec['name']], model, training_plan['settings'], args.methods,
                      plan, perceptual, device, out)
    summary = {'version': VERSION, 'scenes': {}}
    for name in names:
        folder = out / 'scenes' / name
        baselines = read(folder / 'references.json')
        results = {}
        for method in args.methods:
            row = read(folder / method / 'metrics.json')
            row['variants'] = {k: {a: b for a, b in v.items() if a != 'views'} for k, v in row['variants'].items()}
            results[method] = row
        summary['scenes'][name] = dict(references={k: v['metrics'] for k, v in baselines.items()}, methods=results)
    json_write(out / 'review_summary.json', summary)
    archive = out / 'review.tar.gz'
    temporary = archive.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as tar:
        for path in (out / 'plan.json', out / 'review_summary.json', out / 'scenes'):
            tar.add(path, arcname=path.name)
    temporary.replace(archive)
    print(f'Complete. Review: {archive}', flush=True)


if __name__ == '__main__':
    main(parser().parse_args())
