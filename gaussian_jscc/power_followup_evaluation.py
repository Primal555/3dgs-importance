"""Paired image evaluation and clean-attribute interventions for power controls."""
import json
from pathlib import Path

import numpy as np
import torch

from .channel_normalization_metrics import calibrate_and_measure, diagnose_normalization
from .cli import seed_all
from .modular_evaluation import decode_path, evaluate_paths
from .modular_experiment import json_write
from .optimization import preserved_rng
from .render_objective import image_metrics, spaced_indices
from .render_validation import save_panel
from .transport import model_id

VARIANTS = {'normal': None, 'color': (11, None), 'shape': (4, 11), 'opacity': (3, 4)}


def replace_attributes(received, clean, variant):
    """Raw layout: XYZ3, opacity1, log scales3, rotation4, DC+SH remainder."""
    if variant not in VARIANTS or received.ndim != 2 or received.shape != clean.shape or received.shape[1] < 14:
        raise ValueError('matching raw Gaussian arrays and a known replacement required')
    result = received.clone()
    if VARIANTS[variant] is not None:
        lo, hi = VARIANTS[variant]
        result[:, lo:hi] = clean[:, lo:hi]
    return result


def bind_test_cameras(scene, spec, requested):
    from .rendering import load_cameras
    cameras = load_cameras(spec['source'], scene.record['resolution'], scene.white_background,
                           scene.record['images'], 'test')
    if not cameras: raise ValueError('no actual test cameras; training-view fallback is prohibited')
    names = [str(c.image_name) for c in cameras]
    if len(set(names)) != len(names) or set(names) & set(scene.record['train_view_names']):
        raise ValueError('test cameras duplicated or overlap training cameras')
    scene.test_cameras = [cameras[i] for i in spaced_indices(len(cameras), requested or len(cameras))]
    return [str(c.image_name) for c in scene.test_cameras]


@torch.no_grad()
def evaluate_endpoint(model, scene, cache, out, args, step, label, *, scope='test', images=False):
    """Atomic endpoint marker; replay incomplete outputs without duplicate metrics."""
    out = Path(out); folder = out/'evaluation'/label
    cameras = scene.test_cameras if scope == 'test' else scene.val_cameras
    identity = dict(step=step, label=label, scope=scope, view_names=[str(c.image_name) for c in cameras],
                    model_id=model_id(model), render_trials=args.render_trials, lpips=args.lpips,
                    seed=args.seed, diagnostic_blocks=args.diagnostic_blocks,
                    response_blocks=args.response_blocks, diagnostic_trials=args.trials, images=images)
    marker = folder/'complete.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != identity:
            raise ValueError('completed evaluation identity differs: '+label)
        return
    folder.mkdir(parents=True, exist_ok=True)
    history = folder/'metrics/validation.jsonl'
    if history.exists(): history.write_text('', encoding='utf-8')
    original = scene.val_cameras; modes = {m: m.training for m in model.modules()}
    device = next(model.parameters()).device
    try:
        with preserved_rng(device):
            model.eval(); scene.val_cameras = cameras
            json_write(folder/'power.json', calibrate_and_measure(model, cache, calibrate=False))
            diagnose_normalization(model, scene, cache, folder/'diagnostics', blocks=args.diagnostic_blocks,
                response_blocks=args.response_blocks, trials=args.trials, seed=args.seed)
            evaluate_paths(model, scene, step=step, phase=label, snrs=[10.], trials=args.render_trials,
                seed=args.seed, save_images=images, out=folder, condition_policy='matched',
                color_metrics=True, lpips_metrics=args.lpips, save_noiseless_images=True)
    finally:
        scene.val_cameras = original
        for module, mode in modes.items(): module.training = mode
    if model_id(model) != identity['model_id']: raise RuntimeError('evaluation changed model state')
    json_write(marker, identity)


@torch.no_grad()
def attribute_ablation(model, scene, out, args):
    """Rendering-only interventions, same received scene across all replacements.

    The teacher is the initial frozen semantic decoder, not source attributes.
    These restored values are unavailable at deployment; no rate/quality claim.
    """
    from .appearance_metrics import appearance_metrics
    from .perceptual_metrics import LPIPSEvaluator
    from .rendering import render
    out = Path(out)/'ablation'; out.mkdir(parents=True, exist_ok=True)
    identity = dict(model_id=model_id(model), view_names=[str(c.image_name) for c in scene.test_cameras],
                    trials=args.render_trials, seed=args.seed, lpips=args.lpips)
    marker = out/'complete.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != identity: raise ValueError('ablation identity changed')
        return
    device = next(model.parameters()).device; modes = {m: m.training for m in model.modules()}
    records = []
    try:
        with preserved_rng(device):
            model.eval(); seed_all(args.seed)
            qs = [torch.where(ids >= 0, 3, 0) for ids in scene.group_ids]
            clean, _ = decode_path(model, scene, qs, 'clean', 10.)
            perceptual = LPIPSEvaluator(device) if args.lpips else None
            for path in ('finite_noiseless', 'awgn'):
                for tier in (1, 2, 3):
                    qs = [torch.where(ids >= 0, tier, 0) for ids in scene.group_ids]
                    measurements = {v: [] for v in VARIANTS}
                    for trial in range(args.render_trials if path == 'awgn' else 1):
                        seed_all(args.seed+1000+trial)
                        received, _ = decode_path(model, scene, qs, path, 10.)
                        for variant in VARIANTS:
                            restored = replace_attributes(received, clean, variant)
                            for i, camera in enumerate(scene.test_cameras):
                                image = render(restored, camera, scene.degree, scene.white_background)
                                source = scene.reference.get(camera, device); photo = camera.original_image.to(device)
                                values = {p+'_'+k: v for p, target in (('source', source), ('photo', photo))
                                          for k, v in image_metrics(image, target).items()}
                                if perceptual is not None:
                                    for prefix, target in (('source', source), ('photo', photo)):
                                        values[prefix+'_lpips'] = perceptual(image, target)
                                values.update({'source_color_'+k: v for k, v in appearance_metrics(image, source).items()})
                                measurements[variant].append(dict(values, trial=trial, view=i, image_name=str(camera.image_name)))
                                if tier == 3 and trial == 0 and i in (0, len(scene.test_cameras)-1):
                                    save_panel(out/'images'/f'{path}_{variant}_view{i:04d}.png', photo, source, image,
                                               f'q3 {path}; diagnostic {variant}')
                            del restored
                        del received
                    normal = None
                    for variant, observations in measurements.items():
                        keys = [k for k in observations[0] if k not in ('trial', 'view', 'image_name')]
                        means = {k: float(np.mean([r[k] for r in observations])) for k in keys}
                        if variant == 'normal': normal = means
                        row = dict(path=path, tier=tier, symbols=model.cfg.rates[tier], replacement=variant,
                            views=len(scene.test_cameras), trials=args.render_trials if path == 'awgn' else 1,
                            **means, per_view_trial=observations)
                        row.update({f'delta_{k}': means[k]-normal[k] for k in ('source_psnr', 'source_ssim', 'source_mse')})
                        if args.lpips: row['delta_source_lpips'] = means['source_lpips']-normal['source_lpips']
                        records.append(row)
                    print(f'attribute ablation {path}/q{tier}: '+', '.join(
                        f'{r["replacement"]}={r["source_psnr"]:.3f}dB' for r in records[-4:]), flush=True)
    finally:
        for module, mode in modes.items(): module.training = mode
    if model_id(model) != identity['model_id']: raise RuntimeError('ablation modified source weights')
    from .channel_audit import write_csv
    json_write(out/'results.json', dict(rows=records, protocol=identity,
        reference='same initial clean semantic decoder; original reliable XYZ; source PLY is the image target',
        interpretation='diagnostic ideal replacements, not deployable; component gains are not additive'))
    write_csv(out/'summary.csv', [{k: v for k, v in r.items() if k != 'per_view_trial'} for r in records])
    json_write(marker, identity)
