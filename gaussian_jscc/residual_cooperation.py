"""Read-only-source Truck probes: gradient alignment, one-step effects, suffix use.

No new training objective, no detach/freeze strategy, no allocator, no saved
updated weights. Source and output directories must differ. Full-scene replay
is used for gradients; fixed full-slot AWGN draws couple positive prefixes.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .codec import channel
from .data import to_raw
from .optimization import preserved_rng
from .position_delivery import delivered_positions
from .render_objective import MultiViewRenderTask, image_metrics
from .training import full_scene_step
from .transport import load_checkpoint, model_id


VARIANTS = ('normal', 'no_delta3', 'zero_suffix', 'shuffle_suffix')
OPTIMIZER_NOTE = 'fresh Adam for EACH independent one-step probe; no historical moments'


def diagnostic_group(name):
    if name.startswith('learned.refinements.'):
        _, _, index, part, *_ = name.split('.')
        return f'refinement{int(index)+2}_' + ('input' if part == 'previous' else 'heads')
    if name.startswith('learned.heads.'):
        return 'xyz' if name.split('.')[2] == 'xyz' else 'base_attributes'
    if name.startswith(('learned.tier.', 'learned.snr.')):
        return 'conditioning'
    if name.startswith('learned.dec'):
        return 'shared_decoder'
    return 'shared_encoder'


def gradient_snapshot(model):
    """Keep a fixed trainable-parameter basis, including unused zero entries."""
    snapshot = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach())
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f'nonfinite gradient: {name}')
        snapshot[name] = {'group': diagnostic_group(name), 'present': parameter.grad is not None,
                          'gradient': gradient.flatten().cpu().clone()}
    return snapshot


def gradient_relations(snapshots):
    """Pairwise cosines by real module and on the complete fixed parameter basis."""
    tiers = sorted(snapshots)
    names = list(snapshots[tiers[0]])
    if any(list(snapshots[t]) != names for t in tiers):
        raise ValueError('gradient snapshots must share the identical parameter basis')
    groups = defaultdict(list)
    for name in names:
        groups[snapshots[tiers[0]][name]['group']].append(name)
    groups['all_trainable'] = names
    rows = []
    for i, first in enumerate(tiers):
        for second in tiers[i+1:]:
            for group, selected in groups.items():
                a = torch.cat([snapshots[first][n]['gradient'] for n in selected]).double()
                b = torch.cat([snapshots[second][n]['gradient'] for n in selected]).double()
                norm_a, norm_b = float(a.norm()), float(b.norm())
                dot = float(a @ b)
                valid = norm_a > 0 and norm_b > 0
                rows.append({'tier_a': first, 'tier_b': second, 'group': group,
                    'norm_a': norm_a, 'norm_b': norm_b, 'dot': dot,
                    'cosine': max(-1., min(1., dot/(norm_a*norm_b))) if valid else None,
                    'status': 'defined' if valid else 'zero_norm', 'parameter_count': a.numel(),
                    'active_parameters_a': sum(snapshots[first][n]['present'] for n in selected),
                    'active_parameters_b': sum(snapshots[second][n]['present'] for n in selected)})
    return rows


def ablate_suffix(signal, received, valid, variant, seed=42, start=24):
    """Intervene only on enhancement-3 slots; shuffle SIGNAL, not point noise.

    Deranged row permutation inside each local block preserves the suffix
    multiset and energy. Padding never participates. Tiny blocks with <2 valid
    rows have no permutation; their number is logged by the caller's plan.
    """
    if variant not in ('normal', 'zero_suffix', 'shuffle_suffix'):
        raise ValueError('unknown received-symbol intervention')
    if signal.shape != received.shape or signal.shape[:2] != valid.shape or signal.ndim != 3:
        raise ValueError('signal/received/valid shapes must match [blocks,points,real slots]')
    result = received.clone()
    if variant == 'zero_suffix':
        result[..., start:] = 0
    elif variant == 'shuffle_suffix':
        generator = torch.Generator(device='cpu').manual_seed(seed)
        for block in range(len(valid)):
            indices = valid[block].nonzero(as_tuple=False).flatten()
            if len(indices) < 2:
                continue
            order = torch.randperm(len(indices), generator=generator).to(indices.device)
            destinations = indices[order]
            sources = destinations.roll(1)
            noise = received[block, destinations, start:] - signal[block, destinations, start:]
            result[block, destinations, start:] = signal[block, sources, start:] + noise
    return result * valid[..., None]


def probe_batch(model, features, q, geometry, snr, kind, variant='normal',
                shuffle_seed=42, return_stages=False):
    if model.cfg.decoder_mode != 'residual' or tuple(model.cfg.rates) != (0, 4, 12, 24):
        raise ValueError('cooperation probes require residual decoder with rates 0/4/12/24')
    if variant not in VARIANTS or (variant != 'normal' and (q[q > 0] != 3).any()):
        raise ValueError('suffix interventions require uniform q3')
    if kind not in ('awgn', 'none'):
        raise ValueError('paired signal/noise interventions require AWGN or noiseless channel')
    active = q > 0
    signal = model.learned.encode_full(features, features[..., :3], active, snr)
    # One fixed noise draw for EVERY point/symbol slot, independent of tier.
    received = channel(signal.reshape(-1, 2), snr, kind).reshape_as(signal) * active[..., None]
    if variant in ('zero_suffix', 'shuffle_suffix'):
        received = ablate_suffix(signal, received, active, variant, shuffle_seed,
                                start=2*model.cfg.rates[2])
    stages, _ = model.learned.decode_stages(received, q, snr)
    delivered = delivered_positions(features[..., :3], q, model.cfg)
    index = 1 if variant == 'no_delta3' else 2
    prediction = model.apply_position_delivery(stages[index], q, delivered)
    raw = to_raw(prediction[active], geometry, model)
    if return_stages:
        return raw, [model.apply_position_delivery(s, q, delivered).detach() for s in stages]
    return raw


def render_gradient(model, scene, cameras, tier, snr, kind, seed, white_background):
    from .cli import seed_all
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)
    batches = [(features, torch.where(ids >= 0, tier, 0))
               for features, ids in zip(scene.groups, scene.group_ids)]
    task = MultiViewRenderTask(cameras, scene.reference, scene.degree, white_background)
    def forward(features, q):
        raw = probe_batch(model, features, q, scene.geometry, snr, kind)
        return raw, raw.sum()*0, {}
    with preserved_rng(device):
        seed_all(seed)
        loss, details = full_scene_step(model, batches, scene.geometry, snr, kind, task,
            attr_weight=0., mode='replay', batch_forward=forward)
    return float(loss), gradient_snapshot(model), details


@torch.no_grad()
def decode_scene(model, scene, tier, snr, kind, seed, variant='normal'):
    from .cli import seed_all
    device = next(model.parameters()).device
    rows = []
    with preserved_rng(device):
        seed_all(seed)
        for batch_index, (features, ids) in enumerate(zip(scene.groups, scene.group_ids)):
            q = torch.where(ids.to(device) >= 0, tier, 0)
            rows.append(probe_batch(model, features.to(device), q, scene.geometry, snr, kind,
                                   variant, shuffle_seed=seed+100003+batch_index))
    return torch.cat(rows)


def _save_image(image, target, path):
    import numpy as np
    from PIL import Image
    # Display only; metrics retain unclipped values.
    panels = torch.cat((target.clamp(0, 1), image.clamp(0, 1),
                        ((image-target).abs()*4).clamp(0, 1)), dim=2)
    array = (panels.permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@torch.no_grad()
def evaluate_scene(model, scene, cameras, tier, snr, kind, seed, white_background,
                   variant='normal', image_directory=None, image_views=0):
    from .rendering import render
    raw = decode_scene(model, scene, tier, snr, kind, seed, variant)
    views = []
    for i, camera in enumerate(cameras):
        image = render(raw, camera, scene.degree, white_background)
        target = scene.reference.get(camera, raw.device)
        views.append({'view': str(camera.image_name), **image_metrics(image, target)})
        if image_directory is not None and i < image_views:
            _save_image(image, target, Path(image_directory)/f'{variant}_view{i:02d}.png')
    return {**{key: sum(r[key] for r in views)/len(views) for key in ('mse','psnr','ssim','l1')},
            'views': views}


def _append(path, row):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, allow_nan=False)+'\n')


def _csv(path, rows):
    flat = [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader()
        writer.writerows(flat)


def aggregate(rows, keys, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    results = []
    for key, values in grouped.items():
        entry = dict(zip(keys, key), samples=len(values))
        for field in fields:
            numbers = [r[field] for r in values if r[field] is not None]
            entry[field+'_valid_samples'] = len(numbers)
            entry[field+'_mean'] = sum(numbers)/len(numbers) if numbers else None
            entry[field+'_negative_fraction'] = sum(x < 0 for x in numbers)/len(numbers) if numbers else None
            entry[field+'_min'] = min(numbers) if numbers else None
            entry[field+'_max'] = max(numbers) if numbers else None
        results.append(entry)
    return results


def run_probes(model, scene, out, probes=8, views_per_probe=2, heldout_views=4,
               snr=10., kind='awgn', seed=42, lrs=(1e-4, 1e-5), white_background=False,
               make_plots=True, image_views=1):
    """All updates are independent and disposable, including on exceptions."""
    from .render_objective import spaced_indices
    if probes < 1 or not 1 <= views_per_probe <= len(scene.cameras) or heldout_views < 1 or image_views < 0:
        raise ValueError('invalid probe/view counts')
    if not 0 <= seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    if not scene.val_cameras or set(map(id, scene.cameras)) & set(map(id, scene.val_cameras)):
        raise ValueError('need disjoint training and heldout cameras')
    if not lrs or len(set(lrs)) != len(lrs) or any(not math.isfinite(lr) or lr <= 0 for lr in lrs):
        raise ValueError('learning rates must be distinct finite positive values')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/f).exists() for f in ('gradients.jsonl','updates.jsonl','ablations.jsonl','summary.json')):
        raise FileExistsError('diagnostic outputs already exist; use a new output directory')
    device = next(model.parameters()).device
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    original_grads = {n: None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
    initial_mode, identity = model.training, model_id(model)
    heldout = [scene.val_cameras[i] for i in spaced_indices(len(scene.val_cameras), heldout_views)]
    gradient_rows, update_rows, ablation_rows = [], [], []
    def reset():
        model.load_state_dict(initial, strict=True)
        model.eval()
        model.zero_grad(set_to_none=True)
    try:
        with preserved_rng(device):
            for probe in range(probes):
                started = time.perf_counter()
                reset()
                channel_seed = (seed + probe*104729) % 2**32
                generator = torch.Generator().manual_seed(seed+probe*7919)
                selected = torch.randperm(len(scene.cameras), generator=generator)[:views_per_probe].tolist()
                cameras = [scene.cameras[i] for i in selected]
                splits = {'update_views': (cameras, channel_seed),
                          'heldout_views': (heldout, (channel_seed+2000003) % 2**32)}
                snapshots, losses, baseline = {}, {}, {}
                for tier in (1, 2, 3):
                    loss, snapshot, _ = render_gradient(model, scene, cameras, tier, snr, kind,
                                                       channel_seed, white_background)
                    snapshots[tier], losses[tier] = snapshot, loss
                    print(f'probe {probe+1}/{probes}: q{tier} gradient collected, image MSE={loss:.7g}', flush=True)
                    for split, (view_batch, noise_seed) in splits.items():
                        metrics = evaluate_scene(model, scene, view_batch, tier, snr, kind, noise_seed, white_background)
                        baseline[split, tier] = metrics
                        if split == 'update_views' and abs(metrics['mse']-loss) > max(1e-7, abs(loss)*1e-5):
                            raise RuntimeError('gradient/evaluation pairing failed')
                for row in gradient_relations(snapshots):
                    record = {'probe': probe, 'noise_seed': channel_seed,
                              'update_views': [str(c.image_name) for c in cameras],
                              'loss_a': losses[row['tier_a']], 'loss_b': losses[row['tier_b']], **row}
                    gradient_rows.append(record)
                    _append(out/'gradients.jsonl', record)
                for update_tier in (1, 2, 3):
                    for lr in lrs:
                        reset()
                        start_identity = model_id(model)
                        if start_identity != identity:
                            raise RuntimeError('single-step probe did not start from original weights')
                        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
                        for name, parameter in model.named_parameters():
                            if name in snapshots[update_tier] and snapshots[update_tier][name]['present']:
                                parameter.grad = snapshots[update_tier][name]['gradient'].reshape_as(parameter).to(parameter).clone()
                        optimizer.step()
                        update_norm = math.sqrt(sum(float((value.detach().cpu()-initial[name]).double().square().sum())
                                                    for name, value in model.named_parameters()))
                        for split, (view_batch, noise_seed) in splits.items():
                            for evaluated_tier in (1, 2, 3):
                                metrics = evaluate_scene(model, scene, view_batch, evaluated_tier, snr, kind, noise_seed, white_background)
                                before = baseline[split, evaluated_tier]
                                record = {'probe': probe, 'update_tier': update_tier, 'evaluated_tier': evaluated_tier,
                                    'split': split, 'lr': lr, 'start_model_id': start_identity,
                                    'optimizer': 'fresh_adam', 'update_norm': update_norm, 'noise_seed': noise_seed,
                                    'before_mse': before['mse'], 'after_mse': metrics['mse'],
                                    'delta_mse': metrics['mse']-before['mse'],
                                    'before_psnr': before['psnr'], 'after_psnr': metrics['psnr'],
                                    'delta_psnr': metrics['psnr']-before['psnr'],
                                    'before_ssim': before['ssim'], 'after_ssim': metrics['ssim'],
                                    'delta_ssim': metrics['ssim']-before['ssim'],
                                    'views_before': before['views'], 'views_after': metrics['views']}
                                update_rows.append(record)
                                _append(out/'updates.jsonl', record)
                        del optimizer
                        print(f'probe {probe+1}/{probes}: independent q{update_tier} update, lr={lr:g}, evaluated all prefixes', flush=True)
                reset()
                for split, (view_batch, noise_seed) in splits.items():
                    for variant in VARIANTS:
                        metrics = evaluate_scene(model, scene, view_batch, 3, snr, kind, noise_seed, white_background,
                            variant, image_directory=out/'images' if probe == 0 and split == 'heldout_views' else None,
                            image_views=image_views)
                        base, q2 = baseline[split, 3], baseline[split, 2]
                        if variant == 'no_delta3' and abs(metrics['mse']-q2['mse']) > max(1e-7, abs(q2['mse'])*1e-5):
                            raise RuntimeError('no_delta3 is not paired with q2; possible future-prefix leakage')
                        record = {'probe': probe, 'split': split, 'variant': variant, 'noise_seed': noise_seed,
                            'mse': metrics['mse'], 'psnr': metrics['psnr'], 'ssim': metrics['ssim'],
                            'delta_psnr_vs_normal': metrics['psnr']-base['psnr'],
                            'delta_ssim_vs_normal': metrics['ssim']-base['ssim'],
                            'delta_psnr_vs_q2': metrics['psnr']-q2['psnr'], 'views': metrics['views']}
                        ablation_rows.append(record)
                        _append(out/'ablations.jsonl', record)
                reset()
                elapsed = time.perf_counter()-started
                _append(out/'progress.jsonl', {'completed_probe': probe+1, 'total_probes': probes, 'seconds': elapsed})
                print(f'cooperation probe {probe+1}/{probes}: independent updates and suffix controls complete; sec={elapsed:.2f}', flush=True)
    finally:
        model.load_state_dict(initial, strict=True)
        model.train(initial_mode)
        for name, parameter in model.named_parameters():
            parameter.grad = original_grads[name]
    if model_id(model) != identity:
        raise RuntimeError('diagnostic changed source model')
    summary = {'completed_probes': probes, 'source_model_unchanged': True, 'source_model_id': identity,
        'optimizer': OPTIMIZER_NOTE, 'learning_rates': list(lrs), 'noise': 'full fixed point/symbol slots, paired across tiers and before/after',
        'metric_target': 'original PLY rendering, not photos; mean per-view PSNR, unclipped MSE, displayed-RGB SSIM',
        'gradient_summary': aggregate(gradient_rows, ('tier_a','tier_b','group'), ('cosine','dot','norm_a','norm_b')),
        'update_summary': aggregate(update_rows, ('update_tier','evaluated_tier','split','lr'), ('delta_mse','delta_psnr','delta_ssim')),
        'ablation_summary': aggregate(ablation_rows, ('split','variant'), ('psnr','ssim','delta_psnr_vs_normal','delta_psnr_vs_q2')),
        'limits': ['one checkpoint and small paired sample, not convergence or causal proof',
                   'fresh Adam, not the historical optimizer trajectory',
                   'suffix zero/shuffle are out-of-distribution interventions',
                   'gradient alignment is local; norm-zero cosines remain null',
                   'the heldout camera set repeats across probes; samples are not independent scenes']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    for name, rows in (('gradients',gradient_rows), ('updates',update_rows), ('ablations',ablation_rows)):
        _csv(out/(name+'.csv'), rows)
    if make_plots:
        plot_results(out, summary)
    return summary


def plot_results(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    folder = Path(out)/'charts'
    folder.mkdir(exist_ok=True)
    # Static test-log figures: signed matrices + prefix control bars, not trends.
    # Blue/neutral/gold with signed numeric labels; shared scales within each
    # update comparison. CSV/JSONL retain per-probe source rows and measures.
    signed = LinearSegmentedColormap.from_list('signed_probe', ['#316D9C','#F5F5F5','#D9A526'])
    sample_note = f'Truck; {summary["completed_probes"]} paired probes; original PLY target'
    groups = list(dict.fromkeys(r['group'] for r in summary['gradient_summary']))
    pairs = [(1,2), (1,3), (2,3)]
    matrix = np.full((len(groups),3), np.nan)
    for row in summary['gradient_summary']:
        value = row['cosine_mean']
        if value is not None:
            matrix[groups.index(row['group']), pairs.index((row['tier_a'],row['tier_b']))] = value
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    im = ax.imshow(matrix, cmap=signed, vmin=-1, vmax=1, aspect='auto')
    ax.set(xticks=range(3), xticklabels=['q1 / q2','q1 / q3','q2 / q3'], yticks=range(len(groups)), yticklabels=groups,
           title=f'Mean paired gradient cosine\n{sample_note}; N/A = zero norm')
    for i in range(len(groups)):
        for j in range(3):
            ax.text(j,i, 'N/A' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:+.2f}', ha='center',va='center',
                    color='#202020')
    fig.colorbar(im, ax=ax, label='cosine [-1,1]')
    fig.savefig(folder/'gradient_cosines.png',dpi=140); plt.close(fig)
    for lr in summary['learning_rates']:
        fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
        matrices = []
        for split in ('update_views','heldout_views'):
            matrix = np.zeros((3,3))
            for row in summary['update_summary']:
                if row['lr'] == lr and row['split'] == split:
                    matrix[row['update_tier']-1,row['evaluated_tier']-1] = row['delta_psnr_mean']
            matrices.append(matrix)
        limit = max(*(float(np.abs(m).max()) for m in matrices), 1e-6)
        for ax, split, matrix in zip(axes, ('update_views','heldout_views'), matrices):
            im = ax.imshow(matrix,cmap=signed,vmin=-limit,vmax=limit)
            ax.set(xticks=range(3),xticklabels=['q1','q2','q3'],yticks=range(3),yticklabels=['q1','q2','q3'],
                   xlabel='Evaluated tier',ylabel='Tier supplying update gradient',title=split)
            for i in range(3):
                for j in range(3):
                    ax.text(j,i,f'{matrix[i,j]:+.3g}',ha='center',va='center',color='#202020')
        fig.colorbar(im, ax=list(axes), label='PSNR change (dB); same scale on both panels')
        fig.suptitle(f'One fresh-Adam step: mean PSNR change; lr={lr:g}\n{sample_note}; positive means better')
        fig.savefig(folder/f'update_effect_lr{lr:g}.png',dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4),layout='constrained')
    labels, values = [], []
    for row in summary['ablation_summary']:
        if row['split'] == 'heldout_views':
            labels.append(row['variant']); values.append(row['delta_psnr_vs_q2_mean'])
    ax.barh(labels,values,color='#316D9C',edgecolor='#333333',linewidth=.8)
    ax.axvline(0,color='#333333',linewidth=1)
    ax.invert_yaxis()
    for i, value in enumerate(values):
        ax.annotate(f'{value:+.3g}',(value,i),xytext=(5 if value>=0 else -5,0),textcoords='offset points',
                    va='center',ha='left' if value>=0 else 'right',color='#202020')
    ax.margins(x=.25)
    ax.set(xlabel='PSNR change vs paired q2 (dB)',title=f'Third-stage controls\n{sample_note}; heldout cameras; zero/shuffle are OOD probes')
    fig.savefig(folder/'suffix_controls.png',dpi=140); plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='paired residual experiment root OR its residual subdirectory')
    p.add_argument('--out',required=True,help='NEW output directory outside source run')
    p.add_argument('--checkpoint',help='optional checkpoint inside this source experiment checkpoints directory')
    p.add_argument('--manifest',help='optional migrated paths; exact original Truck identity/split required')
    p.add_argument('--probes',type=int,default=8)
    p.add_argument('--views-per-probe',type=int,default=2)
    p.add_argument('--heldout-views',type=int,default=4)
    p.add_argument('--lrs',type=float,nargs='+',default=[1e-4,1e-5])
    p.add_argument('--blocks-per-batch',type=int,help='default from source training')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--image-views',type=int,default=1)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def main(args):
    source = Path(args.run).resolve()
    run = source/'residual' if (source/'residual'/'training.json').is_file() else source
    out = Path(args.out).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('output must be outside the source run, not its ancestor or descendant')
    if out.exists():
        raise FileExistsError(f'use a new output directory: {out}')
    if args.probes < 1 or args.views_per_probe < 1 or args.heldout_views < 1 or args.image_views < 0 or args.cpu_threads < 1:
        raise ValueError('invalid diagnostic counts')
    if not args.lrs or len(set(args.lrs)) != len(args.lrs) or any(not math.isfinite(x) or x <= 0 for x in args.lrs):
        raise ValueError('invalid learning rates')
    if args.blocks_per_batch is not None and args.blocks_per_batch < 1:
        raise ValueError('invalid blocks per batch')
    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    record = json.loads((run/'training.json').read_text(encoding='utf-8'))
    checkpoint = (Path(args.checkpoint) if args.checkpoint else run/'checkpoints'/'final'/'codec.pt').resolve()
    if (run/'checkpoints').resolve() not in checkpoint.parents:
        raise ValueError('checkpoint must be inside the selected source experiment checkpoints directory')
    if record.get('decoder_mode') != 'residual':
        raise ValueError('source run must be the experimental residual decoder, not direct')
    if record.get('channel', 'awgn') not in ('awgn', 'none'):
        raise ValueError('cooperation probes currently support AWGN/noiseless sources only')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    scene_identity = json.loads((run/'scenes'/'truck'/'scene.json').read_text(encoding='utf-8'))
    plan = {**vars(args), 'source_run':str(run), 'checkpoint':str(checkpoint),
            'manifest':args.manifest or record['manifest'], 'optimizer':OPTIMIZER_NOTE,
            'scope':'Truck only, full scene; 3 gradient objectives, independent one-step controls, 4 suffix conditions',
            'source_weights_saved':False, 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:
        return
    from .cli import device_for
    from .data import read_ply
    from .multiscene import Scene, configure_codec_training, manifest, parser as training_parser
    device = device_for(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    model = load_checkpoint(checkpoint,device)
    if (model.cfg.decoder_mode != 'residual' or tuple(model.cfg.rates) != (0,4,12,24) or
            model.cfg.position_delivery != 'quantized' or model.cfg.position_bits != 16):
        raise ValueError('checkpoint must use residual attributes, 0/4/12/24 rates and reliable 16bit XYZ')
    configure_codec_training(model,'all')
    opts = training_parser().parse_args(['--out',str(out)])
    for name in ('blocks_per_batch','resolution','white_background','images','validation_views','net_bits_per_use'):
        setattr(opts,name,record[name])
    opts.blocks_per_batch = args.blocks_per_batch or opts.blocks_per_batch
    opts.views_per_step = args.views_per_probe
    spec = next(s for s in manifest(plan['manifest']) if s['name'] == 'truck')
    raw, degree = read_ply(spec['ply'])
    if degree != model.cfg.sh_degree:
        raise ValueError('Truck SH degree differs from checkpoint')
    out.mkdir(parents=True,exist_ok=False)
    scene = Scene(spec,raw,degree,model,opts,out,with_mask=False)
    current = json.loads((scene.out/'scene.json').read_text(encoding='utf-8'))
    for name in ('fingerprint','source_gaussians','train_view_names','validation_view_names'):
        if current[name] != scene_identity[name]:
            raise ValueError(f'Truck identity/split differs from source: {name}')
    plan.update(source_model_id=model_id(model), source_step=int(torch.load(checkpoint,map_location='cpu',weights_only=True)['step']),
                snr=record['snr'],channel=record['channel'],blocks_per_batch=opts.blocks_per_batch,
                source_gaussians=len(raw),scene_identity_verified=True,
                unshufflable_blocks=sum(int((ids>=0).sum(1).lt(2).sum()) for ids in scene.group_ids))
    del raw
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    summary = run_probes(model,scene,out,args.probes,args.views_per_probe,args.heldout_views,
        record['snr'],record['channel'],args.seed,args.lrs,opts.white_background,
        make_plots=not args.no_plots,image_views=args.image_views)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != plan['checkpoint_sha256']:
        raise RuntimeError('source checkpoint file changed during diagnostic')
    (out/'complete.json').write_text(json.dumps({'completed_probes':summary['completed_probes'],
        'source_model_unchanged':True,'source_checkpoint_unchanged':True,'optimizer':OPTIMIZER_NOTE,
        'status':'complete','output':str(out)},indent=2),encoding='utf-8')
    print(f'Cooperation diagnostics complete: {out}',flush=True)


if __name__ == '__main__':
    main(parser().parse_args())
