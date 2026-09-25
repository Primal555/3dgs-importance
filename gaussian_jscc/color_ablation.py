"""Evaluation-only attribute swaps; source attributes are diagnostic oracles.

One decode per tier/trial is reused for every intervention. XYZ is never swapped.
No optimization, recoloring, image registration or checkpoint writes occur.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import prepare, read_ply, to_features
from .learned_training import decode_batches, hard_layout
from .render_objective import image_metrics, split_cameras
from .render_validation import save_panel
from .transport import load_checkpoint, model_id


# Raw PLY tensor: XYZ 0:3, opacity 3, log-scale 4:7, quaternion 7:11,
# DC RGB 11:14, channel-major higher-order SH 14:.
SWAPS = {
    'received': (),
    'source_color': ((11, None),),
    'source_dc': ((11, 14),),
    'source_sh': ((14, None),),
    'source_shape': ((4, 11),),
    'source_opacity': ((3, 4),),
    'source_shape_opacity': ((3, 11),),
    'source_attributes': ((3, None),),
}
LABELS = {
    'received': 'Received (baseline)',
    'source_color': 'Source DC + SH; predicted shape/alpha',
    'source_dc': 'Source DC only',
    'source_sh': 'Source higher-order SH only',
    'source_shape': 'Source scale + rotation only',
    'source_opacity': 'Source opacity only',
    'source_shape_opacity': 'Source shape/alpha; predicted DC + SH',
    'source_attributes': 'Source all attributes; delivered XYZ',
}


def swap_attributes(received, source, variant):
    if received.shape != source.shape or received.ndim != 2 or received.shape[1] < 14:
        raise ValueError('Expected matching row-aligned raw Gaussian tensors [N, >=14]')
    result = received.clone()
    for start, end in SWAPS[variant]:
        result[:, start:end] = source[:, start:end]
    return result


def color_metrics(image, target):
    """Signed bias is separate from spatial error; no display clipping in metrics."""
    delta = image - target
    result = {}
    for i, channel in enumerate('rgb'):
        result[f'{channel}_bias'] = float(delta[i].mean())
        result[f'{channel}_mae'] = float(delta[i].abs().mean())
    # Cancels equal RGB brightness offsets. Not a perceptual Delta-E metric.
    result['rg_error_rmse'] = float((delta[0]-delta[1]).square().mean().sqrt())
    result['gb_error_rmse'] = float((delta[1]-delta[2]).square().mean().sqrt())
    return result


def save_grid(path, images):
    """Render-generated comparison sheet, consistent display range, no correction."""
    from PIL import Image, ImageDraw
    _, height, width = images[0][1].shape
    canvas = Image.new('RGB', (3*width, 3*(height+24)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, tensor) in enumerate(images):
        pixels = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype('uint8')
        x, y = (i % 3)*width, (i // 3)*(height+24)
        draw.text((x+4, y+5), label, fill='black')
        canvas.paste(Image.fromarray(pixels), (x, y+24))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def evaluate_decoded(received, source, cameras, degree, white_background, out, tier, trial):
    from .rendering import render
    device = received.device
    targets = [render(source, c, degree, white_background) for c in cameras]
    sheets = [[('Source PLY (original XYZ)', t.cpu())] for t in targets] if trial == 0 else None
    observations = []
    for variant in SWAPS:
        scene = swap_attributes(received, source, variant)
        for view, (camera, target) in enumerate(zip(cameras, targets)):
            image = render(scene, camera, degree, white_background)
            metrics = image_metrics(image, target)
            observations.append({
                'tier': tier, 'trial': trial, 'view_index': view,
                'view': str(camera.image_name), 'variant': variant,
                **{'source_'+k: v for k, v in metrics.items()},
                **color_metrics(image, target),
            })
            if trial == 0:
                save_panel(out/'panels'/f'q{tier}'/f'view{view:02d}_{variant}.png',
                           camera.original_image[:3].to(device), target, image)
                sheets[view].append((LABELS[variant], image.cpu()))
        del scene
    if sheets is not None:
        for view, sheet in enumerate(sheets):
            save_grid(out/'comparisons'/f'q{tier}_view{view:02d}.png', sheet)
    return observations


@torch.no_grad()
def run(args):
    from .cli import seed_all
    from .rendering import load_cameras
    training_dir = Path(args.training)
    config = json.loads((training_dir/'training.json').read_text(encoding='utf-8'))
    checkpoint = Path(args.checkpoint) if args.checkpoint else training_dir/'codec.pt'
    ply, source_path = args.ply or config['ply'], args.source or config['source']
    out = Path(args.out) if args.out else training_dir/'color_ablation'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing diagnostic: {out}')
    device = torch.device(args.device)
    model = load_checkpoint(checkpoint, device)
    if any(not 1 <= q < len(model.cfg.rates) for q in args.tiers):
        raise ValueError('requested tier outside checkpoint rate table')
    if model.cfg.position_delivery == 'learned':
        raise ValueError('This diagnostic expects fixed delivered XYZ, not learned positions')
    before = model_id(model)
    original, degree = read_ply(ply)
    if degree != model.cfg.sh_degree or len(original) != config['source_gaussians']:
        raise ValueError('PLY degree/count does not match the training record')
    raw, geometry, _ = prepare(original, model.cfg.morton_bits)
    del original
    blocks, ids = [], []
    for start in range(0, len(raw), model.cfg.block_size):
        f, _ = to_features(raw[start:start+model.cfg.block_size].to(device), geometry, model)
        blocks.append(f.cpu())
        ids.append(torch.arange(start, start+len(f)))
    # Keep training batch grouping: changing it changes noise draw layout.
    batch_size = config['blocks_per_batch']
    groups = [pad_sequence(blocks[i:i+batch_size], batch_first=True)
              for i in range(0, len(blocks), batch_size)]
    group_ids = [pad_sequence(ids[i:i+batch_size], batch_first=True, padding_value=-1)
                 for i in range(0, len(ids), batch_size)]
    white = config.get('white_background', False)
    all_cameras = load_cameras(source_path, config['resolution'], white, config.get('images', 'images'), 'train')
    _, cameras, _, view_indices = split_cameras(all_cameras, config['validation_views'], config.get('train_views', 0))
    names = [str(c.image_name) for c in cameras]
    if config.get('validation_view_names', names) != names:
        raise ValueError('Validation camera names differ from the training run')
    del all_cameras
    paired = model.cfg.prefix_mode == 'progressive'
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'checkpoint': str(checkpoint.resolve()), 'model_id': before,
        'ply': str(Path(ply).resolve()), 'source': str(Path(source_path).resolve()),
        'variants': LABELS, 'tiers': args.tiers, 'trials': args.trials,
        'snr': config['snr'], 'channel': config['channel'], 'seed': config['seed'],
        'prefix_mode': model.cfg.prefix_mode, 'blocks_per_batch': batch_size,
        'view_names': names, 'view_indices': view_indices, 'resolution': config['resolution'],
        'noise': 'Original validation seed schedule; single decoded scene reused for all swaps within each tier/trial',
        'coordinates': 'All eight interventions retain exactly the received XYZ. Source PLY image uses original XYZ.',
        'scope': 'Oracle diagnostics only, NOT deployable decoding or extra training. No source row reordering after prepare.',
        'metrics': 'Unclipped RGB MSE/PSNR and bias; display-clipped SSIM. RGB units 0..1 nominal. No color correction.',
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows = []
    raw_device = raw.to(device)
    for tier in args.tiers:
        qs = [hard_layout(gi, tier, 0.,len(model.cfg.rates)) for gi in group_ids]
        for trial in range(args.trials):
            seed_all(config['seed']+20000+(0 if paired else (tier-1)*1000)+trial)
            received = decode_batches(model, groups, qs, config['snr'], config['channel'], geometry, paired_noise=paired)
            if received.shape != raw_device.shape:
                raise ValueError('Decoded/source row alignment failed')
            observations = evaluate_decoded(received, raw_device, cameras, degree, white, out, tier, trial)
            rows.extend(observations)
            with (out/'metrics.jsonl').open('a', encoding='utf-8') as handle:
                for row in observations:
                    handle.write(json.dumps(row, allow_nan=False)+'\n')
            del received
            print(f'q{tier} trial {trial+1}/{args.trials}: saved all attribute-swap renders and metrics', flush=True)
    if model_id(model) != before:
        raise RuntimeError('Diagnostic unexpectedly changed model state')
    keys = [k for k in rows[0] if k not in ('tier', 'trial', 'view_index', 'view', 'variant')]
    summary = []
    for tier in args.tiers:
        means = {}
        for variant in SWAPS:
            selected = [r for r in rows if r['tier'] == tier and r['variant'] == variant]
            means[variant] = {k: sum(r[k] for r in selected)/len(selected) for k in keys}
        for variant, scores in means.items():
            row = {'tier': tier, 'variant': variant, **scores,
                   'psnr_gain_vs_received': scores['source_psnr']-means['received']['source_psnr'],
                   'mse_reduction_vs_received': means['received']['source_mse']-scores['source_mse']}
            summary.append(row)
            print(f"q{tier} {variant}: PSNR={scores['source_psnr']:.3f}, gain={row['psnr_gain_vs_received']:+.3f} dB", flush=True)
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out/'summary.json').write_text(json.dumps({'weights_unchanged': True, 'results': summary}, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Complete (weights unchanged): {out}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, help='Completed run containing training.json')
    parser.add_argument('--checkpoint', help='Defaults to training/codec.pt, NOT best checkpoint')
    parser.add_argument('--ply', help='Override original PLY path after migration')
    parser.add_argument('--source', help='Override scene path after migration')
    parser.add_argument('--out', help='New directory, defaults to training/color_ablation')
    parser.add_argument('--tiers', type=int, nargs='+', default=[3])
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.trials < 1 or len(set(args.tiers)) != len(args.tiers):
        parser.error('trials must be positive and tiers must not repeat')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.error('Real scene rendering requires CUDA PyTorch and diff_gaussian_rasterization; CPU is not a substitute')
    # Cameras currently construct tensors on the current CUDA device.
    torch.cuda.set_device(torch.device(args.device).index or 0)
    run(args)


if __name__ == '__main__':
    main()
