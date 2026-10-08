"""One-shot, fixed-histogram allocation control; no training or packet dumping."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .data import prepare, read_ply, to_features
from .learned_training import decode_batches
from .optimization import preserved_rng
from .render_objective import image_metrics, split_cameras
from .render_validation import save_panel
from .transport import load_checkpoint, model_id


def shuffled_tiers(q, seed):
    """Shuffle labels across ALL original rows, including q0; exact histogram."""
    q = torch.as_tensor(q, dtype=torch.long).cpu()
    if q.ndim != 1 or not len(q):
        raise ValueError('tiers must be a nonempty one-dimensional map')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return q[torch.randperm(len(q), generator=generator)]


def summarize(rows):
    keys = ('source_mse', 'source_psnr', 'source_ssim', 'photo_mse', 'photo_psnr', 'photo_ssim')
    return {k: sum(r[k] for r in rows)/len(rows) for k in keys}


@torch.no_grad()
def run(args):
    from .cli import device_for, seed_all
    from .allocation_diagnostics import AllocationCostMeter
    from .position_delivery import PositionCostMeter
    from .rendering import load_cameras, render, RenderReference
    from .route2 import load_mask, hard_tiers

    training = Path(args.training)
    config = json.loads((training/'training.json').read_text(encoding='utf-8'))
    out = Path(args.out) if args.out else training/'allocation_shuffle'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    if args.trials < 1:
        raise ValueError('trials must be positive')
    if not args.device.startswith('cuda'):
        raise ValueError('scene comparison requires CUDA')
    device = device_for(args.device)
    checkpoint = training/'codec_best_joint.pt'
    model = load_checkpoint(checkpoint, device).eval()
    if model.cfg.prefix_mode != 'progressive':
        raise ValueError('paired-noise comparison requires progressive prefixes')
    original, degree = read_ply(args.ply or config['ply'])
    if degree != model.cfg.sh_degree:
        raise ValueError('PLY SH degree differs from checkpoint')
    mask = load_mask(training/'route2_best_joint.pt', original, model, device)
    snr, kind, seed = config['snr'], config['channel'], config['seed']
    q = hard_tiers(mask, snr, seed)
    before = model_id(model)
    maps = {'learned': q, 'shuffled': shuffled_tiers(q, args.shuffle_seed)}
    counts = torch.bincount(q, minlength=len(model.cfg.rates))
    for value in maps.values():
        if not torch.equal(torch.bincount(value, minlength=len(counts)), counts):
            raise RuntimeError('shuffle changed tier histogram')

    raw, geometry, packet_q = prepare(original, model.cfg.morton_bits, q)
    # prepare() sorts each map with the SAME source geometry; never regroup
    # retained points or discard q0 holes before codec inference.
    packet_maps = {'learned': packet_q,
                   'shuffled': prepare(original, model.cfg.morton_bits, maps['shuffled'])[2]}
    blocks, ids = [], []
    for start in range(0, len(raw), model.cfg.block_size):
        features, _ = to_features(raw[start:start+model.cfg.block_size].to(device), geometry, model)
        blocks.append(features.cpu())
        ids.append(torch.arange(start, start+len(features)))
    batch_size = config['blocks_per_batch']
    groups = [pad_sequence(blocks[i:i+batch_size], batch_first=True)
              for i in range(0, len(blocks), batch_size)]
    group_ids = [pad_sequence(ids[i:i+batch_size], batch_first=True, padding_value=-1)
                 for i in range(0, len(ids), batch_size)]
    white = config.get('white_background', False)
    all_cameras = load_cameras(args.source or config['source'], config['resolution'], white,
                               config.get('images', 'images'), 'train')
    _, cameras, _, view_indices = split_cameras(all_cameras, config['validation_views'], config.get('train_views', 0))
    names = [str(getattr(c, 'image_name', i)) for i, c in enumerate(cameras)]
    if config.get('validation_view_names', names) != names:
        raise ValueError('validation camera names differ from training')
    reference = RenderReference(raw, degree, white, 'source')
    meter = AllocationCostMeter(model.cfg, PositionCostMeter(model.cfg, geometry.normalize(raw[:, :3])),
                                len(raw), config['position_net_bits_per_use'])
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'checkpoint': str(checkpoint), 'codec_id': before,
                'shuffle_seed': args.shuffle_seed, 'allocation_seed': seed,
                'snr': snr, 'channel': kind, 'trials': args.trials,
                'view_names': names, 'view_indices': view_indices,
                'tier_counts': counts.tolist(), 'source_gaussians': len(raw),
                'scope': 'same tier histogram and payload, NOT guaranteed equal compressed total cost',
                'noise': 'same full point/symbol-slot draws per trial; original validation seed schedule',
                'cost': 'same AllocationCostMeter as training; measured XYZ + tier-map proxy, excludes packet framing/FEC/shared weights',
                'no_training': True}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows, summaries = [], {}
    with preserved_rng(device):
        for label, packet_map in packet_maps.items():
            np.save(out/f'{label}_tiers.npy', maps[label].numpy().astype(np.uint8))
            qs = [torch.where(gi >= 0, packet_map[gi.clamp_min(0)], 0) for gi in group_ids]
            observations = []
            for trial in range(args.trials):
                seed_all(seed+20000+trial)
                scene = decode_batches(model, groups, qs, snr, kind, geometry, paired_noise=True)
                for view, camera in enumerate(cameras):
                    decoded = render(scene, camera, degree, white)
                    target, photo = reference.get(camera, device), camera.original_image[:3].to(device)
                    row = {'layout': label, 'trial': trial, 'view_index': view, 'view': names[view],
                           **{'source_'+k: v for k, v in image_metrics(decoded, target).items()},
                           **{'photo_'+k: v for k, v in image_metrics(decoded, photo).items()}}
                    observations.append(row)
                    if trial == 0:
                        save_panel(out/'images'/f'{label}_view{view:02d}.png', photo, target, decoded)
                del scene
                print(f'{label}: trial {trial+1}/{args.trials} complete', flush=True)
            summaries[label] = {**summarize(observations), **meter.details(packet_map),
                                'tier_counts': counts.tolist(),
                                'payload_complex_symbols': int(torch.tensor(model.cfg.rates)[packet_map].sum())}
            rows.extend(observations)
            print(json.dumps({'layout': label, **summaries[label]}), flush=True)
    if model_id(model) != before:
        raise RuntimeError('diagnostic unexpectedly changed codec weights')
    deltas = [{k: a[k]-b[k] for k in summarize([rows[0]])}
              for a, b in zip(rows[:len(rows)//2], rows[len(rows)//2:])]
    result = {'layouts': summaries, 'paired_learned_minus_shuffled': summarize(deltas),
              'sign': 'positive PSNR/SSIM favors learned; negative MSE favors learned',
              'codec_unchanged': True,
              'caveat': 'one shuffled map; descriptive paired comparison, not a significance test'}
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved comparison: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--out')
    parser.add_argument('--ply')
    parser.add_argument('--source')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--shuffle-seed', type=int, default=2026)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
