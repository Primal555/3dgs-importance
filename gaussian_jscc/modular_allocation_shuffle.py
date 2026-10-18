"""Frozen reliable-XYZ codec: shuffle positive tiers, never the keep/drop set."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .cli import device_for, seed_all
from .data import read_ply
from .modular_allocation import measured_cost
from .modular_checkpoint import initialize_codec_run
from .modular_evaluation import decode_path
from .modular_experiment import inputs, make_scene, json_write
from .optimization import preserved_rng
from .perceptual_metrics import LPIPSEvaluator, image_metric_protocol
from .render_objective import image_metrics, spaced_indices
from .render_validation import append_json, save_panel
from .transport import model_id


def validated_tiers(q):
    q = torch.as_tensor(q).detach().cpu()
    if (q.ndim != 1 or not q.numel() or q.is_floating_point() or
            q.is_complex() or q.dtype == torch.bool or q.min() < 0 or q.max() > 3):
        raise ValueError('tiers must be nonempty integer [N] labels in q0..q3')
    return q.long()


def shuffle_retained(q, seed):
    """Original PLY row order; independently permute positive labels only."""
    q = validated_tiers(q)
    result = q.clone()
    ids = (q > 0).nonzero().flatten()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    result[ids] = q[ids[torch.randperm(len(ids), generator=generator)]]
    return result


def tier_batches(q, group_ids):
    """Map original-row labels to unchanged padded Morton blocks."""
    qs = [torch.where(ids >= 0, q[ids.clamp_min(0)], 0) for ids in group_ids]
    return qs, torch.cat([v[ids >= 0] for v, ids in zip(qs, group_ids)])


def metric_means(rows):
    if not rows:
        raise ValueError('no metric observations')
    keys = [k for k in rows[0] if k.startswith(('source_', 'photo_'))]
    result = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    if not keys or not all(math.isfinite(v) for v in result.values()):
        raise ValueError('missing or nonfinite image metrics')
    return result


def comparison_summary(observations):
    """Paired differences; spread across permutations, NOT training confidence."""
    def indexed(rows):
        result = {(r['trial'], r['view']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate trial/view observations')
        return result

    baseline = indexed(observations['learned'])
    per_shuffle = {}
    for label, rows in observations.items():
        if label == 'learned':
            continue
        other = indexed(rows)
        if baseline.keys() != other.keys():
            raise ValueError('unmatched trial/view observations')
        differences = [{k: row[k] - other[pair][k] for k in metric_means([row])}
                       for pair, row in baseline.items()]
        per_shuffle[label] = metric_means(differences)
    if not per_shuffle:
        raise ValueError('at least one shuffled layout is required')
    spread = {k: dict(mean=float(np.mean(values)), std=float(np.std(values)),
                      min=float(min(values)), max=float(max(values)))
              for k in next(iter(per_shuffle.values()))
              for values in [[v[k] for v in per_shuffle.values()]]}
    return dict(per_shuffle=per_shuffle, across_shuffle_seeds=spread,
                sign='learned minus shuffled: positive PSNR/SSIM or negative MSE/LPIPS favors learned',
                caveat='permutation spread is descriptive; not a training-seed confidence interval')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='completed modular allocation run (beta=0.005 by default)')
    p.add_argument('--out', required=True, help='NEW directory; never overwrite training results')
    p.add_argument('--manifest', help='optional relocated dataset manifest; identity must match')
    p.add_argument('--device', default='cuda')
    p.add_argument('--expected-beta', type=float, default=.005, help='reject accidental comparison of another beta')
    p.add_argument('--shuffle-seeds', type=int, nargs='+', default=[101, 102, 103, 104, 105])
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--test-views', type=int, default=0, help='0: all held-out test cameras')
    p.add_argument('--image-views', type=int, default=2, help='learned + first shuffle only; 0 disables images')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    return p


@torch.no_grad()
def evaluate_maps(model, scene, q, out, args, config):
    from .rendering import render

    device = next(model.parameters()).device
    before = model_id(model)
    counts = torch.bincount(q, minlength=4).tolist()
    perceptual = LPIPSEvaluator(device) if args.lpips else None
    labels = ['learned'] + [f'shuffle_{seed}' for seed in args.shuffle_seeds]
    selected = set(spaced_indices(len(scene.val_cameras), args.image_views)) if args.image_views else set()
    seed = config['seed']
    observations, layouts = {}, {}
    for index, label in enumerate(labels):
        current = q if index == 0 else shuffle_retained(q, args.shuffle_seeds[index-1])
        if (not torch.equal(current == 0, q == 0) or
                torch.bincount(current, minlength=4).tolist() != counts):
            raise RuntimeError('shuffle changed the keep set or tier histogram')
        qs, packet_q = tier_batches(current, scene.group_ids)
        rows = []
        for trial in range(args.trials):
            # decode_path draws every padded point's full maximum-prefix slots,
            # independently of tier. Identical blocks/seeds give paired AWGN.
            seed_all(seed + 1000 + trial)
            received, _ = decode_path(model, scene, qs, 'awgn', config['snr'])
            for view, camera in enumerate(scene.val_cameras):
                image = render(received, camera, scene.degree, scene.white_background)
                source = scene.reference.get(camera, device)
                photo = camera.original_image[:3].to(device)
                metrics = {prefix+'_'+k: v
                           for prefix, target in (('source', source), ('photo', photo))
                           for k, v in image_metrics(image, target).items()}
                if perceptual is not None:
                    metrics.update(source_lpips=perceptual(image, source), photo_lpips=perceptual(image, photo))
                row = dict(layout=label, trial=trial, view=view, image_name=str(camera.image_name), **metrics)
                rows.append(row)
                append_json(out/'metrics.jsonl', row)
                if index <= 1 and trial == 0 and view in selected:
                    save_panel(out/'images'/f'{label}_view{view:04d}.png', photo, source, image, label)
            del received
            print(f'{label}: channel trial {trial+1}/{args.trials} complete', flush=True)
        observations[label] = rows
        cost = measured_cost(model, scene, packet_q, config['snr'], before)
        layouts[label] = dict(**metric_means(rows), **cost, tier_counts=counts,
            tier_shares=[n/len(q) for n in counts], retained_gaussians=int((q > 0).sum()),
            changed_positive_labels=int((current != q).sum()),
            tier_map_sha256=hashlib.sha256(current.numpy().astype(np.uint8).tobytes()).hexdigest())
        baseline = layouts['learned']
        if (cost['payload_complex_symbols'] != baseline['payload_complex_symbols'] or
                cost['xyz_bytes'] != baseline['xyz_bytes']):
            raise RuntimeError('fixed keep-set control changed payload or coordinate bytes')
        layouts[label]['total_channel_uses_minus_learned'] = cost['total_channel_uses'] - baseline['total_channel_uses']
        if index <= 1:
            np.save(out/f'{label}_tiers.npy', current.numpy().astype(np.uint8), allow_pickle=False)
        json_write(out/'layouts'/f'{label}.json', layouts[label])
        print(f"{label}: PSNR={layouts[label]['source_psnr']:.4f}, "
              f"SSIM={layouts[label]['source_ssim']:.4f}; total uses={cost['total_channel_uses']}", flush=True)
    if model_id(model) != before:
        raise RuntimeError('evaluation changed codec weights')
    result = dict(layouts=layouts, paired_learned_minus_shuffled=comparison_summary(observations),
                  codec_unchanged=True, keep_set_unchanged=True,
                  caveat='isolates positive-tier placement, including receiver neighbor interactions; not deletion benefit. '
                         'Payload and XYZ bytes match, but compressed tier metadata and total costs can differ.')
    json_write(out/'summary.json', result)
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        rows = [r for values in observations.values() for r in values]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_write(out/'complete.json', dict(status='complete', codec_id=before, layouts=labels))
    return result


def main(args):
    from .rendering import load_cameras

    if (args.trials < 1 or args.test_views < 0 or args.image_views < 0 or args.cpu_threads < 1 or
            not args.shuffle_seeds or len(set(args.shuffle_seeds)) != len(args.shuffle_seeds) or
            not math.isfinite(args.expected_beta) or args.expected_beta < 0):
        raise ValueError('invalid evaluation settings or duplicate shuffle seeds')
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'refusing to overwrite: {out}')
    config = json.loads((run/'training.json').read_text(encoding='utf-8'))
    complete = json.loads((run/'complete.json').read_text(encoding='utf-8'))
    if not math.isclose(config['beta'], args.expected_beta, rel_tol=0, abs_tol=1e-12):
        raise ValueError('source beta differs from --expected-beta')
    allocation = json.loads((run/'deployment/final/allocation.json').read_text(encoding='utf-8'))
    q = validated_tiers(np.load(run/'deployment/final/tiers.npy', allow_pickle=False))
    if (len(q) != allocation['source_gaussians'] or not (q > 0).any() or
            torch.bincount(q, minlength=4).tolist() != allocation['hard_tier_counts']):
        raise ValueError('saved deployment map counts do not match final allocation record')
    device = device_for(args.device)
    torch.set_num_threads(args.cpu_threads)
    source, manifest, spec, old = inputs(args)
    with preserved_rng(device):
        seed_all(config['seed'])
        checkpoint = run/'checkpoints/final/codec.pt'
        model, progress, saved_scene = initialize_codec_run(checkpoint, device)
        if (model.cfg.position_delivery != 'quantized' or model.cfg.position_bits != 16 or
                model.cfg.rates != (0, 8, 24, 48) or model.cfg.prefix_mode != 'progressive'):
            raise ValueError('requires reliable 16bit XYZ and progressive 8/24/48 modular codec')
        identity = model_id(model)
        if (identity != complete['final_codec_id'] or json.loads(json.dumps(model.cfg.to_dict())) != config['codec_config'] or
                progress['step'] != allocation['step'] or allocation['rates'] != list(model.cfg.rates) or
                allocation['snr'] != config['snr']):
            raise ValueError('final checkpoint / deployment map / training configuration mismatch')
        model.set_training_snrs(progress.get('train_snrs', [config['snr']]))
        model.eval().requires_grad_(False)
        raw, degree = read_ply(spec['ply'])
        if degree != model.cfg.sh_degree:
            raise ValueError('source SH degree differs from checkpoint')
        # make_scene verifies fingerprint and original camera splits; all writes
        # are to the NEW diagnostic directory, never the source training run.
        scene = make_scene(spec, raw, degree, model, source, old, out)
        del raw
        if scene.record != saved_scene or len(scene.raw) != len(q):
            raise ValueError('source scene differs from final checkpoint')
        cameras = load_cameras(spec['source'], scene.record['resolution'], scene.white_background,
                               scene.record['images'], 'test')
        if not cameras:
            raise ValueError('no held-out test cameras')
        scene.val_cameras = ([cameras[i] for i in spaced_indices(len(cameras), args.test_views)]
                             if args.test_views else cameras)
        names = [str(c.image_name) for c in scene.val_cameras]
        if len(set(names)) != len(names) or set(names) & set(scene.record['train_view_names']):
            raise ValueError('duplicate test names or test/train overlap')
        json_write(out/'manifest.json', dict(source_run=str(run), checkpoint=str(checkpoint),
            codec_id=identity, beta=config['beta'], rates=list(model.cfg.rates), snr=config['snr'],
            scene_fingerprint=scene.fingerprint, manifest=str(manifest),
            tier_counts=allocation['hard_tier_counts'], deployment='saved final ten-draw map; NOT resampled',
            shuffle_seeds=args.shuffle_seeds, trials=args.trials, noise_seed_base=config['seed']+1000,
            noise_pairing='same padded Morton blocks and full maximum-prefix AWGN draws per point/symbol slot',
            view_names=names, image_views=args.image_views, image_metric_protocol=image_metric_protocol(args.lpips),
            no_training=True, keep_set_fixed=True,
            cost='measured payload + retained XYZ + complete metadata; shared model/FEC excluded'))
        return evaluate_maps(model, scene, q, out, args, config)


if __name__ == '__main__':
    main(parser().parse_args())
