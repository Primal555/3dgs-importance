"""Read-only center checkpoint diagnostics: neighbor dependence and layer probes.

Coordinates/bounds/ordering stay fixed. No renderer, optimizer step on the codec,
communication channel, or declaration that sensitivity alone proves harm.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .allocation import scene_fingerprint
from .cli import seed_all, device_for
from .data import Geometry, read_ply, morton_order
from .render_objective import spaced_indices
from .transport import load_checkpoint, model_id


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@torch.no_grad()
def layer_features(model, xyz):
    active = torch.ones(xyz.shape[:2], dtype=torch.bool, device=xyz.device)
    z = model.learned.center_encoder(xyz*2-1, xyz, active)
    decoder = model.learned.center_decoder
    h = decoder.input(z)
    features = {'input': h}
    tap = 0
    for i, block in enumerate(decoder.blocks):
        h = block(h, active, self_only=decoder.self_only)
        features[f'block_{i+1}'] = h
        if i in decoder.tap_indices:
            features[f'tap_{i+1}'] = decoder.norms[tap](h)
            tap += 1
    return features


def distribution(values):
    x = torch.tensor(values, dtype=torch.float64)
    return {'mean': float(x.mean()), 'median': float(x.median()),
            'p95': float(x.quantile(.95)), 'max': float(x.max())}


@torch.no_grad()
def neighbor_sensitivity(model, blocks, indices, span, factors, targets=4, trials=3, seed=42):
    """Perturb source neighbors in world units; fix target/source slot and bounds.

    Decoder-only replaces neighbor latents with their re-encoded perturbed values
    while restoring target latent. Encoder-only does the converse. These hybrid
    latent interventions are not independent channel noise or proof of causality.
    """
    device = next(model.parameters()).device
    span = span.to(device)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    rows = []
    for index in indices:
        x = blocks[index].to(device)[None]
        n = x.shape[1]
        if n < 2:
            continue
        active = torch.ones((1, n), dtype=torch.bool, device=device)
        world = x[0]*span
        distances = torch.cdist(world.float(), world.float(), compute_mode='donot_use_mm_for_euclid_dist')
        distances.fill_diagonal_(float('inf'))
        nearest = distances.min(-1).values
        positive = nearest[nearest > 1e-8]
        if not len(positive):
            raise ValueError(f'block {index} has no nonzero nearest spacing for perturbation calibration')
        spacing = float(positive.median())
        z = model.learned.center_encoder(x*2-1, x, active)
        base = model.learned.center_decoder(z, active)
        for target in spaced_indices(n, min(targets, n)):
            for trial in range(trials):
                direction = torch.randn((1, n, 3), generator=generator).to(device)
                direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                direction[:, target] = 0
                for factor in factors:
                    epsilon = spacing*factor
                    changed = x+direction*epsilon/span
                    assert torch.equal(changed[:, target], x[:, target])
                    zp = model.learned.center_encoder(changed*2-1, changed, active)
                    decoder_only = zp.clone()
                    decoder_only[:, target] = z[:, target]
                    encoder_only = z.clone()
                    encoder_only[:, target] = zp[:, target]
                    # Separate calls preserve the baseline batch shape, including
                    # exact-zero perturbation. No Morton re-sort or bounds refit.
                    for route, received in (('end_to_end', zp), ('decoder_only', decoder_only), ('encoder_only', encoder_only)):
                        pred = model.learned.center_decoder(received, active)[:, target]
                        shift = float(((pred-base[:, target])*span).norm())
                        error = float(((pred-x[:, target])*span).norm())
                        rows.append({'block': index, 'target_slot': target, 'trial': trial, 'route': route,
                            'factor': factor, 'spacing_world': spacing, 'perturbation_world': epsilon,
                            'target_shift_world': shift, 'target_error_world': error,
                            'baseline_error_world': float(((base[:, target]-x[:, target])*span).norm()),
                            'reencoded_target_latent_shift': float((zp[:, target]-z[:, target]).norm()),
                            'received_target_latent_shift': float((received[:, target]-z[:, target]).norm()),
                            'response_per_neighbor_displacement': shift/epsilon if epsilon else None})
    if not rows:
        raise ValueError('no valid sensitivity blocks')
    summary = []
    for route in ('end_to_end', 'decoder_only', 'encoder_only'):
        for factor in factors:
            group = [r for r in rows if r['route'] == route and r['factor'] == factor]
            summary.append({'route': route, 'factor': factor, 'count': len(group),
                'shift_world': distribution([r['target_shift_world'] for r in group]),
                'error_change_world': distribution([r['target_error_world']-r['baseline_error_world'] for r in group]),
                'gain': distribution([r['response_per_neighbor_displacement'] for r in group]) if factor else None})
    return {'summary': summary, 'samples': rows,
            'interpretation': 'conditional sensitivity, not proof of harmful mixing; encoder and decoder routes need not add linearly'}


@torch.no_grad()
def feature_cache(model, blocks, indices):
    device = next(model.parameters()).device
    result, target, dispersion = {}, [], {}
    for index in indices:
        x = blocks[index].to(device)[None]
        target.append(x[0].cpu())
        for name, value in layer_features(model, x).items():
            h = value[0].cpu()
            result.setdefault(name, []).append(h)
            # Center across points, not channels: rank alone is not correctness.
            centered = h.double()-h.double().mean(0)
            singular = torch.linalg.svdvals(centered)
            probability = singular/singular.sum().clamp_min(1e-30)
            rank = float(torch.exp(-(probability*probability.clamp_min(1e-30).log()).sum())) if singular.sum() > 0 else 0.
            dispersion.setdefault(name, []).append({'block': index, 'effective_rank': rank,
                                                    'centered_rms': float(centered.square().mean().sqrt())})
    return {k: torch.cat(v) for k, v in result.items()}, torch.cat(target), dispersion


def fit_layer_probes(train_features, train_xyz, held_features, held_xyz, span,
                     device, steps=1000, width=64, batch_size=512, lr=1e-3, seed=42):
    """Same initialized two-layer MLP, batches and budget at every depth.

    Train-only invertible per-channel standardization removes a trivial feature
    scale conditioning difference. Targets also use train-only normalization.
    Validation never updates probes or selects stopping/checkpoints.
    """
    seed_all(seed)
    names = list(train_features)
    dim = train_features[names[0]].shape[-1]
    template = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, 3))
    heads = nn.ModuleDict()
    xf, xh = {}, {}
    standardization = {}
    for name in names:
        head = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, 3))
        head.load_state_dict(template.state_dict())
        heads[name] = head
        mean = train_features[name].mean(0)
        std = train_features[name].std(0, unbiased=False).clamp_min(1e-6)
        xf[name] = ((train_features[name]-mean)/std).to(device)
        xh[name] = ((held_features[name]-mean)/std).to(device)
        standardization[name] = {'mean': mean.tolist(), 'std': std.tolist()}
    heads.to(device)
    mean = train_xyz.mean(0).to(device)
    std = train_xyz.std(0, unbiased=False).clamp_min(1e-6).to(device)
    target = (train_xyz.to(device)-mean)/std
    span = span.to(device)
    optimizer = torch.optim.Adam(heads.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed+1)
    history = []

    @torch.no_grad()
    def measure(step):
        row = {'step': step, 'layers': {}}
        for name in names:
            metrics = {}
            for split, feats, truth in (('fit', xf[name], train_xyz), ('heldout', xh[name], held_xyz)):
                preds = torch.cat([heads[name](part) for part in feats.split(4096)])*std+mean
                delta = (preds-truth.to(device))*span
                metrics[split+'_world_rmse'] = float(delta.double().square().mean().sqrt())
                metrics[split+'_distance_median'] = float(delta.norm(dim=-1).median())
            row['layers'][name] = metrics
        history.append(row)
        print(f'layer probes {step}/{steps}: '+', '.join(f'{k}={v["heldout_world_rmse"]:.4g}' for k,v in row['layers'].items()), flush=True)

    measure(0)
    for step in range(1, steps+1):
        indices = torch.randint(len(target), (batch_size,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = [(heads[name](xf[name][indices])-target[indices]).square().mean() for name in names]
        loss = torch.stack(losses).sum()  # disjoint heads: each receives its own unscaled loss
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite probe loss')
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps:
            measure(step)
    if any(not math.isfinite(v) for r in history for m in r['layers'].values() for v in m.values()):
        raise FloatingPointError('nonfinite probe metric')
    return {'history': history, 'final': history[-1]['layers'], 'steps': steps, 'width': width,
            'batch_size': batch_size, 'lr': lr, 'seed': seed, 'input_statistics': standardization,
            'target_mean': mean.cpu().tolist(), 'target_std': std.cpu().tolist(),
            'interpretation': 'finite-budget supervised probes, not codec outputs or achievable communication quality'}


def plot_diagnostics(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for route in ('end_to_end', 'decoder_only', 'encoder_only'):
        rows = [r for r in report['sensitivity']['summary'] if r['route'] == route]
        axes[0].plot([r['factor'] for r in rows], [r['shift_world']['mean'] for r in rows], 'o-', label=route)
    axes[0].set_title('Fixed target: mean prediction shift')
    axes[0].set_xlabel('Neighbor displacement / median nearest spacing')
    axes[0].set_ylabel('World distance')
    names = list(report['probes']['final'])
    for split in ('fit', 'heldout'):
        axes[1].plot(names, [report['probes']['final'][n][split+'_world_rmse'] for n in names], 'o-', label=split)
    axes[1].set_title('Frozen-layer MLP probes: world RMSE')
    axes[1].tick_params(axis='x', rotation=50)
    for name in names:
        axes[2].plot([r['step'] for r in report['probes']['history']],
                     [r['layers'][name]['heldout_world_rmse'] for r in report['probes']['history']], label=name)
    axes[2].set_title('Probe optimization / heldout RMSE')
    axes[2].set_xlabel('Probe step (not codec training)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out)/'diagnostics.png', dpi=160)
    plt.close(fig)


def run(args):
    if min(args.fit_blocks, args.heldout_blocks, args.targets, args.trials, args.probe_steps,
           args.probe_width, args.probe_batch, args.cpu_threads) < 1:
        raise ValueError('counts must be positive')
    if not math.isfinite(args.probe_lr) or args.probe_lr <= 0 or any(not math.isfinite(x) or x < 0 for x in args.factors):
        raise ValueError('invalid probe LR or perturbation factors')
    factors = sorted(set([0., *args.factors]))
    out = Path(args.out).resolve()
    if out.exists() and any(p.name not in ('console.log', 'run.pid') for p in out.iterdir()):
        raise FileExistsError('choose a new diagnostics output')
    seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = device_for(args.device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = load_checkpoint(args.checkpoint, device).requires_grad_(False)
    if (not model.cfg.center_latent_dim or model.cfg.center_decoder_kind != 'transformer'
            or model.cfg.center_position_layout != 'absolute'):
        raise ValueError('layer interaction probes require absolute-layout independent Transformer centers; '
                         'use center-drift diagnostics for centroid_residual')
    before = model_id(model)
    info = saved['training']
    if not all(k in info for k in ('geometry', 'fitted_blocks', 'heldout_blocks', 'fingerprint')):
        raise ValueError('checkpoint lacks original geometry/split provenance')
    raw, degree = read_ply(args.ply)
    if degree != model.cfg.sh_degree:
        raise ValueError('PLY SH degree differs')
    fingerprint = scene_fingerprint(raw)
    match = fingerprint == info['fingerprint']
    if not match and not args.allow_ply_mismatch:
        raise ValueError('PLY fingerprint differs; use the original PLY, or explicitly --allow-ply-mismatch for non-exact diagnostics')
    geometry = Geometry(**info['geometry'])
    xyz = raw[:, :3]
    bounds = (xyz-geometry.lower)/geometry.span
    if bounds.min() < -1e-5 or bounds.max() > 1.00001:
        raise ValueError('PLY lies outside saved geometry; will not silently refit bounds')
    order = torch.from_numpy(morton_order(geometry.quantize(xyz).numpy()).astype(np.int64))
    unit = geometry.normalize(xyz[order])
    blocks = list(unit.split(model.cfg.block_size))
    fitted, heldout = info['fitted_blocks'], info['heldout_blocks']
    if set(fitted) & set(heldout) or set(fitted+heldout) != set(range(len(blocks))):
        raise ValueError('saved split overlaps or does not cover this PLY')
    fitted = [fitted[i] for i in spaced_indices(len(fitted), min(args.fit_blocks, len(fitted))) ]
    heldout = [heldout[i] for i in spaced_indices(len(heldout), min(args.heldout_blocks, len(heldout))) ]
    out.mkdir(parents=True, exist_ok=True)
    print(f'Diagnostics: checkpoint step={saved["step"]}, readout={model.cfg.center_readout_norm}, '
          f'PLY fingerprint match={match}; codec weights frozen', flush=True)
    sensitivity = neighbor_sensitivity(model, blocks, heldout, geometry.span, factors, args.targets, args.trials, args.seed)
    write_json(out/'sensitivity.json', sensitivity)
    print('Neighbor interventions complete; extracting frozen layer features.', flush=True)
    xf, yf, df = feature_cache(model, blocks, fitted)
    xh, yh, dh = feature_cache(model, blocks, heldout)
    probes = fit_layer_probes(xf, yf, xh, yh, geometry.span, device, args.probe_steps,
                              args.probe_width, args.probe_batch, args.probe_lr, args.seed)
    after = model_id(model)
    if before != after:
        raise RuntimeError('diagnostic unexpectedly changed codec weights')
    report = {'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_step': saved['step'],
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'model_id_before': before, 'model_id_after': after, 'weights_unchanged': True,
              'config': model.cfg.to_dict(), 'geometry': geometry.to_dict(), 'arguments': vars(args),
              'ply_fingerprint_match': match, 'ply_fingerprint': fingerprint,
              'fitted_blocks': fitted, 'heldout_blocks': heldout, 'sensitivity': sensitivity,
              'probes': probes, 'feature_dispersion': {'fitted': df, 'heldout': dh},
              'limitations': ['No PSNR measured by these probes.', 'Hybrid latent interventions may be off-distribution.',
                             'No block reassignment, re-sorting or bounds refit.',
                             'Sensitivity/rank alone do not establish harmful overmixing.']}
    write_json(out/'diagnostics.json', report)
    plot_diagnostics(report, out)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ply', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--fit-blocks', type=int, default=64)
    p.add_argument('--heldout-blocks', type=int, default=8)
    p.add_argument('--targets', type=int, default=4)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--factors', type=float, nargs='+', default=[0., .01, .05, .1])
    p.add_argument('--probe-steps', type=int, default=1000)
    p.add_argument('--probe-width', type=int, default=64)
    p.add_argument('--probe-batch', type=int, default=512)
    p.add_argument('--probe-lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--allow-ply-mismatch', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
