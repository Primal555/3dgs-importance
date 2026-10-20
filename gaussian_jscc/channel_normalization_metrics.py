"""Original-block diagnostics and full-population power for normalization controls."""
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .channel_audit import (accumulate, attribute_response, averaged, distribution,
                           physical_response, response_metrics, trace_channel, write_csv)
from .codec import channel, prefix_mask
from .latent_error_diagnostics import error_moments
from .modular_experiment import json_write
from .optimization import preserved_rng
from .position_delivery import delivered_positions


@torch.no_grad()
def cache_teacher(model, scene):
    """CPU cache; each row retains its ORIGINAL Morton block and padding mask."""
    device = next(model.parameters()).device
    model.eval(); model.set_scene_geometry(scene.geometry)
    cache = []
    for features, ids in zip(scene.groups, scene.group_ids):
        f = features.to(device); active = (ids >= 0).to(device)
        y = model.semantic_encoder(f, f[..., :3], active)
        xyz = delivered_positions(f[..., :3], active.long(), model.cfg)
        cache.append(dict(y=y.cpu(), active=active.cpu(), xyz=xyz.cpu()))
    return cache


@torch.no_grad()
def calibrate_and_measure(model, cache, *, calibrate, tolerance=.02):
    """Calibration uses ALL training points, never reference images or test errors.

    Eval freezes scales. Returns actual segment/prefix energy, instantaneous
    symbol PAPR, and a budget flag. AWGN is always referenced to P=1, not to
    the observed instantaneous energies. No receiver inverse gain is supplied.
    """
    encoder = model.channel_encoder; device = next(model.parameters()).device
    mode = encoder.training; encoder.eval()
    rates = model.cfg.rates; lengths = torch.tensor([b-a for a, b in zip(rates, rates[1:])], dtype=torch.float64)
    raw_sum = torch.zeros(len(lengths), dtype=torch.float64); count = 0
    post = []; peaks = torch.zeros(len(lengths), dtype=torch.float64)
    before = getattr(getattr(encoder, 'normalizer', None), 'running_energy', None)
    before = before.cpu().tolist() if before is not None else None
    try:
        for batch in cache:
            y, active = batch['y'].to(device), batch['active'].to(device)
            z = encoder.raw_signal(y, active, 10.)[active].double()
            for j, (a, b) in enumerate(zip(rates, rates[1:])):
                raw_sum[j] += z[:, 2*a:2*b].square().sum().cpu()/(b-a)
            count += len(z)
        if not count: raise ValueError('power accounting requires a nonempty teacher population')
        raw_mean = raw_sum/count
        if calibrate and model.cfg.channel_power_mode == 'statistical':
            encoder.normalizer.set_population_energy(raw_mean)
        for batch in cache:
            y, active = batch['y'].to(device), batch['active'].to(device)
            z = encoder(y, active, 10.)[active].double()
            energies = []
            for j, (a, b) in enumerate(zip(rates, rates[1:])):
                symbols = z[:, 2*a:2*b].reshape(-1, b-a, 2).square().sum(-1)
                energies.append(symbols.mean(-1).cpu())
                if symbols.numel(): peaks[j] = max(float(peaks[j]), float(symbols.max()))
            post.append(torch.stack(energies, -1))
    finally:
        encoder.train(mode)
    energy = torch.cat(post); segments = []; prefixes = []
    for j, (a, b) in enumerate(zip(rates, rates[1:])):
        measured = distribution(energy[:, j]); mean = measured['mean']
        segments.append(dict(start=a, end=b, raw_mean_complex_power=float(raw_mean[j]),
            mean_complex_power=mean, point_energy=measured,
            symbol_papr=float(peaks[j])/max(mean, 1e-30),
            effective_snr_db=10+10*math.log10(max(mean, 1e-30))))
        cumulative = (energy[:, :j+1]*lengths[:j+1]).sum(-1)/b
        prefixes.append(dict(tier=j+1, symbols=b, mean_complex_power=float(cumulative.mean()),
                             point_energy=distribution(cumulative)))
    after = getattr(getattr(encoder, 'normalizer', None), 'running_energy', None)
    return dict(population_points=count, segments=segments, prefixes=prefixes,
        calibration_applied=bool(calibrate and model.cfg.channel_power_mode == 'statistical'),
        population_energy_before=before, population_energy_after=after.cpu().tolist() if after is not None else None,
        reference_complex_power=1., noise_variance_per_real=.05, power_tolerance=tolerance,
        within_average_power_budget=all(r['mean_complex_power'] <= 1+tolerance for r in segments),
        scope='all retained original scene points, uniform positive tiers; not learned deployment',
        receiver_scale_side_information=False)


def selected_blocks(scene, cache, count, seed):
    available = [(g, b) for g, batch in enumerate(cache) for b in range(len(batch['y']))
                 if batch['active'][b].any()]
    indices = torch.randperm(len(available), generator=torch.Generator().manual_seed(seed))[:count].sort().values
    selected = []
    for index in indices.tolist():
        g, b = available[index]
        item = {k: v[b:b+1] for k, v in cache[g].items()}
        item.update(features=scene.groups[g][b:b+1], source_ids=scene.group_ids[g][b:b+1], block_index=index)
        selected.append(item)
    return selected


@torch.no_grad()
def diagnose_normalization(model, scene, cache, out, *, blocks=32, response_blocks=8, trials=8, seed=42):
    """Repeated paired noise; geometry/teacher context never regrouped.

    Physical response samples are spread across the selected original blocks,
    not just the first few Morton blocks. Node deltas are conditional forward
    observations, not retrained causal ablations.
    """
    device = next(model.parameters()).device; out = Path(out); model.eval()
    selected = selected_blocks(scene, cache, blocks, seed)
    measure = set(torch.linspace(0, len(selected)-1, min(response_blocks, len(selected))).round().long().tolist())
    source_ids = torch.cat([b['source_ids'][b['active']] for b in selected])
    directions = F.normalize(torch.randn(8, 3, generator=torch.Generator().manual_seed(seed+901)), dim=-1).to(device)
    rows = []; stage_rows = []; dimension_rows = []
    with preserved_rng(device):
        for tier in (1, 2, 3):
            targets = []; predictions = [[] for _ in range(trials+1)]; nodes = {}; responses = {}
            for number, block in enumerate(selected):
                y, active, xyz = [block[k].to(device) for k in ('y', 'active', 'xyz')]
                q = active.long()*tier
                z = model.channel_encoder(y, active, 10.)
                mask = prefix_mask(q.flatten(), model.cfg.rates).reshape_as(z)
                targets.append(y[active].cpu())
                if number in measure:
                    clean = physical_response(model, y, active, xyz, directions)
                    attrs = block['features'].to(device)[active][:, 3:]*model.attr_std+model.attr_mean
                    source = attribute_response(attrs, scene.degree, directions)
                for trial in range(-1, trials):
                    generator = torch.Generator(device=device).manual_seed(seed+100003*block['block_index']+1000+max(trial, 0))
                    received = channel(z.reshape(-1, 2), 10., 'none' if trial < 0 else 'awgn', generator).reshape_as(z)*mask
                    trace = trace_channel(model.channel_decoder, received, mask, q, 10., xyz)
                    final = trace['final']['value']; predictions[trial+1].append(final[active].cpu())
                    path = 'noiseless' if trial < 0 else 'awgn10'; n = int(active.sum())
                    for name, node in trace.items():
                        own = node['active']
                        if not own.any(): continue
                        metrics = {key+'_mse': float((value[own]-y[own]).double().square().mean())
                                   for key, value in (('before', node['before']), ('after', node['value']))}
                        metrics['delta_mse'] = metrics['after_mse']-metrics['before_mse']
                        accumulate(nodes, (path, name), metrics, int(own.sum()))
                    if number in measure:
                        final_response = physical_response(model, final, active, xyz, directions)
                        for reference, target in (('clean', clean), ('source', source)):
                            accumulate(responses, (path, 'final', reference), response_metrics(final_response, target), n)
                        # The input of the last compensation module, or the final
                        # stage if this architecture has no compensation branch.
                        residuals = [v for k, v in trace.items() if 'denoiser' in k or k == 'neighbor']
                        before = residuals[-1]['before'] if residuals else final
                        response = physical_response(model, before, active, xyz, directions)
                        accumulate(responses, (path, 'before_last_compensation', 'clean'), response_metrics(response, clean), n)
            target = torch.cat(targets); values = [torch.cat(v) for v in predictions]
            moments = error_moments(target, values[0], torch.stack(values[1:]))
            rows.append(dict(tier=tier, symbols=model.cfg.rates[tier], moments=moments,
                semantic_response=[dict(path=p, node=n, reference=r, **metrics)
                    for (p, n, r), metrics in averaged(responses).items()]))
            stage_rows.extend(dict(tier=tier, path=p, node=n, **metrics)
                              for (p, n), metrics in averaged(nodes).items())
            dimension_rows.extend(dict(tier=tier, **r) for r in moments['dimensions'])
            print(f"diagnostic q{tier}: no-noise NMSE={moments['summary']['noiseless_nmse']:.6g}; "
                  f"10dB NMSE={moments['summary']['noisy_nmse']:.6g}", flush=True)
    result = dict(sample_sha256=hashlib.sha256(source_ids.numpy().tobytes()).hexdigest(),
        sample_points=len(source_ids), block_indices=[b['block_index'] for b in selected],
        response_block_indices=[selected[i]['block_index'] for i in sorted(measure)],
        trials=trials, tiers=rows, noise_variance_per_real=.05,
        limitation='finite-trial mean errors contain sampling noise; responses are not rendered image metrics; stage traces are conditional contributions')
    json_write(out/'latent_errors.json', result)
    write_csv(out/'stage_metrics.csv', stage_rows); write_csv(out/'latent_dimensions.csv', dimension_rows)
    return result
