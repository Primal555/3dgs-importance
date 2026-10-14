"""Read-only, fixed-point latent bias/noise diagnostics; no deployment inputs."""
import csv
import hashlib
import math
from pathlib import Path

import torch

from .codec import channel, prefix_mask
from .modular_experiment import json_write
from .optimization import preserved_rng
from .transport import model_id


def error_moments(target, noiseless, noisy):
    """Finite-trial population identity; not an unbiased estimate of E_noise.

    Global error mean across points is distinct from each point's error mean
    across trials. The latter plus within-point variance exactly sums to MSE.
    """
    if (target.ndim != 2 or not target.numel() or noiseless.shape != target.shape
            or noisy.ndim != 3 or noisy.shape[1:] != target.shape or noisy.shape[0] < 2
            or any(not torch.isfinite(v).all() for v in (target, noiseless, noisy))):
        raise ValueError('finite [points,dimensions] targets and >=2 matching noise trials required')
    y, finite, predictions = [v.detach().double().cpu() for v in (target, noiseless, noisy)]
    errors = predictions-y
    mean_error = errors.mean(0)
    finite_error = finite-y
    mse = errors.square().mean((0,1))
    point_bias = mean_error.square().mean(0)
    variance = (errors-mean_error).square().mean((0,1))
    bias = mean_error.mean(0)
    shift = predictions.mean(0)-finite
    power = y.square().mean(0)
    total = float(mse.sum())
    dims = []
    for j in range(y.shape[1]):
        dims.append(dict(dimension=j, target_mean=float(y[:,j].mean()), target_std=float(y[:,j].std(unbiased=False)),
            target_power=float(power[j]), noiseless_bias=float(finite_error[:,j].mean()),
            noiseless_mse=float(finite_error[:,j].square().mean()), noisy_bias=float(bias[j]),
            noisy_mse=float(mse[j]), error_std=float(errors[:,:,j].std(unbiased=False)),
            pointwise_mean_error_mse=float(point_bias[j]), noise_variance=float(variance[j]),
            global_bias_mse=float(bias[j].square()), noise_induced_mean_shift_mse=float(shift[:,j].square().mean()),
            mse_contribution=float(mse[j])/total if total else 0.,
            relative_mse=float(mse[j]/power[j]) if float(power[j])>1e-12 else None))
    centered = mean_error-bias
    summary = dict(noiseless_mse=float(finite_error.square().mean()), noisy_mse=float(mse.mean()),
        pointwise_mean_error_mse=float(point_bias.mean()), noise_variance=float(variance.mean()),
        global_bias_mse=float(bias.square().mean()),
        noise_induced_mean_shift_mse=float(shift.square().mean()), target_power=float(power.mean()),
        decomposition_residual=float((mse-point_bias-variance).mean()))
    summary['noiseless_nmse'] = summary['noiseless_mse']/max(summary['target_power'],1e-12)
    summary['noisy_nmse'] = summary['noisy_mse']/max(summary['target_power'],1e-12)
    return dict(summary=summary, dimensions=dims,
        point_mean_error_covariance=(centered.T@centered/len(y)).tolist())


@torch.no_grad()
def sample_latents(model, scene, points, seed):
    """Sample points AFTER teacher context extraction; never reblock the scene."""
    total = sum(int((ids>=0).sum()) for ids in scene.group_ids)
    if total < 1: raise ValueError('diagnostics requires active scene points')
    selected = torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:min(points,total)].sort().values
    device = next(model.parameters()).device
    values=[]; offset=0
    for features, ids in zip(scene.groups,scene.group_ids):
        active=ids>=0; count=int(active.sum())
        local=selected[(selected>=offset)&(selected<offset+count)]-offset
        if local.numel():
            f=features.to(device); mask=active.to(device)
            y=model.semantic_encoder(f,f[...,:3],mask)[mask]
            values.append(y[local.to(device)].cpu())
        offset+=count
    return torch.cat(values), hashlib.sha256(selected.numpy().tobytes()).hexdigest(), total


@torch.no_grad()
def diagnose(model, scene, out, *, points=32768, trials=8, batch_size=2048, seed=42, snr=10.):
    if min(points,batch_size)<1 or trials<2 or not math.isfinite(snr):
        raise ValueError('positive sample/batch sizes, >=2 trials and finite SNR required')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device; mode=model.training; identity=model_id(model)
    reports=[]; energies=[]
    try:
        with preserved_rng(device):
            model.eval()
            target, sample_hash, population=sample_latents(model,scene,points,seed)
            chunks=[]; raw_chunks=[]
            for y in target.split(batch_size):
                y=y.to(device)[None]; active=torch.ones(y.shape[:-1],device=device,dtype=torch.bool)
                values={}
                hook=model.channel_encoder.readout.register_forward_hook(lambda m,a,o: values.update(raw=o))
                try: signal=model.channel_encoder(y,active,snr)
                finally: hook.remove()
                chunks.append(signal[0].cpu()); raw_chunks.append(values['raw'][0].cpu())
            signal=torch.cat(chunks); raw=torch.cat(raw_chunks)
            for start,end in zip(model.cfg.rates[:-1],model.cfg.rates[1:]):
                pre=raw[:,2*start:2*end].double().square().sum(-1)/(end-start)
                post=signal[:,2*start:2*end].double().square().sum(-1)/(end-start)
                energies.append(dict(start=start,end=end,raw_power_mean=float(pre.mean()),
                    raw_power_median=float(pre.median()), normalized_power_mean=float(post.mean()),
                    normalized_power_max=float(post.max()),
                    radial_log_gain_median=float((model.cfg.power_floor/(pre+model.cfg.power_floor)).median())))
            for tier in (1,2,3):
                predictions=[]
                # Private fixed-slot noise: same selected points/trials across tiers.
                for trial in range(-1,trials):
                    generator=torch.Generator(device=device).manual_seed(seed+1000+max(trial,0))
                    decoded=[]
                    for z in signal.split(batch_size):
                        z=z.to(device)[None]; q=torch.full(z.shape[:-1],tier,device=device,dtype=torch.long)
                        mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
                        received=channel(z.reshape(-1,2),snr,'none' if trial<0 else 'awgn',generator).reshape_as(z)
                        decoded.append(model.channel_decoder(received*mask,mask,q,snr)[0].cpu())
                    predictions.append(torch.cat(decoded))
                record=error_moments(target,predictions[0],torch.stack(predictions[1:]))
                record.update(tier=tier,symbols=model.cfg.rates[tier])
                reports.append(record)
                print(f"latent diagnostic q{tier}: finite NMSE={record['summary']['noiseless_nmse']:.6g}, "
                      f"10dB NMSE={record['summary']['noisy_nmse']:.6g}, "
                      f"bias={record['summary']['pointwise_mean_error_mse']:.6g}, variance={record['summary']['noise_variance']:.6g}",flush=True)
    finally: model.train(mode)
    if model_id(model)!=identity: raise RuntimeError('diagnostic changed model state')
    result=dict(version=1,sample_points=len(target),population_points=population,sample_sha256=sample_hash,
        trials=trials,batch_size=batch_size,actual_snr=snr,tx_condition_snr=snr,rx_condition_snr=snr,
        parameters_unchanged=True,tiers=reports,segment_power=energies,
        scope='fixed uniform point sample; teacher extracted with original full codec-batch context; padding excluded',
        variance_definition='population variance over finite trials (divide by T), not unbiased population-noise variance',
        warning='finite-trial mean error includes estimation noise; not proof of population bias or per-dimension semantic meaning')
    json_write(out/'latent_errors.json',result)
    csv_rows=[dict(tier=r['tier'],symbols=r['symbols'],**d) for r in reports for d in r['dimensions']]
    with (out/'latent_dimensions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    return result
