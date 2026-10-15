"""Read-only joint error directions and frozen-decoder color sensitivity."""
import math
import csv
from pathlib import Path

import torch


@torch.no_grad()
def finite_difference_gram(response,point,epsilon,*,return_jacobian=False):
    """J^T J / output_count for one vector, without an autograd Jacobian.

    Callers preserve the entire original point block in response(). Only the
    chosen point changes; output neighbors remain in the measured response.
    """
    if point.ndim!=1 or not point.numel() or not torch.isfinite(point).all() or not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError('finite vector and positive finite difference epsilon required')
    columns=[]
    for dimension in range(len(point)):
        delta=torch.zeros_like(point);delta[dimension]=epsilon
        plus=response(point+delta).detach().flatten().double().cpu()
        minus=response(point-delta).detach().flatten().double().cpu()
        if not plus.numel() or plus.shape!=minus.shape or not torch.isfinite(plus).all() or not torch.isfinite(minus).all():
            raise ValueError('response must return matching finite nonempty tensors')
        if columns and plus.shape!=columns[0].shape:raise ValueError('response shape changed across probes')
        columns.append((plus-minus)/(2*epsilon))
    jacobian=torch.stack(columns,1)
    gram=jacobian.T@jacobian/len(jacobian)
    return (gram,jacobian) if return_jacobian else gram


def direction_risks(second_moment,gram):
    """Error PCA directions ranked by lambda * v^T G v, summing to tr(G S).

    These are correlated latent directions, never hand-labeled RGB dimensions.
    The local linearization is diagnostic only and does not set loss weights.
    """
    s,g=(v.detach().double().cpu() for v in (second_moment,gram))
    if (s.ndim!=2 or not s.numel() or s.shape[0]!=s.shape[1] or g.shape!=s.shape or
            not torch.isfinite(s).all() or not torch.isfinite(g).all()):
        raise ValueError('matching finite square error moment and sensitivity matrices required')
    if any(not torch.allclose(v,v.T,rtol=1e-6,atol=1e-10) for v in (s,g)):
        raise ValueError('error moment and sensitivity must be symmetric')
    s=(s+s.T)/2;g=(g+g.T)/2
    eigenvalues,vectors=torch.linalg.eigh(s)
    if float(eigenvalues.min()) < -1e-7*max(1.,float(s.abs().max())) or float(torch.linalg.eigvalsh(g).min()) < -1e-7*max(1.,float(g.abs().max())):
        raise ValueError('error moment and sensitivity must be positive semidefinite')
    directions=[]
    for index,value in enumerate(eigenvalues.clamp_min(0)):
        vector=vectors[:,index]
        sensitivity=float(vector@g@vector)
        directions.append(dict(error_energy=float(value),sensitivity=sensitivity,
            risk=float(value)*sensitivity,vector=vector.tolist()))
    directions.sort(key=lambda row:row['risk'],reverse=True)
    return dict(total_linearized_risk=float(torch.trace(g@s)),directions=directions,
        diagonal_only_contributions=(s.diag()*g.diag()).tolist(),
        warning='local linear response; eigenvector signs arbitrary; does not label semantic dimensions')


def decoded_colors(model,latent,active,xyz,directions):
    """Unclipped directional SH RGB from a fixed semantic decoder, no renderer."""
    from utils.sh_utils import eval_sh
    attrs=model.semantic_decoder(latent,active,xyz)[active]
    coefficients=attrs[:,8:]*model.attr_std[8:]+model.attr_mean[8:]
    dc=coefficients[:,:3,None]
    rest=coefficients[:,3:].reshape(len(attrs),3,(model.cfg.sh_degree+1)**2-1)
    sh=torch.cat((dc,rest),-1)
    return eval_sh(model.cfg.sh_degree,sh[:,None],directions[None])+.5


@torch.no_grad()
def diagnose_geometry(model,scene,out,*,blocks=4,trials=8,color_views=8,relative_step=.01,seed=42,snr=10.):
    """Probe one point per fixed original block; never regroup sampled points.

    A Gram matrix is measured at clean teacher y. Conditional error moments are
    measured at the SAME anchor point, rather than multiplying unrelated global
    averages. Cross-point error covariances are not included in that attribution.
    """
    from .codec import channel,prefix_mask
    from .modular_experiment import json_write
    from .optimization import preserved_rng
    from .position_delivery import delivered_positions
    from .transport import model_id
    if (type(blocks) is not int or blocks<1 or type(trials) is not int or trials<2 or
            type(color_views) is not int or color_views<1 or not math.isfinite(relative_step) or relative_step<=0 or
            not math.isfinite(snr)):
        raise ValueError('positive block/view counts, >=2 trials, positive probe step and finite SNR required')
    population=[(g,b) for g,ids in enumerate(scene.group_ids) for b,row in enumerate(ids) if (row>=0).any()]
    if not population:raise ValueError('geometry diagnostic requires active scene points')
    local=torch.Generator().manual_seed(seed)
    selected=[population[i] for i in torch.randperm(len(population),generator=local)[:blocks].tolist()]
    mode=model.training;identity=model_id(model);device=next(model.parameters()).device
    anchors=[];csv_rows=[]
    try:
        with preserved_rng(device):
            model.eval();model.set_scene_geometry(scene.geometry)
            for number,(group,block) in enumerate(selected):
                f=scene.groups[group][block:block+1].to(device)
                ids=scene.group_ids[group][block:block+1].to(device);active=ids>=0
                valid=active[0].nonzero().flatten()
                point=int(valid[torch.randint(len(valid),(1,),generator=local)].item())
                y=model.semantic_encoder(f,f[...,:3],active)
                xyz=delivered_positions(f[...,:3],active.long(),model.cfg)
                directions=torch.randn(color_views,3,generator=local).to(device=device,dtype=y.dtype)
                directions=torch.nn.functional.normalize(directions,dim=-1)
                def response(vector):
                    changed=y.clone();changed[0,point]=vector
                    return decoded_colors(model,changed,active,xyz,directions)
                clean=response(y[0,point])
                epsilon=relative_step*max(float(y[0,point].square().mean().sqrt()),1e-3)
                gram=finite_difference_gram(response,y[0,point],epsilon)
                half,jacobian=finite_difference_gram(response,y[0,point],epsilon/2,return_jacobian=True)
                rgb_jacobian=jacobian.reshape(-1,3,y.shape[-1])
                gram_rgb={color:rgb_jacobian[:,c].T@rgb_jacobian[:,c]/len(rgb_jacobian) for c,color in enumerate('rgb')}
                discrepancy=float((gram-half).norm()/half.norm().clamp_min(1e-12))
                record=dict(group=group,block=block,point=point,source_id=int(ids[0,point]),
                    active_points=int(active.sum()),epsilon=epsilon,gram=half.tolist(),
                    gram_rgb={c:v.tolist() for c,v in gram_rgb.items()},
                    gram_halfstep_relative_error=discrepancy,layouts=[])
                z=model.channel_encoder(y,active,snr)
                for layout in ('1','2','3','mixed'):
                    q=(torch.randint(1,4,active.shape,generator=torch.Generator().manual_seed(seed+900001*group+1009*block)).to(device)
                       if layout=='mixed' else torch.full_like(ids,int(layout)))
                    q=q.masked_fill(~active,0);mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
                    errors=[];isolated=[];full=[];isolated_rgb=[];full_bias_rgb=[]
                    finite=None
                    for trial in range(-1,trials):
                        generator=torch.Generator(device=device).manual_seed(seed+170003*group+17011*block+max(trial,0))
                        rx=channel(z.reshape(-1,2),snr,'none' if trial<0 else 'awgn',generator).reshape_as(z)
                        options=({'delivered_xyz':xyz} if model.cfg.channel_denoiser_mode!='none' or model.cfg.channel_neighbor_mode!='none' else {})
                        recovered=model.channel_decoder(rx,mask,q,snr,**options)
                        error=(recovered[0,point]-y[0,point]).double().cpu()
                        if trial<0:
                            finite=error
                            finite_color=float((response(recovered[0,point])-clean).double().square().mean())
                            continue
                        errors.append(error)
                        difference=(response(recovered[0,point])-clean).double()
                        isolated.append(float(difference.square().mean()))
                        isolated_rgb.append(difference.square().mean((0,1)).cpu())
                        full_difference=(decoded_colors(model,recovered,active,xyz,directions)-clean).double()
                        full.append(float(full_difference.square().mean()))
                        full_bias_rgb.append(full_difference.mean((0,1)).cpu())
                    errors=torch.stack(errors);mean=errors.mean(0);random=errors-mean
                    second=errors.T@errors/trials;cov=random.T@random/trials
                    risks=direction_risks(second,half)
                    row=dict(layout=layout,anchor_tier=int(q[0,point]),error_second_moment=second.tolist(),
                        noise_covariance=cov.tolist(),point_mean_error=mean.tolist(),
                        noiseless_error=finite.tolist(),latent_point_mse=float(second.diag().mean()),
                        linearized_noiseless_risk=float(finite@half@finite),measured_noiseless_isolated_color_mse=finite_color,
                        linearized_mean_risk=float(mean@half@mean),linearized_noise_risk=float(torch.trace(half@cov)),
                        linearized_risk_rgb={c:float(torch.trace(g@second)) for c,g in gram_rgb.items()},
                        measured_isolated_mse_rgb=dict(zip('rgb',torch.stack(isolated_rgb).mean(0).tolist())),
                        measured_full_block_bias_rgb=dict(zip('rgb',torch.stack(full_bias_rgb).mean(0).tolist())),
                        measured_isolated_color_mse=sum(isolated)/trials,
                        measured_full_block_color_mse=sum(full)/trials,directions=risks)
                    record['layouts'].append(row)
                    cumulative=0.
                    for rank,direction in enumerate(risks['directions'],1):
                        cumulative+=direction['risk']
                        csv_rows.append(dict(anchor=number,layout=layout,anchor_tier=row['anchor_tier'],rank=rank,
                            error_energy=direction['error_energy'],sensitivity=direction['sensitivity'],risk=direction['risk'],
                            cumulative_fraction=cumulative/max(risks['total_linearized_risk'],1e-30)))
                anchors.append(record)
                print(f'error geometry {number+1}/{len(selected)}: block={group}/{block}, point={point}, '
                      f'half-step Gram difference={discrepancy:.3g}',flush=True)
    finally:model.train(mode)
    if model_id(model)!=identity:raise RuntimeError('geometry diagnostic changed model state')
    result=dict(version=1,anchors=anchors,trials=trials,color_views=color_views,relative_step=relative_step,
        actual_snr=snr,seed=seed,parameters_unchanged=True,
        target='unclipped directional RGB from frozen semantic decoder(clean teacher y), not original source attributes',
        scope='one anchor per sampled original complete block; all active output points measured; independent private fixed-slot noise',
        limitations=['local finite differences; inspect half-step stability and measured nonlinear response',
            'anchor-wise conditional moments are not full cross-point scene error covariance',
            'finite-trial means include sampling noise; anchor set is diagnostic, not an independent held-out training replication',
            'read-only diagnostic; no automatic loss weighting or transmitter adaptation'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    json_write(out/'error_geometry.json',result)
    with (out/'direction_risks.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    json_write(out/'complete.json',dict(status='complete',anchors=len(anchors),parameters_unchanged=True))
    return result
