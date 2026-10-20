"""Read-only paired channel audit on original complete Morton blocks.

No training, model surgery, latent swapping or CUDA rasterizer is required.
Each checkpoint uses its OWN semantic coordinate system on the same Gaussians.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .codec import channel, prefix_mask
from .data import Geometry, morton_order, read_ply, to_features
from .latent_error_diagnostics import error_moments
from .modular_experiment import json_write
from .optimization import preserved_rng
from .position_delivery import delivered_positions
from .transport import load_checkpoint, model_id


def distribution(values):
    values=values.detach().double().cpu().flatten()
    if not values.numel() or not torch.isfinite(values).all():
        raise ValueError('nonempty finite diagnostic values required')
    return dict(mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()),
                **{f'p{int(p*100):02d}':float(values.quantile(p)) for p in (.01,.05,.5,.95,.99)})


def power_report(raw,signal,rates,epsilon):
    if raw.shape!=signal.shape or raw.shape[-1]!=2*rates[-1] or epsilon<=0:
        raise ValueError('matching raw/normalized signals and positive power floor required')
    rows=[]
    for start,end in zip(rates[:-1],rates[1:]):
        energy=raw[...,2*start:2*end].double().square().sum(-1)/(end-start)
        post=signal[...,2*start:2*end].double().square().sum(-1)/(end-start)
        rows.append(dict(start=start,end=end,raw_power=distribution(energy),
            normalized_power=distribution(post),radial_log_gain=distribution(epsilon/(energy+epsilon)),
            power_ceiling_fraction_over_099=float((post>.99).double().mean())))
    return rows


def error_tail(errors,tail_fraction=.01):
    if (errors.ndim!=3 or not errors.numel() or not torch.isfinite(errors).all()
            or not 0<tail_fraction<=1):raise ValueError('finite [trials,points,dimensions] errors required')
    point_mse=errors.detach().double().square().mean((0,2)).cpu()
    count=max(1,math.ceil(len(point_mse)*tail_fraction))
    worst=point_mse.topk(count)
    return dict(point_mean_mse=distribution(point_mse),tail_fraction=tail_fraction,
                top_fraction_error_share=float(worst.values.sum()/point_mse.sum().clamp_min(1e-30)),
                worst_point_indices=worst.indices.tolist())


def select_blocks(raw,geometry,block_size,blocks,seed):
    """Sample original blocks, never regroup sampled output points into new blocks."""
    if type(blocks) is not int or blocks<1 or type(block_size) is not int or block_size<1:
        raise ValueError('positive integer block count/size required')
    order=torch.from_numpy(morton_order(geometry.quantize(raw[:,:3]).numpy()).copy()).long()
    count=math.ceil(len(raw)/block_size)
    chosen=torch.randperm(count,generator=torch.Generator().manual_seed(seed))[:blocks].sort().values
    return [dict(block_index=int(i),source_ids=order[i*block_size:(i+1)*block_size],
                 raw=raw[order[i*block_size:(i+1)*block_size]]) for i in chosen.tolist()]


@torch.no_grad()
def pointwise_collapse_error(module,args,output):
    """Numerically check the self-only attention's algebraic Q/K redundancy."""
    if module.mode!='pointwise':raise ValueError('collapse probe requires a pointwise denoiser')
    previous,received,receive_mask,xyz,active,snr=args
    stage_mask=receive_mask.bool().clone();stage_mask[...,2*module.symbols:]=False
    stage_mask &= active[...,None]
    signal=received.masked_fill(~stage_mask,0)
    inputs=torch.cat((previous.masked_fill(~active[...,None],0),signal,stage_mask.to(signal),
                      signal.new_full((*active.shape,1),float(snr)/20)),-1)
    h=module.input(inputs).masked_fill(~active[...,None],0)
    # Repeated keys/values all belong to the SAME point; the weights sum to 1.
    context=(module.value(h)+module.relative(h.new_zeros(*active.shape,4))).masked_fill(~active[...,None],0)
    simple=module.readout(module.fuse(torch.cat((h,context),-1))).masked_fill(~active[...,None],0)
    return float((simple-output).abs().max())


@torch.no_grad()
def trace_channel(decoder,received,mask,q,snr,xyz,*,probe_pointwise=False):
    """Observe an ordinary forward with hooks; do not replace its computation.

    before/value/correction retain the FULL original block. active denotes the
    points actually updated by that node (q>=stage), not all output points.
    """
    if not hasattr(decoder,'stages'):raise ValueError('audit requires progressive channel decoder')
    rows={};hooks=[]
    state={'value':received.new_zeros(*q.shape,decoder.cfg.latent_dim)}
    def stage_hook(tier):
        def record(module,args,output):
            active=q>=tier;before=state['value'];after=before.clone()
            if not torch.allclose(args[1],before[active],rtol=1e-5,atol=1e-7):
                raise RuntimeError('stage trace diverged from actual progressive input')
            after[active]+=output
            rows[f'stage{tier}']=dict(before=before,value=after,correction=after-before,active=active)
            state['value']=after
        return record
    def residual_hook(name):
        def record(module,args,output):
            before=state['value']
            if not torch.equal(args[0],before):raise RuntimeError('residual trace input diverged from actual forward')
            after=before+output
            rows[name]=dict(before=before,value=after,correction=output,active=args[4])
            if probe_pointwise and module.mode=='pointwise':
                rows[name]['pointwise_collapse_max_abs']=pointwise_collapse_error(module,args,output)
            state['value']=after
        return record
    try:
        for tier,stage in enumerate(decoder.stages,1):
            hooks.append(stage.register_forward_hook(stage_hook(tier)))
        if hasattr(decoder,'denoisers'):
            for tier,denoiser in enumerate(decoder.denoisers,1):
                name=f'stage{tier}_denoiser' if decoder.cfg.channel_denoiser_scope=='stage' else 'final_denoiser'
                hooks.append(denoiser.register_forward_hook(residual_hook(name)))
        if hasattr(decoder,'neighbor_refiner'):
            hooks.append(decoder.neighbor_refiner.register_forward_hook(residual_hook('neighbor')))
        result=decoder(received,mask,q,snr,delivered_xyz=xyz)
        if not torch.equal(result,state['value']):raise RuntimeError('trace does not match official decoder output')
        # Reuse the numerically verified last state for response-cache identity.
        previous=state['value']
        rows['final']=dict(before=previous,value=previous,correction=torch.zeros_like(result),active=q>0)
        return rows
    finally:
        for hook in hooks:hook.remove()


def physical_response(model,latent,active,xyz,directions):
    """Unclipped SH RGB plus rendered-parameter opacity and covariance.

    Log covariance is rotation/sign invariant and avoids tiny-scale division.
    Groups are reported separately: this is NOT another weighted training loss.
    """
    from .losses import rotation_matrix
    from utils.sh_utils import eval_sh
    attrs=model.semantic_decoder(latent,active,xyz)[active]*model.attr_std+model.attr_mean
    return attribute_response(attrs,model.cfg.sh_degree,directions,rotation_matrix,eval_sh)


def attribute_response(attrs,degree,directions,rotation_matrix=None,eval_sh=None):
    if rotation_matrix is None:
        from .losses import rotation_matrix
    if eval_sh is None:
        from utils.sh_utils import eval_sh
    count=len(attrs)
    coefficients=torch.cat((attrs[:,8:11,None],attrs[:,11:].reshape(count,3,(degree+1)**2-1)),-1)
    rgb=eval_sh(degree,coefficients[:,None],directions[None])+.5
    rotation=attrs[:,4:8]
    identity=torch.zeros_like(rotation);identity[:,0]=1
    rotation=torch.where(rotation.norm(dim=-1,keepdim=True)>1e-8,F.normalize(rotation,dim=-1),identity)
    matrix=rotation_matrix(rotation)
    logs=attrs[:,1:4].clamp(-20,10)
    covariance=(matrix*logs.mul(2).exp()[:,None])@matrix.transpose(-1,-2)
    log_covariance=(matrix*logs.mul(2)[:,None])@matrix.transpose(-1,-2)
    return dict(rgb=rgb,opacity=attrs[:,:1].clamp(-20,20).sigmoid(),
                covariance=covariance,log_covariance=log_covariance,log_scale=logs)


def response_metrics(prediction,target):
    delta=(prediction['rgb']-target['rgb']).double()
    covariance=(prediction['covariance']-target['covariance']).double()
    result=dict(rgb_mse=float(delta.square().mean()),
        rg_mse=float((delta[...,0]-delta[...,1]).square().mean()),
        gb_mse=float((delta[...,1]-delta[...,2]).square().mean()),
        relative_covariance_mse=float((covariance.square().sum((-1,-2))/
                                      target['covariance'].double().square().sum((-1,-2)).clamp_min(1e-30)).mean()))
    for j,c in enumerate('rgb'):
        result[c+'_bias']=float(delta[...,j].mean());result[c+'_mse']=float(delta[...,j].square().mean())
    for key in ('opacity','log_scale','log_covariance'):
        result[key+'_mse']=float((prediction[key]-target[key]).double().square().mean())
    return result


def subset_response(response,active):
    return {key:value[active] for key,value in response.items()}


def accumulate(store,key,metrics,weight):
    row=store.setdefault(key,dict(weight=0.,sums={}))
    row['weight']+=weight
    for name,value in metrics.items():row['sums'][name]=row['sums'].get(name,0.)+float(value)*weight


def averaged(store):
    return {key:{name:value/row['weight'] for name,value in row['sums'].items()} for key,row in store.items()}


def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


@torch.no_grad()
def audit_model(model,blocks,geometry,out,*,trials=32,snr=10.,seed=42,response_blocks=4,
                directions=8,probe_nmse=.02):
    if (not blocks or trials<2 or response_blocks<1 or directions<1 or
            not math.isfinite(snr) or not math.isfinite(probe_nmse) or probe_nmse<=0):
        raise ValueError('nonempty blocks, >=2 trials and positive response/probe settings required')
    device=next(model.parameters()).device;identity=model_id(model)
    condition_snr=model.resolve_condition_snr(snr)
    modes={module:module.training for module in model.modules()}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source_ids=torch.cat([b['source_ids'] for b in blocks])
    digest=hashlib.sha256(source_ids.numpy().tobytes()).hexdigest()
    records=[];stage_rows=[];dimension_rows=[];raw_signals=[];signals=[];pointwise_checks={}
    try:
        with preserved_rng(device):
            model.eval();model.set_scene_geometry(geometry)
            sphere=torch.randn(directions,3,generator=torch.Generator().manual_seed(seed+71))
            sphere=F.normalize(sphere.to(device),dim=-1)
            # Cache complete original semantic blocks, not sampled/regrouped y.
            cache=[]
            for block in blocks:
                f,_=to_features(block['raw'].to(device),geometry,model);f=f[None]
                active=torch.ones(f.shape[:2],device=device,dtype=torch.bool)
                y=model.semantic_encoder(f,f[...,:3],active)
                xyz=delivered_positions(f[...,:3],active.long(),model.cfg)
                captured={}
                hook=model.channel_encoder.readout.register_forward_hook(lambda m,a,o:captured.update(raw=o))
                try:z=model.channel_encoder(y,active,condition_snr)
                finally:hook.remove()
                raw_signals.append(captured['raw'][active].cpu());signals.append(z[active].cpu())
                cache.append(dict(y=y.cpu(),z=z.cpu(),xyz=xyz.cpu(),raw=block['raw'],block_index=block['block_index']))
            for layout in ('1','2','3','mixed'):
                targets=[];finite=[];noisy=[[] for _ in range(trials)]
                node_store={};response_store={};probe_store={};tier_ids=[]
                for number,data in enumerate(cache):
                    y,z,xyz=(data[k].to(device) for k in ('y','z','xyz'))
                    active=torch.ones(y.shape[:2],device=device,dtype=torch.bool)
                    block_seed=seed+100003*data['block_index']
                    if layout=='mixed':
                        q=torch.randint(1,4,active.shape,generator=torch.Generator().manual_seed(block_seed+991)).to(device)
                    else:q=torch.full(active.shape,int(layout),device=device,dtype=torch.long)
                    mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
                    targets.append(y[active].cpu());tier_ids.append(q[active].cpu())
                    measure=number<response_blocks
                    if measure:
                        clean=physical_response(model,y,active,xyz,sphere)
                        source=attribute_response(data['raw'][:,3:].to(device),model.cfg.sh_degree,sphere)
                        accumulate(response_store,'clean_vs_source',response_metrics(clean,source),int(active.sum()))
                    for trial in range(-1,trials):
                        generator=torch.Generator(device=device).manual_seed(block_seed+17011+max(trial,0))
                        rx=channel(z.reshape(-1,2),snr,'none' if trial<0 else 'awgn',generator).reshape_as(z)
                        trace=trace_channel(model.channel_decoder,rx,mask,q,condition_snr,xyz,
                                            probe_pointwise=number==0 and trial==-1)
                        response_cache={}
                        def node_response(value):
                            key=id(value)
                            if key not in response_cache:
                                response_cache[key]=physical_response(model,value,active,xyz,sphere)
                            return response_cache[key]
                        recovered=trace['final']['value'];error=recovered-y
                        if trial<0:finite.append(recovered[active].cpu())
                        else:noisy[trial].append(recovered[active].cpu())
                        condition='noiseless' if trial<0 else 'awgn'
                        for name,node in trace.items():
                            if 'pointwise_collapse_max_abs' in node:
                                pointwise_checks[f'{layout}/{name}']=node['pointwise_collapse_max_abs']
                            selected=node['active'];n=int(selected.sum())
                            if not n:continue
                            before=(node['before'][selected]-y[selected]).double().square().mean()
                            after=(node['value'][selected]-y[selected]).double().square().mean()
                            metrics=dict(before_mse=float(before),after_mse=float(after),delta_mse=float(after-before),
                                target_power=float(y[selected].double().square().mean()),
                                correction_power=float(node['correction'][selected].double().square().mean()))
                            if measure:
                                # Decode with the full original block. Only then
                                # restrict metrics to this node's updated points.
                                response=node_response(node['value'])
                                own=selected[active]
                                metrics.update({'response_'+k:v for k,v in response_metrics(
                                    subset_response(response,own),subset_response(clean,own)).items()})
                                # Separate denominator: physical response only
                                # covers the declared response block subset.
                                physical={k:v for k,v in metrics.items() if k.startswith('response_')}
                                before_response=node_response(node['before'])
                                before_physical=response_metrics(subset_response(before_response,own),subset_response(clean,own))
                                for key,value in before_physical.items():
                                    physical['response_before_'+key]=value
                                    physical['response_delta_'+key]=physical['response_'+key]-value
                                accumulate(node_store,(condition,name,'response'),physical,n)
                                metrics={k:v for k,v in metrics.items() if not k.startswith('response_')}
                                if name=='final':
                                    accumulate(response_store,condition+'_vs_clean',response_metrics(response,clean),n)
                                    accumulate(response_store,condition+'_vs_source',response_metrics(response,source),n)
                            accumulate(node_store,(condition,name,'latent'),metrics,n)
                        if measure and trial>=0:
                            desired=(y.double().square().sum()*probe_nmse).sqrt().to(y)
                            random=torch.randn(y.shape,generator=torch.Generator(device=device).manual_seed(block_seed+880001+trial),device=device)
                            for key,direction in (('actual_direction_equal_nmse',error),('isotropic_equal_nmse',random)):
                                norm=direction.double().norm()
                                if norm<=1e-30:continue
                                changed=y+direction*(desired/norm.to(y))
                                response=physical_response(model,changed,active,xyz,sphere)
                                metrics=response_metrics(response,clean)
                                metrics['measured_nmse']=float((changed-y).double().square().sum()/y.double().square().sum().clamp_min(1e-30))
                                accumulate(probe_store,key,metrics,int(active.sum()))
                    print(f'audit q{layout}: original block {number+1}/{len(cache)}, {trials} noise trials',flush=True)
                target=torch.cat(targets);finite=torch.cat(finite);noisy=torch.stack([torch.cat(v) for v in noisy])
                moments=error_moments(target,finite,noisy)
                finite_error=(finite-target).double()
                mean_shift=noisy.double().mean(0)-finite.double()
                cross=float(2*(finite_error*mean_shift).mean())
                moments['summary']['noiseless_mean_shift_cross_term']=cross
                moments['summary']['noiseless_shift_variance_identity_residual']=(
                    moments['summary']['noisy_mse']-moments['summary']['noiseless_mse']-
                    moments['summary']['noise_induced_mean_shift_mse']-moments['summary']['noise_variance']-cross)
                tail=error_tail(noisy-target)
                tail['worst_source_ids']=source_ids[tail.pop('worst_point_indices')].tolist()
                tier_ids=torch.cat(tier_ids)
                by_tier=[]
                for tier in (1,2,3):
                    use=tier_ids==tier
                    if use.any():
                        mse=(noisy[:,use]-target[use]).double().square().mean()
                        power=target[use].double().square().mean()
                        by_tier.append(dict(tier=tier,points=int(use.sum()),mse=float(mse),nmse=float(mse/power.clamp_min(1e-12))))
                nodes=averaged(node_store)
                names=list(dict.fromkeys((condition,name) for condition,name,_ in nodes))
                for condition,name in names:
                    metrics=nodes[(condition,name,'latent')]
                    row=dict(layout=layout,path=condition,node=name,
                        latent_point_observations=int(node_store[(condition,name,'latent')]['weight']),**metrics)
                    row['after_nmse']=row['after_mse']/max(row['target_power'],1e-12)
                    physical=nodes.get((condition,name,'response'),{})
                    row['response_point_observations']=int(node_store.get((condition,name,'response'),{}).get('weight',0))
                    for prefix in ('response_','response_before_','response_delta_'):
                        row.update({prefix+key:physical.get(prefix+key) for key in (
                            'rgb_mse','rg_mse','gb_mse','opacity_mse','log_scale_mse',
                            'log_covariance_mse','relative_covariance_mse')})
                    stage_rows.append(row)
                semantic=averaged(response_store)
                semantic['noisy_vs_clean']=semantic.pop('awgn_vs_clean')
                semantic['noisy_vs_source']=semantic.pop('awgn_vs_source')
                records.append(dict(layout=layout,moments=moments,tail=tail,by_deployed_tier=by_tier,
                    semantic_response=semantic,sensitivity=averaged(probe_store)))
                dimension_rows.extend(dict(layout=layout,**r) for r in moments['dimensions'])
    finally:
        for module,mode in modes.items():module.training=mode
    if identity!=model_id(model):raise RuntimeError('audit changed model weights/configuration')
    result=dict(version=1,optimizer_updates=0,parameters_unchanged=True,model_id=identity,
        sample_sha256=digest,sample_points=len(source_ids),source_ids=source_ids.tolist(),
        block_indices=[b['block_index'] for b in blocks],trials=trials,snr=snr,seed=seed,
        tx_condition_snr=condition_snr,rx_condition_snr=condition_snr,training_snrs=list(model.training_snrs),
        response_blocks=min(response_blocks,len(blocks)),directions=directions,probe_nmse=probe_nmse,
        pointwise_attention=dict(max_abs_difference_from_no_qk_mapping=pointwise_checks,
            interpretation='Repeated self keys make Q/K scores irrelevant in pointwise mode. Numerical equivalence is NOT proof of quality benefit/harm.'),
        power=power_report(torch.cat(raw_signals),torch.cat(signals),model.cfg.rates,model.cfg.power_floor),
        layouts=records,
        definitions=dict(target='each checkpoint own clean semantic y; no cross-model latent swap',
            noise='independent real components, variance 10**(-snr/10)/2; paired full slot draws before prefix masking',
            variance='finite-trial divisor T: noisy MSE = point-mean-error MSE + within-point variance',
            stages='actual-forward observations; negative delta_mse means this residual reduces its input error, conditionally',
            responses='unclipped directional SH RGB, physical opacity and covariance; not rendered image PSNR/SSIM/LPIPS',
            sensitivity='same per-block relative NMSE, actual channel error direction vs paired isotropic direction'),
        limitations=['No optimizer updates or retraining; module trace is conditional contribution, not a causal removal experiment.',
            'Sampled original blocks, not full-scene rendered quality or independent training replications.',
            'Physical response metrics cover only the specified response block subset; latent metrics cover all sampled blocks.',
            'Finite-trial mean error contains sampling noise and is not proven population bias.',
            'Latent dimensions are model-specific; never label a dimension red or compare same index across different teachers.',
            'Mixed uses fixed uniform positive tiers, not learned allocation or q0 pruning.',
            'Power normalization with positive floor is radially invertible but can be ill-conditioned; not strict amplitude erasure.'])
    json_write(out/'audit.json',result)
    write_csv(out/'stage_metrics.csv',stage_rows);write_csv(out/'latent_dimensions.csv',dimension_rows)
    json_write(out/'complete.json',dict(status='complete',optimizer_updates=0,parameters_unchanged=True))
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='current modular run with training.json')
    p.add_argument('--reference-run',required=True,help='older modular run; keep its OWN semantic weights')
    p.add_argument('--checkpoint',help='current checkpoint override; default checkpoints/final/codec.pt')
    p.add_argument('--reference-checkpoint',help='older checkpoint override; default checkpoints/final/codec.pt')
    p.add_argument('--manifest',help='override Truck PLY location without changing source identity')
    p.add_argument('--out',required=True,help='new independent directory, not inside either source run')
    p.add_argument('--device',default='cuda')
    p.add_argument('--blocks',type=int,default=32)
    p.add_argument('--response-blocks',type=int,default=4)
    p.add_argument('--trials',type=int,default=32)
    p.add_argument('--directions',type=int,default=8)
    p.add_argument('--probe-nmse',type=float,default=.02)
    p.add_argument('--snr',type=float,default=10.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--dry-run',action='store_true',help='validate source metadata/checkpoints without loading PLY or writing output')
    return p


def build_plan(args):
    if (min(args.blocks,args.response_blocks,args.directions,args.cpu_threads)<1 or args.trials<2 or
            not math.isfinite(args.snr) or not math.isfinite(args.probe_nmse) or args.probe_nmse<=0):
        raise ValueError('invalid audit budget/SNR')
    out=Path(args.out).resolve();sources={}
    for label,run_arg,checkpoint_arg in (('current',args.run,args.checkpoint),
                                        ('reference',args.reference_run,args.reference_checkpoint)):
        run=Path(run_arg).resolve()
        checkpoint=Path(checkpoint_arg).resolve() if checkpoint_arg else run/'checkpoints/final/codec.pt'
        if out.is_relative_to(run) or run.is_relative_to(out) or checkpoint.is_relative_to(out):
            raise ValueError('audit output must be separate from source runs/checkpoints')
        meta=json.loads((run/'training.json').read_text(encoding='utf-8'))
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg=saved['config']
        if (cfg.get('architecture')!='modular_jscc_v1' or cfg.get('channel_variant')!='progressive' or
                len(cfg['rates'])!=4 or json.loads(json.dumps(cfg))!=meta['codec_config']):
            raise ValueError('requires a matching modular three-tier progressive checkpoint/config')
        from .modular_checkpoint import validate_scene
        validate_scene(saved['training']['scene_record'],meta['scene'])
        sources[label]=dict(run=str(run),checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            training_progress=saved.get('training',{}).get('progress',{}),
            step=int(saved['step']),scene=meta['scene'],config=cfg,manifest=str(args.manifest or meta['manifest']))
    left,right=(sources[k] for k in ('current','reference'))
    if left['scene']!=right['scene']:raise ValueError('source scene/settings/split differ; refusing unpaired comparison')
    for key in ('block_size','morton_bits','rates','sh_degree','position_bits','position_delivery'):
        if left['config'][key]!=right['config'][key]:raise ValueError('unmatched '+key)
    return dict(version=1,sources=sources,settings={k:getattr(args,k) for k in (
        'blocks','response_blocks','trials','directions','probe_nmse','snr','seed')},optimizer_updates=0)


def compare(reports,out):
    hashes={r['sample_sha256'] for r in reports.values()}
    if len(hashes)!=1:raise ValueError('paired Gaussian samples differ')
    rows=[]
    for label,report in reports.items():
        for r in report['layouts']:
            s=r['moments']['summary'];p=r['semantic_response']['noisy_vs_clean']
            rows.append(dict(model=label,layout=r['layout'],noiseless_nmse=s['noiseless_nmse'],
                noisy_nmse=s['noisy_nmse'],point_mean_error_mse=s['pointwise_mean_error_mse'],
                noise_variance=s['noise_variance'],worst_1pct_error_share=r['tail']['top_fraction_error_share'],
                rgb_mse=p['rgb_mse'],rg_mse=p['rg_mse'],gb_mse=p['gb_mse'],
                log_covariance_mse=p['log_covariance_mse'],opacity_mse=p['opacity_mse']))
    write_csv(Path(out)/'comparison.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key in zip(axes,('noisy_nmse','rgb_mse','log_covariance_mse')):
        for label in reports:
            subset=[r for r in rows if r['model']==label]
            ax.plot([r['layout'] for r in subset],[r[key] for r in subset],marker='o',label=label)
        ax.set(title=key,xlabel='Tier (mixed: uniform positive tiers)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(Path(out)/'comparison.png',dpi=150);plt.close(fig)
    result=dict(status='complete',rows=rows,sample_sha256=next(iter(hashes)),optimizer_updates=0,
        interpretation='Compare normalized latent error AND physical response; own teachers differ. No rendered PSNR claim.')
    json_write(Path(out)/'summary.json',result)
    return result


def main(args):
    plan=build_plan(args)
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError('use a NEW audit output directory')
    torch.set_num_threads(args.cpu_threads)
    from .allocation import scene_fingerprint
    from .multiscene import manifest
    current=plan['sources']['current']
    specs=[s for s in manifest(current['manifest']) if s['name']=='truck' and s['role']=='train']
    if len(specs)!=1:raise ValueError('manifest must identify exactly one training Truck')
    raw,degree=read_ply(specs[0]['ply']);record=current['scene'];cfg=current['config']
    geometry=Geometry.fit(raw[:,:3],cfg['morton_bits'])
    if (scene_fingerprint(raw)!=record['fingerprint'] or len(raw)!=record['source_gaussians'] or
            degree!=cfg['sh_degree'] or geometry.to_dict()!=record['geometry']):
        raise ValueError('source PLY/fingerprint/geometry differs; do not substitute unpruned PLY')
    blocks=select_blocks(raw,geometry,cfg['block_size'],args.blocks,args.seed);del raw
    json_write(out/'plan.json',plan)
    reports={}
    from .cli import device_for
    device=device_for(args.device)
    for label,source in plan['sources'].items():
        print(f'Audit {label}: checkpoint step={source["step"]}; NO training or rasterizer',flush=True)
        with preserved_rng(device):model=load_checkpoint(source['checkpoint'],device)
        reports[label]=audit_model(model,blocks,geometry,out/label,**{k:getattr(args,k) for k in (
            'trials','snr','seed','response_blocks','directions','probe_nmse')})
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    result=compare(reports,out)
    json_write(out/'complete.json',dict(status='complete',optimizer_updates=0,parameters_unchanged=True))
    print(f'Complete: {out}/summary.json',flush=True)
    return result


if __name__=='__main__':main(parser().parse_args())
