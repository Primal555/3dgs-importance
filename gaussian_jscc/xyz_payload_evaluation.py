"""Clean vs finite-noiseless vs AWGN; explicit XYZ/attribute diagnostics."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import to_raw
from .optimization import preserved_rng
from .local_response import directional_color_loss
from .render_objective import image_metrics
from .render_validation import append_json,save_panel
from .xyz_payload_training import payload_loss,diagnostics
from .xyz_payload_transport import rate_record
from .transport import model_id


def position_statistics(target,pred,block_sizes):
    error=pred.detach().cpu().double()-target.detach().cpu().double()
    if error.ndim!=2 or error.shape[1]!=3 or not len(error) or sum(block_sizes)!=len(error):
        raise ValueError('matching nonempty coordinates and complete block sizes required')
    if not torch.isfinite(error).all(): raise ValueError('nonfinite position error')
    mean=error.mean(0); distance=error.norm(dim=-1)
    residual=0.; offset=0
    for count in block_sizes:
        if count:
            block=error[offset:offset+count]; residual+=float((block-block.mean(0)).square().sum())
        offset+=count
    return dict(world_coordinate_mse=float(error.square().mean()),
                world_distance_rmse=float(distance.square().mean().sqrt()),
                world_axis_rmse=error.square().mean(0).sqrt().tolist(),
                world_mean_error_xyz=mean.tolist(),world_bias_norm=float(mean.norm()),
                world_distance_p50=float(distance.quantile(.5)),world_distance_p95=float(distance.quantile(.95)),
                world_distance_max=float(distance.max()),
                world_translation_removed_rmse=float((error-mean).square().sum(-1).mean().sqrt()),
                world_block_translation_removed_rmse=(residual/len(error))**.5)


@torch.no_grad()
def decode_path(model,scene,qs,path,snr):
    model.set_scene_geometry(scene.geometry)
    device=next(model.parameters()).device
    rows=[]; statistics={}; sizes=[]; target_rows=[]; attr_sums={}; color_sums={}; count=0
    for features,q in zip(scene.groups,qs):
        f=features.to(device); q=q.to(device); active=q>0
        if path=='clean':
            y=model.encode_representation(f,f[...,:3],active); yh=y
        else:
            y,yh,_=model.forward_channel_batches(f,f[...,:3],F.one_hot(q,4).to(f),snr,
                                                'awgn' if path=='awgn' else 'none',paired_noise=True)
        pred=model.decode_representation(yh,q)
        if not active.any(): continue
        raw=to_raw(pred[active],scene.geometry,model)
        if not torch.isfinite(raw).all(): raise RuntimeError('nonfinite decoded Gaussian')
        rows.append(raw); target_rows.append(to_raw(f[active],scene.geometry,model)[:,:3].cpu())
        sizes.extend(active.sum(-1).cpu().tolist())
        n=int(active.sum()); count+=n
        _,metrics=payload_loss(y,yh,q)
        for key,value in metrics.items(): statistics[key]=statistics.get(key,0.)+float(value.detach())*n
        delta=(pred[active]-f[active])[:,3:].square()
        for name,a,b in (('opacity',0,1),('scale',1,4),('rotation',4,8),('dc',8,11),('sh',11,model.cfg.attr_dim)):
            if b>a: attr_sums[name]=attr_sums.get(name,0.)+float(delta[:,a:b].sum())/(b-a)
        dirs=torch.cat((torch.eye(3,device=device),-torch.eye(3,device=device)),0)
        _,colors=directional_color_loss(pred[active],f[active],model,dirs)
        for key in ('directional_color_mse','directional_color_r_bias','directional_color_g_bias','directional_color_b_bias'):
            color_sums[key]=color_sums.get(key,0.)+colors[key]*n
    if not rows: raise ValueError('no retained points')
    raw=torch.cat(rows)
    diag=diagnostics({k:v/count for k,v in statistics.items()})
    diag.update(position_statistics(torch.cat(target_rows),raw[:,:3],sizes),
                attribute_normalized_mse={k:v/count for k,v in attr_sums.items()},
                **{k:v/count for k,v in color_sums.items()})
    return raw,diag


@torch.no_grad()
def evaluate_paths(model,scene,*,step,phase,out,include_channel=True,trials=2,seed=42,
                   save_images=False,lpips_metrics=False):
    from .cli import seed_all
    from .rendering import render
    from .perceptual_metrics import LPIPSEvaluator,image_metric_protocol
    from .appearance_metrics import appearance_metrics
    if not scene.val_cameras or trials<1: raise ValueError('validation views and trials required')
    out=Path(out); (out/'metrics').mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device; was_training=model.training; records=[]
    try:
        with preserved_rng(device):
            seed_all(seed); model.eval()
            perceptual=LPIPSEvaluator(device) if lpips_metrics else None
            identity=model_id(model)
            mixed=[torch.where(ids>=0,torch.randint(1,4,ids.shape),0) for ids in scene.group_ids]
            paths=[('clean','clean',1)]
            if include_channel:
                paths += [(p,q,trials if p=='awgn' else 1) for p in ('finite_noiseless','awgn') for q in ('1','2','3','mixed')]
            for path,layout,ntrials in paths:
                qs=mixed if layout=='mixed' else [torch.where(ids>=0,3 if layout=='clean' else int(layout),0) for ids in scene.group_ids]
                qfull=torch.cat([q[ids>=0] for q,ids in zip(qs,scene.group_ids)])
                diag_rows=[]; view_rows=[]; diagnostic_rows=[]
                for trial in range(ntrials):
                    seed_all(seed+1000+trial)
                    raw,diag=decode_path(model,scene,qs,path,10.)
                    diag_rows.append(diag)
                    original=scene.raw[qfull>0].to(device)
                    for i,camera in enumerate(scene.val_cameras):
                        image=render(raw,camera,scene.degree,scene.white_background)
                        source=scene.reference.get(camera,device); photo=camera.original_image.to(device)
                        metrics={}
                        for prefix,target in (('source',source),('photo',photo)):
                            metrics.update({prefix+'_'+k:v for k,v in image_metrics(image,target).items()})
                            metrics.update({prefix+'_color_'+k:v for k,v in appearance_metrics(image,target).items()})
                            if perceptual is not None: metrics[prefix+'_lpips']=perceptual(image,target)
                        view_rows.append(dict(metrics,trial=trial,view=i,image_name=str(camera.image_name)))
                        representative=i in (0,len(scene.val_cameras)-1)
                        if representative and trial==0:
                            # Teacher replacements are diagnostics on explicitly named
                            # representative views, never deployable reconstruction.
                            center=torch.cat((raw[:,:3],original[:,3:]),-1)
                            attrs=torch.cat((original[:,:3],raw[:,3:]),-1)
                            diagnostic_rows.append(dict(image_name=str(camera.image_name),
                                center_only_source_psnr=image_metrics(render(center,camera,scene.degree,scene.white_background),source)['psnr'],
                                teacher_xyz_source_psnr=image_metrics(render(attrs,camera,scene.degree,scene.white_background),source)['psnr']))
                            if save_images and (path=='awgn' or path=='clean'):
                                save_panel(out/'images'/phase/f'{path}_{layout}_view{i:04d}.png',photo,source,image,f'{path} / {layout}')
                row=dict(step=step,phase=phase,path=path,layout=layout,
                         actual_snr=None if path=='clean' else 10.,tx_condition_snr=None if path=='clean' else 10.,
                         rx_condition_snr=None if path=='clean' else 10.,condition_policy='matched',
                         channel_trained=bool(getattr(scene,'channel_trained',False)),views=len(scene.val_cameras),
                         trials=ntrials,per_view_trial=view_rows,teacher_replacement_diagnostics=diagnostic_rows,
                         image_metric_protocol=image_metric_protocol(lpips_metrics),
                         diagnostic_scope='teacher replacement only on first/last validation views, trial 0; not deployment',
                         mixed_scope='fixed random positive tiers, not a learned allocator')
                for key in diag_rows[0]:
                    first=diag_rows[0][key]
                    if isinstance(first,dict): row[key]={k:float(np.mean([d[key][k] for d in diag_rows])) for k in first}
                    elif isinstance(first,list): row[key]=np.mean([d[key] for d in diag_rows],axis=0).tolist()
                    elif first is None: row[key]=None
                    else: row[key]=float(np.mean([d[key] for d in diag_rows]))
                metric_keys=[k for k in view_rows[0] if k not in ('trial','view','image_name')]
                for key in metric_keys:
                    means=[float(np.mean([r[key] for r in view_rows if r['trial']==t])) for t in range(ntrials)]
                    row[key]=float(np.mean(means)); row[key+'_std_trials']=float(np.std(means))
                if path=='clean':
                    row.update(payload_complex_symbols=None,total_channel_uses=None,xyz_bytes=None,metadata_bytes=None,
                               symbols_per_gaussian=None,rate_scope='unquantized clean payload bypass, not a communication rate')
                else:
                    row.update(rate_record(model,scene.geometry,qfull,10.,'awgn' if path=='awgn' else 'none',
                                           scene.net_bits_per_use,identity))
                records.append(row)
    finally: model.train(was_training)
    for row in records: append_json(out/'metrics'/('test.jsonl' if phase=='test' else 'validation.jsonl'),row)
    print('validation '+str(step)+' '+phase+': '+', '.join(
        f"{r['path']}/q{r['layout']} PSNR={r['source_psnr']:.2f}, XYZ_RMSE={r['world_distance_rmse']:.5g}" for r in records),flush=True)
    return records
