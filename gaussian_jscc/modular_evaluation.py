"""Current clean representation vs finite-noiseless vs noisy communication.

All prefixes receive noise at fixed symbol slots. Mixed is a fixed positive
tier map, NOT a trained allocation. Digital overhead is measured in bytes;
analogue payload is measured in complex channel uses, not float file sizes.
"""
import math
import time
from pathlib import Path
import numpy as np
import torch
from .data import to_raw
from .optimization import preserved_rng
from .position_delivery import encode_positions
from .render_objective import image_metrics
from .render_validation import save_panel,append_json
from .transport import model_id,encode_metadata
from .modular_training import predict

CONDITION_POLICIES=('trained_range','matched','fixed_both','fixed_tx','fixed_rx')


def conditioning_pair(model,actual_snr,policy='trained_range',reference_snr=10.):
    if policy not in CONDITION_POLICIES: raise ValueError('unknown SNR condition policy')
    actual=model.resolve_condition_snr(actual_snr,actual_snr)
    reference=model.resolve_condition_snr(actual_snr,reference_snr)
    if policy=='trained_range':
        supported=model.resolve_condition_snr(actual)
        return supported,supported
    return (reference if policy in ('fixed_both','fixed_tx') else actual,
            reference if policy in ('fixed_both','fixed_rx') else actual)


def rate_record(model,scene,q,snr,kind,identity,tx_snr=None,rx_snr=None):
    if not hasattr(scene,'_modular_xyz_bytes'):
        stream=encode_positions(scene.geometry.normalize(scene.raw[:,:3]),torch.ones(len(scene.raw),dtype=torch.long),model.cfg)
        scene._modular_xyz_bytes=len(stream)
    header={'version':2,'config':model.cfg.to_dict(),'model_id':identity,'count':len(q),'source_count':len(q),
            'geometry':scene.geometry.to_dict(),'snr_db':float(snr),'channel':kind}
    header.update(tx_condition_snr_db=model.resolve_condition_snr(snr,tx_snr),
                  rx_condition_snr_db=model.resolve_condition_snr(snr,rx_snr))
    meta=encode_metadata(header,q.numpy())
    payload=int(torch.tensor(model.cfg.rates)[q].sum())
    efficiency=getattr(scene,'net_bits_per_use',2.)
    if not math.isfinite(efficiency) or efficiency<=0: raise ValueError('invalid digital efficiency assumption')
    xyz_uses=math.ceil(8*scene._modular_xyz_bytes/efficiency); meta_uses=math.ceil(8*len(meta)/efficiency)
    return {'payload_complex_symbols':payload,'symbols_per_gaussian':payload/len(q),
            'xyz_bytes':scene._modular_xyz_bytes,'metadata_bytes':len(meta),'total_channel_uses':payload+xyz_uses+meta_uses,
            'coordinate_channel_uses':xyz_uses,'metadata_channel_uses':meta_uses,'net_bits_per_use_assumption':efficiency,
            'reliability':'XYZ/metadata assumed reliable; no FEC or packet errors simulated',
            'rate_scope':'analogue payload + measured XYZ and metadata framing; shared model excluded'}


@torch.no_grad()
def decode_path(model,scene,qs,path,snr,*,tx_condition_snr=None,rx_condition_snr=None,noise_reference_symbols=None):
    model.set_scene_geometry(scene.geometry)
    device=next(model.parameters()).device
    rows=[]; count=0; attr={n:0. for n in ('opacity','scale','rotation','dc','sh')}
    latent_error=latent_total=latent_second=0.; latent_count=0
    for f,q in zip(scene.groups,qs):
        f,q=f.to(device),q.to(device); active=q>0; values={}
        hooks=[model.semantic_encoder.register_forward_hook(lambda m,a,o:values.update(y=o))]
        if path!='clean': hooks.append(model.channel_decoder.register_forward_hook(lambda m,a,o:values.update(y_hat=o)))
        try:
            pred=predict(model,f,q,snr,'none' if path!='awgn' else 'awgn',
                         'semantic_render' if path=='clean' else 'channel_render',paired_noise=True,
                         tx_condition_snr=tx_condition_snr,rx_condition_snr=rx_condition_snr,
                         noise_reference_symbols=noise_reference_symbols)
        finally:
            for hook in hooks: hook.remove()
        rows.append(to_raw(pred[active],scene.geometry,model))
        target=f[active]
        differences=pred[active][:,3:]-target[:,3:]
        for name,a,b in (('opacity',0,1),('scale',1,4),('rotation',4,8),('dc',8,11),('sh',11,model.cfg.attr_dim)):
            if b>a: attr[name]+=float(differences[:,a:b].double().square().sum())/(b-a)
        count+=len(target)
        y=values['y'][active].double(); yh=values.get('y_hat',values['y'])[active].double()
        latent_total+=float(y.sum()); latent_second+=float(y.square().sum())
        latent_error+=float((yh-y).square().sum()); latent_count+=y.numel()
    mean=latent_total/latent_count
    power=latent_second/latent_count; mse=latent_error/latent_count
    return torch.cat(rows),{'attribute_normalized_mse':{n:v/count for n,v in attr.items()},
        'latent_mse':mse,'latent_target_power':power,'latent_nmse':mse/max(power,1e-12),
        'latent_mean':mean,'latent_std':max(0,power-mean**2)**.5}


@torch.no_grad()
def evaluate_paths(model,scene,*,step,phase,snrs,trials,seed,save_images,out,condition_policy='trained_range',reference_snr=10.,verbose=False,
                   clean_only=False,color_metrics=False,save_phase_images=False,lpips_metrics=False,noise_reference_symbols=None,
                   save_noiseless_images=False):
    from .rendering import render
    from .cli import seed_all
    from .perceptual_metrics import LPIPSEvaluator,image_metric_protocol
    if not scene.val_cameras or trials<1 or not snrs or any(not math.isfinite(s) for s in snrs):
        raise ValueError('validation requires cameras, positive trials and finite SNRs')
    conditioning_pair(model,snrs[0],condition_policy,reference_snr)
    out=Path(out); (out/'metrics').mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device; mode=model.training; records=[]
    identity=None if clean_only else model_id(model)
    try:
        with preserved_rng(device):
            model.eval(); seed_all(seed)
            perceptual=LPIPSEvaluator(device) if lpips_metrics else None
            maximum=len(model.cfg.rates)-1
            # Generate once on CPU; independent of padding and subsequent noise.
            mixed=[torch.where(ids>=0,torch.randint(1,maximum+1,ids.shape),0) for ids in scene.group_ids]
            paths=[('clean','clean',10.,1)]
            if not clean_only:
                layouts=tuple(map(str,range(1,maximum+1)))+('mixed',)
                paths += [('finite_noiseless',q,10.,1) for q in layouts]
                paths += [('awgn',q,float(snr),trials) for snr in snrs for q in layouts]
            for path,layout,snr,ntrials in paths:
                started=time.perf_counter()
                tx_snr,rx_snr=conditioning_pair(model,snr,condition_policy,reference_snr)
                qs=mixed if layout=='mixed' else [torch.where(ids>=0,int(layout) if layout!='clean' else maximum,0) for ids in scene.group_ids]
                qfull=torch.cat([q[ids>=0] for q,ids in zip(qs,scene.group_ids)])
                measurements=[]; diagnostics=[]; per_view=[]
                for trial in range(ntrials):
                    # Same fixed-slot noise across prefixes and SNRs, per trial.
                    seed_all(seed+1000+trial)
                    raw,diag=decode_path(model,scene,qs,path,snr,tx_condition_snr=tx_snr,rx_condition_snr=rx_snr,
                                         noise_reference_symbols=noise_reference_symbols); diagnostics.append(diag)
                    views=[]
                    for i,camera in enumerate(scene.val_cameras):
                        image=render(raw,camera,scene.degree,getattr(scene,'white_background',False))
                        source=scene.reference.get(camera,device); photo=camera.original_image.to(device)
                        metrics={prefix+'_'+k:v for prefix,target in [('source',source),('photo',photo)] for k,v in image_metrics(image,target).items()}
                        if perceptual is not None:
                            for prefix,target in [('source',source),('photo',photo)]:
                                metrics[prefix+'_lpips']=perceptual(image,target)
                        if color_metrics:
                            from .appearance_metrics import appearance_metrics
                            for prefix,target in [('source',source),('photo',photo)]:
                                metrics.update({prefix+'_color_'+k:v for k,v in appearance_metrics(image,target).items()})
                        views.append(metrics); per_view.append(dict(metrics,trial=trial,view=i,image_name=str(camera.image_name)))
                        endpoint=phase in ('initial','final') or (save_phase_images and phase=='semantic_bootstrap_end')
                        image_paths=('clean','awgn','finite_noiseless') if save_noiseless_images else ('clean','awgn')
                        if save_images and endpoint and trial==0 and path in image_paths and snr==10 and i in (0,len(scene.val_cameras)-1):
                            folder='end_semantic_bootstrap' if phase=='semantic_bootstrap_end' else phase
                            save_panel(out/'images'/folder/f'{path}_{layout}_view{i:04d}.png',photo,source,image,f'{path} / {layout}')
                    measurements.append({k:float(np.mean([v[k] for v in views])) for k in views[0]})
                row={'step':step,'phase':phase,'path':path,'layout':layout,'actual_snr':None if path=='clean' else snr,
                     'tx_condition_snr':None if path=='clean' else tx_snr,
                     'rx_condition_snr':None if path=='clean' else rx_snr,'condition_policy':condition_policy,
                     'training_snrs':list(model.training_snrs),'condition_support_known':bool(model.training_snrs),
                     'channel_trained':getattr(scene,'channel_trained',False),
                     'noise_reference_complex_symbols':noise_reference_symbols or model.cfg.rates[-1],
                     'image_metric_protocol':image_metric_protocol(lpips_metrics),
                     'views':len(scene.val_cameras),'trials':ntrials,'per_view_trial':per_view,
                     'latent_mse':float(np.mean([d['latent_mse'] for d in diagnostics])),
                     'latent_target_power':float(np.mean([d['latent_target_power'] for d in diagnostics])),
                     'latent_nmse':float(np.mean([d['latent_nmse'] for d in diagnostics])),
                     'latent_mean':float(np.mean([d['latent_mean'] for d in diagnostics])),
                     'latent_std':float(np.mean([d['latent_std'] for d in diagnostics])),
                     'attribute_normalized_mse':{n:float(np.mean([d['attribute_normalized_mse'][n] for d in diagnostics])) for n in diagnostics[0]['attribute_normalized_mse']}}
                for k in measurements[0]:
                    row[k]=float(np.mean([v[k] for v in measurements]))
                    row[k+'_std_trials']=float(np.std([v[k] for v in measurements]))
                if path=='clean':
                    row.update(payload_complex_symbols=None,total_channel_uses=None,xyz_bytes=None,metadata_bytes=None,
                               symbols_per_gaussian=None,rate_scope='clean bypass diagnostic, not a guaranteed upper bound; latent float storage is NOT a wire rate')
                else:
                    row.update(rate_record(model,scene,qfull,snr,'none' if path=='finite_noiseless' else 'awgn',identity,tx_snr,rx_snr))
                records.append(row)
                if verbose:
                    quality=f", SSIM={row['source_ssim']:.4f}" + (f", LPIPS={row['source_lpips']:.4f}" if lpips_metrics else '')
                    print(f"{condition_policy} {path}/q{layout}: actual={row['actual_snr']}, tx={row['tx_condition_snr']}, rx={row['rx_condition_snr']}, "
                          f"PSNR={row['source_psnr']:.3f}{quality}, latent_MSE={row['latent_mse']:.6g}, sec={time.perf_counter()-started:.2f}",flush=True)
    finally:
        model.train(mode)
    for row in records: append_json(out/'metrics'/('test.jsonl' if phase=='test' else 'validation.jsonl'),row)
    summary=', '.join(f"{r['path']}/{r['layout']}: PSNR={r['source_psnr']:.2f}, SSIM={r['source_ssim']:.4f}"
                      +(f", LPIPS={r['source_lpips']:.4f}" if lpips_metrics else '')
                      for r in records if r['path']=='clean' or r['actual_snr']==10)
    print(f'validation {step} {phase} [SNR conditions: {condition_policy}]: {summary}',flush=True)
    return records
