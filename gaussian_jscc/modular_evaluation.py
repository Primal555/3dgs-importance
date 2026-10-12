"""Current clean representation vs finite-noiseless vs noisy communication.

All prefixes receive noise at fixed symbol slots. Mixed is a fixed positive
tier map, NOT a trained allocation. Digital overhead is measured in bytes;
analogue payload is measured in complex channel uses, not float file sizes.
"""
import math
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


def rate_record(model,scene,q,snr,kind,identity):
    if not hasattr(scene,'_modular_xyz_bytes'):
        stream=encode_positions(scene.geometry.normalize(scene.raw[:,:3]),torch.ones(len(scene.raw),dtype=torch.long),model.cfg)
        scene._modular_xyz_bytes=len(stream)
    header={'version':2,'config':model.cfg.to_dict(),'model_id':identity,'count':len(q),'source_count':len(q),
            'geometry':scene.geometry.to_dict(),'snr_db':float(snr),'channel':kind}
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
def decode_path(model,scene,qs,path,snr):
    device=next(model.parameters()).device
    rows=[]; count=0; attr={n:0. for n in ('opacity','scale','rotation','dc','sh')}
    latent_error=latent_total=latent_second=0.; latent_count=0
    for f,q in zip(scene.groups,qs):
        f,q=f.to(device),q.to(device); active=q>0; values={}
        hooks=[model.semantic_encoder.register_forward_hook(lambda m,a,o:values.update(y=o))]
        if path!='clean': hooks.append(model.channel_decoder.register_forward_hook(lambda m,a,o:values.update(y_hat=o)))
        try:
            pred=predict(model,f,q,snr,'none' if path!='awgn' else 'awgn',
                         'semantic_render' if path=='clean' else 'channel_render',paired_noise=True)
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
    return torch.cat(rows),{'attribute_normalized_mse':{n:v/count for n,v in attr.items()},
        'latent_mse':latent_error/latent_count,'latent_mean':mean,'latent_std':max(0,latent_second/latent_count-mean**2)**.5}


@torch.no_grad()
def evaluate_paths(model,scene,*,step,phase,snrs,trials,seed,save_images,out):
    from .rendering import render
    from .cli import seed_all
    if not scene.val_cameras or trials<1 or not snrs or any(not math.isfinite(s) for s in snrs):
        raise ValueError('validation requires cameras, positive trials and finite SNRs')
    out=Path(out); (out/'metrics').mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device; mode=model.training; records=[]
    identity=model_id(model)
    try:
        with preserved_rng(device):
            model.eval(); seed_all(seed)
            # Generate once on CPU; independent of padding and subsequent noise.
            mixed=[torch.where(ids>=0,torch.randint(1,4,ids.shape),0) for ids in scene.group_ids]
            paths=[('clean','clean',10.,1)]
            paths += [('finite_noiseless',q,10.,1) for q in ('1','2','3','mixed')]
            paths += [('awgn',q,float(snr),trials) for snr in snrs for q in ('1','2','3','mixed')]
            for path,layout,snr,ntrials in paths:
                qs=mixed if layout=='mixed' else [torch.where(ids>=0,int(layout) if layout!='clean' else 3,0) for ids in scene.group_ids]
                qfull=torch.cat([q[ids>=0] for q,ids in zip(qs,scene.group_ids)])
                measurements=[]; diagnostics=[]; per_view=[]
                for trial in range(ntrials):
                    # Same fixed-slot noise across prefixes and SNRs, per trial.
                    seed_all(seed+1000+trial)
                    raw,diag=decode_path(model,scene,qs,path,snr); diagnostics.append(diag)
                    views=[]
                    for i,camera in enumerate(scene.val_cameras):
                        image=render(raw,camera,scene.degree,getattr(scene,'white_background',False))
                        source=scene.reference.get(camera,device); photo=camera.original_image.to(device)
                        metrics={prefix+'_'+k:v for prefix,target in [('source',source),('photo',photo)] for k,v in image_metrics(image,target).items()}
                        views.append(metrics); per_view.append(dict(metrics,trial=trial,view=i,image_name=str(camera.image_name)))
                        if save_images and phase in ('initial','final') and trial==0 and path in ('clean','awgn') and snr==10 and i in (0,len(scene.val_cameras)-1):
                            save_panel(out/'images'/phase/f'{path}_{layout}_view{i:04d}.png',photo,source,image,f'{path} / {layout}')
                    measurements.append({k:float(np.mean([v[k] for v in views])) for k in views[0]})
                row={'step':step,'phase':phase,'path':path,'layout':layout,'actual_snr':None if path=='clean' else snr,
                     'tx_condition_snr':None if path=='clean' else snr,'channel_trained':getattr(scene,'channel_trained',False),
                     'views':len(scene.val_cameras),'trials':ntrials,'per_view_trial':per_view,
                     'latent_mse':float(np.mean([d['latent_mse'] for d in diagnostics])),
                     'latent_mean':float(np.mean([d['latent_mean'] for d in diagnostics])),
                     'latent_std':float(np.mean([d['latent_std'] for d in diagnostics])),
                     'attribute_normalized_mse':{n:float(np.mean([d['attribute_normalized_mse'][n] for d in diagnostics])) for n in diagnostics[0]['attribute_normalized_mse']}}
                for k in measurements[0]:
                    row[k]=float(np.mean([v[k] for v in measurements]))
                    row[k+'_std_trials']=float(np.std([v[k] for v in measurements]))
                if path=='clean':
                    row.update(payload_complex_symbols=None,total_channel_uses=None,xyz_bytes=None,metadata_bytes=None,
                               symbols_per_gaussian=None,rate_scope='clean upper reference; latent float storage is NOT a wire rate')
                else:
                    row.update(rate_record(model,scene,qfull,snr,'none' if path=='finite_noiseless' else 'awgn',identity))
                records.append(row)
    finally:
        model.train(mode)
    for row in records: append_json(out/'metrics'/('test.jsonl' if phase=='test' else 'validation.jsonl'),row)
    summary=', '.join(f"{r['path']}/{r['layout']}={r['source_psnr']:.2f}" for r in records if r['path']=='clean' or r['actual_snr']==10)
    print(f'validation {step} {phase}: {summary}',flush=True)
    return records
