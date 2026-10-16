"""Paired finite-prefix evaluation; no nonexistent clean-latent bypass."""
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .cli import seed_all
from .data import to_raw
from .optimization import preserved_rng
from .position_delivery import encode_positions
from .render_objective import image_metrics
from .render_validation import append_json,save_panel
from .transport import model_id,encode_metadata


@torch.no_grad()
def decode(model, scene, qs, kind):
    device=next(model.parameters()).device;rows=[];sums={};count=0
    slices=(('opacity',3,4),('scale',4,7),('rotation',7,11),('dc',11,14),('sh',14,3+model.cfg.attr_dim))
    for f,q in zip(scene.groups,qs):
        f,q=f.to(device),q.to(device);active=q>0
        pred=model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).to(f),10.,kind,paired_noise=True)[0]
        rows.append(to_raw(pred[active],scene.geometry,model));count+=int(active.sum())
        for name,a,b in slices:
            if b>a:sums[name]=sums.get(name,0.)+float((pred[active][:,a:b]-f[active][:,a:b]).double().square().sum())/(b-a)
    return torch.cat(rows),{n:v/count for n,v in sums.items()}


def rate_record(model,scene,q,kind,identity):
    if not hasattr(scene,'_integrated_xyz_bytes'):
        scene._integrated_xyz_bytes=len(encode_positions(scene.geometry.normalize(scene.raw[:,:3]),
            torch.ones(len(scene.raw),dtype=torch.long),model.cfg))
    meta=encode_metadata(dict(version=2,config=model.cfg.to_dict(),model_id=identity,count=len(q),
        source_count=len(q),geometry=scene.geometry.to_dict(),snr_db=10.,channel=kind),q.numpy())
    efficiency=scene.net_bits_per_use
    if not math.isfinite(efficiency) or efficiency<=0:raise ValueError('invalid net bits per use assumption')
    payload=int(torch.tensor(model.cfg.rates)[q].sum());xyz=scene._integrated_xyz_bytes
    side=math.ceil(8*xyz/efficiency)+math.ceil(8*len(meta)/efficiency)
    return dict(payload_complex_symbols=payload,symbols_per_gaussian=payload/len(q),xyz_bytes=xyz,
        metadata_bytes=len(meta),total_channel_uses=payload+side,net_bits_per_use_assumption=efficiency,
        reliability='XYZ/metadata assumed reliable; no FEC/packet errors simulated; shared model excluded')


@torch.no_grad()
def stage_diagnostics(model,scene,seed=42):
    """Audit the already validated scene; never reload a manifest or prior."""
    from .codec import channel,prefix_mask
    device=next(model.parameters()).device;mode=model.training
    errors=[0.,0.,0.];corrections=[0.,0.];count=0;group_sums=[{},{}]
    slices=(('opacity',0,1),('scale',1,4),('rotation',4,8),('dc',8,11),('sh',11,model.cfg.attr_dim))
    try:
        with preserved_rng(device):
            model.eval();seed_all(seed+20000)
            for f,ids in zip(scene.groups,scene.group_ids):
                f=f.to(device);q=torch.where(ids.to(device)>=0,3,0);active=q>0
                z=model.learned.encode(f,f[...,:3],q,10.)
                mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
                received=channel(z.reshape(-1,2),10.,'awgn').reshape_as(z)*mask
                stages,deltas=model.learned.decode_stages(received,q,10.)
                target=f[active][:,3:];count+=len(target)
                for i,pred in enumerate(stages):errors[i]+=float((pred[active][:,3:]-target).double().square().sum())
                for i,delta in enumerate(deltas):
                    values=delta[active].double();corrections[i]+=float(values.square().sum())
                    for name,a,b in slices:
                        if b>a:group_sums[i][name]=group_sums[i].get(name,0.)+float(values[:,a:b].square().sum())
    finally:model.train(mode)
    return dict(source_gaussians=count,scope='q3/full scene/fixed-slot AWGN; normalized pre-physical attributes; diagnostic only',
        stage_normalized_attribute_mse=[e/count/model.cfg.attr_dim for e in errors],
        corrections=[dict(tier=i+2,normalized_rms=(v/count/model.cfg.attr_dim)**.5,
            groups={name:(group_sums[i][name]/count/(b-a))**.5 for name,a,b in slices if b>a})
            for i,v in enumerate(corrections)])


@torch.no_grad()
def evaluate(model,scene,*,out,step,phase,trials=2,seed=42,save_images=False):
    from .rendering import render
    from .appearance_metrics import appearance_metrics
    if not scene.val_cameras or trials<1:raise ValueError('evaluation requires cameras and positive trials')
    out=Path(out);device=next(model.parameters()).device;mode=model.training;records=[]
    identity=model_id(model)
    try:
        with preserved_rng(device):
            model.eval();seed_all(seed)
            mixed=[torch.where(ids>=0,torch.randint(1,4,ids.shape),0) for ids in scene.group_ids]
            for path,kind,ntrials in [('finite_noiseless','none',1),('awgn','awgn',trials)]:
                for layout in ('1','2','3','mixed'):
                    qs=mixed if layout=='mixed' else [torch.where(ids>=0,int(layout),0) for ids in scene.group_ids]
                    qfull=torch.cat([q[ids>=0] for q,ids in zip(qs,scene.group_ids)])
                    observations=[];attr=[]
                    for trial in range(ntrials):
                        seed_all(seed+1000+trial)
                        raw,errors=decode(model,scene,qs,kind);attr.append(errors)
                        for i,camera in enumerate(scene.val_cameras):
                            image=render(raw,camera,scene.degree,scene.white_background)
                            target=scene.reference.get(camera,device);photo=camera.original_image[:3].to(device)
                            metrics={}
                            for prefix,reference in [('source',target),('photo',photo)]:
                                metrics.update({prefix+'_'+k:v for k,v in image_metrics(image,reference).items()})
                                metrics.update({prefix+'_color_'+k:v for k,v in appearance_metrics(image,reference).items()})
                            observations.append(dict(metrics,trial=trial,view=i,image_name=str(camera.image_name)))
                            if (save_images and phase in ('initial','final') and path=='awgn' and trial==0
                                    and i in (0,len(scene.val_cameras)-1)):
                                save_panel(out/'images'/phase/f'q{layout}_view{i:04d}.png',photo,target,image,'Received: q'+layout)
                    keys=[k for k in observations[0] if k not in ('trial','view','image_name')]
                    row=dict(step=step,phase=phase,path=path,layout=layout,snr=10.,trials=ntrials,
                        views=len(scene.val_cameras),per_view_trial=observations,
                        attribute_normalized_mse={n:float(np.mean([d[n] for d in attr])) for n in attr[0]},
                        **{k:float(np.mean([o[k] for o in observations])) for k in keys},
                        **rate_record(model,scene,qfull,kind,identity))
                    row['source_psnr_std_trials']=float(np.std([np.mean([o['source_psnr'] for o in observations if o['trial']==t]) for t in range(ntrials)]))
                    records.append(row)
    finally:model.train(mode)
    metrics=out/'metrics';metrics.mkdir(parents=True,exist_ok=True)
    dest=metrics/('test.jsonl' if phase=='test' else 'validation.jsonl')
    for row in records:append_json(dest,row)
    for path in ('finite_noiseless','awgn'):
        positive=[r for r in records if r['path']==path and r['layout']!='mixed']
        for a,b in zip(positive,positive[1:]):
            differences=[x['source_mse']-y['source_mse'] for x,y in zip(a['per_view_trial'],b['per_view_trial'])]
            append_json(metrics/'prefix_gains.jsonl',dict(step=step,phase=phase,path=path,
                lower=a['layout'],upper=b['layout'],source_psnr_gain_db=b['source_psnr']-a['source_psnr'],
                mse_reduction_per_extra_symbol=float(np.mean(differences))/(b['symbols_per_gaussian']-a['symbols_per_gaussian']),
                paired_view_trial_improved_fraction=sum(v>0 for v in differences)/len(differences)))
    print(f'validation {step} {phase}: '+', '.join(f"{r['path']}/q{r['layout']}={r['source_psnr']:.2f}" for r in records),flush=True)
    return records
