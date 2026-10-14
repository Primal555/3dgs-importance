"""Read-only enhancement-symbol shuffle: paired channel noise, fixed tiers.

Shuffle complete symbol segments across active points within each codec batch.
Unchanged channel normalization, base symbols, suffix lengths and noise slots
make the untransmitted-suffix controls exact negatives. No retraining or extra
deployment inputs. This is an intervention, not a new rate-distortion point.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from .appearance_metrics import appearance_metrics
from .cli import device_for,seed_all
from .data import read_ply
from .modular_checkpoint import validate_scene
from .modular_evaluation import conditioning_pair,decode_path,rate_record
from .modular_experiment import make_scene,json_write
from .modular_snr_controls import control_inputs
from .optimization import preserved_rng
from .render_objective import image_metrics
from .render_validation import append_json
from .transport import load_checkpoint,model_id


def shuffle_suffix(signal,active,start,end,generator):
    """Derange active rows with a private CPU RNG; preserve all other values."""
    if (active.shape!=signal.shape[:-1] or active.dtype!=torch.bool or
            type(start) is not int or type(end) is not int or
            not 0<=start<end<=signal.shape[-1]//2):
        raise ValueError('invalid suffix shuffle shape or range')
    positions=active.flatten().nonzero().flatten()
    count=len(positions)
    if count<2:
        return signal,0
    order=torch.randperm(count,generator=generator).to(positions.device)
    recipients=positions[order]; donors=positions[order.roll(1)]
    shuffled=signal.clone(); flat=shuffled.reshape(-1,signal.shape[-1])
    flat[recipients,2*start:2*end]=signal.reshape_as(flat)[donors,2*start:2*end]
    return shuffled,count


def check_negative_control(row,baseline):
    for key,value in baseline.items():
        if key.startswith(('source_','photo_','latent_')) and isinstance(value,(int,float)):
            if row.get(key)!=value:
                raise RuntimeError(f'negative control changed {key}; inspect prefix isolation/noise pairing')


@torch.no_grad()
def evaluate_probe(model,scene,*,out,snr=10.,trials=2,seed=42):
    from .rendering import render
    if not math.isfinite(snr) or trials<1 or not scene.val_cameras:
        raise ValueError('finite SNR, positive trials and test cameras required')
    out=Path(out); (out/'metrics').mkdir(parents=True,exist_ok=True)
    records=[]; mode=model.training
    identity=model_id(model); device=next(model.parameters()).device
    tx_snr,rx_snr=conditioning_pair(model,snr)
    cases={1:('baseline','shuffle_all'),2:('baseline','shuffle_2','shuffle_3'),
           3:('baseline','shuffle_2','shuffle_3','shuffle_all')}
    _,base,middle,maximum=model.cfg.rates
    ranges={'shuffle_2':(base,middle),'shuffle_3':(middle,maximum),'shuffle_all':(base,maximum)}
    try:
        with preserved_rng(device):
            model.eval()
            for path in ('finite_noiseless','awgn'):
                for tier,actions in cases.items():
                    qs=[torch.where(ids>=0,tier,0) for ids in scene.group_ids]
                    qfull=torch.cat([q[ids>=0] for q,ids in zip(qs,scene.group_ids)])
                    baseline=None
                    for action in actions:
                        measurements=[]; diagnostics=[]; details=[]; shuffle_audits=[]
                        for trial in range(1 if path=='finite_noiseless' else trials):
                            # Same global noise draws irrespective of shuffle.
                            seed_all(seed+1000+trial)
                            generator=torch.Generator(device='cpu').manual_seed(seed+2000+trial)
                            audit={'deranged_rows':0,'singleton_groups':0}
                            hook=None
                            if action!='baseline':
                                start,end=ranges[action]
                                def intervene(module,args,signal):
                                    shuffled,count=shuffle_suffix(signal,args[1],start,end,generator)
                                    audit['deranged_rows']+=count
                                    audit['singleton_groups']+=int(int(args[1].sum())==1)
                                    return shuffled
                                hook=model.channel_encoder.register_forward_hook(intervene)
                            try:
                                raw,diag=decode_path(model,scene,qs,path,snr,
                                    tx_condition_snr=tx_snr,rx_condition_snr=rx_snr)
                            finally:
                                if hook is not None: hook.remove()
                            diagnostics.append(diag); views=[]
                            shuffle_audits.append(audit)
                            for index,camera in enumerate(scene.val_cameras):
                                image=render(raw,camera,scene.degree,scene.white_background)
                                targets={'source':scene.reference.get(camera,device),'photo':camera.original_image.to(device)}
                                metrics={}
                                for name,target in targets.items():
                                    metrics.update({name+'_'+k:v for k,v in image_metrics(image,target).items()})
                                    metrics.update({name+'_color_'+k:v for k,v in appearance_metrics(image,target).items()})
                                views.append(metrics)
                                details.append(dict(metrics,trial=trial,view=index,image_name=str(camera.image_name)))
                            measurements.append({k:sum(v[k] for v in views)/len(views) for k in views[0]})
                        row=dict(path=path,layout=str(tier),intervention=action,actual_snr=snr,
                                 tx_condition_snr=tx_snr,rx_condition_snr=rx_snr,
                                 views=len(scene.val_cameras),trials=len(measurements),per_view_trial=details)
                        for key in measurements[0]: row[key]=sum(v[key] for v in measurements)/len(measurements)
                        for key in ('latent_mse','latent_nmse','latent_target_power'):
                            row[key]=sum(d[key] for d in diagnostics)/len(diagnostics)
                        row.update(rate_record(model,scene,qfull,snr,'none' if path=='finite_noiseless' else 'awgn',identity,tx_snr,rx_snr))
                        if action=='baseline': baseline=row
                        row.update(source_psnr_drop_db=baseline['source_psnr']-row['source_psnr'],
                                   latent_mse_increase=row['latent_mse']-baseline['latent_mse'],
                                   negative_control=action!='baseline' and (tier==1 or (tier==2 and action=='shuffle_3')))
                        row['shuffle_audit']={
                            'eligible_active_rows_per_trial':len(qfull),
                            'deranged_rows_per_trial':[a['deranged_rows'] for a in shuffle_audits],
                            'singleton_groups_per_trial':[a['singleton_groups'] for a in shuffle_audits],
                            'affected_transmitted_rows_per_trial':[0 if row['negative_control'] else a['deranged_rows'] for a in shuffle_audits]}
                        if row['negative_control']: check_negative_control(row,baseline)
                        records.append(row); append_json(out/'metrics/prefix_probe.jsonl',row)
                        print(f"{path} q{tier} {action}: PSNR={row['source_psnr']:.3f}, "
                              f"drop={row['source_psnr_drop_db']:.3f} dB, latent_MSE={row['latent_mse']:.6g}",flush=True)
    finally:
        model.train(mode)
    if model_id(model)!=identity: raise RuntimeError('prefix probe changed model parameters')
    return records


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='modular experiment arm, e.g. OUT/progressive')
    p.add_argument('--out',required=True,help='new diagnostic directory, normally RUN/diagnostics/prefix_probe')
    p.add_argument('--checkpoint',help='default RUN/checkpoints/final/codec.pt')
    p.add_argument('--manifest',help='override paths while preserving scene identity and split')
    p.add_argument('--device',default='cuda')
    p.add_argument('--snr',type=float,default=10.)
    p.add_argument('--trials',type=int,default=2)
    p.add_argument('--test-views',type=int,default=4,help='evenly spaced original test views; 0 means all')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    return p


def main(args):
    from .rendering import load_cameras
    if args.test_views<0 or args.cpu_threads<1 or args.trials<1 or not math.isfinite(args.snr):
        raise ValueError('invalid view, thread, trial budget or SNR')
    run=Path(args.run).resolve(); out=Path(args.out).resolve()
    if out.exists(): raise FileExistsError('use a NEW prefix probe directory')
    config=json.loads((run/'training.json').read_text(encoding='utf-8'))
    checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else run/'checkpoints/final/codec.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    device=device_for(args.device); torch.set_num_threads(args.cpu_threads)
    model=load_checkpoint(checkpoint,device)
    if model.cfg.architecture!='modular_jscc_v1' or len(model.cfg.rates)!=4:
        raise ValueError('probe requires modular three-tier progressive prefixes')
    if json.loads(json.dumps(model.cfg.to_dict()))!=config['codec_config']:
        raise ValueError('run/checkpoint config mismatch')
    if not model.training_snrs: model.set_training_snrs(config.get('train_snrs',[]))
    source,manifest_path,spec,old,_=control_inputs(config,args.manifest)
    raw,degree=read_ply(spec['ply'])
    scene=make_scene(spec,raw,degree,model,source,old,out); del raw
    validate_scene(config['scene'],scene.record)
    if 'scene_record' in saved.get('training',{}): validate_scene(saved['training']['scene_record'],scene.record)
    cameras=load_cameras(spec['source'],scene.record['resolution'],scene.white_background,scene.record['images'],'test')
    if not cameras: raise ValueError('missing original test cameras')
    if 0<args.test_views<len(cameras):
        count=args.test_views
        indices=[round(i*(len(cameras)-1)/(count-1)) for i in range(count)] if count>1 else [len(cameras)//2]
        cameras=[cameras[i] for i in indices]
    scene.val_cameras=cameras
    records=evaluate_probe(model,scene,out=out,snr=args.snr,trials=args.trials,seed=args.seed)
    summary={'status':'complete','source_run':str(run),'checkpoint':str(checkpoint),'manifest':str(manifest_path),
             'optimizer_updates':0,'parameters_unchanged':True,'test_view_names':[str(c.image_name) for c in cameras],
             'shuffle_scope':'all active points within each codec batch, not global whole-scene permutation',
             'same_prefix_tier_power_and_noise':True,
             'interpretation':'content-dependence intervention; does not alone distinguish encoder redundancy from decoder failure',
             'comparison':[{k:v for k,v in r.items() if k!='per_view_trial'} for r in records]}
    json_write(out/'summary.json',summary)
    json_write(out/'complete.json',{'status':'complete','optimizer_updates':0,'parameters_unchanged':True})
    return summary


if __name__=='__main__': main(parser().parse_args())
