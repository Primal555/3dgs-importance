"""Checkpoint-only controls separating AWGN noise and encoder/decoder conditions."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path
import torch
from .cli import device_for
from .data import read_ply
from .modular_checkpoint import validate_scene
from .modular_experiment import make_scene,json_write
from .multiscene import manifest
from .modular_evaluation import evaluate_paths
from .transport import load_checkpoint,model_id

POLICIES=('matched','fixed_both','fixed_tx','fixed_rx')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='completed modular experiment directory')
    p.add_argument('--out',required=True,help='new directory; source files and weights stay untouched')
    p.add_argument('--checkpoint',help='default: RUN/checkpoints/final/codec.pt')
    p.add_argument('--manifest',help='override source paths after migration, with identical PLY and camera split')
    p.add_argument('--device',default='cuda')
    p.add_argument('--snrs',nargs='+',type=float,default=[0,5,10,15,20])
    p.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES))
    p.add_argument('--reference-snr',type=float,help='defaults to the median recorded training SNR; 10 for the current model')
    p.add_argument('--trials',type=int,default=2)
    p.add_argument('--test-views',type=int,default=0,help='0: all original test cameras; positive: evenly spaced subset')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=False,
                   help='evaluate pretrained VGG LPIPS v0.1 without any optimizer updates')
    return p


def comparisons(records):
    awgn=[r for r in records if r['path']=='awgn']
    result=[]
    for snr in sorted({r['actual_snr'] for r in awgn}):
        for layout in ('1','2','3','mixed'):
            group={r['condition_policy']:r for r in awgn if r['actual_snr']==snr and r['layout']==layout}
            row=dict(actual_snr=snr,layout=layout,views=next(iter(group.values()))['views'])
            for policy,r in group.items():
                row[policy]=dict(source_psnr=r['source_psnr'],photo_psnr=r['photo_psnr'],
                    source_ssim=r['source_ssim'],latent_mse=r['latent_mse'],latent_nmse=r['latent_nmse'],
                    tx_condition_snr=r['tx_condition_snr'],rx_condition_snr=r['rx_condition_snr'])
                row[policy].update({k:r[k] for k in ('photo_ssim','source_lpips','photo_lpips') if k in r})
            if 'matched' in group:
                row['gain_over_matched_db']={p:r['source_psnr']-group['matched']['source_psnr'] for p,r in group.items()}
            result.append(row)
    return result


def control_inputs(config,override_manifest=None):
    path=Path(override_manifest or config['manifest']).resolve()
    selected=[s for s in manifest(path) if s['name']=='truck' and s['role']=='train']
    if len(selected)!=1: raise ValueError('control manifest must contain one training Truck scene')
    spec=dict(selected[0],prior='none')
    for key in ('ply','source'):
        if not Path(spec[key]).exists(): raise FileNotFoundError(spec[key])
    # The modular run preserves identity, split and all rendering/batch options.
    # Only the legacy digital accounting assumption may need an optional source.
    source=dict(config['scene'])
    source['net_bits_per_use']=config.get('net_bits_per_use',2.)
    origin='modular run' if 'net_bits_per_use' in config else 'fallback assumption (2 net bits/use)'
    old_metadata=Path(config['source_run'])/'training.json'
    if 'net_bits_per_use' not in config and old_metadata.exists():
        old_source=json.loads(old_metadata.read_text(encoding='utf-8'))
        if 'net_bits_per_use' in old_source:
            source['net_bits_per_use']=old_source['net_bits_per_use']; origin='original source metadata'
    return source,path,spec,config['scene'],origin


def plot_controls(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    styles={'matched':('#356d9b','o','-'),'fixed_both':('#d88931','s','--'),
            'fixed_tx':('#7b8139','^','-.'),'fixed_rx':('#b06a8e','d',':')}
    for row,tier in enumerate(('1','2','3')):
        for policy in POLICIES:
            series=sorted((r for r in records if r['path']=='awgn' and r['layout']==tier and r['condition_policy']==policy),
                          key=lambda r:r['actual_snr'])
            if not series: continue
            color,marker,line=styles[policy]
            for ax,key in zip(axes[row],('source_psnr','latent_nmse')):
                ax.plot([r['actual_snr'] for r in series],[r[key] for r in series],color=color,marker=marker,linestyle=line,label=policy)
                ax.set(title=f'q{tier}',xlabel='Physical AWGN SNR (dB)',ylabel='Source PLY PSNR (dB)' if key=='source_psnr' else 'Latent NMSE')
                ax.grid(alpha=.2); ax.legend(fontsize=8)
                if key=='latent_nmse': ax.set_yscale('log')
    fig.suptitle('Same checkpoint, test cameras, tiers and paired noise; only SNR conditions differ')
    path=Path(out)/'charts/snr_controls.png'; path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150); plt.close(fig)
    return path


def main(args):
    from .rendering import load_cameras
    if (not args.snrs or any(not math.isfinite(s) for s in args.snrs) or args.trials<1 or args.cpu_threads<1 or args.test_views<0
            or (args.reference_snr is not None and not math.isfinite(args.reference_snr))):
        raise ValueError('invalid SNR, trials, threads or view budget')
    if len(set(args.snrs))!=len(args.snrs) or len(set(args.policies))!=len(args.policies):
        raise ValueError('duplicate SNRs or condition policies')
    run=Path(args.run).resolve(); out=Path(args.out).resolve()
    if out.exists(): raise FileExistsError('use a NEW SNR control output directory')
    config=json.loads((run/'training.json').read_text(encoding='utf-8'))
    checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else run/'checkpoints/final/codec.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    device=device_for(args.device); torch.set_num_threads(args.cpu_threads)
    model=load_checkpoint(checkpoint,device)
    if model.cfg.architecture!='modular_jscc_v1': raise ValueError('SNR controls require modular_jscc_v1')
    if model.cfg.to_dict()!=config['codec_config']:
        # JSON serializes tuples as arrays.
        if json.loads(json.dumps(model.cfg.to_dict()))!=config['codec_config']: raise ValueError('run/checkpoint config mismatch')
    if not model.training_snrs: model.set_training_snrs(config.get('train_snrs',[]))
    if not model.training_snrs: raise ValueError('missing recorded training SNR support')
    reference=float(args.reference_snr) if args.reference_snr is not None else statistics.median(model.training_snrs)
    source,manifest_path,spec,old,rate_origin=control_inputs(config,args.manifest)
    raw,degree=read_ply(spec['ply'])
    scene=make_scene(spec,raw,degree,model,source,old,out); del raw
    validate_scene(config['scene'],scene.record)
    if 'scene_record' in saved.get('training',{}): validate_scene(saved['training']['scene_record'],scene.record)
    cameras=load_cameras(spec['source'],scene.record['resolution'],scene.white_background,scene.record['images'],'test')
    if not cameras: raise ValueError('missing test cameras; no training-view substitution permitted')
    if args.test_views and args.test_views<len(cameras):
        n=args.test_views
        indices=[round(i*(len(cameras)-1)/(n-1)) for i in range(n)] if n>1 else [len(cameras)//2]
        cameras=[cameras[i] for i in indices]
    scene.val_cameras=cameras
    completed=json.loads((run/'complete.json').read_text(encoding='utf-8')) if (run/'complete.json').exists() else {}
    scene.channel_trained=saved.get('training',{}).get('progress',{}).get('channel_trained',completed.get('channel_trained',False))
    identity=model_id(model); model.eval()
    from .perceptual_metrics import image_metric_protocol
    metadata=dict(image_metric_protocol=image_metric_protocol(args.lpips),
                  source_run=str(run),checkpoint=str(checkpoint),manifest=str(manifest_path),model_id=identity,
                  reference_snr=reference,training_snrs=list(model.training_snrs),physical_snrs=args.snrs,policies=args.policies,
                  test_view_names=[str(c.image_name) for c in cameras],trials=args.trials,seed=args.seed,
                  net_bits_per_use=scene.net_bits_per_use,net_bits_per_use_origin=rate_origin,
                  optimizer_updates=0,noise_pairing='same fixed complex-symbol slots and random draws across all controls',
                  deployment_policy='trained_range; only network conditions are bounded, physical noise is untouched')
    json_write(out/'experiment.json',metadata)
    records=[]; timings={}
    for policy in args.policies:
        started=time.perf_counter()
        print(f'SNR control: {policy}; {len(cameras)} test views; {args.trials} trials; reference={reference:g} dB',flush=True)
        records+=evaluate_paths(model,scene,step=int(saved['step']),phase='test',snrs=args.snrs,trials=args.trials,
                                seed=args.seed,save_images=False,out=out,condition_policy=policy,reference_snr=reference,verbose=True,
                                lpips_metrics=args.lpips)
        timings[policy]=time.perf_counter()-started
        if model_id(model)!=identity: raise RuntimeError('evaluation changed checkpoint parameters')
    comparison=comparisons(records)
    plot_controls(records,out)
    summary=dict(metadata,status='complete',parameters_unchanged=True,comparison=comparison,policy_seconds=timings,
                 interpretation='Compare fixed_both with matched first; fixed_tx/fixed_rx isolate sender/receiver condition extrapolation. Monotonic PSNR is not guaranteed by bounding conditions.')
    json_write(out/'summary.json',summary); json_write(out/'complete.json',dict(status='complete',optimizer_updates=0,parameters_unchanged=True))
    for r in comparison:
        if r['layout']=='3': print(json.dumps(r,ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__': main(parser().parse_args())
