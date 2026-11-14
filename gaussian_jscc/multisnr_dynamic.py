"""Frozen initial/final codecs under a changing, paired per-delivery SNR trace."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
import statistics
import tarfile

import torch

from comparison_baselines.side_link_policy import POLICIES,side_link_config
from .cli import device_for
from .data import read_ply
from .model_factory import build_codec,config_from_dict
from .modular_experiment import json_write
from .multisnr_evaluation import read,fixed_map,deliver_trial
from .optimization import preserved_rng
from .perceptual_metrics import LPIPSEvaluator
from .successive_multiscene_inputs import make_scene
from .transport import model_id

TRACE = [10.,9.,7.5,5.,6.,10.,15.,12.5,5.,15.,7.5,10.]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='completed multi-SNR experiment')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',default=['truck'])
    p.add_argument('--layouts',nargs='+',choices=('1','2','3','mixed'),default=['2'])
    p.add_argument('--snr-trace',nargs='+',type=float,default=TRACE)
    p.add_argument('--trials',type=int,help='independent paired traces; inherits the source setting')
    p.add_argument('--seed',type=int,default=20261114)
    p.add_argument('--side-link-policy',choices=POLICIES,default='snr_qpsk_half_v1')
    p.add_argument('--device',default='cuda');p.add_argument('--digital-device',default='cuda')
    p.add_argument('--digital-python');p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    return p


def summarize(rows):
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ('scene','method','endpoint','trace_step'))
        groups.setdefault(key,[]).append(row)
    results=[];cumulative={}
    for (scene,method,endpoint,step),items in sorted(groups.items()):
        good=[r for r in items if r['status']=='success']
        metrics=None;deviations=None
        if good:
            keys=set.intersection(*(set(r['metrics']) for r in good))
            metrics={k:statistics.mean(r['metrics'][k] for r in good) for k in sorted(keys)}
            deviations={k:statistics.pstdev(r['metrics'][k] for r in good) for k in sorted(keys)}
        cost=statistics.mean(r['total_channel_uses'] for r in items)
        key=(scene,method,endpoint);cumulative[key]=cumulative.get(key,0)+cost
        results.append(dict(scene=scene,method=method,endpoint=endpoint,trace_step=step,
            snr_db=items[0]['snr_db'],trials=len(items),successes=len(good),success_rate=len(good)/len(items),
            metrics=metrics,metrics_std=deviations,total_channel_uses=cost,
            cumulative_channel_uses=cumulative[key],quality_conditioning='successful_complete_delivery'))
    paired=[]
    for scene,method,step in sorted({(r['scene'],r['method'],r['trace_step']) for r in rows}):
        selected=[r for r in rows if (r['scene'],r['method'],r['trace_step'])==(scene,method,step)]
        lookup={(r['endpoint'],r['trial']):r for r in selected}
        pairs=[(lookup['initial',t],lookup['final',t]) for t in {r['trial'] for r in selected}
               if ('initial',t) in lookup and ('final',t) in lookup
               and lookup['initial',t]['status']==lookup['final',t]['status']=='success']
        delta=None
        if pairs:
            keys=set.intersection(*(set(a['metrics']) & set(b['metrics']) for a,b in pairs))
            delta={k:statistics.mean(b['metrics'][k]-a['metrics'][k] for a,b in pairs) for k in sorted(keys)}
        paired.append(dict(scene=scene,method=method,trace_step=step,both_successes=len(pairs),
                           final_minus_initial=delta))
    return results,paired


@torch.no_grad()
def main(args):
    source,out=args.run.resolve(),args.out.resolve()
    if source==out or source.is_relative_to(out): raise ValueError('use a dedicated evaluation output directory')
    metadata=read(source/'training.json');complete=read(source/'complete.json');settings=metadata['settings']
    if complete.get('status')!='complete' or metadata.get('version') not in (
            'successive_multisnr_v1','successive_multisnr_source_v2'):
        raise ValueError('completed multi-SNR experiment required')
    if min(args.cpu_threads,args.trials or settings['trials'])<1 or args.trials==0:
        raise ValueError('positive threads/trials required')
    low,high=settings['snr_range']
    if len(args.snr_trace)<2 or any(not math.isfinite(s) or not low<=s<=high for s in args.snr_trace):
        raise ValueError('trace needs at least two finite in-range SNR samples')
    held=metadata.get('heldout_scene',[])
    if isinstance(held,dict): held=[held]
    lookup={s['name']:s for s in metadata['selected_scenes']+held}
    if not args.scenes or len(set(args.scenes))!=len(args.scenes) or not set(args.scenes)<=set(lookup):
        raise ValueError('unique scenes from the source experiment required')
    if len(set(args.layouts))!=len(args.layouts): raise ValueError('unique layouts required')
    digital_python=args.digital_python or settings['digital_python']
    if not Path(digital_python).is_file(): raise FileNotFoundError(digital_python)
    device=device_for(args.device);torch.set_num_threads(args.cpu_threads)
    saved={label:torch.load(source/f'checkpoints/{label}/codec.pt',map_location='cpu',weights_only=True)
           for label in ('initial','final')}
    if saved['initial']['config']!=saved['final']['config']: raise ValueError('initial/final config mismatch')
    models={}
    for label,state in saved.items():
        model=build_codec(config_from_dict(state['config'])).to(device)
        model.load_state_dict(state['state_dict']);model.eval().requires_grad_(False)
        model.set_training_snrs([10.] if label=='initial' else settings['snr_range'])
        models[label]=model
    ids={label:model_id(model) for label,model in models.items()}
    if ids['final']!=complete['codec_id']: raise ValueError('final checkpoint does not match complete.json')
    trials=args.trials if args.trials is not None else settings['trials']
    plan=dict(version='multisnr_dynamic_delivery_v1',source_run=str(source),codec_ids=ids,
        scenes=args.scenes,layouts=args.layouts,snr_trace=args.snr_trace,trials=trials,seed=args.seed,
        side_link_policy=args.side_link_policy,digital_python=str(digital_python),
        digital_device=args.digital_device,device=str(device),cpu_threads=args.cpu_threads,
        links=[asdict(side_link_config(s,args.side_link_policy,args.digital_device)) for s in args.snr_trace],
        sampling='independent AWGN per delivery; identical seeds across models and layouts',
        channel='SNR fixed during each complete delivery, changes between deliveries; exact current CSI',
        time_axis='delivery index; no physical seconds or intra-file fading assumed',
        scopes=dict(initial='fixed 10dB condition',final='matched current SNR'),
        selection=False,optimizer_updates=0,arq=False,
        quality_conditioning='successful complete delivery; missing quality when any side file fails',
        failure_cost='all three side files and attribute symbols charged; no retransmission or fallback image')
    if out.exists():
        if not args.resume or not (out/'plan.json').exists() or read(out/'plan.json')!=plan:
            raise ValueError('use a new output or resume the identical evaluation plan')
    else: json_write(out/'plan.json',plan)
    del saved
    options=argparse.Namespace(**settings);options.net_bits_per_use=2.
    options.device=str(device);options.out=str(out)
    perceptual=LPIPSEvaluator(device) if settings['lpips'] else None
    rows=[]
    with preserved_rng(device):
        for name in args.scenes:
            spec=lookup[name];raw,degree=read_ply(spec['ply'])
            old=metadata['scene_records'].get(name)
            if old is None:
                record=read(source/f'scenes/{name}/scene.json')
                keys=next(iter(metadata['scene_records'].values()))
                old={key:record[key] for key in keys}
            scene=make_scene(spec,raw,degree,models['initial'],options,out,old=old)
            previous=read(source/f'evaluation/final/{name}/summary.json')['rows'][0]['views']
            if [str(c.image_name) for c in scene.test_cameras]!=[r['view'] for r in previous]:
                raise ValueError('dynamic evaluation must use the original test cameras')
            json_write(scene.out/'scene.json',scene.record)
            for step,snr in enumerate(args.snr_trace,1):
                config=side_link_config(snr,args.side_link_policy,args.digital_device)
                for layout in args.layouts:
                    q=fixed_map(scene,layout,settings['seed'])
                    for trial in range(trials):
                        draw=(step-1)*trials+trial
                        seed=args.seed+900000+draw;digital_seed=args.seed+1800000+3*draw
                        for label,model in models.items():
                            folder=out/'deliveries'/name/layout/label/f'{step:03d}'/f'{trial:03d}'
                            file=folder/'result.json'
                            identity=dict(scene=name,method=layout,endpoint=label,trace_step=step,trial=trial,
                                seed=seed,digital_seed=digital_seed,snr_db=snr,codec_id=ids[label],
                                condition_snr_db=10. if label=='initial' else snr)
                            if file.exists():
                                row=read(file)
                                if any(row.get(k)!=v for k,v in identity.items()) or row['comparison_accounting']['link']!=asdict(config):
                                    raise ValueError('saved delivery identity changed')
                            else:
                                row=deliver_trial(model,scene,q,snr,seed,digital_seed,config,digital_python,
                                                  folder,perceptual=perceptual)
                                if any(row[k]!=identity[k] for k in ('codec_id','snr_db','condition_snr_db','seed','digital_seed')):
                                    raise RuntimeError('delivery conditions differ from paired plan')
                                row.update(identity);json_write(file,row)
                            rows.append(row)
                            quality='unavailable' if row['metrics'] is None else f'{row["metrics"]["source_psnr"]:.4f}'
                            print(f'dynamic {step}/{len(args.snr_trace)} {name}/q{layout}/{label} trial={trial+1}: '
                                  f'SNR={snr:g}, {row["status"]}, source PSNR={quality}, '
                                  f'cost={row["total_channel_uses"]/1e6:.5f}M',flush=True)
            del scene
    if any(model_id(m)!=ids[k] for k,m in models.items()): raise RuntimeError('frozen codec changed')
    results,paired=summarize(rows)
    report=dict(status='complete',plan=plan,trials=rows,results=results,paired_quality=paired,
                frozen_codecs_verified=True)
    json_write(out/'summary.json',report)
    from .multisnr_dynamic_plots import write_figures
    write_figures(out,report)
    json_write(out/'complete.json',dict(status='complete',deliveries=len(rows),optimizer_updates=0,
                                      frozen_codecs_verified=True))
    with tarfile.open(out/'review.tar.gz','w:gz') as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.csv','.png','.pdf','.log'):
                archive.add(path,arcname=path.relative_to(out))
    print(f'Dynamic evaluation complete: {out / "review.tar.gz"}',flush=True)
    return report


if __name__=='__main__': main(parser().parse_args())
