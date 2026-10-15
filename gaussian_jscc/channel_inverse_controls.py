"""Paired channel-only inverse, prefix reread and receiver cooperation controls."""
import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from .cli import device_for
from .data import read_ply
from .latent_error_diagnostics import diagnose
from .modular_checkpoint import capture_rng, restore_rng, restore_training_checkpoint, save_training_checkpoint, validate_scene
from .modular_evaluation import evaluate_paths
from .modular_experiment import json_write, make_scene, repair_logs
from .modular_snr_controls import control_inputs
from .modular_training import bootstrap_step, finish_update

ARMS={'noiseless':'none','awgn10':'awgn'}
PREFIX_ARMS={'suffix10':'awgn','full_prefix10':'awgn'}
SPATIAL_ARMS={'pointwise10':'awgn','spatial10':'awgn'}
COOPERATIVE_ARMS={'stage_spatial10':'awgn','final_pointwise10':'awgn','final_spatial10':'awgn'}


def arms_for(args):
    if args.suite=='cooperative': return COOPERATIVE_ARMS
    return SPATIAL_ARMS if args.suite=='spatial' else PREFIX_ARMS if args.suite=='prefix' else ARMS


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='completed pure-y run, or suite root containing progressive/')
    p.add_argument('--out',required=True,help='NEW suite output; source run is read-only')
    p.add_argument('--checkpoint',help='default RUN/checkpoints/final/codec.pt; must be completed channel_bootstrap')
    p.add_argument('--manifest',help='path override with identical PLY/split')
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=2000,help='ADDITIONAL real updates PER ARM')
    p.add_argument('--lr',type=float,help='optional identical LR override; otherwise retain source Adam LR')
    p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--diagnostic-points',type=int,default=32768)
    p.add_argument('--diagnostic-trials',type=int,default=8)
    p.add_argument('--diagnostic-batch',type=int,default=2048)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--resume-suite',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--suite',choices=['inverse','prefix','spatial','cooperative'],default='inverse',help='opt-in channel controls')
    p.add_argument('--denoiser-hidden',type=int,default=64)
    p.add_argument('--denoiser-neighbors',type=int,default=16)
    p.add_argument('--arm',choices=(*ARMS,*PREFIX_ARMS,*SPATIAL_ARMS,*COOPERATIVE_ARMS),help='internal single-arm entry; normally omit')
    return p


def source_files(args):
    run=Path(args.run).resolve()
    if args.suite in ('spatial','cooperative') and not (run/'training.json').exists() and (run/'full_prefix10/training.json').exists():
        run=run/'full_prefix10'
    if not (run/'training.json').exists() and (run/'progressive/training.json').exists(): run=run/'progressive'
    metadata=json.loads((run/'training.json').read_text(encoding='utf-8'))
    checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else run/'checkpoints/final/codec.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    training=saved.get('training',{}); progress=training.get('progress',{})
    if (saved.get('version')!=4 or training.get('modular_version')!=1 or
            progress.get('phase')!='channel_bootstrap' or not progress.get('phase_complete') or
            progress.get('completed',{}).get('channel_bootstrap',0)<1 or
            progress.get('completed',{}).get('channel_render',0) or
            not progress.get('freeze_semantic') or not metadata.get('freeze_semantic') or
            metadata.get('phase_budgets',{}).get('channel_render',0) or
            progress.get('train_snrs')!=[10.] or training.get('optimizer') is None):
        raise ValueError('source requires completed frozen-semantic pure-y 10 dB training with Adam state')
    if json.loads(json.dumps(saved['config']))!=metadata['codec_config']:
        raise ValueError('source checkpoint/config mismatch')
    validate_scene(training['scene_record'],metadata['scene'])
    return run,checkpoint,metadata,saved


def build_plan(args):
    if (min(args.steps,args.validate_every,args.save_every,args.diagnostic_points,args.diagnostic_batch,args.cpu_threads)<1
            or args.diagnostic_trials<2 or (args.lr is not None and (not math.isfinite(args.lr) or args.lr<=0))):
        raise ValueError('invalid continuation or diagnostic settings')
    run,checkpoint,metadata,saved=source_files(args)
    arms=arms_for(args)
    if args.arm is not None and args.arm not in arms: raise ValueError('arm does not belong to selected suite')
    if args.suite=='prefix' and (metadata['codec_config'].get('channel_variant')!='progressive' or
                                metadata['codec_config'].get('channel_prefix_mode','suffix')!='suffix'):
        raise ValueError('prefix controls require an original progressive suffix checkpoint')
    if args.suite in ('spatial','cooperative') and (metadata['codec_config'].get('channel_prefix_mode')!='full_prefix' or
            metadata['codec_config'].get('channel_denoiser_mode','none')!='none' or metadata.get('channel_kind')!='awgn' or
            min(args.denoiser_hidden,args.denoiser_neighbors)<1):
        raise ValueError('spatial controls require a completed 10 dB full-prefix source without a denoiser')
    destination=Path(args.out).resolve()
    for protected in (Path(args.run).resolve(),run):
        if destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise ValueError('output must be separate from the read-only source run/suite')
    if checkpoint.is_relative_to(destination):
        raise ValueError('output must not contain the source checkpoint')
    source_step=saved['training']['progress']['step']
    common=['--run',str(run),'--out',str(Path(args.out).resolve()),'--checkpoint',str(checkpoint)]
    for key in ('manifest','device','steps','lr','validate_every','save_every','diagnostic_points','diagnostic_trials',
                'diagnostic_batch','seed','cpu_threads'):
        value=getattr(args,key)
        if value is not None: common+=['--'+key.replace('_','-'),str(value)]
    if args.suite!='inverse': common+=['--suite',args.suite]
    if args.suite in ('spatial','cooperative'):common+=['--denoiser-hidden',str(args.denoiser_hidden),'--denoiser-neighbors',str(args.denoiser_neighbors)]
    plan=dict(version=1,source_run=str(run),checkpoint=str(checkpoint),
        source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_step=source_step,
        target_step=source_step+args.steps,additional_updates_per_arm=args.steps,rates=metadata['codec_config']['rates'],
        optimizer='inherit source Adam parameters, step counters and moments; no reset',
        semantic_modules='frozen; original XYZ path and semantic context unchanged',
        conditioning_snr=10.,sampling='private deterministic per-update point batch and positive-tier mask; independent of AWGN RNG',
        diagnostics=dict(points=args.diagnostic_points,trials=args.diagnostic_trials,batch_size=args.diagnostic_batch,seed=args.seed),
        jobs=[dict(arm=arm,kind=kind,command=[sys.executable,'-u','-m','gaussian_jscc.channel_inverse_controls',*common,'--arm',arm])
              for arm,kind in arms.items()])
    if args.suite=='prefix':
        plan['prefix_control']=dict(readout=['suffix','full_prefix'],
            initialization='function-preserving zero earlier-prefix projection; existing Adam retained; new weights have fresh state',
            parameter_matched=False,transmitted_payload_matched=True)
    if args.suite in ('spatial','cooperative'):
        plan['optimizer']='new Adam for zero-output denoiser branches ONLY; original source modules frozen'
        plan['receiver_context']=dict(modes=['pointwise','spatial'],hidden=args.denoiser_hidden,neighbors=args.denoiser_neighbors,
            initialization='identical hidden weights; zero residual outputs; no clean y at receiver',
            scope='original codec blocks; reliable 16bit XYZ; current-stage prefixes only',
            parameter_matched=True,flop_matched=False,transmitted_payload_matched=True)
        if args.suite=='cooperative':
            plan['receiver_context'].update(modes=list(COOPERATIVE_ARMS),
                scope='stage-spatial reference vs final-only pointwise/spatial using every actually received neighbor prefix',
                parameter_matched=False,final_pair_parameter_matched=True,
                stage_reference='three stage branches vs one final branch; not a capacity-controlled causal comparison')
    return plan


def training_batch(scene, phase_step, seed):
    """Same point batch and tier map in both arms despite different noise draws."""
    local=random.Random(seed+100003*phase_step)
    index=local.randrange(len(scene.groups)); ids=scene.group_ids[index]
    layout=('1','2','3','mixed')[(phase_step-1)%4]
    if layout=='mixed':
        values=torch.randint(1,4,ids.shape,generator=torch.Generator().manual_seed(seed+200003*phase_step)).to(ids.device)
    else: values=torch.full_like(ids,int(layout))
    return index,scene.groups[index],torch.where(ids>=0,values,0),layout


def evaluate_endpoint(model,scene,folder,phase,step,seed):
    """Publish a complete endpoint once; replay replaces rather than appends."""
    with tempfile.TemporaryDirectory(prefix='endpoint_'+phase+'_',dir=folder) as directory:
        temporary=Path(directory)
        evaluate_paths(model,scene,step=step,phase=phase,snrs=[10],trials=2,
                       seed=seed,save_images=True,out=temporary,color_metrics=True)
        path=folder/'metrics/validation.jsonl'
        existing=path.read_text(encoding='utf-8').splitlines() if path.exists() else []
        retained=[line for line in existing if json.loads(line)['phase']!=phase]
        complete=(temporary/'metrics/validation.jsonl').read_text(encoding='utf-8').splitlines()
        pending=path.with_name(path.name+'.endpoint.tmp')
        pending.write_text('\n'.join(retained+complete)+'\n',encoding='utf-8'); pending.replace(path)
        for image in (temporary/'images'/phase).glob('*.png'):
            target=folder/'images'/phase/image.name; target.parent.mkdir(parents=True,exist_ok=True); image.replace(target)


def run_arm(args,arm,max_updates=None):
    arms=arms_for(args)
    if arm not in arms: raise ValueError('unknown arm for selected suite')
    plan=build_plan(args); run,checkpoint,metadata,source_saved=source_files(args)
    folder=Path(args.out).resolve()/arm; kind=arms[arm]
    settings=dict(plan=plan,arm=arm,channel_kind=kind)
    if folder.exists() and not args.resume_suite: raise FileExistsError('use a NEW output or --resume-suite')
    resumed=False
    if folder.exists():
        control=folder/'control.json'
        if not control.exists() or json.loads(control.read_text(encoding='utf-8'))!=settings:
            raise ValueError('unrecognized or mismatched continuation directory; refusing to overwrite')
        if (folder/'training.json').exists():
            previous=json.loads((folder/'training.json').read_text(encoding='utf-8'))
            if previous['inverse_control']!=settings: raise ValueError('exact continuation settings mismatch')
        if (folder/'complete.json').exists(): return json.loads((folder/'complete.json').read_text())
        latest=folder/'checkpoints/latest/codec.pt'
        resumed=latest.exists()
        if not resumed and (folder/'metrics/loss.jsonl').exists():
            raise ValueError('updates exist without a resumable checkpoint')
        loading=latest if resumed else checkpoint
    else:
        json_write(folder/'control.json',settings)
        loading=checkpoint
    device=device_for(args.device); torch.set_num_threads(args.cpu_threads)
    model,optimizer,progress,rng,saved_scene=restore_training_checkpoint(loading,device)
    expected=copy.deepcopy(metadata['codec_config'])
    if arm=='full_prefix10':
        expected['channel_prefix_mode']='full_prefix'
        if not resumed:
            from .progressive_channel import enable_full_prefix
            enable_full_prefix(model,optimizer)
    if args.suite in ('spatial','cooperative'):
        from .channel_spatial import enable_channel_denoising
        from .modular_training import configure_phase
        expected.update(channel_denoiser_mode='pointwise' if arm in ('pointwise10','final_pointwise10') else 'spatial',
                        channel_denoiser_hidden=args.denoiser_hidden,channel_denoiser_neighbors=args.denoiser_neighbors)
        scope='final' if arm.startswith('final_') else 'stage'
        if scope=='final':expected['channel_denoiser_scope']=scope
        if not resumed:
            # New hidden weights are paired across independently launched arms.
            restore_rng(rng)
            enable_channel_denoising(model,expected['channel_denoiser_mode'],hidden=args.denoiser_hidden,
                                     neighbors=args.denoiser_neighbors,scope=scope)
            progress=copy.deepcopy(progress);progress['channel_denoiser_only']=True
            progress['module_lrs']['channel_denoiser']=args.lr or progress['module_lrs'].get('channel_decoder',1e-4)
            optimizer=configure_phase(model,'channel_bootstrap',progress['module_lrs'],freeze_semantic=True,denoiser_only=True)
    if json.loads(json.dumps(model.cfg.to_dict()))!=expected: raise ValueError('resume codec mismatch')
    if resumed and progress.get('inverse_control')!=settings: raise ValueError('checkpoint continuation mismatch')
    if args.lr is not None:
        for group in optimizer.param_groups:
            group['lr']=args.lr; progress['module_lrs'][group['name']]=args.lr
    source,manifest_path,spec,old,_=control_inputs(metadata,args.manifest)
    raw,degree=read_ply(spec['ply']); scene=make_scene(spec,raw,degree,model,source,old,folder); del raw
    validate_scene(saved_scene,scene.record)
    scene.channel_trained=True
    if not resumed:
        progress=copy.deepcopy(progress)
        progress.update(phase_complete=False,inverse_control=settings)
        progress['phase_budgets']['channel_bootstrap']+=args.steps
        record=copy.deepcopy(metadata)
        record.update(source_run=str(run),manifest=str(manifest_path),inverse_control=settings,
            codec_config=model.cfg.to_dict(),
            parameters={name:sum(p.numel() for p in getattr(model,name).parameters()) for name in metadata['parameters']},
            phase_budgets=progress['phase_budgets'],module_lrs=progress['module_lrs'],
            initialization='channel continuation; full source weights AND Adam retained',
            initializer=str(checkpoint),source_weights_loaded=True,random_initialization=False,
            source_usage='completed channel weights, Adam and scene identity; source files are read-only',
            channel_kind=kind,clip_mode='none',color_metrics=True)
        json_write(folder/'training.json',record)
        if args.suite in ('spatial','cooperative'):
            record.update(initialization='full-prefix source weights frozen; only new zero-output denoisers have fresh Adam',
                          source_usage='completed source weights and scene identity; new denoisers use fresh Adam; source files are read-only',
                          channel_denoiser_only=True,
                          trainable_modules={p:(['channel_decoder.denoisers'] if p=='channel_bootstrap' else []) for p in record['trainable_modules']})
            json_write(folder/'training.json',record)
    else: repair_logs(folder,progress['step'])
    restore_rng(rng)

    def save(name):
        save_training_checkpoint(folder/'checkpoints'/name/'codec.pt',model,optimizer,progress,capture_rng(),scene.record)
        json_write(folder/'progress.json',progress)

    if not resumed:
        save('initial'); save('latest')
    if not (folder/'diagnostics/initial/complete.json').exists():
        if progress['step']!=plan['source_step']:
            raise ValueError('missing initial diagnostics after updates; refusing to label trained weights as initial')
        diagnose(model,scene,folder/'diagnostics/initial',points=args.diagnostic_points,trials=args.diagnostic_trials,
                 batch_size=args.diagnostic_batch,seed=args.seed)
        evaluate_endpoint(model,scene,folder,'initial',progress['step'],args.seed)
        json_write(folder/'diagnostics/initial/complete.json',dict(step=progress['step'],status='complete'))
    updates=0
    while progress['step']<plan['target_step']:
        phase_step=progress['completed']['channel_bootstrap']+1
        index,features,q,layout=training_batch(scene,phase_step,args.seed)
        model.train(); optimizer.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        if device.type=='cuda': torch.cuda.synchronize(device)
        started=time.perf_counter()
        loss,stats=bootstrap_step(model,features.to(device),q.to(device),scene.geometry,10.,kind,'channel_bootstrap')
        stats.update(finish_update(model,optimizer,before))
        if device.type=='cuda': torch.cuda.synchronize(device)
        progress['step']+=1; progress['phase_step']=phase_step; progress['completed']['channel_bootstrap']=phase_step
        progress['phase_complete']=progress['step']==plan['target_step']
        continuation_step=progress['step']-plan['source_step']
        row=dict(stats,step=progress['step'],phase_step=phase_step,continuation_step=continuation_step,
                 phase='channel_bootstrap',loss=float(loss),layout=layout,batch_index=index,channel_kind=kind,
                 condition_snr=10.,actual_snr=10. if kind=='awgn' else None,step_seconds=time.perf_counter()-started,
                 module_lrs={g['name']:g['lr'] for g in optimizer.param_groups},frozen_parameters_unchanged=True)
        from .render_validation import append_json
        append_json(folder/'metrics/loss.jsonl',row)
        if continuation_step==1 or continuation_step%10==0 or progress['phase_complete']:
            print(f"{arm} {continuation_step}/{args.steps} (total {progress['step']}): "
                  f"q={layout}, loss={float(loss):.6g}, NMSE={stats['latent_nmse']:.6g}, "
                  f"grad={stats['grad_norm']:.4g}, sec={row['step_seconds']:.2f}",flush=True)
        if continuation_step%args.validate_every==0:
            evaluate_paths(model,scene,step=progress['step'],phase='channel_bootstrap',snrs=[10],trials=2,
                           seed=args.seed,save_images=False,out=folder,color_metrics=True)
        if continuation_step%args.save_every==0: save('latest')
        updates+=1
        if max_updates is not None and updates>=max_updates: save('latest'); return progress
    # Save before heavy read-only evaluation so an interrupted endpoint is resumable.
    save('end_channel_bootstrap'); save('latest')
    diagnose(model,scene,folder/'diagnostics/final',points=args.diagnostic_points,trials=args.diagnostic_trials,
             batch_size=args.diagnostic_batch,seed=args.seed)
    evaluate_endpoint(model,scene,folder,'final',progress['step'],args.seed)
    save('final'); save('latest')
    done=dict(status='complete',source_step=plan['source_step'],additional_updates=args.steps,
              total_step=progress['step'],channel_kind=kind,optimizer_moments_retained=args.suite not in ('spatial','cooperative'),semantic_modules_frozen=True)
    if args.suite in ('spatial','cooperative'):done.update(source_modules_frozen=True,only_new_denoisers_trained=True)
    json_write(folder/'complete.json',done)
    return done


def summarize(out):
    from .modular_plots import read_rows
    out=Path(out); destination=out/'comparison'; destination.mkdir(parents=True,exist_ok=True)
    arms=(COOPERATIVE_ARMS if (out/'stage_spatial10').is_dir() else SPATIAL_ARMS if (out/'spatial10').is_dir()
          else PREFIX_ARMS if (out/'full_prefix10').is_dir() else ARMS)
    groups=list(arms); first,second=groups[:2]
    records={arm:read_rows(out/arm/'metrics/validation.jsonl') for arm in arms}
    finals={arm:{(r['path'],r['layout']):r for r in rs if r['phase']=='final'} for arm,rs in records.items()}
    comparison=[]
    for path in ('finite_noiseless','awgn'):
        for tier in ('1','2','3','mixed'):
            pair={arm:finals[arm][(path,tier)] for arm in arms}
            row=dict(path=path,tier=tier)
            for arm,r in pair.items():
                row[arm]={k:r[k] for k in ('source_psnr','photo_psnr','source_ssim','latent_mse','latent_nmse','symbols_per_gaussian')}
            if arms is COOPERATIVE_ARMS:
                row['final_spatial_minus_final_pointwise_psnr_db']=pair['final_spatial10']['source_psnr']-pair['final_pointwise10']['source_psnr']
                row['final_spatial_minus_stage_spatial_psnr_db']=pair['final_spatial10']['source_psnr']-pair['stage_spatial10']['source_psnr']
            else:
                delta_key=('spatial_minus_pointwise_psnr_db' if arms is SPATIAL_ARMS else
                           'full_prefix_minus_suffix_psnr_db' if arms is PREFIX_ARMS else 'noiseless_minus_awgn10_psnr_db')
                row[delta_key]=(pair[second]['source_psnr']-pair[first]['source_psnr'] if arms is not ARMS
                                else pair[first]['source_psnr']-pair[second]['source_psnr'])
            if arms is SPATIAL_ARMS or arms is COOPERATIVE_ARMS:
                baseline=next(r for r in records[first] if r['phase']=='initial' and r['path']==path and r['layout']==tier)
                row['full_prefix_source']={k:baseline[k] for k in ('source_psnr','photo_psnr','latent_mse','latent_nmse')}
                for arm,r in pair.items():row[arm+'_minus_source_psnr_db']=r['source_psnr']-baseline['source_psnr']
            comparison.append(row)
    # Compare all arms to their identical baseline; do not silently mix view sets.
    for rs in records.values():
        initial=[r for r in rs if r['phase']=='initial']
        for r in initial:
            final=next(v for v in rs if v['phase']=='final' and v['path']==r['path'] and v['layout']==r['layout'])
            if r['views']!=final['views']: raise ValueError('comparison view count changed')
    initial={arm:{(r['path'],r['layout']):r for r in records[arm] if r['phase']=='initial'} for arm in arms}
    matched=True
    required={('clean','clean')}|{(p,t) for p in ('finite_noiseless','awgn') for t in ('1','2','3','mixed')}
    if any(set(initial[a])!=required for a in arms):raise ValueError('missing initial control metrics')
    for arm in groups[1:]:
        if initial[first].keys()!=initial[arm].keys(): matched=False
        else:
            for key in initial[first]:
                for metric in ('latent_mse','source_psnr','photo_psnr'):
                    if not math.isclose(initial[first][key][metric],initial[arm][key][metric],rel_tol=1e-6,abs_tol=1e-6): matched=False
    if not matched: raise ValueError('control initial functions do not match; comparison is not a paired warm start')
    diagnostics={arm:{endpoint:json.loads((out/arm/'diagnostics'/endpoint/'latent_errors.json').read_text(encoding='utf-8'))
                      for endpoint in ('initial','final')} for arm in arms}
    hashes={d['sample_sha256'] for stages in diagnostics.values() for d in stages.values()}
    if len(hashes)!=1: raise ValueError('diagnostic point sample changed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={first:('#356d9b','-','o'),second:('#d88931','--','s')}
    if arms is COOPERATIVE_ARMS:styles[groups[2]]=('#3c9465','-.','^')
    tiers=('1','2','3','mixed') if arms is COOPERATIVE_ARMS else ('1','2','3')
    fig,axes=plt.subplots(2,len(tiers),figsize=(4*len(tiers)+1,7),layout='constrained')
    for arm,rs in records.items():
        color,line,marker=styles[arm]
        for row,path in enumerate(('finite_noiseless','awgn')):
            for col,tier in enumerate(tiers):
                series={r['step']:r for r in rs if r['path']==path and r['layout']==tier}
                steps=sorted(series)
                axes[row,col].plot(steps,[series[s]['source_psnr'] for s in steps],color=color,linestyle=line,marker=marker,label=arm)
                axes[row,col].set(title=f'{path} q{tier}',xlabel='Total optimizer steps',ylabel='Source PLY PSNR (dB)')
                axes[row,col].grid(alpha=.2); axes[row,col].legend(fontsize=8)
    fig.savefig(destination/'psnr_curves.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    labels=[]; values=[[],[],[]]
    for arm,stages in diagnostics.items():
        for endpoint,d in stages.items():
            for tier in d['tiers']:
                labels.append(f'{arm}\n{endpoint}\nq{tier["tier"]}')
                for col,key in enumerate(('noiseless_nmse','pointwise_mean_error_mse','noise_variance')):
                    values[col].append(tier['summary'][key])
    for ax,key,values_col in zip(axes,('Noiseless NMSE','Finite-trial point mean error MSE','Within-point noise variance'),values):
        ax.bar(range(len(labels)),values_col,color='#356d9b'); ax.set(xticks=range(len(labels)),xticklabels=labels,title=key)
        ax.tick_params(axis='x',labelsize=6); ax.grid(axis='y',alpha=.2)
    fig.savefig(destination/'latent_errors.png',dpi=150); plt.close(fig)
    result=dict(status='complete',groups=groups,matched_initial_function=matched,comparison=comparison,diagnostic_sample_sha256=next(iter(hashes)),
                warning='one training checkpoint; finite-trial bias estimates are not independent training replications')
    if arms is COOPERATIVE_ARMS:
        result['mixed_latent_by_tier']={arm:{endpoint:dict(overall=d['mixed']['overall']['summary'],
            by_tier=[dict(tier=v['tier'],points=v['points'],summary=v['summary']) for v in d['mixed']['by_tier']],
            sampling=d['mixed']['sampling'],symbols_per_sampled_gaussian=d['mixed']['symbols_per_sampled_gaussian'])
            for endpoint,d in stages.items()} for arm,stages in diagnostics.items()}
        result['control_scope']='final pair: equal nominal parameters and initialization; stage reference has three branches, final pair one'
    json_write(destination/'summary.json',result)
    return result


def main(args):
    plan=build_plan(args); out=Path(args.out).resolve()
    if args.dry_run:
        if args.arm: plan=dict(plan,jobs=[j for j in plan['jobs'] if j['arm']==args.arm])
        print(json.dumps(plan,indent=2)); return plan
    if args.arm: return run_arm(args,args.arm)
    if out.exists():
        if not args.resume_suite: raise FileExistsError('use a NEW suite directory or --resume-suite')
        if json.loads((out/'plan.json').read_text(encoding='utf-8'))!=plan: raise ValueError('suite plan differs')
    else: json_write(out/'plan.json',plan)
    for job in plan['jobs']:
        arm=job['arm']; command=list(job['command'])
        if (out/arm/'complete.json').exists(): continue
        if args.resume_suite: command+=['--resume-suite']
        print(f'Starting {arm}; log: {out/(arm+".log")}',flush=True)
        with (out/(arm+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{arm} failed; inspect {out/(arm+".log")}')
    summary=summarize(out)
    json_write(out/'complete.json',dict(status='complete',additional_updates_per_arm=args.steps,groups=list(arms_for(args))))
    print(f'Complete: {out}/comparison/summary.json',flush=True)
    return summary


if __name__=='__main__': main(parser().parse_args())
