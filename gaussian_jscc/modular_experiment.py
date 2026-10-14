"""Truck-only modular JSCC: random representation, then finite channel training."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from .cli import seed_all,device_for
from .data import read_ply
from .multiscene import Scene,manifest,shared_statistics,parser as scene_parser
from .modular_config import ModularCodecConfig
from .modular_codec import ModularGaussianCodec
from .modular_training import MODULES,PHASE_MODULES,configure_phase,bootstrap_step,color_bootstrap_step,render_step,finish_update,objective_settings
from .modular_checkpoint import (capture_rng,restore_rng,save_training_checkpoint,restore_training_checkpoint,
                                initialize_channel_run,validate_scene)
from .modular_evaluation import evaluate_paths
from .render_objective import MultiViewRenderTask
from .render_validation import append_json

PHASES=tuple(PHASE_MODULES)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='source metadata only; no pretrained codec is loaded')
    p.add_argument('--out',required=True)
    p.add_argument('--manifest',help='override scene paths after migration; identity/split must match')
    p.add_argument('--device',default='cuda')
    p.add_argument('--decoder-geometry',choices=['xyz','none'],default='xyz')
    p.add_argument('--latent-dim',type=int,default=48)
    p.add_argument('--context-variant',choices=['baseline','encoder','decoder','both'],default='baseline',
                   help='opt-in PointNet++-style spatial context experiment')
    p.add_argument('--context-centers',nargs=2,type=int,default=[64,16])
    p.add_argument('--context-neighbors',nargs=2,type=int,default=[16,32])
    p.add_argument('--context-radii',nargs=2,type=float,default=[2.,4.])
    p.add_argument('--blocks-per-batch',type=int,help='override source batch size, identically in all control arms')
    p.add_argument('--channel-variant',choices=['baseline','residual','progressive'],default='baseline')
    p.add_argument('--channel-hidden',type=int,default=192,help='width of opt-in residual/progressive channel modules')
    p.add_argument('--rates',type=int,nargs=3,default=[4,12,24],metavar=('Q1','Q2','Q3'),
                   help='cumulative complex-symbol budgets per Gaussian; q0 remains zero')
    p.add_argument('--freeze-semantic',action='store_true',help='channel-only control; freeze both semantic modules in ALL communication updates')
    for phase,steps in zip(PHASES,(2000,1000,2000,2000)):
        p.add_argument('--'+phase.replace('_','-')+'-steps',type=int,default=steps)
    for name in MODULES:
        p.add_argument('--'+name.replace('_','-')+'-lr',type=float,default=1e-4)
    p.add_argument('--train-snrs',nargs='+',type=float,default=[10.])
    p.add_argument('--channel-latent-weight',type=float,default=1.,
                   help='positive latent MSE weight in communication render training; tunable engineering coefficient')
    p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--resume',help='exact optimizer/RNG/progress continuation in SAME out directory')
    group.add_argument('--init-semantic',help='completed semantic_render checkpoint; new communication experiment')
    p.add_argument('--with-direct-control',action='store_true',help='additional random 3000+2000 direct control; not FLOP matched')
    p.add_argument('--clean-only',action='store_true',help='no channel training, invocation or communication evaluation')
    p.add_argument('--semantic-color-every',type=int,default=0,
                   help='every Nth semantic bootstrap update uses independent unclipped direction RGB MSE; 0 disables')
    p.add_argument('--semantic-color-directions',type=int,default=8,help='even number; half training cameras, half uniform sphere')
    p.add_argument('--color-metrics',action='store_true',help='log uncorrected RGB bias and channel-difference errors')
    p.add_argument('--save-semantic-end-images',action='store_true',help='also save representative images after semantic bootstrap')
    return p


def json_write(path,record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def inputs(args):
    run=Path(args.run).resolve()
    source=json.loads((run/'training.json').read_text(encoding='utf-8'))
    source_manifest=Path(args.manifest or source['manifest'])
    if not source_manifest.is_absolute(): source_manifest=Path.cwd()/source_manifest
    specs=manifest(source_manifest)
    selected=[s for s in specs if s['name']=='truck' and s['role']=='train']
    if len(selected)!=1 or 'truck' not in [s['name'] for s in source['selected_scenes']]:
        raise ValueError('Truck must be a trained scene in source run and manifest')
    spec=dict(selected[0],prior='none')
    for key in ('ply','source'):
        if not Path(spec[key]).exists(): raise FileNotFoundError(spec[key])
    if source.get('codec_config',{}).get('architecture')=='modular_jscc_v1':
        old=source['scene']
        source={**source,**old}
    else:
        old=json.loads((run/'scenes/truck/scene.json').read_text(encoding='utf-8'))
    for key in ('fingerprint','source_gaussians','train_view_names','validation_view_names'):
        if not old.get(key): raise ValueError('source scene lacks '+key)
    return source,source_manifest.resolve(),spec,old


def make_scene(spec,raw,degree,model,source,old,out):
    from .rendering import load_cameras,RenderReference
    opts=scene_parser().parse_args(['--out',str(out)])
    for key in ('blocks_per_batch','resolution','images','white_background','views_per_step','net_bits_per_use'):
        if key in source: setattr(opts,key,source[key])
    scene=Scene(spec,raw,degree,model,opts,out/'metrics',with_mask=False,cameras=False)
    model.set_scene_geometry(scene.geometry)
    if scene.fingerprint!=old['fingerprint'] or len(raw)!=old['source_gaussians']:
        raise ValueError('source PLY differs from recorded Truck')
    cameras=load_cameras(spec['source'],opts.resolution,opts.white_background,opts.images,'train')
    names=[str(c.image_name) for c in cameras]
    if len(set(names))!=len(names): raise ValueError('duplicate camera names')
    lookup=dict(zip(names,cameras)); tn=old['train_view_names']; vn=old['validation_view_names']
    if set(tn)&set(vn) or not set(tn+vn).issubset(lookup): raise ValueError('source camera split changed')
    scene.cameras=[lookup[n] for n in tn]; scene.val_cameras=[lookup[n] for n in vn]
    if opts.views_per_step>len(scene.cameras): raise ValueError('insufficient training cameras')
    scene.reference=RenderReference(scene.raw,degree,opts.white_background,'source')
    scene.white_background=opts.white_background; scene.net_bits_per_use=opts.net_bits_per_use
    scene.views_per_step=opts.views_per_step
    scene.record={'fingerprint':scene.fingerprint,'source_gaussians':len(raw),'sh_degree':degree,
                  'geometry':scene.geometry.to_dict(),'train_view_names':tn,'validation_view_names':vn,
                  'resolution':opts.resolution,'images':opts.images,'white_background':opts.white_background,
                  'blocks_per_batch':opts.blocks_per_batch,'views_per_step':opts.views_per_step}
    json_write(scene.out/'scene.json',dict(scene.record,**spec))
    return scene


def tier_batches(scene,layout):
    return [torch.where(ids>=0,torch.randint(1,4,ids.shape) if layout=='mixed' else int(layout),0)
            for ids in scene.group_ids]


def repair_logs(out,step):
    # A process may have logged uncheckpointed updates before interruption.
    # Archive those rows before truncating so exact resume has one row/update.
    for name in ('loss','validation','test'):
        path=out/'metrics'/f'{name}.jsonl'
        if path.exists():
            lines=path.read_text(encoding='utf-8').splitlines()
            retained=[line for line in lines if json.loads(line)['step']<=step]
            if retained!=lines:
                archive=path.with_name(f'{name}.uncheckpointed_{time.time_ns()}.jsonl')
                archive.write_text('\n'.join(line for line in lines if json.loads(line)['step']>step)+'\n',encoding='utf-8')
                path.write_text('\n'.join(retained)+'\n',encoding='utf-8')


def direct_control(args,out):
    """Existing direct path, identical source split/XYZ assistance, fresh weights."""
    import subprocess
    import sys
    from .residual_controls import build_plan
    options=argparse.Namespace(run=args.run,out=str(out/'direct_control'),manifest=args.manifest,groups=['direct'],
        bootstrap_steps=3000,render_steps=2000,validate_every=args.validate_every,lr=args.semantic_encoder_lr,
        test_views=0,trials=2,image_views=2,color_rois=None,device=args.device,dry_run=False)
    plan=build_plan(options)
    for job in plan['jobs']:
        command=job['command']
        if job['kind']=='fit': command += ['--decoder-geometry',args.decoder_geometry]
        subprocess.run(command,check=True)
    json_write(out/'direct_control/comparison_design.json',{'plan':plan,'capacity_matched':False,'flop_matched':False})


def main(args,max_updates=None):
    from .rendering import load_cameras
    objectives=objective_settings(args.channel_latent_weight)
    counts={phase:getattr(args,phase+'_steps') for phase in PHASES}
    if args.freeze_semantic and not (args.init_semantic or args.resume):
        raise ValueError('freeze-semantic requires a semantic initializer or exact resume')
    if args.clean_only and (counts['channel_bootstrap'] or counts['channel_render'] or args.init_semantic or args.with_direct_control):
        raise ValueError('clean-only requires zero channel budgets, no semantic initializer or direct control')
    if args.semantic_color_every<0 or args.semantic_color_directions<2 or args.semantic_color_directions%2:
        raise ValueError('color interval must be nonnegative; direction count must be even and >=2')
    if args.semantic_color_every:
        objectives=dict(objectives,semantic_bootstrap_color={'version':'directional_color_mse_unclipped_v1',
            'every':args.semantic_color_every,'directions':args.semantic_color_directions,
            'sampling':'half training camera-to-point; half uniform sphere; unclipped; no alpha or footprint'})
    if args.clean_only: objectives=dict(objectives,clean_only=True)
    evaluation_options={'clean_only':args.clean_only,'color_metrics':args.color_metrics,
                        'save_phase_images':args.save_semantic_end_images}
    if min(counts.values())<0 or min(args.validate_every,args.save_every,args.cpu_threads)<1 or not args.train_snrs or any(not math.isfinite(s) for s in args.train_snrs):
        raise ValueError('invalid step budgets, intervals or training SNRs')
    lrs={name:getattr(args,name+'_lr') for name in MODULES}
    if any(not math.isfinite(v) or v<=0 for v in lrs.values()): raise ValueError('invalid module learning rate')
    if args.init_semantic: counts.update(semantic_bootstrap=0,semantic_render=0)
    out=Path(args.out).resolve()
    if out.exists() and not args.resume: raise FileExistsError(f'use a NEW output directory: {out}')
    if args.resume and not out.exists(): raise FileNotFoundError('resume requires original output directory')
    source,manifest_path,spec,old=inputs(args)
    if args.blocks_per_batch is not None:
        if args.blocks_per_batch<1: raise ValueError('blocks per batch must be positive')
        source=dict(source,blocks_per_batch=args.blocks_per_batch)
    device=device_for(args.device); torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    raw,degree=read_ply(spec['ply'])
    common=source.get('codec_config',{})
    context = {} if args.context_variant=='baseline' else dict(context_centers=tuple(args.context_centers),
        context_neighbors=tuple(args.context_neighbors),context_radii=tuple(args.context_radii))
    cfg=ModularCodecConfig(sh_degree=degree,latent_dim=args.latent_dim,decoder_geometry=args.decoder_geometry,
        channel_variant=args.channel_variant,channel_hidden=args.channel_hidden,rates=(0,*args.rates),
        context_variant=args.context_variant,**context,
        **{k:common[k] for k in ('hidden','grid_dim','levels','planes','depth','block_size','morton_bits','decoder_window','attention_heads') if k in common})
    resumed_rng=saved_scene=None
    if args.resume:
        previous=json.loads((out/'training.json').read_text(encoding='utf-8'))
        if previous.get('freeze_semantic',False)!=args.freeze_semantic:
            raise ValueError('exact resume freeze-semantic mismatch')
        if previous.get('training_objective')!=objectives:
            raise ValueError('exact resume objective mismatch; start a NEW run with --init-semantic from end_semantic_render instead')
        # A semantic-only fork skips these phases permanently, including when
        # resumed without the mutually exclusive --init-semantic option.
        if previous.get('initializer'):
            counts.update(semantic_bootstrap=0,semantic_render=0)
        for key,current in [('phase_budgets',counts),('module_lrs',lrs),('train_snrs',args.train_snrs),('seed',args.seed)]:
            if previous[key]!=current: raise ValueError('exact resume settings mismatch: '+key)
        for key in ('color_metrics','save_semantic_end_images'):
            if previous.get(key,False)!=getattr(args,key): raise ValueError('exact resume evaluation settings mismatch: '+key)
        model,optimizer,progress,resumed_rng,saved_scene=restore_training_checkpoint(args.resume,device)
        if progress.get('training_objective')!=objectives:
            raise ValueError('checkpoint objective mismatch')
        if model.cfg!=cfg: raise ValueError('exact resume codec config mismatch')
    else:
        model=initialize_channel_run(args.init_semantic,device,cfg=cfg) if args.init_semantic else ModularGaussianCodec(cfg).to(device)
        if model.cfg!=cfg: raise ValueError('semantic initialization config mismatch')
        if not args.init_semantic:
            mean,std=shared_statistics([raw]); model.attr_mean.copy_(mean.to(device)); model.attr_std.copy_(std.to(device))
        optimizer=None
        progress={'phase':'semantic_bootstrap','phase_step':0,'step':0,'completed':{p:0 for p in PHASES},
                  'channel_trained':False,'module_lrs':lrs,'phase_complete':False,'phase_budgets':counts,'train_snrs':args.train_snrs,
                  'training_objective':objectives}
        if args.freeze_semantic: progress['freeze_semantic']=True
    model.set_training_snrs(args.train_snrs)
    scene=make_scene(spec,raw,degree,model,source,old,out); del raw
    color_camera_centers=None
    if args.semantic_color_every:
        color_camera_centers=torch.stack([torch.as_tensor(c.camera_center).detach().cpu() for c in scene.cameras]).to(device)
    if saved_scene is not None: validate_scene(saved_scene,scene.record)
    if args.init_semantic:
        init=torch.load(args.init_semantic,map_location='cpu',weights_only=True)
        validate_scene(init['training']['scene_record'],scene.record)
    if args.resume:
        repair_logs(out,progress['step']); restore_rng(resumed_rng)
    else:
        for folder in ('metrics','checkpoints','charts','images'): (out/folder).mkdir(parents=True,exist_ok=True)
        json_write(out/'training.json',{'source_run':str(Path(args.run).resolve()),'manifest':str(manifest_path),
            'selected_scenes':[spec],'codec_config':cfg.to_dict(),'phase_budgets':counts,'module_lrs':lrs,'train_snrs':args.train_snrs,
            'seed':args.seed,'random_initialization':not bool(args.init_semantic),'initializer':args.init_semantic,
            'source_weights_loaded':False,'reference_scene_identity_and_splits_verified':True,
            'scene':scene.record,'net_bits_per_use':scene.net_bits_per_use,
            'step_definition':'one real optimizer update; no accumulation',
            'trainable_modules':{p:[n for n in ns if not (args.freeze_semantic and n.startswith('semantic_'))] for p,ns in PHASE_MODULES.items()},
            'freeze_semantic':args.freeze_semantic,
            'parameters':{n:sum(p.numel() for p in getattr(model,n).parameters()) for n in MODULES},
            'training_objective':objectives,
            'clean_only':args.clean_only,'color_metrics':args.color_metrics,'save_semantic_end_images':args.save_semantic_end_images,
            'semantic_color_every':args.semantic_color_every,'semantic_color_directions':args.semantic_color_directions,
            'color_update_counts':{'directional':counts['semantic_bootstrap']//args.semantic_color_every if args.semantic_color_every else 0,
                'local_response':counts['semantic_bootstrap']-(counts['semantic_bootstrap']//args.semantic_color_every if args.semantic_color_every else 0)},
            'snr_condition_policy':'trained_range; actual channel SNR remains unchanged',
            'losses':{'semantic_bootstrap':'local response MSE; separately scheduled directional RGB MSE' if args.semantic_color_every else 'local response MSE',
                      'semantic_render':'source PLY RGB MSE',
                      'channel_bootstrap':'latent MSE against frozen semantic encoder',
                      'channel_render':'source PLY RGB MSE + channel_latent_weight * latent MSE'},
            'clip_mode':'none','render_backward':'replay','shared_model_cost':'excluded, assumed shared in advance',
            'source_usage':'paths, camera split and model dimensions only; never source checkpoint weights',
            'initialization': 'semantic-only fork; fresh channel modules and Adam' if args.init_semantic else 'all four modules random'})
    scene.channel_trained=progress['channel_trained']
    if cfg.context_variant!='baseline' and not args.resume:
        from .pointnet_context import geometry_diagnostics
        from .position_delivery import delivered_positions
        block_indices=torch.linspace(0,len(scene.blocks)-1,min(16,len(scene.blocks))).round().long().tolist()
        sampled=[scene.blocks[i] for i in block_indices]
        f=torch.nn.utils.rnn.pad_sequence(sampled,batch_first=True)
        active=torch.arange(f.shape[1])[None,:]<torch.tensor([len(b) for b in sampled])[:,None]
        report={'block_indices':block_indices,'scope':'evenly spaced sample of processing blocks; padding excluded'}
        for side in ('encoder','decoder'):
            xyz=f[...,:3] if side=='encoder' else delivered_positions(f[...,:3],active.long(),cfg)
            report[side]=geometry_diagnostics(xyz,active,scene.geometry,cfg)
        json_write(out/'metrics/spatial_neighborhoods.json',report)
    print(f"Spatial context: {cfg.context_variant}; internal centers={cfg.context_centers}; neighbors={cfg.context_neighbors}; radii={cfg.context_radii}",flush=True)
    if args.clean_only:
        print('CLEAN ONLY: semantic latent is handed directly to semantic decoder; both channel modules unused. Not a communication-rate result.',flush=True)
    else:
        print(f"Channel objective: bootstrap=latent MSE; render=RGB MSE + {args.channel_latent_weight:g} * latent MSE; semantic encoder frozen; semantic decoder {'FROZEN' if args.freeze_semantic else 'trainable in render'}.",flush=True)
    print(f'Channel variant: {cfg.channel_variant}; hidden={cfg.hidden if cfg.channel_variant=="baseline" else cfg.channel_hidden}',flush=True)
    print(f'Channel budgets (complex symbols/Gaussian): {cfg.rates[1:]}; latent dimension: {cfg.latent_dim}',flush=True)
    if args.semantic_color_every:
        print(f'Color supervision: every {args.semantic_color_every} bootstrap updates; {args.semantic_color_directions} directions, unclipped RGB; all render updates retain scene RGB MSE.',flush=True)
    if not args.resume:
        # Same data/layout/view/noise stream despite different module sizes.
        if args.freeze_semantic: seed_all(args.seed+17)
        evaluate_paths(model,scene,step=0,phase='initial',snrs=[10],trials=2,seed=args.seed,save_images=True,out=out,**evaluation_options)

    def save(label):
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,capture_rng(),scene.record)
        json_write(out/'progress.json',dict(progress,status='running'))

    updated=0
    for phase in PHASES:
        if progress['completed'][phase]>=counts[phase]:
            if (phase=='semantic_bootstrap' and counts[phase]>0 and args.save_semantic_end_images
                    and progress['phase']==phase and progress.get('phase_complete')
                    and not (out/'images/end_semantic_bootstrap/complete.json').exists()):
                evaluate_paths(model,scene,step=progress['step'],phase=phase+'_end',snrs=[10],trials=2,
                               seed=args.seed,save_images=True,out=out,**evaluation_options)
                json_write(out/'images/end_semantic_bootstrap/complete.json',{'step':progress['step']})
            # A latest checkpoint can be written at the last update just before
            # interruption, before the phase-end snapshot is materialized.
            if (counts[phase]>0 and progress['phase']==phase and progress.get('phase_complete') and
                    not (out/'checkpoints'/('end_'+phase)/'codec.pt').exists()):
                save('end_'+phase)
            continue
        continuing=progress['phase']==phase and optimizer is not None and not progress.get('phase_complete')
        if not continuing: optimizer=configure_phase(model,phase,lrs,freeze_semantic=args.freeze_semantic)
        progress.update(phase=phase,phase_step=progress['completed'][phase],phase_complete=False)
        model.train()
        evaluate_paths(model,scene,step=progress['step'],phase=phase+'_start',snrs=[10],trials=2,seed=args.seed,save_images=False,out=out,**evaluation_options)
        while progress['completed'][phase]<counts[phase]:
            phase_step=progress['completed'][phase]+1
            snr=random.choice(args.train_snrs)  # Exactly one draw per real update, never per block/replay.
            layout='clean' if phase.startswith('semantic') else ('1','2','3','mixed')[(phase_step-1)%4]
            qs=tier_batches(scene,'3' if layout=='clean' else layout)
            optimizer.zero_grad(set_to_none=True)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
            start=time.perf_counter(); indices=[]
            if phase.endswith('bootstrap'):
                index=random.randrange(len(scene.groups)); features=scene.groups[index].to(device); q=qs[index].to(device)
                if phase=='semantic_bootstrap' and args.semantic_color_every and phase_step%args.semantic_color_every==0:
                    loss,stats=color_bootstrap_step(model,features,q,scene.geometry,color_camera_centers,args.semantic_color_directions)
                else:
                    loss,stats=bootstrap_step(model,features,q,scene.geometry,snr,'awgn',phase)
            else:
                indices=random.sample(range(len(scene.cameras)),scene.views_per_step)
                task=MultiViewRenderTask([scene.cameras[i] for i in indices],scene.reference,degree,scene.white_background)
                loss,stats=render_step(model,list(zip(scene.groups,qs)),scene.geometry,snr,'awgn',task,phase,
                                       channel_latent_weight=args.channel_latent_weight)
                stats.update(task.stats)
            stats.update(finish_update(model,optimizer,before))
            if device.type=='cuda': torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-start
            progress['step']+=1; progress['phase_step']=phase_step; progress['completed'][phase]=phase_step
            if phase.startswith('channel'): progress['channel_trained']=scene.channel_trained=True
            row=dict(stats,step=progress['step'],phase_step=phase_step,phase=phase,loss=float(loss),layout=layout,
                     snr=None if args.clean_only else snr,
                     objective_version=objectives['version'],
                     actual_snr=snr if phase.startswith('channel') else None,tx_condition_snr=None if args.clean_only else snr,
                     rx_condition_snr=None if args.clean_only else snr,condition_policy='trained_range',
                     step_seconds=elapsed,training_view_indices=indices,module_lrs={g['name']:g['lr'] for g in optimizer.param_groups},
                     trainable_parameters={n:sum(p.numel() for p in getattr(model,n).parameters() if p.requires_grad) for n in MODULES},
                     frozen_parameters_unchanged=True,
                     peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                     peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
            append_json(out/'metrics/loss.jsonl',row)
            if phase_step==1 or phase_step%10==0 or phase_step==counts[phase]:
                detail=f", latent_mse={stats['latent_mse']:.6g}, latent_nmse={stats['latent_nmse']:.6g}" if phase.startswith('channel') else ''
                if phase=='channel_render': detail+=f", render_mse={stats['render_loss']:.6g}, weighted_latent={stats['weighted_latent_loss']:.6g}"
                if args.semantic_color_every and phase=='semantic_bootstrap': detail+=f", objective={stats['objective']}"
                displayed_snr='bypass' if args.clean_only else f'{snr:g}'
                print(f"{phase} {phase_step}/{counts[phase]}: loss={float(loss):.6g}, q={layout}, snr={displayed_snr}, grad={stats['grad_norm']:.4g}, sec={elapsed:.2f}{detail}",flush=True)
            if progress['step']%args.validate_every==0:
                evaluate_paths(model,scene,step=progress['step'],phase=phase,snrs=[10],trials=2,seed=args.seed,save_images=False,out=out,**evaluation_options)
            progress['phase_complete']=phase_step==counts[phase]
            if progress['step']%args.save_every==0: save('latest')
            updated+=1
            if max_updates is not None and updated>=max_updates:
                save('latest'); return progress
        evaluate_paths(model,scene,step=progress['step'],phase=phase+'_end',snrs=[10],trials=2,seed=args.seed,
                       save_images=args.save_semantic_end_images and phase=='semantic_bootstrap',out=out,**evaluation_options)
        if args.save_semantic_end_images and phase=='semantic_bootstrap':
            json_write(out/'images/end_semantic_bootstrap/complete.json',{'step':progress['step']})
        save('end_'+phase); save('latest')
    # Final endpoint images use the SAME validation cameras as the initial endpoint.
    evaluate_paths(model,scene,step=progress['step'],phase='final',snrs=[10],trials=2,seed=args.seed,save_images=True,out=out,**evaluation_options)
    tests=load_cameras(spec['source'],scene.record['resolution'],scene.white_background,scene.record['images'],'test')
    if not tests: raise ValueError('no test cameras; refusing to substitute training views for final test')
    scene.val_cameras=tests
    evaluate_paths(model,scene,step=progress['step'],phase='test',snrs=[10] if args.clean_only else [0,5,10,15,20],
                   trials=2,seed=args.seed,save_images=False,out=out,**evaluation_options)
    save('final'); save('latest')
    rows=[json.loads(l) for l in (out/'metrics/loss.jsonl').read_text(encoding='utf-8').splitlines()] if (out/'metrics/loss.jsonl').exists() else []
    summaries={p:{'updates':len(part),'total_step_seconds':sum(r['step_seconds'] for r in part),
                    'mean_step_seconds':sum(r['step_seconds'] for r in part)/len(part) if part else None,
                    'optimizer_reset_at_boundary':True} for p in PHASES if (part:=[r for r in rows if r['phase']==p]) or counts[p]==0}
    json_write(out/'metrics/phase_summary.json',summaries)
    from .modular_plots import plot_run
    plot_run(out)
    if args.with_direct_control:
        direct_control(args,out)
    json_write(out/'complete.json',dict(progress,status='complete',test_views=len(tests),cuda_render_verified=device.type=='cuda'))
    print(json.dumps({'status':'complete','completed':progress['completed'],'out':str(out)}),flush=True)
    return progress


if __name__=='__main__': main(parser().parse_args())
