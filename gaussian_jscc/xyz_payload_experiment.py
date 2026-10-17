"""Truck XYZ-in-payload: semantic pretrain, channel recovery, joint decoding."""
import argparse
import json
import math
import random
import time
import uuid
from pathlib import Path
import torch
from .cli import seed_all,device_for
from .data import read_ply
from .multiscene import shared_statistics
from .modular_checkpoint import restore_rng,validate_scene
from .modular_experiment import json_write,tier_batches
from .modular_training import MODULES,PHASE_MODULES
from .render_objective import MultiViewRenderTask
from .render_validation import append_json
from .xyz_payload_config import XYZPayloadConfig
from .xyz_payload_codec import XYZPayloadCodec
from .xyz_payload_scene import inputs,make_scene
from .xyz_payload_training import (objective_settings,configure_phase,bootstrap_step,
    color_bootstrap_step,render_step,finish_update)
from .xyz_payload_checkpoint import save_checkpoint,restore_checkpoint,read_checkpoint
from .xyz_payload_evaluation import evaluate_paths

PHASES=tuple(PHASE_MODULES)


def repair_payload_logs(out,step):
    """Archive uncheckpointed/torn tail records, including the empty-log case."""
    for name in ('loss','validation','test'):
        path=Path(out)/'metrics'/(name+'.jsonl')
        if not path.exists(): continue
        lines=path.read_text(encoding='utf-8').splitlines(); kept=[]; removed=[]
        last=next((i for i in range(len(lines)-1,-1,-1) if lines[i].strip()),-1)
        for i,line in enumerate(lines):
            if not line.strip(): removed.append(line); continue
            try: record=json.loads(line)
            except json.JSONDecodeError:
                if i!=last: raise ValueError('corrupt non-tail training log: '+str(path))
                removed.append(line); continue
            (kept if record['step']<=step else removed).append(line)
        if removed:
            path.with_name(f'{name}.uncheckpointed_{time.time_ns()}.txt').write_text('\n'.join(removed),encoding='utf-8')
            temporary=path.with_suffix('.repair.tmp')
            temporary.write_text(''.join(line+'\n' for line in kept),encoding='utf-8'); temporary.replace(path)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='Truck paths/split/dimensions only; NEVER implicit source weights')
    p.add_argument('--out',required=True); p.add_argument('--manifest')
    p.add_argument('--device',default='cuda'); p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--attribute-latent-dim',type=int,default=48)
    p.add_argument('--rates',type=int,nargs=3,default=[8,24,48])
    p.add_argument('--channel-hidden',type=int,default=192)
    p.add_argument('--context-variant',choices=['both','encoder','decoder','baseline'],default='both')
    p.add_argument('--context-centers',nargs=2,type=int,default=[64,16])
    p.add_argument('--context-neighbors',nargs=2,type=int,default=[16,32])
    p.add_argument('--context-radii',nargs=2,type=float,default=[2.,4.])
    p.add_argument('--blocks-per-batch',type=int); p.add_argument('--views-per-step',type=int)
    for phase,steps in zip(PHASES,(4000,1000,4000,2000)):
        p.add_argument('--'+phase.replace('_','-')+'-steps',type=int,default=steps)
    for name in MODULES: p.add_argument('--'+name.replace('_','-')+'-lr',type=float,default=1e-4)
    p.add_argument('--semantic-color-every',type=int,default=4)
    p.add_argument('--semantic-color-directions',type=int,default=8)
    p.add_argument('--xyz-latent-weight',type=float,default=1.)
    p.add_argument('--channel-latent-weight',type=float,default=1.)
    p.add_argument('--joint-semantic-encoder',action='store_true',help='explicit opt-in; default joint keeps semantic encoder frozen')
    p.add_argument('--validate-every',type=int,default=500); p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--validation-trials',type=int,default=2); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--no-plots',action='store_true'); p.add_argument('--no-packet',action='store_true')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--resume',help='exact same-run model/Adam/RNG continuation')
    g.add_argument('--init-semantic',help='explicit NEW-architecture semantic initializer; new channel weights')
    g.add_argument('--init-codec',help='explicit NEW-architecture full initializer; fresh Adam/schedule')
    return p


def main(args,max_updates=None):
    from .rendering import load_cameras
    objectives=objective_settings(args.xyz_latent_weight,args.channel_latent_weight)
    budgets={p:getattr(args,p+'_steps') for p in PHASES}
    lrs={n:getattr(args,n+'_lr') for n in MODULES}
    if (min(budgets.values())<0 or min(args.validate_every,args.save_every,args.cpu_threads,args.validation_trials)<1
            or args.semantic_color_every<0 or args.semantic_color_directions<2 or args.semantic_color_directions%2
            or any(not math.isfinite(v) or v<=0 for v in lrs.values())
            or (args.blocks_per_batch is not None and args.blocks_per_batch<1)
            or (args.views_per_step is not None and args.views_per_step<1)):
        raise ValueError('invalid training budgets/intervals/learning rates')
    objectives.update(color_every=args.semantic_color_every,color_directions=args.semantic_color_directions)
    out=Path(args.out).resolve()
    if out.exists() and not args.resume: raise FileExistsError('use a NEW output directory: '+str(out))
    if args.resume and not out.exists(): raise FileNotFoundError('resume requires original output directory')
    source,manifest_path,spec,old=inputs(args)
    device=device_for(args.device); torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    raw,degree=read_ply(spec['ply']); common=source.get('codec_config',{})
    cfg=XYZPayloadConfig(sh_degree=degree,attribute_latent_dim=args.attribute_latent_dim,latent_dim=args.attribute_latent_dim+3,
        rates=(0,*args.rates),channel_hidden=args.channel_hidden,context_variant=args.context_variant,
        context_centers=args.context_centers,context_neighbors=args.context_neighbors,context_radii=args.context_radii,
        **{k:common[k] for k in ('hidden','grid_dim','levels','planes','depth','block_size','morton_bits','decoder_window','attention_heads') if k in common})
    saved_scene=rng=None; optimizer=None; initializer=args.init_codec or args.init_semantic
    if args.resume:
        previous=json.loads((out/'training.json').read_text(encoding='utf-8'))
        initializer=previous.get('initializer')
    if initializer: budgets.update(semantic_bootstrap=0,semantic_render=0)
    signature=dict(codec_config=cfg.to_dict(),phase_budgets=budgets,module_lrs=lrs,training_objective=objectives,
                   joint_semantic_encoder=args.joint_semantic_encoder,seed=args.seed,
                   validation_trials=args.validation_trials,lpips_metrics=args.lpips)
    # JSON canonicalization makes tuple/list config forms compare identically.
    signature=json.loads(json.dumps(signature))
    if args.resume:
        if previous.get('resume_signature')!=signature: raise ValueError('exact resume settings/objective mismatch')
        checkpoint=read_checkpoint(args.resume); saved_progress=checkpoint['progress']
        if (saved_progress.get('run_id')!=previous.get('run_id') or not previous.get('run_id')
                or saved_progress.get('resume_signature')!=signature
                or saved_progress.get('phase_budgets')!=budgets or saved_progress.get('module_lrs')!=lrs
                or saved_progress.get('joint_semantic_encoder')!=args.joint_semantic_encoder):
            raise ValueError('exact resume requires the same-run identity and checkpoint settings signature')
        model,optimizer,progress,rng,saved_scene=restore_checkpoint(args.resume,device)
        if model.cfg!=cfg or progress['training_objective']!=objectives:
            raise ValueError('checkpoint settings/objective mismatch')
    else:
        model=XYZPayloadCodec(cfg).to(device)
        mean,std=shared_statistics([raw]); model.attr_mean.copy_(mean.to(device)); model.attr_std.copy_(std.to(device))
        trained=False
        if initializer:
            saved=read_checkpoint(initializer); previous_cfg=XYZPayloadConfig.from_dict(saved['config'])
            if previous_cfg!=cfg: raise ValueError('initializer config mismatch; explicit compatible architecture required')
            saved_scene=saved['scene_record']
            if args.init_semantic:
                if (saved['progress']['phase']!='semantic_render' or not saved['progress']['completed'].get('semantic_render')
                        or not saved['progress'].get('phase_complete')):
                    raise ValueError('semantic initializer must complete semantic render')
                state={k:v for k,v in saved['state_dict'].items() if k.startswith('semantic_') or k in ('attr_mean','attr_std')}
                expected={k for k in model.state_dict() if k.startswith('semantic_') or k in ('attr_mean','attr_std')}
                if set(state)!=expected: raise ValueError('incomplete or unexpected semantic initializer keys')
                model.load_state_dict(state,strict=False)
            else:
                if not saved['progress'].get('channel_trained'): raise ValueError('initializer has no trained channel')
                model.load_state_dict(saved['state_dict']); trained=True
        progress=dict(phase=PHASES[0],phase_step=0,step=0,phase_complete=False,completed={p:0 for p in PHASES},
                      phase_budgets=budgets,module_lrs=lrs,training_objective=objectives,
                      joint_semantic_encoder=args.joint_semantic_encoder,train_snrs=[10.],channel_trained=trained,
                      run_id=uuid.uuid4().hex,resume_signature=signature)
    model.set_training_snrs([10.])
    scene=make_scene(spec,raw,degree,model,source,old,args); del raw
    if saved_scene is not None: validate_scene(saved_scene,scene.record)
    scene.channel_trained=progress['channel_trained']
    centers=torch.stack([torch.as_tensor(c.camera_center).detach().cpu() for c in scene.cameras]).to(device)
    if args.resume:
        stamp=str(time.time_ns())
        for name in ('complete','failure'):
            marker=out/(name+'.json')
            if marker.exists(): marker.rename(out/f'{name}.before_resume_{stamp}.json')
        if progress['step']<sum(budgets.values()):
            future={'packet','charts','checkpoints/final','images/final'}
            end_step=0
            for phase in PHASES:
                end_step+=budgets[phase]
                if end_step>progress['step']:
                    future.update(('checkpoints/end_'+phase,'images/'+phase+'_end'))
            for relative in sorted(future):
                path=out/relative
                if path.exists():
                    archived=out/'archive'/('before_resume_'+stamp)/relative
                    archived.parent.mkdir(parents=True,exist_ok=True); path.rename(archived)
        repair_payload_logs(out,progress['step']); restore_rng(rng)
    else:
        for folder in ('metrics','charts','checkpoints','images'): (out/folder).mkdir(parents=True,exist_ok=True)
        json_write(out/'training.json',dict(run_id=progress['run_id'],resume_signature=signature,**signature,source_run=str(Path(args.run).resolve()),
            manifest=str(manifest_path),selected_scenes=[spec],scene=scene.record,initializer=initializer,
            random_initialization=not bool(initializer),source_weights_loaded=bool(args.init_codec),
            metadata_only_source=not bool(initializer),position_delivery='inside_analogue_payload',
            net_bits_per_use=scene.net_bits_per_use,semantic_encoder_frozen_in_joint=not args.joint_semantic_encoder,
            clip_mode='none',render_backward='replay',train_snrs=[10.],shared_model_cost='excluded; shared in advance'))

    def evaluate(label,images=False,include=None):
        return evaluate_paths(model,scene,step=progress['step'],phase=label,out=out,
            include_channel=scene.channel_trained if include is None else include,
            trials=args.validation_trials,seed=args.seed,save_images=images,lpips_metrics=args.lpips)

    def save(label):
        save_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,scene.record)
        json_write(out/'progress.json',dict(progress,status='running'))

    def endpoint(phase):
        label='end_'+phase
        evaluate(phase+'_end',images=phase in ('semantic_render','channel_bootstrap'))
        save(label)
        save('latest')

    if not args.resume:
        evaluate('initial',images=True); save('initial'); save('latest')
    updated=0
    try:
        for phase in PHASES:
            if progress['completed'][phase]>=budgets[phase]:
                if budgets[phase] and progress['phase']==phase and progress.get('phase_complete'): endpoint(phase)
                continue
            continuing=progress['phase']==phase and optimizer is not None and not progress.get('phase_complete')
            if not continuing:
                optimizer=configure_phase(model,phase,lrs,joint_semantic_encoder=args.joint_semantic_encoder)
            progress.update(phase=phase,phase_step=progress['completed'][phase],phase_complete=False)
            model.train(); evaluate(phase+'_start',include=phase.startswith('channel'))
            while progress['completed'][phase]<budgets[phase]:
                phase_step=progress['completed'][phase]+1
                layout='clean' if phase.startswith('semantic') else ('1','2','3','mixed')[(phase_step-1)%4]
                qs=tier_batches(scene,'3' if layout=='clean' else layout)
                optimizer.zero_grad(set_to_none=True)
                before={n:p.detach().clone() for n,p in model.named_parameters()}
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
                start=time.perf_counter(); indices=[]
                if phase.endswith('bootstrap'):
                    index=random.randrange(len(scene.groups)); features=scene.groups[index].to(device); q=qs[index].to(device)
                    if phase=='semantic_bootstrap' and args.semantic_color_every and phase_step%args.semantic_color_every==0:
                        loss,stats=color_bootstrap_step(model,features,q,scene.geometry,centers,args.semantic_color_directions)
                    else:
                        loss,stats=bootstrap_step(model,features,q,scene.geometry,10.,'awgn',phase,args.xyz_latent_weight)
                else:
                    indices=random.sample(range(len(scene.cameras)),scene.views_per_step)
                    task=MultiViewRenderTask([scene.cameras[i] for i in indices],scene.reference,degree,scene.white_background)
                    loss,stats=render_step(model,list(zip(scene.groups,qs)),scene.geometry,10.,'awgn',task,phase,
                        xyz_weight=args.xyz_latent_weight,channel_latent_weight=args.channel_latent_weight)
                    stats.update(task.stats)
                stats.update(finish_update(model,optimizer,before))
                if device.type=='cuda': torch.cuda.synchronize(device)
                progress['step']+=1; progress['phase_step']=phase_step; progress['completed'][phase]=phase_step
                progress['phase_complete']=phase_step==budgets[phase]
                if phase.startswith('channel'): progress['channel_trained']=scene.channel_trained=True
                row=dict(stats,step=progress['step'],phase_step=phase_step,phase=phase,layout=layout,loss=float(loss),
                    step_seconds=time.perf_counter()-start,actual_snr=10. if phase.startswith('channel') else None,
                    training_view_indices=indices,module_lrs={g['name']:g['lr'] for g in optimizer.param_groups},
                    frozen_parameters_unchanged=True,xyz_latent_weight=args.xyz_latent_weight,
                    channel_latent_weight=args.channel_latent_weight,
                    peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
                append_json(out/'metrics/loss.jsonl',row)
                if phase_step==1 or phase_step%10==0 or progress['phase_complete']:
                    suffix=f", XYZ_MSE={stats['payload_xyz_mse']:.6g}, attr_y_MSE={stats['payload_attr_mse']:.6g}" if phase.startswith('channel') else ''
                    print(f"{phase} {phase_step}/{budgets[phase]}: loss={float(loss):.6g}, q={layout}, objective={stats['objective']}, sec={row['step_seconds']:.2f}{suffix}",flush=True)
                if progress['step']%args.validate_every==0: evaluate(phase)
                if progress['step']%args.save_every==0: save('latest')
                updated+=1
                if max_updates is not None and updated>=max_updates:
                    save('latest'); return progress
            endpoint(phase)
        evaluate('final',images=True)
        tests=load_cameras(spec['source'],scene.record['resolution'],scene.white_background,scene.record['images'],'test')
        if not tests: raise ValueError('no test cameras; refusing training-view fallback')
        scene.val_cameras=tests; evaluate('test')
        save('final'); save('latest')
        if not args.no_packet and scene.channel_trained:
            from .xyz_payload_transport import export_packet,receive_packet
            if (out/'packet').exists():
                receive_packet(model,out/'packet')  # Never silently accept a partial/wrong-model export.
            else:
                export_packet(model,scene.raw,torch.full((len(scene.raw),),3,dtype=torch.long),10.,'awgn',args.seed,
                              out/'packet',scene.net_bits_per_use)
        if not args.no_plots:
            from .xyz_payload_plots import plot_run
            plot_run(out)
        rows=[json.loads(s) for s in (out/'metrics/loss.jsonl').read_text().splitlines()] if (out/'metrics/loss.jsonl').exists() else []
        json_write(out/'metrics/phase_summary.json',{p:dict(updates=len(part),
            mean_step_seconds=sum(r['step_seconds'] for r in part)/len(part) if part else None)
            for p in PHASES if (part:=[r for r in rows if r['phase']==p]) or budgets[p]==0})
        json_write(out/'complete.json',dict(progress,status='complete',test_views=len(tests),cuda_render_verified=device.type=='cuda'))
        print(json.dumps(dict(status='complete',completed=progress['completed'],out=str(out))),flush=True)
        return progress
    except BaseException as error:
        json_write(out/'failure.json',dict(step=progress['step'],phase=progress['phase'],error_type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__': main(parser().parse_args())
