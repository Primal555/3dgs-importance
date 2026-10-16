"""Independent one-pair progressive JSCC: noisy attributes/color, then rendering."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from .cli import seed_all,device_for
from .codec import CodecConfig,GaussianCodec
from .data import read_ply
from .multiscene import shared_statistics
from .modular_experiment import json_write,tier_batches,repair_logs
from .modular_checkpoint import restore_rng,validate_scene
from .integrated_scene import inputs,make_scene
from .integrated_training import update_spec,make_optimizer,bootstrap_step,finish_update,save_state,restore_state
from .integrated_evaluation import evaluate,stage_diagnostics
from .render_objective import MultiViewRenderTask
from .render_validation import append_json
from .training import full_scene_step


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='scene metadata only; never source codec weights')
    p.add_argument('--out',required=True);p.add_argument('--manifest');p.add_argument('--device',default='cuda')
    p.add_argument('--bootstrap-steps',type=int,default=5000);p.add_argument('--render-steps',type=int,default=1000)
    p.add_argument('--color-every',type=int,default=4);p.add_argument('--color-directions',type=int,default=8)
    p.add_argument('--rates',type=int,nargs=3,default=[12,24,48])
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--blocks-per-batch',type=int)
    p.add_argument('--validate-every',type=int,default=500);p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--trials',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--resume',help='exact checkpoint continuation in original output directory')
    return p


def main(args,max_updates=None):
    from .rendering import load_cameras
    if (min(args.bootstrap_steps,args.render_steps,args.color_every)<0 or args.bootstrap_steps+args.render_steps<1
            or min(args.validate_every,args.save_every,args.trials,args.cpu_threads)<1
            or args.color_directions<2 or args.color_directions%2 or not math.isfinite(args.lr) or args.lr<=0
            or (args.blocks_per_batch is not None and args.blocks_per_batch<1)):
        raise ValueError('invalid budgets, intervals, directions or learning rate')
    out=Path(args.out).resolve()
    if out.exists() and not args.resume:raise FileExistsError('use a NEW output directory: '+str(out))
    if args.resume and (not out.exists() or not Path(args.resume).resolve().is_relative_to(out/'checkpoints')):
        raise ValueError('exact resume requires a checkpoint inside the original output directory')
    if args.resume and (out/'complete.json').exists():
        raise ValueError('output is already complete; refusing a rewind with stale completion/endpoint artifacts')
    source,manifest_path,spec,old=inputs(args)
    if args.blocks_per_batch is not None:source['blocks_per_batch']=args.blocks_per_batch
    device=device_for(args.device);torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    raw,degree=read_ply(spec['ply']);common=source.get('codec_config',{})
    cfg=CodecConfig(sh_degree=degree,rates=(0,*args.rates),prefix_mode='progressive',decoder_mode='residual',
        position_delivery='quantized',position_bits=16,position_compression='delta_zlib',position_compression_level=6,
        **{k:common[k] for k in ('hidden','grid_dim','levels','planes','depth','block_size','morton_bits','decoder_window','attention_heads') if k in common})
    budgets={'bootstrap':args.bootstrap_steps,'render':args.render_steps}
    settings=dict(version=1,phase_budgets=budgets,color_every=args.color_every,color_directions=args.color_directions,
        lr=args.lr,snr=10.,channel='awgn',seed=args.seed,validate_every=args.validate_every,save_every=args.save_every,
        trials=args.trials,cpu_threads=args.cpu_threads,codec_config=cfg.to_dict())
    # JSON-normalized syntax keeps checkpoint and human-readable settings equal.
    settings=json.loads(json.dumps(settings))
    state=None
    if args.resume:
        model,optimizer,state=restore_state(args.resume,device)
        if state['settings']!=settings:raise ValueError('exact resume settings mismatch; do not silently change experiment')
        progress=state['progress']
        marker=out/'progress.json'
        if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))['step']>progress['step']:
            raise ValueError('cannot rewind to an older checkpoint in an existing output directory')
    else:
        model=GaussianCodec(cfg).to(device);mean,std=shared_statistics([raw])
        model.attr_mean.copy_(mean.to(device));model.attr_std.copy_(std.to(device))
        optimizer=make_optimizer(model,args.lr)
        progress=dict(step=0,phase='bootstrap',completed={'bootstrap':0,'render':0},
            objective_layout_updates={obj:{q:0 for q in ('1','2','3','mixed')} for obj in ('local_response','direction_color','render')})
    scene=make_scene(spec,raw,degree,model,source,old,out);del raw
    if state is not None:validate_scene(state['scene_record'],scene.record)
    stats_identity={n:model.state_dict()[n].detach().clone() for n in ('attr_mean','attr_std','learned.heads.xyz.weight','learned.heads.xyz.bias')}
    centers=None
    if args.color_every and args.bootstrap_steps:
        centers=torch.stack([torch.as_tensor(c.camera_center).detach().cpu() for c in scene.cameras]).to(device)
    if args.resume:
        repair_logs(out,progress['step'])
        # Prefix-gain rows, too, may have been written beyond the last save.
        path=out/'metrics/prefix_gains.jsonl'
        if path.exists():
            rows=[l for l in path.read_text(encoding='utf-8').splitlines() if json.loads(l)['step']<=progress['step']]
            path.write_text('\n'.join(rows)+'\n',encoding='utf-8')
        restore_rng(state['rng'])
    else:
        for folder in ('metrics','checkpoints','images','charts'):(out/folder).mkdir(parents=True,exist_ok=True)
        json_write(out/'training.json',dict(source_run=str(Path(args.run).resolve()),manifest=str(manifest_path),
            selected_scenes=[spec],codec_config=cfg.to_dict(),scene=scene.record,settings=settings,
            random_initialization=True,source_weights_loaded=False,source_usage='metadata only; source checkpoints never read',
            blocks_per_batch=scene.record['blocks_per_batch'],net_bits_per_use=scene.net_bits_per_use,
            training_family='integrated_progressive_color_v1',render_backward='replay',clip_mode='none',
            initialization='all one-pair network weights random; single-scene attribute statistics',
            step_definition='one optimizer update; color replaces every fourth bootstrap update',
            losses={'local_response':'local Gaussian RGB response MSE','direction_color':'unclipped direction RGB MSE',
                    'render':'source PLY rendered RGB MSE only'},
            decoding='full received prefix + previous normalized attributes; cumulative residual; no latent reconstruction objective'))

    def save(label):
        save_state(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,settings,scene.record)
        json_write(out/'progress.json',dict(progress,status='running'))

    def validate(phase,images=False):
        return evaluate(model,scene,out=out,step=progress['step'],phase=phase,
                        trials=args.trials,seed=args.seed,save_images=images)

    initialization='EXACT resume: weights/Adam/RNG restored' if args.resume else 'RANDOM weights'
    print(f'INTEGRATED one-pair JSCC; {initialization}; {budgets}; cumulative budgets={cfg.rates[1:]}; 10 dB AWGN; LR={args.lr:g}',flush=True)
    print('Color and local-response use independent layout counters; no latent/channel pretraining or learned allocation.',flush=True)
    if not args.resume:
        validate('initial',True);save('initial');save('latest')
    updated=0
    for phase in ('bootstrap','render'):
        if progress['completed'][phase]>=budgets[phase]:
            if (budgets[phase]>0 and progress['phase']==phase and
                    not (out/'checkpoints'/('end_'+phase)/'codec.pt').exists()):
                validate(phase+'_end');save('end_'+phase)
            continue
        if progress['phase']!=phase:
            optimizer=make_optimizer(model,args.lr);progress['phase']=phase
        model.train()
        while progress['completed'][phase]<budgets[phase]:
            phase_step=progress['completed'][phase]+1
            objective,layout=update_spec(phase,phase_step,args.color_every)
            qs=tier_batches(scene,layout)
            optimizer.zero_grad(set_to_none=True)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
            started=time.perf_counter();indices=[]
            if phase=='bootstrap':
                i=random.randrange(len(scene.groups));features=scene.groups[i].to(device);q=qs[i].to(device)
                loss,stats=bootstrap_step(model,features,q,scene.geometry,centers,
                    color=objective=='direction_color',color_directions=args.color_directions)
            else:
                indices=random.sample(range(len(scene.cameras)),scene.views_per_step)
                task=MultiViewRenderTask([scene.cameras[i] for i in indices],scene.reference,degree,scene.white_background)
                loss,stats=full_scene_step(model,list(zip(scene.groups,qs)),scene.geometry,10.,'awgn',task,
                    attr_weight=0,mode='replay',profile=True)
                stats.update(task.stats,objective='source_ply_rgb_mse')
            stats.update(finish_update(model,optimizer,before))
            if device.type=='cuda':torch.cuda.synchronize(device)
            progress['step']+=1;progress['completed'][phase]=phase_step
            progress['objective_layout_updates'][objective][layout]+=1
            append_json(out/'metrics/loss.jsonl',dict(stats,step=progress['step'],phase_step=phase_step,phase=phase,
                layout=layout,loss=float(loss),snr=10.,lr=args.lr,training_view_indices=indices,
                step_seconds=time.perf_counter()-started,
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
            if phase_step==1 or phase_step%10==0 or phase_step==budgets[phase]:
                print(f"{phase} {phase_step}/{budgets[phase]}: objective={objective}, q={layout}, loss={float(loss):.6g}, grad={stats['grad_norm']:.4g}",flush=True)
            if progress['step']%args.validate_every==0:validate(phase)
            if progress['step']%args.save_every==0:save('latest')
            updated+=1
            if max_updates is not None and updated>=max_updates:save('latest');return progress
        validate(phase+'_end');save('end_'+phase);save('latest')
    validate('final',True)
    tests=load_cameras(spec['source'],scene.record['resolution'],scene.white_background,scene.record['images'],'test')
    if not tests:raise ValueError('no independent test cameras; refusing training-view substitution')
    scene.val_cameras=tests;validate('test')
    for n,v in stats_identity.items():
        if not torch.equal(model.state_dict()[n],v):raise RuntimeError('fixed XYZ/statistics changed: '+n)
    save('final');save('latest')
    # Runs once against the validated in-memory scene; no unused prior loading.
    json_write(out/'stage_diagnostics.json',stage_diagnostics(model,scene,args.seed))
    from .integrated_plots import plot_run
    plot_run(out)
    json_write(out/'complete.json',dict(progress,status='complete',test_views=len(tests),cuda_render_verified=device.type=='cuda'))
    print(json.dumps(dict(status='complete',completed=progress['completed'],out=str(out))),flush=True)
    return progress


if __name__=='__main__':main(parser().parse_args())
