"""Truck: trained 8/24/48 modular codec -> 2000 allocator -> 1000 joint updates."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .allocation import GaussianTierMask
from .allocation_diagnostics import record_allocation
from .cli import device_for,seed_all
from .data import read_ply
from .mask_gradient import local_mask_step
from .modular_checkpoint import capture_rng,restore_rng,initialize_codec_run,save_training_checkpoint
from .modular_evaluation import decode_path,conditioning_pair
from .modular_experiment import inputs,make_scene,json_write,repair_logs
from .modular_training import MODULES
from .multiscene import preserve_prefix
from .optimization import preserved_rng,clip_codec_gradients,update_stats
from .perceptual_metrics import LPIPSEvaluator,image_metric_protocol
from .position_delivery import encode_positions
from .render_objective import image_metrics,MaskedMultiViewRenderTask,MultiViewRenderTask,spaced_indices
from .render_validation import append_json,save_panel
from .tier_visualization import render_tier_map,save_tier_panel
from .transport import model_id,encode_metadata
from .training import full_scene_step


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='completed trained three-tier modular run; full final codec retained')
    p.add_argument('--out',required=True)
    p.add_argument('--manifest',help='override moved data paths; scene identity must still match')
    p.add_argument('--device',default='cuda')
    p.add_argument('--allocation-steps',type=int,default=2000)
    p.add_argument('--joint-steps',type=int,default=1000)
    p.add_argument('--beta',type=float,default=.01)
    p.add_argument('--mask-lr',type=float,default=.001)
    p.add_argument('--keep-lr',type=float,default=.01)
    p.add_argument('--mask-adam-eps',type=float,default=1e-15)
    p.add_argument('--codec-lr',type=float,default=1e-5,help='joint-only LR for all four trained modules')
    p.add_argument('--image-objective',choices=['mse','mse_ssim','mse_lpips'],default='mse',
                   help='opt-in fixed-deployment finetuning only; evaluation metrics remain unchanged')
    p.add_argument('--image-aux-weight',type=float,
                   help='positive SSIM/LPIPS coefficient; default paper-relative 1/3 (SSIM) or 1 (LPIPS); ignored for mse')
    p.add_argument('--snr',type=float,default=10.)
    p.add_argument('--prefix-anchor-every',type=int,default=4,help='joint-only uniform q1/q2/q3 anchor, cycling independently')
    p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--save-every',type=int,default=250)
    p.add_argument('--trials',type=int,default=2)
    p.add_argument('--test-views',type=int,default=0,help='0 = all test views; positive explicitly selects a subset')
    p.add_argument('--image-views',type=int,default=2)
    p.add_argument('--rate-chunk-size',type=int,default=65536,help='single-draw expectation only; no ten-draw pattern graph')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--resume',action='store_true',help='restore codec/table/Adam/RNG from OUT/checkpoints/latest/state.pt')
    p.add_argument('--dry-run',action='store_true',help='check configuration/checkpoint and scene paths without rendering or writing')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--fixed-deployment',action='store_true',help='finetune the saved final tier map; no allocator or rate gradients; allocation-steps must be 0')
    mode.add_argument('--continue-allocation',action='store_true',help='control: initialize the learned final allocation table as well as codec; fresh Adam')
    return p


def configure_allocation_phase(model,phase,lr):
    if phase not in ('allocation','joint'): raise ValueError('unknown allocation phase')
    model.requires_grad_(phase=='joint')
    model.zero_grad(set_to_none=True)
    return (torch.optim.Adam([dict(name=n,params=list(getattr(model,n).parameters()),lr=lr) for n in MODULES])
            if phase=='joint' else None)


def measured_cost(model,scene,q,snr,identity):
    """q is packet/Morton order; source XYZ only for transmitted rows is charged."""
    q=q.detach().cpu().long()
    xyz=encode_positions(scene.geometry.normalize(scene.raw[:,:3]),q,model.cfg)
    tx,rx=conditioning_pair(model,snr)
    header={'version':2,'config':model.cfg.to_dict(),'model_id':identity,
        'count':int((q>0).sum()),'source_count':len(q),'geometry':scene.geometry.to_dict(),
        'snr_db':snr,'channel':'awgn','tx_condition_snr_db':tx,'rx_condition_snr_db':rx}
    meta=encode_metadata(header,q.numpy())
    payload=int(torch.tensor(model.cfg.rates)[q].sum())
    bits=scene.net_bits_per_use
    xyz_uses=math.ceil(8*len(xyz)/bits);meta_uses=math.ceil(8*len(meta)/bits)
    record=dict(payload_complex_symbols=payload,symbols_per_source_gaussian=payload/len(q),
        xyz_bytes=len(xyz),metadata_bytes=len(meta),coordinate_channel_uses=xyz_uses,
        metadata_channel_uses=meta_uses,total_channel_uses=payload+xyz_uses+meta_uses,
        net_bits_per_use_assumption=bits,
        rate_scope='analogue payload + measured retained XYZ + complete metadata; shared weights excluded',
        reliability='XYZ/metadata assumed reliable; no FEC/packet loss simulated')
    if hasattr(model,'scale_rate_record'):
        record.update(model.scale_rate_record(scene,q,bits))
        record['total_channel_uses']+=record['scale_channel_uses']
        record['rate_scope']='analogue payload + measured retained XYZ + complete metadata + binary16 scale frames; shared weights excluded'
    return record


@torch.no_grad()
def evaluate(model,scene,mask,out,step,phase,args,save_images=False,*,fixed_tiers=None):
    from .rendering import render
    device=next(model.parameters()).device; was_training=model.training
    out=Path(out); deployment=out/'deployment';deployment.mkdir(exist_ok=True)
    records=[]
    with preserved_rng(device):
        try:
            model.eval()
            if fixed_tiers is None:
                q_original,info=record_allocation(deployment,step,mask,args.snr,model.cfg.rates,
                    seed=args.seed,save_probabilities=True)
            else:
                q_original=fixed_tiers.detach().cpu().clone()
                info=dict(step=step,snr=args.snr,rates=list(model.cfg.rates),source_gaussians=len(q_original),
                    hard_tier_counts=torch.bincount(q_original,minlength=4).tolist(),
                    decision='unchanged saved final deployment map; no resampling')
            if phase in ('initial','end_allocation','final'):
                folder=deployment/phase;folder.mkdir(exist_ok=True)
                np.save(folder/'tiers.npy',q_original.numpy().astype(np.uint8),allow_pickle=False)
                json_write(folder/'allocation.json',info)
            identity=model_id(model)
            perceptual=LPIPSEvaluator(device) if args.lpips else None
            chosen_views=set(spaced_indices(len(scene.val_cameras),args.image_views))
            for layout in ('1','2','3','deployment'):
                q=q_original if layout=='deployment' else torch.full_like(q_original,int(layout))
                qs=[torch.where(ids>=0,q[ids.clamp_min(0)],0) for ids in scene.group_ids]
                q_sorted=torch.cat([v[ids>=0] for v,ids in zip(qs,scene.group_ids)])
                counts=torch.bincount(q,minlength=4).tolist()
                observations=[]; diagnostics=[]
                for trial in range(args.trials):
                    seed_all(args.seed+1000+trial)
                    if (q>0).any():
                        received,diag=decode_path(model,scene,qs,'awgn',args.snr)
                    else:
                        received=scene.raw[:0].to(device);diag={'latent_mse':None,'latent_nmse':None}
                    diagnostics.append(diag)
                    for i,camera in enumerate(scene.val_cameras):
                        image=render(received,camera,scene.degree,scene.white_background)
                        source=scene.reference.get(camera,device);photo=camera.original_image[:3].to(device)
                        metrics={prefix+'_'+k:v for prefix,target in (('source',source),('photo',photo))
                                 for k,v in image_metrics(image,target).items()}
                        if perceptual is not None:
                            metrics.update(source_lpips=perceptual(image,source),photo_lpips=perceptual(image,photo))
                        observations.append(dict(trial=trial,view=i,image_name=str(camera.image_name),**metrics))
                        if save_images and trial==0 and i in chosen_views:
                            stem=out/'images'/phase/f'{layout}_view{i:04d}'
                            save_panel(stem.with_suffix('.png'),photo,source,image,layout)
                            if layout=='deployment':
                                colored,labels=render_tier_map(received,q_sorted[q_sorted>0],camera,scene.degree)
                                save_tier_panel(stem.with_name(stem.name+'_tiers.png'),photo,source,image,colored,
                                                model.cfg.rates,counts)
                                from PIL import Image
                                Image.fromarray(labels.cpu().numpy().astype(np.uint8)).save(stem.with_name(stem.name+'_tier_labels.png'))
                row=dict(step=step,phase=phase,path='awgn',layout=layout,snr=args.snr,
                    codec_id=identity,rates=list(model.cfg.rates),tier_counts=counts,
                    tier_shares=[n/len(q) for n in counts],source_gaussians=len(q),
                    retained_gaussians=int((q>0).sum()),views=len(scene.val_cameras),trials=args.trials,
                    per_view_trial=observations,image_metric_protocol=image_metric_protocol(args.lpips),
                    deployment_draws=10 if layout=='deployment' and fixed_tiers is None else None,
                    allocation_training_sampling='single' if fixed_tiers is None else 'fixed_saved_deployment',
                    deployment_seed=args.seed)
                keys=[k for k in observations[0] if k not in ('trial','view','image_name')]
                for key in keys:
                    trial_means=[float(np.mean([r[key] for r in observations if r['trial']==j])) for j in range(args.trials)]
                    row[key]=float(np.mean(trial_means));row[key+'_std_trials']=float(np.std(trial_means))
                for key in ('latent_mse','latent_nmse'):
                    row[key]=float(np.mean([d[key] for d in diagnostics])) if diagnostics[0][key] is not None else None
                row.update(measured_cost(model,scene,q_sorted,args.snr,identity))
                append_json(out/'metrics'/('test.jsonl' if phase=='test' else 'validation.jsonl'),row)
                records.append(row)
                detail=f", LPIPS={row['source_lpips']:.4f}" if args.lpips else ''
                print(f"validation {step} {phase}/{layout}: PSNR={row['source_psnr']:.3f}, SSIM={row['source_ssim']:.4f}{detail}; counts={counts}",flush=True)
        finally:
            model.train(was_training)
    return records


def save_state(out,label,model,mask,mask_optimizer,codec_optimizer,progress,scene,settings,initial_id,*,fixed_tiers=None):
    folder=Path(out)/'checkpoints'/label;folder.mkdir(parents=True,exist_ok=True)
    state=dict(version=1,config=model.cfg.to_dict(),state_dict=model.state_dict(),mask=None if mask is None else mask.state_dict(),
        mask_optimizer=None if mask_optimizer is None else mask_optimizer.state_dict(),codec_optimizer=None if codec_optimizer is None else codec_optimizer.state_dict(),
        progress=dict(progress),rng=capture_rng(),scene=scene.record,settings=settings,initial_codec_id=initial_id)
    if fixed_tiers is not None:state['fixed_tiers']=fixed_tiers.detach().cpu().clone()
    temporary=folder/'state.pt.tmp';torch.save(state,temporary);temporary.replace(folder/'state.pt')
    # Standard inference/trained-codec fork entry; exact allocation resume uses state.pt.
    inference_progress=dict(phase='channel_render',phase_complete=True,step=progress['step'],
        channel_trained=True,train_snrs=[settings['snr']],completed={'channel_render':progress['completed']['joint']})
    save_training_checkpoint(folder/'codec.pt',model,None,inference_progress,capture_rng(),scene.record)
    if mask is not None:
        torch.save(dict(version=2,count=len(scene.raw),scene_fingerprint=scene.fingerprint,snr_conditioned=False,
            state_dict={k:v.detach().cpu() for k,v in mask.state_dict().items()},codec_id=model_id(model),
            rates=list(model.cfg.rates),step=progress['step']),folder/'allocation.pt')
    json_write(Path(out)/'progress.json',dict(progress,status='running'))


def main(args,max_updates=None):
    if args.image_objective!='mse':
        if not args.fixed_deployment:
            raise ValueError('perceptual objectives require --fixed-deployment')
        if args.image_aux_weight is None:
            from .perceptual_objective import PAPER_WEIGHTS
            args.image_aux_weight=PAPER_WEIGHTS[args.image_objective]
        if not math.isfinite(args.image_aux_weight) or args.image_aux_weight<=0:
            raise ValueError('image-aux-weight must be finite and positive')
    if (args.fixed_deployment or args.continue_allocation) and args.allocation_steps!=0:
        raise ValueError('finetuning a completed allocation requires --allocation-steps 0')
    if (min(args.allocation_steps,args.joint_steps,args.test_views,args.prefix_anchor_every)<0 or
            min(args.validate_every,args.save_every,args.trials,args.image_views,args.rate_chunk_size,args.cpu_threads)<1 or
            not math.isfinite(args.snr) or not math.isfinite(args.beta) or args.beta<0):
        raise ValueError('invalid budgets, evaluation settings or beta/SNR')
    for name in ('mask_lr','keep_lr','mask_adam_eps','codec_lr'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0:raise ValueError('invalid '+name)
    out=Path(args.out).resolve();run=Path(args.run).resolve()
    source,manifest_path,spec,old=inputs(args)
    source_checkpoint=run/'checkpoints/final/codec.pt'
    device=device_for(args.device);torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    model,source_progress,saved_scene=initialize_codec_run(source_checkpoint,device)
    if model.cfg.rates!=(0,8,24,48) or model.cfg.channel_variant!='progressive':
        raise ValueError('this experiment requires the trained progressive 8/24/48 codec; no silent architecture upgrade')
    source_id=model_id(model)
    settings={k:v for k,v in vars(args).items() if k not in ('out','resume','dry_run','manifest','run','device')}
    # False opt-ins must not invalidate exact resume of historical runs.
    for flag in ('fixed_deployment','continue_allocation'):
        if not settings[flag]:settings.pop(flag)
    if args.image_objective=='mse':
        # Preserve exact resume compatibility with pre-experiment MSE states.
        settings.pop('image_objective');settings.pop('image_aux_weight')
    settings.update(source_codec_id=source_id,manifest=str(manifest_path))
    fixed_q=None
    if args.fixed_deployment or args.continue_allocation:
        complete=json.loads((run/'complete.json').read_text(encoding='utf-8'))
        if complete['final_codec_id']!=source_id or not math.isclose(source['beta'],args.beta,abs_tol=1e-12,rel_tol=0):
            raise ValueError('source final codec or beta mismatch')
        if args.fixed_deployment:
            fixed_q=torch.from_numpy(np.load(run/'deployment/final/tiers.npy',allow_pickle=False))
            if (fixed_q.ndim!=1 or not fixed_q.numel() or fixed_q.is_floating_point() or
                    fixed_q.is_complex() or fixed_q.dtype==torch.bool or fixed_q.min()<0 or fixed_q.max()>3):
                raise ValueError('saved deployment tiers must be integer [N] q0..q3 labels')
            fixed_q=fixed_q.long()
            info=json.loads((run/'deployment/final/allocation.json').read_text(encoding='utf-8'))
            if (len(fixed_q)!=saved_scene['source_gaussians'] or not (fixed_q>0).any() or
                    torch.bincount(fixed_q,minlength=4).tolist()!=info['hard_tier_counts'] or
                    info['step']!=source_progress['step'] or info['snr']!=args.snr or info['rates']!=list(model.cfg.rates)):
                raise ValueError('saved final deployment map does not match source checkpoint/SNR')
    if args.dry_run:
        print(json.dumps(dict(source_checkpoint=str(source_checkpoint),settings=settings,rates=model.cfg.rates),indent=2))
        return
    if out.exists() and not args.resume:raise FileExistsError('use a NEW output directory or --resume')
    if args.resume and not (out/'checkpoints/latest/state.pt').exists():raise FileNotFoundError('missing latest allocation state')
    image_objective=None
    if args.image_objective!='mse':
        from .perceptual_objective import PerceptualImageObjective
        image_objective=PerceptualImageObjective(args.image_objective,args.image_aux_weight,device)

    def render_task(cameras):
        if image_objective is not None:
            from .perceptual_objective import PerceptualMultiViewRenderTask
            return PerceptualMultiViewRenderTask(cameras,scene.reference,degree,scene.white_background,image_objective)
        return (MultiViewRenderTask if args.fixed_deployment else MaskedMultiViewRenderTask)(
            cameras,scene.reference,degree,scene.white_background)

    raw,degree=read_ply(spec['ply'])
    scene=make_scene(spec,raw,degree,model,source,old,out)
    for key in ('fingerprint','source_gaussians','geometry','train_view_names','validation_view_names','resolution','images','white_background'):
        if scene.record[key]!=saved_scene[key]:raise ValueError('initializer scene mismatch: '+key)
    model.set_training_snrs([args.snr]);scene.channel_trained=True
    if args.fixed_deployment:
        mask=None;mask_optimizer=None
    else:
        if args.continue_allocation:
            from .route2 import load_mask
            mask=load_mask(run/'checkpoints/final/allocation.pt',raw,model,device)
        else:mask=GaussianTierMask(len(scene.raw),tier_count=4).to(device)
        mask_optimizer=torch.optim.Adam([dict(params=[mask.keep_logits],lr=args.keep_lr),
                                          dict(params=[mask.logits],lr=args.mask_lr)],eps=args.mask_adam_eps)
    del raw
    fixed_batches=([(f,torch.where(ids>=0,fixed_q[ids.clamp_min(0)],0))
                    for f,ids in zip(scene.groups,scene.group_ids)] if fixed_q is not None else None)
    progress=dict(step=0,phase='allocation',completed={'allocation':0,'joint':0},frozen_codec_verified=False)
    codec_optimizer=None
    for folder in ('metrics','checkpoints','images','deployment'): (out/folder).mkdir(parents=True,exist_ok=True)
    if args.resume:
        state=torch.load(out/'checkpoints/latest/state.pt',map_location='cpu',weights_only=True)
        if state['settings']!=settings or state['scene']!=scene.record or state['config']!=model.cfg.to_dict():
            raise ValueError('exact resume configuration/source/scene mismatch')
        model.load_state_dict(state['state_dict'])
        if mask is not None:
            mask.load_state_dict(state['mask']);mask_optimizer.load_state_dict(state['mask_optimizer'])
        elif not torch.equal(state.get('fixed_tiers',torch.empty(0)),fixed_q):
            raise ValueError('fixed deployment map changed since resume checkpoint')
        progress=state['progress'];source_id=state['initial_codec_id']
        codec_optimizer=configure_allocation_phase(model,progress['phase'],args.codec_lr)
        if codec_optimizer is not None:codec_optimizer.load_state_dict(state['codec_optimizer'])
        repair_logs(out,progress['step'])
        history=out/'deployment/allocation_history.jsonl'
        if history.exists():
            rows=[r for r in history.read_text(encoding='utf-8').splitlines() if json.loads(r)['step']<=progress['step']]
            history.write_text('\n'.join(rows)+'\n',encoding='utf-8')
        restore_rng(state['rng'])
    else:
        if args.lpips:
            # Fail early if pretrained evaluation weights are missing.
            preflight=LPIPSEvaluator(device);del preflight
        if device.type=='cuda' and not args.fixed_deployment:
            from .mask_checks import check_masked_renderer
            json_write(out/'mask_renderer_check.json',check_masked_renderer(scene.raw.to(device),scene.cameras[0],degree,scene.white_background))
        json_write(out/'training.json',dict(source_run=str(run),initializer=str(source_checkpoint),
            codec_config=model.cfg.to_dict(),
            selected_scenes=[spec],scene=scene.record,**settings,
            allocation_sampling='fixed_saved_deployment' if args.fixed_deployment else 'single',deployment_draws=None if args.fixed_deployment else 10,render_backward='replay',
            joint_trainable_modules=list(MODULES),lpips_metrics=args.lpips,
            objective=('source PLY '+image_objective.protocol()['formula']+'; fixed budget, no rate/latent/attribute loss'
                if image_objective is not None else
                'source PLY RGB MSE; fixed budget, no rate/latent/attribute loss' if args.fixed_deployment else
                'source PLY RGB MSE + beta * normalized expected payload/XYZ/tier proxy'),
            prefix_anchor='joint: every N steps, cyclic uniform q1/q2/q3, equal codec image gradient mean; mask gradients unchanged',
            **(dict(render_objective=image_objective.protocol())
               if image_objective is not None else {}),
            initialization=('full trained codec; saved deployment map; fresh Adam' if args.fixed_deployment else
                'full trained codec and learned allocation table; fresh Adam' if args.continue_allocation else
                'full trained codec; new .99-keep / uniform-positive scene table; fresh Adam')))
        configure_allocation_phase(model,'allocation',args.codec_lr)
        evaluate(model,scene,mask,out,0,'initial',args,save_images=True,fixed_tiers=fixed_q)
        save_state(out,'initial',model,mask,mask_optimizer,None,progress,scene,settings,source_id,fixed_tiers=fixed_q)

    def save(label):save_state(out,label,model,mask,mask_optimizer,codec_optimizer,progress,scene,settings,source_id,fixed_tiers=fixed_q)
    updated=0
    for phase,budget in (('allocation',args.allocation_steps),('joint',args.joint_steps)):
        if phase=='joint' and not progress['frozen_codec_verified']:
            if model_id(model)!=source_id:raise RuntimeError('frozen allocation phase changed the codec')
            progress['frozen_codec_verified']=True
            save('end_allocation')
        if progress['completed'][phase]>=budget:continue
        if progress['phase']!=phase or (phase=='joint' and codec_optimizer is None):
            codec_optimizer=configure_allocation_phase(model,phase,args.codec_lr)
        elif phase=='allocation':configure_allocation_phase(model,phase,args.codec_lr)
        progress['phase']=phase
        while progress['completed'][phase]<budget:
            visit=progress['completed'][phase]+1
            model.train();model.zero_grad(set_to_none=True)
            if mask_optimizer is not None:mask_optimizer.zero_grad(set_to_none=True)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            views=random.sample(range(len(scene.cameras)),scene.views_per_step)
            cameras=[scene.cameras[i] for i in views]
            task=render_task(cameras)
            start=time.perf_counter()
            if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            if args.fixed_deployment:
                loss,stats=full_scene_step(model,fixed_batches,scene.geometry,args.snr,'awgn',
                    task,attr_weight=0.,mode='replay')
                stats.update(rate_loss=0.,allocation_sampling='fixed_saved_deployment',
                    sampled_tier_counts=torch.bincount(fixed_q,minlength=4).tolist(),codec_updated=True)
            else:
                loss,stats=local_mask_step(model,mask,scene.groups,scene.group_ids,scene.geometry,args.snr,'awgn',
                    task,args.beta,scene.rate_meter,train_codec=phase=='joint',mode='replay',
                    sampling='single',rate_chunk_size=args.rate_chunk_size)
            stats.update(task.stats)
            if phase=='joint':
                # For the historical nonfixed mode the anchor has no existence column.
                anchor_task=render_task(cameras) if args.fixed_deployment else MultiViewRenderTask(cameras,scene.reference,degree,scene.white_background)
                stats.update(preserve_prefix(model,scene.groups,scene.group_ids,scene.geometry,args.snr,'awgn',
                    anchor_task,visit,args.prefix_anchor_every,3))
                if image_objective is not None:
                    if stats['prefix_anchor_applied']:
                        stats.update({'prefix_anchor_'+key:value for key,value in anchor_task.stats.items()})
                    stats['optimization_image_loss']=((stats['render_loss']+stats['prefix_anchor_render_loss'])/2
                        if stats['prefix_anchor_applied'] else stats['render_loss'])
            norm,gradient_stats=clip_codec_gradients(model,mode='none')
            if not torch.isfinite(loss) or (mask is not None and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in mask.parameters())):
                raise RuntimeError('nonfinite loss/allocation gradient; no update performed')
            if mask_optimizer is not None:mask_optimizer.step()
            if codec_optimizer is not None:codec_optimizer.step()
            changes=update_stats(model,before)
            if phase=='allocation' and any(v['update_norm']!=0 for v in changes.values()):raise RuntimeError('frozen codec updated')
            if device.type=='cuda':torch.cuda.synchronize(device)
            progress['step']+=1;progress['completed'][phase]=visit
            row=dict(stats,**gradient_stats,step=progress['step'],phase=phase,phase_step=visit,loss=float(loss),
                grad_norm=float(norm),updates=changes,training_view_indices=views,snr=args.snr,
                step_seconds=time.perf_counter()-start,
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            append_json(out/'metrics/loss.jsonl',row)
            if visit==1 or visit%10==0 or visit==budget:
                detail=(f", mse={stats['image_mse']:.6g}, weighted_aux={stats['image_aux_weighted']:.6g}"
                    if image_objective is not None else '')
                print(f"{phase} {visit}/{budget}: render={stats['render_loss']:.6g}{detail}, rate={stats['rate_loss']:.6g}, counts={stats['sampled_tier_counts']}, sec={row['step_seconds']:.2f}",flush=True)
            if progress['step']%args.validate_every==0 or visit==budget:
                evaluate(model,scene,mask,out,progress['step'],phase,args,fixed_tiers=fixed_q)
            if progress['step']%args.save_every==0 or visit==budget:save('latest')
            updated+=1
            if max_updates is not None and updated>=max_updates:save('latest');return progress
        if phase=='allocation':
            if model_id(model)!=source_id:raise RuntimeError('frozen allocation phase changed weights/statistics')
            progress['frozen_codec_verified']=True
        save('end_'+phase);save('latest')
    evaluate(model,scene,mask,out,progress['step'],'final',args,save_images=True,fixed_tiers=fixed_q)
    from .rendering import load_cameras
    cameras=load_cameras(spec['source'],scene.record['resolution'],scene.white_background,scene.record['images'],'test')
    if not cameras:raise ValueError('no held-out test cameras')
    scene.val_cameras=([cameras[i] for i in spaced_indices(len(cameras),args.test_views)] if args.test_views else cameras)
    evaluate(model,scene,mask,out,progress['step'],'test',args,fixed_tiers=fixed_q)
    save('final');save('latest')
    json_write(out/'complete.json',dict(status='complete',completed=progress['completed'],
        frozen_codec_verified=progress['frozen_codec_verified'],rates=list(model.cfg.rates),
        final_codec_id=model_id(model),initial_codec_id=source_id))
    json_write(out/'progress.json',dict(progress,status='complete'))
    return progress


if __name__=='__main__':main(parser().parse_args())
