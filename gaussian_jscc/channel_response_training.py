"""Frozen-semantic channel recovery with latent and isolated Gaussian responses.

Rendering is evaluation-only. No allocator, semantic update or joint phase.
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .channel_capacity_controls import semantic_digest
from .channel_normalization_metrics import cache_teacher
from .cli import device_for
from .codec import channel, prefix_mask
from .data import read_ply
from .decoder_adaptation import build_plan as source_plan, parser as source_parser, evaluate_saved
from .local_response import local_response_loss
from .modular_checkpoint import (capture_rng, restore_rng, restore_training_checkpoint,
                                 save_training_checkpoint, validate_scene)
from .modular_experiment import json_write, make_scene, repair_logs
from .modular_snr_controls import control_inputs
from .modular_training import configure_phase, finish_update, latent_reconstruction_loss
from .optimization import preserved_rng
from .power_followup_evaluation import bind_test_cameras
from .render_validation import append_json


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='block channel-only endpoint, or power-stability suite root')
    p.add_argument('--out',required=True,help='NEW output directory')
    p.add_argument('--manifest',help='relocated paths; same PLY identity and camera split')
    p.add_argument('--steps',type=int,default=2000,help='additional channel updates; no joint phase')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--response-directions',type=int,default=8)
    p.add_argument('--calibration-batches',type=int,default=4)
    p.add_argument('--response-weight',type=float,help='explicit fixed override; otherwise one-off gradient RMS calibration')
    p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--test-views',type=int,default=32)
    p.add_argument('--render-trials',type=int,default=4)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def build_plan(args):
    if min(args.response_directions,args.calibration_batches)<1:
        raise ValueError('response directions/calibration batches must be positive')
    if args.response_weight is not None and (not math.isfinite(args.response_weight) or args.response_weight<0):
        raise ValueError('response weight must be finite and nonnegative')
    # Reuse endpoint/split/overlap validation, never a decoder-adapted initializer.
    base=source_parser().parse_args(['--run',args.run,'--out',args.out])
    for key in ('steps','lr','validate_every','save_every','test_views','render_trials','seed','device','cpu_threads','manifest'):
        setattr(base,key,getattr(args,key))
    old=source_plan(base)
    saved=torch.load(old['checkpoint'],map_location='cpu',weights_only=True)
    if saved['config'].get('channel_power_mode')!='block':
        raise ValueError('this controlled experiment requires the block-normalized channel endpoint')
    return dict(version=1,source_run=old['source_run'],checkpoint=old['checkpoint'],
        source_sha256=old['source_sha256'],source_step=old['source_step'],
        semantic_sha256=semantic_digest(saved['state_dict']),
        settings={k:getattr(args,k) for k in ('steps','lr','response_directions','calibration_batches',
            'response_weight','validate_every','save_every','test_views','render_trials','lpips','seed','device','cpu_threads')},
        manifest_override=old['manifest_override'],
        trainable_modules=['channel_encoder','channel_decoder'],
        frozen_modules=['semantic_encoder','semantic_decoder','attribute normalization'],
        objective='original-y MSE + fixed lambda * frozen-teacher isolated Gaussian black/white RGB response MSE',
        calibration='RMS gradients w.r.t. recovered y on initial training batches; ratio pooled across batches; fixed thereafter',
        training='10dB AWGN; 8/24/48; q1/q2/q3/mixed; original blocks; NEW channel Adam',
        rendering='evaluation only; no joint phase, no allocator, no gradient clipping')


def frozen_teacher(model):
    teacher=copy.deepcopy(model.semantic_decoder).eval().requires_grad_(False)
    for p in teacher.parameters(): p.grad=None
    return teacher


def directions(count,seed,device):
    # Independent of channel noise and evaluator RNG; isotropic directions, not test cameras.
    return F.normalize(torch.randn(count,3,generator=torch.Generator().manual_seed(seed)),dim=-1).to(device)


def response_loss(model,teacher,y,y_hat,q,xyz,geometry,views):
    active=q>0
    if not active.any(): raise ValueError('response loss requires retained points')
    side=xyz.detach().to(y_hat).masked_fill(~active[...,None],0)
    # Complete original blocks reach BOTH decoders: never collapse the point axis
    # before spatial context. Only the isolated-response evaluation flattens it.
    with torch.no_grad(): target=torch.cat((side,teacher(y.detach(),active,side)),-1)
    pred=torch.cat((side,teacher(y_hat,active,side)),-1)
    return local_response_loss(pred[active],target[active],geometry,model,directions=views)


def channel_loss(model,teacher,y,y_hat,q,xyz,geometry,views,weight):
    latent,metrics=latent_reconstruction_loss(y,y_hat,q,len(model.cfg.rates))
    response,parts=response_loss(model,teacher,y,y_hat,q,xyz,geometry,views)
    loss=latent+weight*response
    if not torch.isfinite(loss): raise RuntimeError('nonfinite response channel loss; no optimizer update')
    return loss,dict(latent_mse=float(latent.detach()),latent_nmse=float(latent.detach()/metrics['latent_target_power'].clamp_min(1e-12)),
        response_mse=float(response.detach()),weighted_response_mse=float(weight*response.detach()),
        response_weight=weight,local_black_mse=parts['local_black_mse'],local_white_mse=parts['local_white_mse'],
        objective='latent_mse+frozen_teacher_local_response',weighted_latent_loss=float(latent.detach()))


def gradient_calibration(latent,response,recovered):
    gy,=torch.autograd.grad(latent,recovered,retain_graph=True)
    gr,=torch.autograd.grad(response,recovered)
    a=float(gy.detach().double().square().sum()); b=float(gr.detach().double().square().sum())
    if not all(math.isfinite(x) and x>0 for x in (a,b)):
        raise ValueError('degenerate initial gradients; inspect inputs or explicitly set --response-weight')
    return dict(latent_gradient_squared_sum=a,response_gradient_squared_sum=b,elements=gy.numel(),response_weight=math.sqrt(a/b))


def transmit(model,y,q,xyz,seed):
    z=model.channel_encoder(y.detach(),q>0,10.)
    mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
    noise=torch.Generator(device=y.device).manual_seed(seed)
    received=channel(z.reshape(-1,2),10.,'awgn',noise).reshape_as(z)*mask
    options=({'delivered_xyz':xyz} if model.cfg.channel_denoiser_mode!='none'
             or model.cfg.channel_neighbor_mode!='none' else {})
    return model.channel_decoder(received,mask,q,10.,**options)


def batch_inputs(model,cache,args,local,*,calibration=False):
    device=next(model.parameters()).device
    # Calibration and training use disjoint deterministic draws, but identical rules.
    step=local+(100000000 if calibration else 0)
    index=random.Random(args.seed+100003*step).randrange(len(cache))
    batch=cache[index]
    y,active,xyz=[batch[k].to(device) for k in ('y','active','xyz')]
    layout=('1','2','3','mixed')[(local-1)%4]
    if layout=='mixed':
        q=torch.randint(1,4,active.shape,generator=torch.Generator().manual_seed(args.seed+300007*step)).to(device)
    else: q=torch.full(active.shape,int(layout),dtype=torch.long,device=device)
    q=q.masked_fill(~active,0)
    return y,q,xyz,directions(args.response_directions,args.seed+700001*step,device),index,layout,step


def calibrate(model,teacher,cache,geometry,args):
    if args.response_weight is not None:
        return dict(method='explicit_override',response_weight=args.response_weight,batches=[])
    records=[]
    with preserved_rng(next(model.parameters()).device):
        model.eval()
        for local in range(1,args.calibration_batches+1):
            y,q,xyz,views,index,layout,step=batch_inputs(model,cache,args,local,calibration=True)
            with torch.no_grad(): recovered=transmit(model,y,q,xyz,args.seed+200003*step)
            recovered=recovered.detach().requires_grad_(True)
            latent,_=latent_reconstruction_loss(y,recovered,q,len(model.cfg.rates))
            response,_=response_loss(model,teacher,y,recovered,q,xyz,geometry,views)
            records.append(dict(gradient_calibration(latent,response,recovered),batch_index=index,layout=layout))
    a=sum(r['latent_gradient_squared_sum'] for r in records)
    b=sum(r['response_gradient_squared_sum'] for r in records)
    count=sum(r['elements'] for r in records)
    return dict(method='initial_pooled_gradient_rms',response_weight=math.sqrt(a/b),batches=records,
        latent_gradient_rms=math.sqrt(a/count),response_gradient_rms=math.sqrt(b/count),
        interpretation='initial gradient scale matching only; not optimal weights or dynamic balancing')


def training_mode(model):
    model.eval()
    model.channel_encoder.train(); model.channel_decoder.train()


def summarize(out):
    from .channel_audit import write_csv
    rows=[]
    for label in ('initial','final'):
        for line in (out/'evaluation'/label/'metrics/validation.jsonl').read_text(encoding='utf8').splitlines():
            record=json.loads(line)
            row={k:record[k] for k in ('path','layout','source_psnr','photo_psnr','source_ssim','photo_ssim',
                                     'latent_mse','latent_nmse','views','trials')}
            row.update({k:record[k] for k in ('source_lpips','photo_lpips') if k in record})
            row['endpoint']=label
            original=next((r for r in rows if r['endpoint']=='initial' and r['path']==row['path'] and r['layout']==row['layout']),None)
            row['source_psnr_change_db']=row['source_psnr']-original['source_psnr'] if original else None
            rows.append(row)
    json_write(out/'comparison/summary.json',dict(rows=rows,protocol='paired32 test views and4 noise trials by default',
        clean='must remain unchanged: both semantic modules frozen',mixed='fixed random positive tiers, not learned allocation'))
    write_csv(out/'comparison/endpoints.csv',rows)


def main(args,max_updates=None):
    plan=build_plan(args)
    if args.dry_run: print(json.dumps(plan,indent=2)); return plan
    out=Path(args.out).resolve(); existed=out.exists()
    if existed:
        if not args.resume: raise FileExistsError('use NEW output or --resume')
        if json.loads((out/'plan.json').read_text(encoding='utf8'))!=plan: raise ValueError('resume settings/source differ')
        if (out/'complete.json').exists(): return json.loads((out/'complete.json').read_text(encoding='utf8'))
    else:
        if args.resume: raise FileNotFoundError('resume requires existing output')
        out.mkdir(parents=True); json_write(out/'plan.json',plan)
    device=device_for(args.device); torch.set_num_threads(args.cpu_threads)
    latest=out/'checkpoints/latest/codec.pt'; fresh=not latest.exists()
    model,optimizer,progress,rng,record=restore_training_checkpoint(latest if not fresh else plan['checkpoint'],device)
    if fresh:
        optimizer=configure_phase(model,'channel_bootstrap',{'channel_encoder':args.lr,'channel_decoder':args.lr},freeze_semantic=True)
        progress=copy.deepcopy(progress)
        progress.update(phase='channel_bootstrap',phase_step=0,phase_complete=False,freeze_semantic=True,
            semantic_decoder_only=False,channel_denoiser_only=False,channel_neighbor_only=False,response_updates=0,
            module_lrs={'channel_encoder':args.lr,'channel_decoder':args.lr},initial_done=False,final_done=False,
            source_step=plan['source_step'],train_snrs=[10.],training_objective=dict(version='channel_response_v1',
                channel_bootstrap='latent_mse+frozen_teacher_local_response',response_directions=args.response_directions))
        progress['phase_budgets']['channel_bootstrap']=plan['source_step']+args.steps
        progress['phase_budgets']['channel_render']=0
    else:
        if progress['training_objective']['version']!='channel_response_v1': raise ValueError('not a response experiment checkpoint')
        repair_logs(out,progress['step'])
    if semantic_digest(model.state_dict())!=plan['semantic_sha256']: raise RuntimeError('semantic teacher changed')
    metadata=json.loads((Path(plan['source_run'])/'training.json').read_text(encoding='utf8'))
    original,manifest,spec,old,_=control_inputs(metadata,args.manifest)
    raw,degree=read_ply(spec['ply']); scene=make_scene(spec,raw,degree,model,original,old,out); del raw
    validate_scene(record,scene.record); scene.channel_trained=True
    names=bind_test_cameras(scene,spec,args.test_views)
    model.set_training_snrs([10.])
    with preserved_rng(device): cache=cache_teacher(model,scene)
    # Scene-span/context buffers are runtime-only, absent from state_dict.
    # Snapshot AFTER geometry binding, including on exact resume.
    teacher=frozen_teacher(model)
    restore_rng(rng)
    training=dict(source_run=plan['source_run'],manifest=str(manifest),selected_scenes=[spec],scene=scene.record,
        codec_config=model.cfg.to_dict(),channel_response=plan,test_view_names=names,
        phase_budgets=dict(channel_bootstrap=args.steps,channel_render=0),freeze_semantic=True,
        trainable_modules=['channel_encoder','channel_decoder'],lpips_metrics=args.lpips)
    if (out/'training.json').exists():
        if json.loads((out/'training.json').read_text(encoding='utf8'))!=json.loads(json.dumps(training)):
            raise ValueError('scene/test split changed')
    else: json_write(out/'training.json',training)

    def save(label='latest'):
        if semantic_digest(model.state_dict())!=plan['semantic_sha256']: raise RuntimeError('frozen semantic state changed')
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,capture_rng(),scene.record)

    def finish_evaluation():
        evaluate_saved(model,scene,out,args,progress['step'],**progress['pending_evaluation'])
        progress.pop('pending_evaluation'); save()

    def evaluate(label,scope='test',images=False):
        progress['pending_evaluation']=dict(label=label,scope=scope,images=images); save(); finish_evaluation()

    if fresh: save('initial'); save()
    if progress.get('pending_evaluation'): finish_evaluation()
    if 'response_weight' not in progress:
        calibration=calibrate(model,teacher,cache,scene.geometry,args)
        progress['response_weight']=calibration['response_weight']; progress['response_calibration']=calibration; save()
    json_write(out/'calibration.json',progress['response_calibration'])
    if not progress['initial_done']:
        evaluate('initial',images=True); progress['initial_done']=True; save()
    weight=progress['response_weight']
    print(f'Channel-only response: +{args.steps} updates, lr={args.lr}, lambda={weight:.8g}; '
          'both semantic modules FROZEN; rendered images evaluation-only',flush=True)
    updates=0
    while progress['response_updates']<args.steps:
        local=progress['response_updates']+1; step=plan['source_step']+local
        training_mode(model); optimizer.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        started=time.perf_counter()
        y,q,xyz,views,index,layout,noise_step=batch_inputs(model,cache,args,local)
        recovered=transmit(model,y,q,xyz,args.seed+200003*noise_step)
        loss,stats=channel_loss(model,teacher,y,recovered,q,xyz,scene.geometry,views,weight)
        loss.backward(); stats.update(finish_update(model,optimizer,before))
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            stats.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
        progress.update(step=step,phase_step=local,response_updates=local,phase_complete=local==args.steps)
        progress['completed']['channel_bootstrap']=step
        append_json(out/'metrics/loss.jsonl',dict(stats,loss=float(loss.detach()),step=step,phase='channel_response',phase_step=local,
            layout=layout,batch_index=index,lr=args.lr,snr=10.,step_seconds=time.perf_counter()-started))
        if local==1 or local%10==0:
            print(f'channel {local}/{args.steps}: q={layout}, loss={float(loss.detach()):.6g}, '
                  f'y_mse={stats["latent_mse"]:.6g}, response={stats["response_mse"]:.6g}, '
                  f'weighted_response={stats["weighted_response_mse"]:.6g}, grad={stats["grad_norm"]:.5g}',flush=True)
        if local%args.validate_every==0 and not progress['phase_complete']:
            evaluate(f'response_{local:06d}',scope='validation')
        if local%args.save_every==0 or local%args.validate_every==0 or progress['phase_complete']: save()
        updates+=1
        if max_updates is not None and updates>=max_updates:
            save(); return dict(status='interrupted',step=step)
    save('end_channel_bootstrap')
    if not progress['final_done']:
        evaluate('final',images=True); progress['final_done']=True
    summarize(out); save('final'); save()
    result=dict(status='complete',step=progress['step'],channel_updates=args.steps,joint_updates=0,
        response_weight=weight,frozen_semantic_state_unchanged=True,source_step=plan['source_step'])
    json_write(out/'complete.json',result)
    return result


if __name__=='__main__': main(parser().parse_args())
