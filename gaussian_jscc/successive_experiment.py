"""Conditional progressive channel experiment; render forward is evaluation-only."""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from . import channel_response_training as base
from .channel_capacity_controls import semantic_digest
from .channel_normalization_controls import channel_digest
from .channel_normalization_metrics import cache_teacher
from .cli import device_for,seed_all
from .codec import channel,prefix_mask
from .data import read_ply
from .decoder_adaptation import evaluate_saved
from .model_factory import build_codec,config_from_dict
from .modular_checkpoint import capture_rng,restore_rng,save_training_checkpoint,validate_scene
from .modular_experiment import json_write,make_scene,repair_logs
from .modular_snr_controls import control_inputs
from .modular_training import finish_update
from .optimization import preserved_rng
from .power_followup_evaluation import bind_test_cameras
from .render_validation import append_json
from .successive_training import PHASES,configure_optimizer,inputs,loss_for_batch,optimizer_group_updates


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--manifest')
    for name,value in [('main-steps',8000),('neighbor-steps',1000),('cooperative-steps',1000),
                       ('response-directions',8),('validate-every',500),('save-every',100),
                       ('test-views',32),('render-trials',4),('cpu-threads',4),('seed',42),
                       ('color-every',4),('color-start',500),('color-stat-start',250),('noiseless-every',4)]:
        p.add_argument('--'+name,type=int,default=value)
    for name,value in [('lr',1e-4),('neighbor-lr',1e-4),('cooperative-lr',1e-5),
                       ('response-weight',.1),('color-relative-gradient',.25),('noiseless-weight',.25)]:
        p.add_argument('--'+name,type=float,default=value)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    return p


def build_plan(args):
    budgets={phase:getattr(args,phase+'_steps') for phase in PHASES}
    if (budgets['main']<1 or min(budgets.values())<0 or
        min(args.validate_every,args.save_every,args.test_views,args.render_trials,args.cpu_threads,
            args.color_every,args.noiseless_every)<1 or args.response_directions<2 or args.response_directions%2
        or args.color_stat_start<0 or args.color_start<args.color_stat_start):
        raise ValueError('invalid training/evaluation budgets or color schedule')
    for key in ('lr','neighbor_lr','cooperative_lr','response_weight','color_relative_gradient','noiseless_weight'):
        value=getattr(args,key)
        if not math.isfinite(value) or value<=0:raise ValueError('positive finite '+key+' required')
    opts=base.parser().parse_args(['--run',args.run,'--out',args.out,'--channel-init','random'])
    for key in ('lr','response_directions','validate_every','save_every','test_views','render_trials','lpips','seed','device','cpu_threads','manifest'):
        setattr(opts,key,getattr(args,key))
    opts.steps=sum(budgets.values())
    plan=base.build_plan(opts)
    source=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
    cfg=dict(source['config'],channel_variant='successive',channel_prefix_mode='full_prefix',
             channel_power_mode='block',channel_denoiser_mode='none',channel_denoiser_scope='stage',
             channel_neighbor_mode='plain')
    config_from_dict(cfg)  # validate before loading PLY or creating an output.
    plan.update(version='successive_channel_v1',codec_config=cfg,budgets=budgets,
        settings={k:v for k,v in vars(args).items() if k not in ('run','out','manifest','resume','dry_run')},
        objective='every-step latent+response; measured weak color; pre/post-neighbor anchors',
        calibration='training EMA after warm-up; no initial-four-batch calibration',
        training='random conditional sender; all-prefix main; neighbor then low-LR channel cooperation',
        scale_delivery='SCL1 uint32 count + little-endian binary16 donor-block scales; reliable, counted',
        rendering='evaluation-only; semantics frozen; no allocator',
        trainable_modules_by_phase={'main':['channel_encoder','channel_decoder.stages'],
          'neighbor':['channel_decoder.neighbor_refiner'],
          'cooperative':['channel_encoder','channel_decoder.stages','channel_decoder.neighbor_refiner']})
    # JSON-canonical identity: source checkpoints may store tuples, while the
    # persisted plan necessarily stores lists. Exact resume compares this form.
    return json.loads(json.dumps(plan))


def initialize(source,cfg,seed,device):
    seed_all(seed);model=build_codec(config_from_dict(cfg)).to(device)
    semantic={k:v for k,v in source['state_dict'].items() if k.startswith(('semantic_encoder.','semantic_decoder.')) or k in ('attr_mean','attr_std')}
    result=model.load_state_dict(semantic,strict=False)
    if result.unexpected_keys or any(not n.startswith(('channel_encoder.','channel_decoder.')) for n in result.missing_keys):
        raise ValueError('semantic source shape/configuration mismatch')
    return model


@torch.no_grad()
def diagnostics(model,teacher,cache,scene,centers,args,out,label,step):
    from .channel_supervision_controls import color_directions
    from .successive_objective import common_terms
    from .successive_channel import encode_scales,decode_scales
    batch=cache[0];device=next(model.parameters()).device
    # Four original blocks, not a regrouped set of individual points.
    y,active,xyz=[batch[k][:4].to(device) for k in ('y','active','xyz')]
    rows=[]
    for tier in (1,2,3,'mixed'):
        q=(torch.randint(1,4,active.shape,generator=torch.Generator().manual_seed(args.seed+7001)).to(device)
           if tier=='mixed' else active.long()*tier).masked_fill(~active,0)
        signal,scales=model.channel_encoder(y,q,10.,model.channel_decoder)
        frame=encode_scales(scales,q);scales=decode_scales(frame,q,dtype=y.dtype)
        mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(signal)
        views,_=color_directions(xyz[q>0],scene.geometry,centers,args.response_directions,args.seed+991)
        for kind in ('none','awgn'):
            gen=torch.Generator(device=device).manual_seed(args.seed+1991)
            received=channel(signal.reshape(-1,2),10.,kind,gen).reshape_as(signal)*mask
            stages,final=model.channel_decoder(received,mask,q,10.,scales=scales,delivered_xyz=xyz,return_stages=True)
            details=[]
            for value in (stages[...,-1,:],final):
                _,_,terms=common_terms(model,teacher,y,value,q,xyz,scene.geometry,views,args.response_weight)
                details.append({k:float(v) for k,v in terms.items()})
            rows.append(dict(layout=str(tier),channel=kind,points=int((q>0).sum()),
                main=details[0],final=details[1],scale_stream_bytes=len(frame),
                prefix_latent_mse=[float((stages[...,i,:][q>0]-y[q>0]).square().mean()) for i in range(3)],
                neighbor_enabled=bool(model.channel_decoder.neighbor_enabled)))
    json_write(out/'diagnostics'/f'{label}.json',dict(step=step,rows=rows,
        scope='first up-to-four ORIGINAL blocks from cached training batch0; fixed directions and one paired noise draw; not whole-scene quality'))


def main(args,max_updates=None):
    plan=build_plan(args)
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    out=Path(args.out).resolve();exists=out.exists()
    if exists:
        if not args.resume:raise FileExistsError('use NEW output or --resume')
        if json.loads((out/'plan.json').read_text(encoding='utf8'))!=plan:raise ValueError('resume settings/source differ')
        if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
    else:
        if args.resume:raise FileNotFoundError('resume output missing')
        out.mkdir(parents=True);json_write(out/'plan.json',plan)
    device=device_for(args.device);torch.set_num_threads(args.cpu_threads)
    latest=out/'checkpoints/latest/codec.pt';fresh=not latest.exists()
    rates=dict(main=args.lr,neighbor=args.neighbor_lr,cooperative=args.cooperative_lr)
    if fresh:
        source=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
        model=initialize(source,plan['codec_config'],args.seed,device);record=source['training']['scene_record']
        optimizer=configure_optimizer(model,'main',rates);seed_all(args.seed+17)
        progress=dict(step=0,phase='channel_bootstrap',successive_phase='main',completed={p:0 for p in PHASES},
            module_lrs=rates,freeze_semantic=True,color_state={},initial_done=False,final_done=False,
            training_objective=dict(version='successive_channel_v1'),train_snrs=[10.])
        rng=capture_rng()
        json_write(out/'initialization.json',dict(channel_initialization='random',channel_sha256=channel_digest(model),
            semantic_sha256=semantic_digest(model.state_dict()),source_channel_weights_loaded=False,
            source_optimizer_loaded=False,optimizer_state_entries=len(optimizer.state)))
    else:
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved['config']!=config_from_dict(plan['codec_config']).to_dict():raise ValueError('resume codec differs')
        model=build_codec(config_from_dict(saved['config'])).to(device);model.load_state_dict(saved['state_dict'])
        state=saved['training'];progress=state['progress'];record=state['scene_record'];rng=state['rng']
        optimizer=configure_optimizer(model,progress['successive_phase'],rates)
        optimizer.load_state_dict(state['optimizer']);repair_logs(out,progress['step'])
    if semantic_digest(model.state_dict())!=plan['semantic_sha256']:raise RuntimeError('semantic teacher differs')
    metadata=json.loads((Path(plan['source_run'])/'training.json').read_text(encoding='utf8'))
    original,manifest,spec,old,_=control_inputs(metadata,args.manifest)
    raw,degree=read_ply(spec['ply']);scene=make_scene(spec,raw,degree,model,original,old,out);del raw
    validate_scene(record,scene.record)
    names=bind_test_cameras(scene,spec,args.test_views);model.set_training_snrs([10.])
    with preserved_rng(device):cache=cache_teacher(model,scene)
    teacher=base.frozen_teacher(model)
    centers=torch.stack([torch.as_tensor(c.camera_center).detach().cpu() for c in scene.cameras]).to(device)
    restore_rng(rng)
    training=dict(source_run=plan['source_run'],manifest=str(manifest),selected_scenes=[spec],scene=scene.record,
        codec_config=model.cfg.to_dict(),successive_channel=plan,test_view_names=names,freeze_semantic=True,
        phase_budgets=plan['budgets'],lpips_metrics=args.lpips)
    if (out/'training.json').exists() and json.loads((out/'training.json').read_text(encoding='utf8'))!=json.loads(json.dumps(training)):
        raise ValueError('training scene/test identity changed')
    json_write(out/'training.json',training)

    def save(label='latest'):
        if semantic_digest(model.state_dict())!=plan['semantic_sha256']:raise RuntimeError('frozen semantics changed')
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,capture_rng(),scene.record)

    def finish_evaluation():
        pending=progress['pending_evaluation'];scene.channel_trained=progress['step']>0
        evaluate_saved(model,scene,out,args,progress['step'],**pending)
        with preserved_rng(device):diagnostics(model,teacher,cache,scene,centers,args,out,pending['label'],progress['step'])
        progress.pop('pending_evaluation');save()

    def evaluate(label,scope='validation',images=False):
        progress['pending_evaluation']=dict(label=label,scope=scope,images=images);save();finish_evaluation()

    if fresh:save('initial');save()
    if progress.get('pending_evaluation'):finish_evaluation()
    if not progress['initial_done']:
        evaluate('initial','test',True);progress['initial_done']=True;save()
    updates=0
    for phase in PHASES:
        budget=plan['budgets'][phase]
        if progress['completed'][phase]>=budget:continue
        if progress['successive_phase']!=phase:
            optimizer=configure_optimizer(model,phase,rates)
            progress['successive_phase']=phase
            save()
            print(f'Phase {phase}: NEW Adam; main LR={rates["cooperative"] if phase=="cooperative" else rates["main"]}; neighbor LR={rates["neighbor"]}',flush=True)
        while progress['completed'][phase]<budget:
            model.eval();model.channel_encoder.train();model.channel_decoder.train()
            phase_step=progress['completed'][phase]+1;step=progress['step']+1
            y,q,xyz,index,layout=inputs(cache,phase,phase_step,step,args.seed,device)
            optimizer.zero_grad(set_to_none=True)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
            started=time.perf_counter()
            loss,stats=loss_for_batch(model,teacher,y,q,xyz,scene.geometry,centers,args,progress)
            loss.backward();stats.update(finish_update(model,optimizer,before))
            stats['optimizer_group_updates']=optimizer_group_updates(model,optimizer,before)
            if device.type=='cuda':
                torch.cuda.synchronize(device)
                stats.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                             peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
            progress['step']=step;progress['completed'][phase]=phase_step
            append_json(out/'metrics/loss.jsonl',dict(stats,loss=float(loss.detach()),step=step,phase=phase,
                phase_step=phase_step,layout=layout,batch_index=index,snr=10.,step_seconds=time.perf_counter()-started))
            if phase_step==1 or phase_step%25==0 or phase_step==budget:
                print(f'{phase} {phase_step}/{budget}: loss={float(loss.detach()):.6g}, y={stats["latent_mse"]:.6g}, '
                      f'color_weight={stats["color_weight"]:.5g}, grad={stats["grad_norm"]:.5g}, sec={time.perf_counter()-started:.2f}',flush=True)
            if phase_step==budget:
                save('end_'+phase);evaluate('end_'+phase)
            elif step%args.validate_every==0:evaluate(f'step_{step:06d}')
            if step%args.save_every==0 or phase_step==budget:save()
            updates+=1
            if max_updates is not None and updates>=max_updates:
                save();return dict(status='interrupted',step=step)
    if not progress['final_done']:
        evaluate('final','test',True);progress['final_done']=True
    base.summarize(out);save('final');save()
    result=dict(status='complete',step=progress['step'],completed=progress['completed'],
        frozen_semantic_state_unchanged=True,channel_initialization='random',training_start_step=0,
        scale_delivery='reliable binary16 per original donor-block/segment; included in rate')
    json_write(out/'complete.json',result)
    return result


if __name__=='__main__':main(parser().parse_args())
