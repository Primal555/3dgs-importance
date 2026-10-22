"""Joint source-PLY rendering after the completed successive channel experiment.

Full weights/config are retained. Only the semantic encoder remains frozen;
the two channel modules and semantic decoder use a NEW Adam optimizer.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .cli import device_for,seed_all
from .data import read_ply
from .decoder_adaptation import evaluate_saved
from .model_factory import build_codec,config_from_dict
from .modular_checkpoint import (capture_rng,restore_rng,save_training_checkpoint,
                                 restore_training_checkpoint,validate_scene)
from .modular_experiment import json_write,make_scene,repair_logs
from .modular_snr_controls import control_inputs
from .modular_training import MODULES,configure_phase,render_step,finish_update
from .power_followup import joint_inputs
from .power_followup_evaluation import bind_test_cameras
from .render_objective import MultiViewRenderTask
from .render_validation import append_json


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--manifest')
    for key,value in [('steps',2000),('validate-every',250),('save-every',100),
                      ('test-views',32),('render-trials',4),('seed',42),('cpu-threads',4)]:
        p.add_argument('--'+key,type=int,default=value)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--latent-weight',type=float,default=1.)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    return p


def encoder_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if name.startswith('semantic_encoder.') or name in ('attr_mean','attr_std'):
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_plan(args):
    if (min(args.steps,args.validate_every,args.save_every,args.test_views,args.render_trials,args.cpu_threads)<1
            or any(not math.isfinite(v) or v<=0 for v in (args.lr,args.latent_weight))):
        raise ValueError('positive step/evaluation budgets and finite positive weights/LR required')
    run=Path(args.run).resolve();out=Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):raise ValueError('output must be separate from source run')
    metadata=json.loads((run/'training.json').read_text(encoding='utf8'))
    complete=json.loads((run/'complete.json').read_text(encoding='utf8'))
    checkpoint=run/'checkpoints/final/codec.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    config=saved.get('config',{});state=saved.get('training',{});progress=state.get('progress',{})
    completed=progress.get('completed',{})
    if (saved.get('version')!=4 or config.get('architecture')!='modular_jscc_v1'
            or config.get('channel_variant')!='successive' or tuple(config.get('rates',()))!=(0,8,24,48)
            or metadata.get('successive_channel',{}).get('version')!='successive_channel_v1'
            or complete.get('status')!='complete' or complete.get('step')!=saved.get('step')
            or progress.get('step')!=saved.get('step') or saved.get('step',0)<1
            or completed!=metadata.get('phase_budgets') or completed!=complete.get('completed')
            or set(completed)!={'main','neighbor','cooperative'} or sum(completed.values())!=saved['step']
            or not bool(saved['state_dict'].get('channel_decoder.neighbor_enabled',False))):
        raise ValueError('source must be a completed successive 8/24/48 run with its trained neighbor enabled')
    if json.loads(json.dumps(config))!=metadata.get('codec_config'):raise ValueError('source config/metadata mismatch')
    config_from_dict(config)
    validate_scene(state['scene_record'],metadata['scene'])
    with checkpoint.open('rb') as stream:identity=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(version='successive_joint_v1',source_run=str(run),checkpoint=str(checkpoint),
        source_sha256=identity,source_step=saved['step'],encoder_sha256=encoder_digest(saved['state_dict']),
        codec_config=json.loads(json.dumps(config)),
        manifest_override=str(Path(args.manifest).resolve()) if args.manifest else None,
        settings={k:v for k,v in vars(args).items() if k not in ('run','out','manifest','resume','dry_run')},
        trainable_modules=['channel_encoder','channel_decoder','semantic_decoder'],
        frozen_modules=['semantic_encoder','attribute statistics','delivered XYZ'],
        optimizer_initialization='new Adam; source moments not loaded',
        objective='source PLY render RGB MSE + latent_weight * MSE(y_hat, stop_gradient(y))',
        channel='10 dB AWGN; actual reliable binary16 scale frames retained and counted',
        layout='q1/q2/q3/random-positive mixed cycle; no allocation learning')


def training_mode(model):
    # Root train mode retains the quantized-scale STE; freeze encoder buffers too.
    model.train();model.semantic_encoder.eval()


def joint_step(model,batches,geometry,task,latent_weight,*,mode='replay'):
    training_mode(model)
    loss,stats=render_step(model,batches,geometry,10.,'awgn',task,'channel_render',
                           mode=mode,channel_latent_weight=latent_weight)
    stats.update(task.stats)
    return loss,stats


def summarize(out):
    from .channel_audit import write_csv
    rows=[]
    for label in ('initial','final'):
        for line in (out/'evaluation'/label/'metrics/validation.jsonl').read_text(encoding='utf8').splitlines():
            record=json.loads(line)
            keys=('path','layout','source_psnr','photo_psnr','source_ssim','photo_ssim','latent_mse',
                  'latent_nmse','views','trials','source_lpips','photo_lpips','total_channel_uses','scale_stream_bytes')
            # Clean has no wire payload/scale cost; keep a common CSV schema.
            row={k:record.get(k) for k in keys};row['endpoint']=label
            previous=next((r for r in rows if r['endpoint']=='initial' and r['path']==row['path'] and r['layout']==row['layout']),None)
            row['source_psnr_change_db']=row['source_psnr']-previous['source_psnr'] if previous else None
            rows.append(row)
    json_write(out/'comparison/summary.json',dict(rows=rows,
        clean='diagnostic: decoder changes, so clean is not a guaranteed upper bound',
        mixed='fixed random positive tiers, NOT learned deployment',
        protocol='initial/final use the same actual test views and paired noise; intermediates use validation views'))
    write_csv(out/'comparison/endpoints.csv',rows)


def main(args,max_updates=None):
    plan=build_plan(args)
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    out=Path(args.out).resolve()
    if out.exists():
        if not args.resume:raise FileExistsError('use NEW output or --resume')
        if json.loads((out/'plan.json').read_text(encoding='utf8'))!=plan:raise ValueError('resume source/settings differ')
        if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text(encoding='utf8'))
    else:
        if args.resume:raise FileNotFoundError('resume output missing')
        out.mkdir(parents=True);json_write(out/'plan.json',plan)
    device=device_for(args.device);torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    latest=out/'checkpoints/latest/codec.pt';fresh=not latest.exists()
    if fresh:
        source=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
        model=build_codec(config_from_dict(source['config'])).to(device);model.load_state_dict(source['state_dict'])
        rates={name:args.lr for name in MODULES}
        optimizer=configure_phase(model,'channel_render',rates)
        progress=dict(step=0,phase='channel_render',phase_step=0,phase_complete=False,successive_joint=True,
            completed=dict(channel_render=0),phase_budgets=dict(channel_render=args.steps),module_lrs=rates,
            channel_trained=True,freeze_semantic=False,initial_done=False,final_done=False,
            source_step=plan['source_step'],train_snrs=[10.],
            training_objective=dict(version='successive_joint_v1',channel_render_latent_weight=args.latent_weight))
        record=source['training']['scene_record'];rng=capture_rng()
        json_write(out/'initialization.json',dict(all_source_weights_loaded=True,source_step=plan['source_step'],
            source_checkpoint_sha256=plan['source_sha256'],source_optimizer_loaded=False,
            optimizer_state_entries=len(optimizer.state)))
    else:
        model,optimizer,progress,rng,record=restore_training_checkpoint(latest,device)
        if not progress.get('successive_joint'):raise ValueError('resume is not this joint experiment')
        repair_logs(out,progress['step'])
    if encoder_digest(model.state_dict())!=plan['encoder_sha256']:raise RuntimeError('frozen encoder/statistics differ')
    metadata=json.loads((Path(plan['source_run'])/'training.json').read_text(encoding='utf8'))
    original,manifest,spec,old,_=control_inputs(metadata,args.manifest)
    raw,degree=read_ply(spec['ply']);scene=make_scene(spec,raw,degree,model,original,old,out);del raw
    validate_scene(record,scene.record);scene.channel_trained=True
    names=bind_test_cameras(scene,spec,args.test_views);model.set_training_snrs([10.]);restore_rng(rng)
    training=dict(source_run=plan['source_run'],manifest=str(manifest),selected_scenes=[spec],scene=scene.record,
        codec_config=model.cfg.to_dict(),successive_joint=plan,test_view_names=names,net_bits_per_use=scene.net_bits_per_use,
        training_objective=progress['training_objective'],trainable_modules=plan['trainable_modules'],
        frozen_modules=plan['frozen_modules'],render_backward='replay',lpips_metrics=args.lpips)
    if (out/'training.json').exists() and json.loads((out/'training.json').read_text(encoding='utf8'))!=json.loads(json.dumps(training)):
        raise ValueError('training/test scene identity differs')
    json_write(out/'training.json',training)

    def save(label='latest'):
        if encoder_digest(model.state_dict())!=plan['encoder_sha256']:raise RuntimeError('frozen encoder/statistics changed')
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,capture_rng(),scene.record)

    def finish_evaluation():
        evaluate_saved(model,scene,out,args,progress['step'],**progress['pending_evaluation'])
        progress.pop('pending_evaluation');save()

    def evaluate(label,scope='validation',images=False):
        progress['pending_evaluation']=dict(label=label,scope=scope,images=images);save();finish_evaluation()

    if fresh:save('initial');save()
    if progress.get('pending_evaluation'):finish_evaluation()
    if not progress['initial_done']:
        evaluate('initial','test',True);progress['initial_done']=True;save()
    print(f'Joint A: {args.steps} updates, all trainable LR={args.lr:g}; NEW Adam; semantic encoder FROZEN',flush=True)
    updates=0
    while progress['step']<args.steps:
        step=progress['step']+1;batches,indices,layout=joint_inputs(scene,step,step,args)
        task=MultiViewRenderTask([scene.cameras[i] for i in indices],scene.reference,degree,scene.white_background)
        optimizer.zero_grad(set_to_none=True);before={n:p.detach().clone() for n,p in model.named_parameters()}
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
        started=time.perf_counter()
        loss,stats=joint_step(model,batches,scene.geometry,task,args.latent_weight)
        stats.update(finish_update(model,optimizer,before))
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            stats.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
        progress.update(step=step,phase_step=step,phase_complete=step==args.steps)
        progress['completed']['channel_render']=step
        append_json(out/'metrics/loss.jsonl',dict(stats,step=step,phase_step=step,phase='channel_render',
            loss=float(loss),layout=layout,snr=10.,module_lrs={g['name']:g['lr'] for g in optimizer.param_groups},
            training_view_indices=indices,step_seconds=time.perf_counter()-started))
        if step<=4 or step%25==0 or step==args.steps:
            print(f'joint {step}/{args.steps}: q={layout}, loss={float(loss):.6g}, render={stats["render_loss"]:.6g}, '
                  f'y={stats["latent_mse"]:.6g}, grad={stats["grad_norm"]:.5g}, sec={time.perf_counter()-started:.2f}',flush=True)
        if step%args.validate_every==0 and step<args.steps:evaluate(f'joint_{step:06d}')
        if step%args.save_every==0 or step==args.steps:save()
        updates+=1
        if max_updates is not None and updates>=max_updates:
            save();return dict(status='interrupted',step=step)
    save('end_joint')
    if not progress['final_done']:
        evaluate('final','test',True);progress['final_done']=True
    summarize(out);save('final');save()
    result=dict(status='complete',step=progress['step'],source_step=plan['source_step'],
        joint_updates=args.steps,frozen_encoder_state_unchanged=True,
        trainable_modules=plan['trainable_modules'],source_optimizer_loaded=False)
    json_write(out/'complete.json',result)
    return result


if __name__=='__main__':main(parser().parse_args())
