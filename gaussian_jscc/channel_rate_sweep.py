"""Independent variable-prefix experiment with a frozen semantic encoder AND decoder."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from .cli import seed_all,device_for
from .data import read_ply
from .modular_config import ModularCodecConfig
from .modular_codec import ModularGaussianCodec
from .modular_checkpoint import (save_training_checkpoint,restore_training_checkpoint,
    capture_rng,restore_rng,validate_scene)
from .modular_experiment import json_write,make_scene,repair_logs
from .modular_snr_controls import control_inputs
from .modular_training import configure_phase,bootstrap_step,finish_update,MODULES
from .modular_evaluation import evaluate_paths
from .channel_capacity_controls import semantic_digest
from .render_validation import append_json
from .transport import save_checkpoint

DEFAULT_RATES=(4,8,12,16,24,32,48,64)
CONTROL_RATES=(8,24,48)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='trained modular run; only its semantic pair/statistics are loaded')
    p.add_argument('--out',required=True,help='new suite directory, never the source run')
    p.add_argument('--checkpoint',help='default RUN/checkpoints/final/codec.pt')
    p.add_argument('--manifest',help='path override; scene identity and camera split must match')
    p.add_argument('--rates',nargs='+',type=int,default=list(DEFAULT_RATES))
    p.add_argument('--arms',nargs='+',choices=['three_tier','fine_tiers'],default=['three_tier','fine_tiers'])
    p.add_argument('--updates-per-rate',type=int,default=500,help='real updates PER uniform prefix, not total steps')
    p.add_argument('--mixed-updates',type=int,default=0,help='additional mixed updates; never subtracts from fixed-rate quotas')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--channel-hidden',type=int,help='default: source channel width; same in both arms')
    p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--eval-snrs',type=float,nargs='+',default=[10.])
    p.add_argument('--trials',type=int,default=2)
    p.add_argument('--test-views',type=int,default=0,help='0=all original test cameras')
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--save-images',action=argparse.BooleanOptionalAction,default=True,help='initial/final endpoint panels only')
    p.add_argument('--device',default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true',help='same suite, exact per-arm Adam/RNG/progress')
    p.add_argument('--dry-run',action='store_true',help='validate source/settings and print plan; no GPU/data loading or output creation')
    return p


def build_plan(args):
    rates=tuple(args.rates)
    if (not rates or len(rates)>255 or any(n<1 for n in rates) or
            any(a>=b for a,b in zip(rates,rates[1:]))):
        raise ValueError('rates must be strictly increasing positive symbol budgets (at most 255)')
    if (min(args.updates_per_rate,args.validate_every,args.save_every,args.trials,args.cpu_threads)<1
            or args.mixed_updates<0 or args.test_views<0 or not math.isfinite(args.lr) or args.lr<=0
            or not args.eval_snrs or any(not math.isfinite(v) for v in args.eval_snrs)
            or len(set(args.eval_snrs))!=len(args.eval_snrs)
            or (args.channel_hidden is not None and args.channel_hidden<1)
            or not args.arms or len(set(args.arms))!=len(args.arms)
            or not 0<=args.seed<2**32):
        raise ValueError('invalid sweep settings')
    run=Path(args.run).resolve(); out=Path(args.out).resolve()
    if out==run or out.is_relative_to(run): raise ValueError('output must be outside the read-only source run')
    metadata=json.loads((run/'training.json').read_text(encoding='utf-8'))
    path=Path(args.checkpoint).resolve() if args.checkpoint else run/'checkpoints/final/codec.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    progress=saved.get('training',{}).get('progress',{})
    if (saved.get('version')!=4 or saved.get('config',{}).get('architecture')!='modular_jscc_v1'
            or not progress.get('phase_complete')
            or not (progress.get('completed',{}).get('semantic_render',0)>0 or metadata.get('initializer'))):
        raise ValueError('requires a completed trained modular semantic teacher, not random or mid-phase weights')
    if json.loads(json.dumps(saved['config']))!=metadata['codec_config']:
        raise ValueError('source checkpoint/config mismatch')
    validate_scene(saved['training']['scene_record'],metadata['scene'])
    source_cfg=ModularCodecConfig.from_dict(saved['config'])
    arms={name:list(CONTROL_RATES if name=='three_tier' else rates) for name in args.arms}
    plan=dict(version=1,source_run=str(run),checkpoint=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        teacher_semantic_sha256=semantic_digest(saved['state_dict']),arms=arms,
        updates_per_rate=args.updates_per_rate,mixed_updates=args.mixed_updates,
        expected_updates={a:len(r)*args.updates_per_rate+args.mixed_updates for a,r in arms.items()},
        lr=args.lr,channel_hidden=args.channel_hidden or source_cfg.channel_hidden,seed=args.seed,
        train_snr=10.,eval_snrs=args.eval_snrs,trials=args.trials,test_views=args.test_views,
        lpips=args.lpips,save_images=args.save_images,cpu_threads=args.cpu_threads,device_kind=args.device.split(':')[0],
        validate_every=args.validate_every,save_every=args.save_every,
        noise_reference_symbols=max(max(r) for r in arms.values()),
        semantic_modules='both frozen; source weights/statistics copied exactly',
        channel_initialization='fresh progressive full-prefix channels; source denoiser/neighbor options retained',
        objective='pure unweighted latent MSE; rendering is evaluation only',
        update_sampling='round-robin uniform budgets; per-budget update-index/data/noise seeds; optional mixed quota independent',
        parameter_and_compute_matched=False,
        caveat='More prefixes add refinement stages; compare this as an operating rate-distortion curve, not isolated symbol-count causality.')
    return plan,metadata,saved


def schedule(rates,per_rate,mixed):
    for index in range(max(per_rate,mixed)):
        if index<per_rate:
            for tier,budget in enumerate(rates,1): yield str(tier),budget,index+1
        if index<mixed: yield 'mixed',None,index+1


def _configuration(saved,rates,width):
    cfg=dict(saved['config'])
    cfg.update(rates=(0,*rates),experimental_multirate=True,channel_variant='progressive',
               channel_prefix_mode='full_prefix',channel_hidden=width)
    return ModularCodecConfig.from_dict(cfg)


def _run_arm(args,plan,metadata,saved,name,raw,degree,limit=None):
    from .rendering import load_cameras
    out=Path(args.out).resolve()/name; rates=plan['arms'][name]
    cfg=_configuration(saved,rates,plan['channel_hidden'])
    source,manifest_path,spec,old,_=control_inputs(metadata,args.manifest)
    if degree!=cfg.sh_degree: raise ValueError('source PLY SH degree mismatch')
    device=device_for(args.device); seed_all(args.seed)
    latest=out/'checkpoints/latest/codec.pt'
    total=plan['expected_updates'][name]
    lrs={m:args.lr for m in MODULES}
    if latest.exists():
        model,optimizer,progress,rng,record=restore_training_checkpoint(latest,device)
        if model.cfg!=cfg: raise ValueError('resume codec settings mismatch')
        if (not progress.get('rate_sweep') or not progress.get('freeze_semantic')
                or progress.get('phase')!='channel_bootstrap'
                or progress.get('phase_step')!=progress.get('step')
                or not 0<=progress['step']<=total):
            raise ValueError('invalid rate sweep resume checkpoint')
    else:
        model=ModularGaussianCodec(cfg).to(device)
        semantic={k:v for k,v in saved['state_dict'].items()
                  if k.startswith(('semantic_encoder.','semantic_decoder.')) or k in ('attr_mean','attr_std')}
        missing=model.load_state_dict(semantic,strict=False)
        if missing.unexpected_keys or any(not k.startswith(('channel_encoder.','channel_decoder.')) for k in missing.missing_keys):
            raise ValueError('incomplete semantic teacher copy')
        optimizer=configure_phase(model,'channel_bootstrap',lrs,freeze_semantic=True)
        progress=dict(step=0,phase='channel_bootstrap',phase_step=0,phase_complete=False,channel_trained=False,
            completed=dict(semantic_bootstrap=0,semantic_render=0,channel_bootstrap=0,channel_render=0),
            phase_budgets=dict(semantic_bootstrap=0,semantic_render=0,channel_bootstrap=total,channel_render=0),
            module_lrs=lrs,train_snrs=[10.],freeze_semantic=True,rate_sweep=True)
        record=rng=None
    model.set_training_snrs([10.])
    scene=make_scene(spec,raw,degree,model,source,old,out)
    if record is not None: validate_scene(record,scene.record)
    if semantic_digest(model.state_dict())!=plan['teacher_semantic_sha256']:
        raise RuntimeError('semantic teacher changed')
    train_metadata=dict(source_run=plan['source_run'],manifest=str(manifest_path),selected_scenes=[spec],
        codec_config=cfg.to_dict(),scene=scene.record,net_bits_per_use=scene.net_bits_per_use,
        initializer=plan['checkpoint'],initializer_kind='semantic_pair_only',source_weights_loaded=True,
        source_step=saved['step'],random_initialization=False,freeze_semantic=True,train_snrs=[10.],
        phase_budgets=progress['phase_budgets'],module_lrs=lrs,seed=args.seed,
        rate_sweep=plan,arm=name,lpips_metrics=args.lpips,
        parameters={m:sum(p.numel() for p in getattr(model,m).parameters()) for m in MODULES},
        decoder_refinement_stages=len(rates),training_objective={'channel_bootstrap':'latent_mse'},
        source_usage='only frozen semantic weights/statistics and verified scene metadata; fresh channel modules')
    json_write(out/'training.json',train_metadata)
    if rng is not None: repair_logs(out,progress['step']); restore_rng(rng)
    def save_latest():
        save_training_checkpoint(latest,model,optimizer,progress,capture_rng(),scene.record)
    def evaluate(phase,images=False):
        scene.channel_trained=progress['channel_trained']
        return evaluate_paths(model,scene,step=progress['step'],phase=phase,
            snrs=args.eval_snrs if phase=='test' else [10.],trials=args.trials,seed=args.seed,
            save_images=images and args.save_images,out=out,color_metrics=True,lpips_metrics=args.lpips,
            noise_reference_symbols=plan['noise_reference_symbols'])
    if progress['step']==0:
        save_latest(); evaluate('initial',True)
    actual=0
    for step,(layout,budget,local_update) in enumerate(schedule(rates,args.updates_per_rate,args.mixed_updates),1):
        if step<=progress['step']: continue
        # Common budgets in both arms see the same groups and packed AWGN draws.
        token=budget if budget is not None else plan['noise_reference_symbols']+1
        update_seed=(args.seed+1000003*token+10007*local_update)%(2**32)
        seed_all(update_seed)
        group=random.randrange(len(scene.groups)); f=scene.groups[group].to(device)
        active=(scene.group_ids[group]>=0).to(device)
        q=(torch.randint(1,len(cfg.rates),active.shape,device=device) if layout=='mixed'
           else torch.full(active.shape,int(layout),dtype=torch.long,device=device)).masked_fill(~active,0)
        model.train(); model.semantic_encoder.eval(); model.semantic_decoder.eval()
        optimizer.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        if device.type=='cuda': torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        loss,stats=bootstrap_step(model,f,q,scene.geometry,10.,'awgn','channel_bootstrap')
        stats.update(finish_update(model,optimizer,before))
        if device.type=='cuda': torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-started
        progress.update(step=step,phase_step=step,phase_complete=step==total,channel_trained=True)
        progress['completed']['channel_bootstrap']=step
        append_json(out/'metrics/loss.jsonl',dict(stats,step=step,phase_step=step,phase='channel_bootstrap',
            loss=float(loss),layout=layout,symbols=budget,budget_update=local_update,sample_seed=update_seed,
            training_group_index=group,snr=10.,lr=args.lr,step_seconds=elapsed,arm=name,
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
        actual+=1
        if step==1 or step%10==0 or step==total:
            print(f'{name} {step}/{total}: budget={budget or "mixed"}, per-budget={local_update}, y_MSE={float(loss):.6g}, sec={elapsed:.3f}',flush=True)
        if step%args.validate_every==0 and step!=total: evaluate('channel_bootstrap')
        if step%args.save_every==0 or step==total: save_latest()
        if limit is not None and actual>=limit and step<total:
            save_latest(); return actual,False
    if semantic_digest(model.state_dict())!=plan['teacher_semantic_sha256']:
        raise RuntimeError('training changed frozen semantic weights/statistics')
    evaluate('final',True)
    cameras=load_cameras(spec['source'],scene.record['resolution'],scene.white_background,scene.record['images'],'test')
    if not cameras: raise ValueError('missing test cameras; no train-view substitution')
    if args.test_views and args.test_views<len(cameras):
        ids=torch.linspace(0,len(cameras)-1,args.test_views).round().long().tolist()
        cameras=[cameras[i] for i in ids]
    if set(str(c.image_name) for c in cameras)&set(scene.record['train_view_names']):
        raise ValueError('test cameras overlap training split')
    scene.val_cameras=cameras
    # Remove an incomplete previous final-test export when resuming at last update.
    test_log=out/'metrics/test.jsonl'
    if test_log.exists():
        index=1
        archive=test_log.with_name(f'test.interrupted-{index}.jsonl')
        while archive.exists():
            index+=1; archive=test_log.with_name(f'test.interrupted-{index}.jsonl')
        test_log.rename(archive)
    evaluate('test')
    final=out/'checkpoints/final/codec.pt'; final.parent.mkdir(parents=True,exist_ok=True)
    # Only latest holds Adam/RNG; final is the smaller inference/initializer file.
    save_checkpoint(final,model,progress['step'],training={'modular_version':1,'progress':progress,'scene_record':scene.record})
    json_write(out/'complete.json',dict(status='complete',step=total,arm=name,semantic_weights_unchanged=True,
        test_views=len(cameras),parameters=train_metadata['parameters']))
    return actual,True


def main(args,max_updates=None):
    if max_updates is not None and max_updates<1: raise ValueError('max_updates must be positive')
    plan,metadata,saved=build_plan(args)
    if args.dry_run:
        print(json.dumps(plan,indent=2)); return plan
    out=Path(args.out).resolve()
    if out.exists() and not args.resume: raise FileExistsError('use a NEW output directory, or --resume with the original settings')
    if args.resume:
        old=json.loads((out/'experiment.json').read_text(encoding='utf-8'))
        if old!=plan: raise ValueError('resume suite settings or teacher mismatch')
    else: json_write(out/'experiment.json',plan)
    torch.set_num_threads(args.cpu_threads)
    source,_,spec,_,_=control_inputs(metadata,args.manifest)
    raw,degree=read_ply(spec['ply'])
    remaining=max_updates
    for name in args.arms:
        if (out/name/'complete.json').exists():
            run=out/name
            complete=json.loads((run/'complete.json').read_text(encoding='utf-8'))
            final=torch.load(run/'checkpoints/final/codec.pt',map_location='cpu',weights_only=True)
            if (complete.get('status')!='complete' or complete.get('step')!=plan['expected_updates'][name]
                    or final.get('step')!=complete['step']
                    or ModularCodecConfig.from_dict(final['config'])!=_configuration(saved,plan['arms'][name],plan['channel_hidden'])
                    or semantic_digest(final['state_dict'])!=plan['teacher_semantic_sha256']):
                raise ValueError('completed arm checkpoint/settings mismatch: '+name)
            print(f'{name}: already complete, preserving outputs',flush=True); continue
        used,done=_run_arm(args,plan,metadata,saved,name,raw,degree,remaining)
        if remaining is not None:
            remaining-=used
            if not done or remaining<=0:
                return {'status':'interrupted','output':str(out)}
    from .channel_rate_sweep_results import summarize
    summary=summarize(out,plan)
    json_write(out/'complete.json',dict(status='complete',expected_updates=plan['expected_updates'],
        semantic_weights_unchanged=True,training='channel only; no render/semantic/allocator updates'))
    return summary


if __name__=='__main__': main(parser().parse_args())
