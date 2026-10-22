"""Joint A conditional codec -> 2000 allocation + 1000 encoder-frozen joint."""
import csv
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

from .allocation import GaussianTierMask
from .allocation_diagnostics import AllocationCostMeter
from .cli import device_for,seed_all
from .data import read_ply
from .mask_gradient import local_mask_step
from .modular_allocation import parser as allocation_parser,evaluate
from .modular_checkpoint import capture_rng,restore_rng,initialize_codec_run,save_training_checkpoint,validate_scene
from .modular_experiment import inputs,make_scene,json_write,repair_logs
from .modular_training import configure_phase
from .multiscene import preserve_prefix
from .optimization import clip_codec_gradients,update_stats
from .power_followup_evaluation import bind_test_cameras
from .render_objective import MaskedMultiViewRenderTask,MultiViewRenderTask
from .render_validation import append_json
from .successive_joint import encoder_digest
from .transport import model_id

JOINT_MODULES=('channel_encoder','channel_decoder','semantic_decoder')


def summarize(out):
    """Compact, directly inspectable endpoint tables and presentation figures."""
    out=Path(out);phases=('initial','end_allocation','final')
    keys=('step','phase','layout','source_psnr','source_ssim','source_lpips',
          'photo_psnr','photo_ssim','photo_lpips','retained_gaussians',
          'payload_complex_symbols','xyz_bytes','metadata_bytes','scale_stream_bytes',
          'total_channel_uses','net_bits_per_use_assumption','views','trials')
    rows=[]
    for line in (out/'metrics/validation.jsonl').read_text(encoding='utf-8').splitlines():
        record=json.loads(line)
        if record['phase'] not in phases:continue
        row={k:record.get(k) for k in keys}
        for q in range(4):
            row[f'q{q}_count']=record['tier_counts'][q]
            row[f'q{q}_share']=record['tier_shares'][q]
        rows.append(row)
    expected={(phase,layout) for phase in phases for layout in ('1','2','3','deployment')}
    if len(rows)!=12 or {(r['phase'],r['layout']) for r in rows}!=expected:
        raise ValueError('missing or duplicated endpoint measurements')
    folder=out/'comparison';folder.mkdir(exist_ok=True)
    json_write(folder/'summary.json',dict(rows=rows,
        scope='identical held-out views, trial seeds and source/photo references at all endpoints',
        cost='analogue payload + measured reliable XYZ/metadata/binary16 scale frames; shared weights excluded'))
    with (folder/'endpoints.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'charts';folder.mkdir(exist_ok=True)
    deployed=[next(r for r in rows if r['phase']==phase and r['layout']=='deployment') for phase in phases]
    labels=('Before allocation','After allocation','After joint')
    colors=('#888888','#1f77b4','#ff7f0e','#9467bd')
    fig,ax=plt.subplots(figsize=(8,4));bottom=[0.]*3
    for q,color in enumerate(colors):
        values=[100*r[f'q{q}_share'] for r in deployed]
        ax.bar(labels,values,bottom=bottom,color=color,label=f'q{q}')
        bottom=[a+b for a,b in zip(bottom,values)]
    ax.set(ylabel='Actual ten-draw deployment share (%)',ylim=(0,100))
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.15));fig.tight_layout()
    fig.savefig(folder/'tier_shares.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for phase,label,marker in zip(phases,labels,('o','s','^')):
        chosen=[r for r in rows if r['phase']==phase]
        for r in chosen:
            ax.scatter(r['total_channel_uses']/1e6,r['source_psnr'],marker=marker,
                       color='#222222' if r['layout']=='deployment' else colors[int(r['layout'])])
            ax.annotate('learned' if r['layout']=='deployment' else 'q'+r['layout'],
                        (r['total_channel_uses']/1e6,r['source_psnr']),xytext=(4,3),textcoords='offset points',fontsize=8)
        ax.scatter([],[],marker=marker,color='#444444',label=label)
    ax.set(xlabel='Total complex channel uses (millions)',ylabel='Source-PLY PSNR (dB)')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(folder/'quality_rate.png',dpi=180);plt.close(fig)


def parser():
    p=allocation_parser();p.description=__doc__
    p._option_string_actions['--run'].help='completed successive Joint A run; retain all final codec weights'
    p._option_string_actions['--codec-lr'].help='joint-only LR for channel encoder, channel decoder and semantic decoder'
    p.set_defaults(allocation_steps=2000,joint_steps=1000,beta=.005,test_views=32,trials=4)
    p.add_argument('--channel-latent-weight',type=float,default=1.,
                   help='joint-only actual retained-channel latent anchor; no allocator gradient')
    return p


class SuccessiveAllocationCostMeter(AllocationCostMeter):
    """Retained-XYZ/tier proxy plus exact binary16 scale-frame sizes."""
    def __init__(self,model,scene,positions):
        self.model,self.scene=model,scene
        super().__init__(model.cfg,positions,len(scene.raw),scene.net_bits_per_use)

    def details(self,q):
        result=super().details(q)
        scale=self.model.scale_rate_record(self.scene,q,self.bits)
        uses=scale['scale_channel_uses']
        result.update(scale)
        result['allocation_side_uses_per_source_gaussian']+=uses/self.count
        result['allocation_uses_per_source_gaussian']+=uses/self.count
        result['allocation_rate_scope']+=' + reliable binary16 block-scale frames'
        return result


def source_plan(args):
    if (min(args.allocation_steps,args.joint_steps,args.test_views,args.prefix_anchor_every)<0
            or args.allocation_steps+args.joint_steps<1
            or min(args.validate_every,args.save_every,args.trials,args.image_views,args.cpu_threads,args.rate_chunk_size)<1
            or not math.isfinite(args.beta) or args.beta<0 or args.snr!=10.):
        raise ValueError('positive budgets/settings and fixed 10dB required')
    for key in ('codec_lr','mask_lr','keep_lr','mask_adam_eps','channel_latent_weight'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0:raise ValueError('invalid '+key)
    if args.fixed_deployment or args.continue_allocation or args.image_objective!='mse':
        raise ValueError('new successive allocation uses a new table and source-PLY MSE')
    run=Path(args.run).resolve();out=Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):raise ValueError('source/output must be separate')
    path=run/'checkpoints/final/codec.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    complete=json.loads((run/'complete.json').read_text(encoding='utf-8'))
    metadata=json.loads((run/'training.json').read_text(encoding='utf-8'))
    progress=saved.get('training',{}).get('progress',{})
    if (saved.get('version')!=4 or saved.get('config',{}).get('channel_variant')!='successive'
            or tuple(saved.get('config',{}).get('rates',()))!=(0,8,24,48)
            or not progress.get('successive_joint') or progress.get('train_semantic_encoder',False)
            or not progress.get('phase_complete') or not progress.get('channel_trained')
            or progress.get('phase')!='channel_render' or progress.get('step')!=saved.get('step')
            or progress.get('completed',{}).get('channel_render')!=saved.get('step')
            or progress.get('phase_budgets',{}).get('channel_render')!=saved.get('step')
            or metadata.get('successive_joint',{}).get('settings',{}).get('steps')!=saved.get('step')
            or complete.get('status')!='complete' or complete.get('step')!=saved.get('step')
            or complete.get('joint_updates')!=saved.get('step') or saved.get('step',0)<1
            or complete.get('frozen_encoder_state_unchanged') is not True
            or metadata.get('successive_joint',{}).get('version')!='successive_joint_v1'
            or not bool(saved['state_dict'].get('channel_decoder.neighbor_enabled',False))):
        raise ValueError('source must be completed Joint A, with frozen semantic encoder and trained successive codec')
    if json.loads(json.dumps(saved['config']))!=metadata.get('codec_config'):
        raise ValueError('source config/metadata mismatch')
    validate_scene(saved['training']['scene_record'],metadata['scene'])
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if encoder_digest(saved['state_dict'])!=metadata['successive_joint']['encoder_sha256']:
        raise ValueError('source frozen encoder/statistics fingerprint mismatch')
    return dict(version='successive_allocation_v1',source_checkpoint=str(path),source_sha256=sha,
                frozen_encoder_sha256=encoder_digest(saved['state_dict']),
                source_step=saved['step'],joint_trainable_modules=list(JOINT_MODULES))


def configure_scope(model,phase,lr):
    if phase=='joint':
        return configure_phase(model,'channel_render',{name:lr for name in JOINT_MODULES})
    if phase!='allocation':raise ValueError('unknown allocation phase')
    model.requires_grad_(False);model.zero_grad(set_to_none=True)
    return None


def main(args,max_updates=None):
    plan=source_plan(args)
    source,manifest,spec,old=inputs(args)
    out=Path(args.out).resolve()
    device=device_for('cpu' if args.dry_run else args.device)
    torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    model,_,record=initialize_codec_run(plan['source_checkpoint'],device)
    initial_id=model_id(model)
    settings={k:v for k,v in vars(args).items() if k not in ('out','run','resume','dry_run','manifest','device')}
    settings.update(plan,manifest=str(manifest),initial_codec_id=initial_id)
    if args.dry_run:
        result=dict(settings=settings,rates=list(model.cfg.rates),writes_output=False)
        print(json.dumps(result,indent=2));return result
    if out.exists() and not args.resume:raise FileExistsError('use a NEW output or --resume')
    if args.resume and not (out/'checkpoints/latest/state.pt').exists():raise FileNotFoundError('missing allocation resume state')
    if args.resume and (out/'complete.json').exists():
        state=torch.load(out/'checkpoints/latest/state.pt',map_location='cpu',weights_only=True)
        if state['settings']!=settings:raise ValueError('resume settings differ')
        return json.loads((out/'complete.json').read_text(encoding='utf-8'))
    raw,degree=read_ply(spec['ply']);scene=make_scene(spec,raw,degree,model,source,old,out);del raw
    validate_scene(record,scene.record)
    test_names=bind_test_cameras(scene,spec,args.test_views)
    scene.rate_meter=SuccessiveAllocationCostMeter(model,scene,scene.position_meter)
    scene.channel_trained=True;model.set_training_snrs([10.])
    mask=GaussianTierMask(len(scene.raw)).to(device)
    mask_optimizer=torch.optim.Adam([dict(params=[mask.keep_logits],lr=args.keep_lr),
                                     dict(params=[mask.logits],lr=args.mask_lr)],eps=args.mask_adam_eps)
    progress=dict(step=0,phase='allocation',completed=dict(allocation=0,joint=0),
                  initial_done=False,end_allocation_done=False,final_done=False,frozen_codec_verified=False)
    codec_optimizer=None
    for folder in ('metrics','deployment','checkpoints','images'): (out/folder).mkdir(parents=True,exist_ok=True)
    if args.resume:
        state=torch.load(out/'checkpoints/latest/state.pt',map_location='cpu',weights_only=True)
        if state['settings']!=settings or state['scene']!=scene.record or state['config']!=model.cfg.to_dict():
            raise ValueError('exact resume source/settings/scene mismatch')
        model.load_state_dict(state['state_dict']);mask.load_state_dict(state['mask'])
        mask_optimizer.load_state_dict(state['mask_optimizer']);progress=state['progress']
        codec_optimizer=configure_scope(model,progress['phase'],args.codec_lr)
        if codec_optimizer is not None:codec_optimizer.load_state_dict(state['codec_optimizer'])
        repair_logs(out,progress['step'])
        history=out/'deployment/allocation_history.jsonl'
        if history.exists():
            kept=[s for s in history.read_text(encoding='utf-8').splitlines() if s.strip() and json.loads(s)['step']<=progress['step']]
            history.write_text(''.join(s+'\n' for s in kept),encoding='utf-8')
        restore_rng(state['rng'])
    else:
        configure_scope(model,'allocation',args.codec_lr)
        if args.lpips:
            from .perceptual_metrics import LPIPSEvaluator
            preflight=LPIPSEvaluator(device);del preflight
        if device.type=='cuda':
            from .mask_checks import check_masked_renderer
            json_write(out/'mask_renderer_check.json',check_masked_renderer(scene.raw.to(device),scene.cameras[0],degree,scene.white_background))
    if encoder_digest(model.state_dict())!=plan['frozen_encoder_sha256']:
        raise ValueError('frozen semantic encoder/statistics changed')
    training=dict(settings,source_run=str(Path(args.run).resolve()),selected_scenes=[spec],scene=scene.record,
        codec_config=model.cfg.to_dict(),test_view_names=test_names,net_bits_per_use=scene.net_bits_per_use,
        deployment_draws=10,allocation_training_sampling='single',render_backward='replay',
        objective='source-PLY RGB MSE + beta * rate proxy; joint additionally anchors actual-channel y',
        tier_estimator='detached within-block uniform-positive candidates, same retained set and paired noise',
        prefix_anchor='every N joint updates; uniform q1/q2/q3 independent cycle; codec gradients averaged',
        scale_cost='exact frame sizes measured; block-presence cost is a detached rate term',
        initialization='all Joint A codec weights retained; new table/Adam; semantic encoder stays frozen')
    if (out/'training.json').exists() and json.loads((out/'training.json').read_text(encoding='utf-8'))!=json.loads(json.dumps(training)):
        raise ValueError('training/test protocol changed')
    json_write(out/'training.json',training)

    def save(label='latest'):
        if encoder_digest(model.state_dict())!=plan['frozen_encoder_sha256']:
            raise RuntimeError('semantic encoder/statistics changed')
        folder=out/'checkpoints'/label;folder.mkdir(parents=True,exist_ok=True)
        state=dict(version=1,config=model.cfg.to_dict(),state_dict=model.state_dict(),mask=mask.state_dict(),
            mask_optimizer=mask_optimizer.state_dict(),codec_optimizer=codec_optimizer.state_dict() if codec_optimizer else None,
            settings=settings,scene=scene.record,progress=dict(progress),rng=capture_rng())
        temporary=folder/'state.pt.tmp';torch.save(state,temporary);temporary.replace(folder/'state.pt')
        inference=dict(step=progress['step'],phase='channel_render',phase_complete=True,successive_joint=True,
            channel_trained=True,train_snrs=[10.],completed=dict(channel_render=progress['completed']['joint']),
            module_lrs={name:args.codec_lr for name in JOINT_MODULES})
        save_training_checkpoint(folder/'codec.pt',model,None,inference,capture_rng(),scene.record)
        torch.save(dict(version=2,count=len(scene.raw),scene_fingerprint=scene.fingerprint,snr_conditioned=False,
            state_dict=mask.state_dict(),codec_id=model_id(model),rates=list(model.cfg.rates),step=progress['step']),folder/'allocation.pt')
        json_write(out/'progress.json',dict(progress,status='running'))

    def finish_evaluation():
        pending=progress['pending_evaluation'];label=pending['label']
        # Restore pre-evaluation reporting state before retrying a partial
        # endpoint; record_allocation writes history before rendering completes.
        history=out/'deployment/allocation_history.jsonl'
        history.write_text(pending['history_before'],encoding='utf-8')
        previous=out/'deployment/allocation_latest/tiers.npy'
        if pending['previous_tiers']:
            previous.parent.mkdir(exist_ok=True)
            shutil.copyfile(out/'deployment/evaluation_before_tiers.npy',previous)
        else:previous.unlink(missing_ok=True)
        # A failed endpoint can be repeated without duplicate metric rows.
        path=out/'metrics'/('test.jsonl' if label=='test' else 'validation.jsonl')
        if path.exists():
            kept=[s for s in path.read_text(encoding='utf-8').splitlines() if s.strip() and json.loads(s)['phase']!=label]
            path.write_text(''.join(s+'\n' for s in kept),encoding='utf-8')
        original=scene.val_cameras
        try:
            if pending['scope']=='test':scene.val_cameras=scene.test_cameras
            evaluate(model,scene,mask,out,progress['step'],label,args,save_images=pending['images'])
        finally:scene.val_cameras=original
        progress.pop('pending_evaluation')
        if label in ('initial','end_allocation','final'):progress[label+'_done']=True
        save()

    def validation(label,scope='validation',images=False):
        previous=out/'deployment/allocation_latest/tiers.npy'
        if previous.exists():shutil.copyfile(previous,out/'deployment/evaluation_before_tiers.npy')
        history=out/'deployment/allocation_history.jsonl'
        progress['pending_evaluation']=dict(label=label,scope=scope,images=images,
            previous_tiers=previous.exists(),history_before=history.read_text(encoding='utf-8') if history.exists() else '')
        save();finish_evaluation()

    if not args.resume:save('initial');save()
    if 'pending_evaluation' in progress:finish_evaluation()
    if not progress['initial_done']:validation('initial','test',True)
    print(f'Successive allocation: {args.allocation_steps} frozen-codec updates + {args.joint_steps} Joint A updates; '
          f'beta={args.beta:g}, 8/24/48, 10dB; semantic encoder FROZEN',flush=True)
    updates=0
    for phase,budget in (('allocation',args.allocation_steps),('joint',args.joint_steps)):
        # A mid-joint resume must not re-label its adapted weights as the
        # frozen allocation endpoint or compare them to the initial codec.
        if phase=='allocation' and progress['completed']['joint']>0:continue
        if progress['completed'][phase]<budget:
            if progress['phase']!=phase or (phase=='joint' and codec_optimizer is None):
                codec_optimizer=configure_scope(model,phase,args.codec_lr)
            progress['phase']=phase
        while progress['completed'][phase]<budget:
            visit=progress['completed'][phase]+1
            model.train();model.semantic_encoder.eval();model.zero_grad(set_to_none=True)
            mask_optimizer.zero_grad(set_to_none=True)
            before={name:p.detach().clone() for name,p in model.named_parameters()}
            indices=random.sample(range(len(scene.cameras)),scene.views_per_step)
            cameras=[scene.cameras[i] for i in indices]
            task=MaskedMultiViewRenderTask(cameras,scene.reference,degree,scene.white_background)
            if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            started=time.perf_counter()
            loss,stats=local_mask_step(model,mask,scene.groups,scene.group_ids,scene.geometry,10.,'awgn',
                task,args.beta,scene.rate_meter,train_codec=phase=='joint',mode='replay',
                sampling='single',rate_chunk_size=args.rate_chunk_size,
                channel_latent_weight=args.channel_latent_weight if phase=='joint' else 0.)
            stats.update(task.stats)
            if phase=='joint':
                anchor=MultiViewRenderTask(cameras,scene.reference,degree,scene.white_background)
                stats.update(preserve_prefix(model,scene.groups,scene.group_ids,scene.geometry,10.,'awgn',
                    anchor,visit,args.prefix_anchor_every,3))
            norm,gradient=clip_codec_gradients(model,mode='none')
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in mask.parameters()):
                raise RuntimeError('nonfinite loss/probability gradient; no update')
            mask_optimizer.step()
            if codec_optimizer is not None:codec_optimizer.step()
            changes=update_stats(model,before)
            if changes['semantic_encoder']['update_norm']!=0:raise RuntimeError('frozen encoder updated')
            if phase=='allocation' and any(v['update_norm']!=0 for v in changes.values()):raise RuntimeError('frozen codec updated')
            if device.type=='cuda':torch.cuda.synchronize(device)
            progress['step']+=1;progress['completed'][phase]=visit
            row=dict(stats,**gradient,step=progress['step'],phase=phase,phase_step=visit,loss=float(loss),
                grad_norm=float(norm),updates=changes,training_view_indices=indices,
                module_lrs={g['name']:g['lr'] for g in codec_optimizer.param_groups} if codec_optimizer else {},
                weighted_latent_loss=stats.get('weighted_latent_loss',0.),
                step_seconds=time.perf_counter()-started,
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            append_json(out/'metrics/loss.jsonl',row)
            if visit==1 or visit%10==0 or visit==budget:
                print(f'{phase} {visit}/{budget}: render={stats["render_loss"]:.6g}, rate={stats["rate_loss"]:.6g}, '
                      f'counts={stats["sampled_tier_counts"]}, sec={row["step_seconds"]:.2f}',flush=True)
            if progress['step']%args.validate_every==0 and visit<budget:validation(f'{phase}_{visit:06d}')
            if progress['step']%args.save_every==0 or visit==budget:save()
            updates+=1
            if max_updates is not None and updates>=max_updates:save();return dict(status='interrupted',**progress)
        if phase=='allocation':
            if model_id(model)!=initial_id:raise RuntimeError('allocation changed frozen codec')
            progress['frozen_codec_verified']=True
            if not progress['end_allocation_done']:validation('end_allocation','test',True)
        save('end_'+phase);save()
    if not progress['final_done']:validation('final','test',True)
    # Final already used the exact test protocol: alias its measured rows for
    # existing tools without decoding/rasterizing the same 512 pairs twice.
    final_rows=[json.loads(s) for s in (out/'metrics/validation.jsonl').read_text(encoding='utf-8').splitlines()
                if json.loads(s)['phase']=='final']
    test_path=out/'metrics/test.jsonl';temporary=test_path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(dict(r,phase='test'))+'\n' for r in final_rows),encoding='utf-8')
    temporary.replace(test_path)
    save('final');save()
    summarize(out)
    result=dict(status='complete',step=progress['step'],completed=progress['completed'],
        frozen_encoder_verified=True,frozen_codec_verified=progress['frozen_codec_verified'],
        initial_codec_id=initial_id,final_codec_id=model_id(model),rates=list(model.cfg.rates),beta=args.beta)
    json_write(out/'complete.json',result);json_write(out/'progress.json',dict(progress,status='complete'))
    return result


if __name__=='__main__':main(parser().parse_args())
