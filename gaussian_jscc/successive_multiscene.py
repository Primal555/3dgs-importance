"""Continue Joint B on balanced scenes; evaluate the fourth scene zero-shot."""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from .cli import device_for,seed_all
from .data import read_ply
from .decoder_adaptation import evaluate_saved
from .model_factory import build_codec
from .modular_checkpoint import capture_rng,restore_rng,save_training_checkpoint,restore_training_checkpoint,validate_scene
from .modular_experiment import json_write,repair_logs
from .modular_training import configure_phase,finish_update,MODULES
from .optimization import preserved_rng
from .power_followup import joint_inputs
from .render_objective import MultiViewRenderTask
from .render_validation import append_json
from .successive_joint import joint_step,encoder_digest
from .successive_multiscene_inputs import inputs,make_scene
from .transport import model_id

FAMILY='successive_multiscene_joint_v1'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','manifest','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--scenes',nargs='+',default=['train','truck','playroom'])
    p.add_argument('--heldout-scene',default='drjohnson')
    p.add_argument('--heldout-manifest',help='optional separate heldout-role manifest if pruned manifest contains only training scenes')
    for name,value in [('steps-per-scene',2000),('validate-every',250),('save-every',100),
                       ('validation-views',4),('test-views',32),('render-trials',4),('cpu-threads',4),('seed',42)]:
        p.add_argument('--'+name,type=int,default=value)
    for name in ('blocks-per-batch','resolution','views-per-step'):p.add_argument('--'+name,type=int)
    p.add_argument('--images');p.add_argument('--white-background',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--net-bits-per-use',type=float)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--latent-weight',type=float,default=1.)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def main(args,max_updates=None,*,input_loader=None,family=FAMILY):
    saved,metadata,cfg,specs,heldout,source=(input_loader or inputs)(args)
    heldouts=heldout if isinstance(heldout,list) else [heldout]
    if (min(args.steps_per_scene,args.validate_every,args.save_every,args.validation_views,args.render_trials,
            args.cpu_threads,args.blocks_per_batch,args.views_per_step)<1 or args.test_views<0
            or any(not math.isfinite(x) or x<=0 for x in (args.lr,args.latent_weight,args.net_bits_per_use))):
        raise ValueError('positive update/evaluation budgets, rates and weights required')
    settings={k:v for k,v in vars(args).items() if k not in ('out','run','device','resume','dry_run')}
    plan=json.loads(json.dumps(dict(version=family,**source,settings=settings,selected_scenes=specs,
        heldout_scene=heldout,codec_config=cfg.to_dict(),total_updates=len(specs)*args.steps_per_scene,
        statistics='retain source Joint B attribute mean/std; no refit on new scenes')))
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    out=Path(args.out).resolve();latest=out/'checkpoints/latest/codec.pt'
    if out.exists() and not args.resume:raise FileExistsError('use NEW output or --resume')
    if args.resume:
        if not latest.exists():raise FileNotFoundError('missing exact-resume checkpoint')
        if json.loads((out/'plan.json').read_text(encoding='utf-8'))!=plan:raise ValueError('resume settings/source/scene paths changed')
    device=device_for(args.device);torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    lrs={name:args.lr for name in MODULES}
    if args.resume:
        model,optimizer,progress,rng,old_records=restore_training_checkpoint(latest,device)
        if progress.get('training_family')!=family or model.cfg!=cfg:raise ValueError('resume is not this multi-scene experiment')
    else:
        model=build_codec(cfg).to(device);model.load_state_dict(saved['state_dict'])
        optimizer=configure_phase(model,'channel_render',lrs,train_semantic_encoder=True)
        progress=dict(training_family=family,step=0,phase='channel_render',phase_step=0,
            successive_joint=True,train_semantic_encoder=True,channel_trained=True,phase_complete=False,
            module_lrs=lrs,train_snrs=[10.],source_step=source['source_step'],
            completed=dict(channel_render=0),phase_budgets=dict(channel_render=plan['total_updates']),
            scene_completed={s['name']:0 for s in specs},next_scene=0,initial_done=False,final_done=False,
            training_objective=dict(version=family,channel_render_latent_weight=args.latent_weight))
        rng=capture_rng();old_records=None
    del saved
    if encoder_digest(model.state_dict(),include_encoder=False)!=source['statistics_sha256']:
        raise ValueError('source attribute statistics changed')
    model.set_training_snrs([10.]);scenes=[]
    for spec in specs:
        raw,degree=read_ply(spec['ply'])
        old_record=metadata.get('scene_records',{}).get(spec['name'])
        if old_record is None and spec['name']=='truck':old_record=metadata.get('scene')
        scene=make_scene(spec,raw,degree,model,args,out,old=old_record)
        del raw;scenes.append(scene)
    # make_scene follows Scene's standard OUT/scenes/name layout.
    records={s.name:s.record for s in scenes}
    if len({s.fingerprint for s in scenes})!=len(scenes):raise ValueError('training PLY content aliases')
    if old_records is not None:validate_scene(old_records,records)
    else:
        json_write(out/'plan.json',plan)
        json_write(out/'training.json',dict(plan,source_run=str(Path(args.run).resolve()),
            trainable_modules=list(MODULES),source_optimizer_loaded=False,attribute_statistics_frozen=True,
            training_objective='source PLY RGB MSE + latent_weight*MSE(y_hat, stop_gradient(current y))',
            allocation_training=False,render_backward='replay',snr=10.,step_budgets_scope='per_scene',scene_records=records))
        for s in scenes:json_write(s.out/'scene.json',dict(s.record,**s.spec))
    if args.resume:
        repair_logs(out,progress['step'])
    (out/'metrics').mkdir(parents=True,exist_ok=True)
    restore_rng(rng)

    def save(label='latest'):
        if encoder_digest(model.state_dict(),include_encoder=False)!=source['statistics_sha256']:
            raise RuntimeError('fixed attribute statistics changed')
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,capture_rng(),records)
        json_write(out/'progress.json',dict(progress,status='running'))

    def finish_evaluation():
        pending=progress['pending_evaluation'];label=pending['label']
        with preserved_rng(device):
            for s in scenes:
                evaluate_saved(model,s,s.out,args,progress['step'],label=label,scope=pending['scope'],images=pending['images'])
            seen={x.fingerprint for x in scenes}
            for heldout_spec in heldouts if pending['scope']=='test' else ():
                raw,degree=read_ply(heldout_spec['ply'])
                s=make_scene(heldout_spec,raw,degree,model,args,out);del raw
                if s.fingerprint in seen:raise ValueError('heldout content aliases training/another heldout PLY')
                seen.add(s.fingerprint)
                path=s.out/'scene.json'
                record=dict(s.record,**s.spec)
                if path.exists():validate_scene(json.loads(path.read_text(encoding='utf-8')),record)
                else:json_write(path,record)
                evaluate_saved(model,s,s.out,args,progress['step'],label=label,scope='test',images=pending['images'])
                del s
        progress.pop('pending_evaluation')
        if label in ('initial','final'):progress[label+'_done']=True
        save()

    def evaluate(label,scope='validation',images=False):
        progress['pending_evaluation']=dict(label=label,scope=scope,images=images)
        save();finish_evaluation()

    if not args.resume:save('initial');save()
    if progress.get('pending_evaluation'):finish_evaluation()
    if not progress['initial_done']:evaluate('initial','test',True)
    print(f'Joint B multi-scene: {args.steps_per_scene} updates PER SCENE; {plan["total_updates"]} total; '
          f'four modules LR={args.lr:g}; heldout {",".join(s["name"] for s in heldouts)} evaluation ONLY',flush=True)
    updated=0
    while progress['step']<plan['total_updates']:
        s=scenes[progress['next_scene']];visit=progress['scene_completed'][s.name]+1;step=progress['step']+1
        model.set_scene_geometry(s.geometry)
        batches,indices,layout=joint_inputs(s,step,visit,args)
        optimizer.zero_grad(set_to_none=True);before={n:p.detach().clone() for n,p in model.named_parameters()}
        if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        task=MultiViewRenderTask([s.cameras[i] for i in indices],s.reference,s.degree,s.white_background)
        loss,stats=joint_step(model,batches,s.geometry,task,args.latent_weight,train_semantic_encoder=True)
        stats.update(finish_update(model,optimizer,before))
        if device.type=='cuda':torch.cuda.synchronize(device)
        progress.update(step=step,phase_step=step,phase_complete=step==plan['total_updates'])
        progress['completed']['channel_render']=step;progress['scene_completed'][s.name]=visit
        progress['next_scene']=(progress['next_scene']+1)%len(scenes)
        row=dict(stats,step=step,phase='channel_render',scene=s.name,scene_step=visit,layout=layout,
            loss=float(loss),training_view_indices=indices,module_lrs=lrs,step_seconds=time.perf_counter()-start,
            optimizer_reset_at_scene_switch=False,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None)
        append_json(out/'metrics/loss.jsonl',row)
        if visit==1 or visit%10==0 or visit==args.steps_per_scene:
            print(f'{s.name} joint {visit}/{args.steps_per_scene} (global {step}): q={layout}, '
                  f'render={stats["render_loss"]:.6g}, y={stats["latent_mse"]:.6g}, sec={row["step_seconds"]:.2f}',flush=True)
        balanced=progress['next_scene']==0
        if balanced and (visit%args.save_every==0 or progress['phase_complete']):save()
        if balanced and visit%args.validate_every==0 and not progress['phase_complete']:
            evaluate(f'round_{visit:06d}')
        updated+=1
        if max_updates is not None and updated>=max_updates:save();return dict(progress,status='interrupted')
    if not progress['final_done']:evaluate('final','test',True)
    save('final');save()
    from .successive_multiscene_report import summarize
    summarize(out,specs,heldout)
    result=dict(status='complete',step=progress['step'],scene_completed=progress['scene_completed'],
        heldout_optimizer_updates=0,attribute_statistics_frozen=True,
        initial_source_step=source['source_step'],final_codec_id=model_id(model))
    if isinstance(heldout,list):result['heldout_scenes']=[s['name'] for s in heldouts]
    else:result['heldout_scene']=heldout['name']
    json_write(out/'complete.json',result);json_write(out/'progress.json',dict(progress,status='complete'))
    return result


if __name__=='__main__':main(parser().parse_args())
