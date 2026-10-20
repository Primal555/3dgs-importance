"""Extend a completed Truck joint stage in a NEW directory, preserving Adam/RNG.

Unlike --init-codec this is not fresh optimization. The existing trainer is
reused with an explicitly extended budget and the saved phase cursor.
"""
import argparse
import copy
import json
from pathlib import Path

from .modular_checkpoint import restore_training_checkpoint, save_training_checkpoint
from .modular_experiment import json_write, main as train, parser as training_parser


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='completed modular Truck joint run')
    p.add_argument('--out',required=True,help='NEW output for the extension; original run is read-only')
    p.add_argument('--additional-steps',type=int,default=1000)
    p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--save-every',type=int,default=250)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--manifest',help='optional migrated scene paths; source PLY/splits must still match')
    p.add_argument('--resume',action='store_true',help='resume this extension, NOT add another budget')
    p.add_argument('--dry-run',action='store_true',help='check source checkpoint and plan on CPU; create no output')
    return p


def training_options(args,record,out):
    argv=['--run',record['source_run'],'--out',str(out),'--resume',str(out/'checkpoints/latest/codec.pt'),
          '--use-source-config','--device',args.device,'--cpu-threads',str(args.cpu_threads),
          '--validate-every',str(args.validate_every),'--save-every',str(args.save_every),
          '--seed',str(record['seed'])]
    for phase,count in record['phase_budgets'].items():
        argv+=['--'+phase.replace('_','-')+'-steps',str(count)]
    for name,lr in record['module_lrs'].items():
        argv+=['--'+name.replace('_','-')+'-lr',str(lr)]
    argv+=['--train-snrs',*map(str,record['train_snrs']),
           '--channel-latent-weight',str(record['training_objective']['channel_render_latent_weight'])]
    objective=record['training_objective'].get('render_image_objective',{})
    argv+=['--image-objective',objective.get('kind','mse')]
    if objective.get('aux_weight') is not None:argv+=['--image-aux-weight',str(objective['aux_weight'])]
    if record.get('test_snrs') is not None:argv+=['--test-snrs',*map(str,record['test_snrs'])]
    for name in ('freeze_semantic','color_metrics','save_semantic_end_images'):
        if record.get(name,False):argv+=['--'+name.replace('_','-')]
    argv+=['--lpips' if record.get('lpips_metrics',False) else '--no-lpips',
           '--semantic-color-every',str(record.get('semantic_color_every',0)),
           '--semantic-color-directions',str(record.get('semantic_color_directions',8))]
    if args.manifest:argv+=['--manifest',args.manifest]
    return training_parser().parse_args(argv)


def main(args,max_updates=None):
    if min(args.additional_steps,args.validate_every,args.save_every,args.cpu_threads)<1:
        raise ValueError('additional steps, intervals and threads must be positive')
    source=Path(args.run).resolve();out=Path(args.out).resolve()
    if source==out or out.is_relative_to(source):
        raise ValueError('continuation must use a NEW directory outside the original run')
    if args.resume:
        record=json.loads((out/'training.json').read_text(encoding='utf-8'))
        info=record.get('continuation',{})
        if (info.get('parent_run')!=str(source) or info.get('additional_steps')!=args.additional_steps or
                info.get('validate_every')!=args.validate_every or info.get('save_every')!=args.save_every):
            raise ValueError('extension resume settings mismatch; do not add the budget twice')
        if not (out/'checkpoints/latest/codec.pt').is_file():raise FileNotFoundError('missing continuation latest checkpoint')
    else:
        if out.exists():raise FileExistsError(f'use a NEW output directory or --resume: {out}')
        record=json.loads((source/'training.json').read_text(encoding='utf-8'))
        completion=json.loads((source/'complete.json').read_text(encoding='utf-8'))
        if (record.get('codec_config',{}).get('architecture')!='modular_jscc_v1' or
                record.get('rate_sweep') or record.get('training_family') or record.get('clean_only') or
                not isinstance(record.get('scene'),dict) or
                [s.get('name') for s in record.get('selected_scenes',[])]!=['truck']):
            raise ValueError('expected completed single-scene modular communication run')
        objective=record.get('training_objective',{}).get('render_image_objective',{}).get('kind','mse')
        if objective not in ('mse','mse_ssim'):
            raise ValueError('this continuation supports MSE/SSIM joint stages, not GAN/LPIPS controls')
        checkpoint=source/'checkpoints/final/codec.pt'
        model,optimizer,progress,rng,scene_record=restore_training_checkpoint(checkpoint,'cpu')
        if (completion.get('status')!='complete' or progress.get('phase')!='channel_render' or
                not progress.get('phase_complete') or not progress.get('channel_trained') or
                progress['completed']['channel_render']<1 or completion.get('step')!=progress['step']):
            raise ValueError('source must have completed a nonempty channel_render stage')
        if (progress.get('phase_budgets')!=record['phase_budgets'] or
                progress['completed']!=record['phase_budgets'] or
                progress.get('training_objective')!=record.get('training_objective') or
                progress['module_lrs']!=record['module_lrs'] or scene_record!=record['scene']):
            raise ValueError('source checkpoint and training metadata disagree')
        if json.loads(json.dumps(model.cfg.to_dict()))!=record['codec_config']:
            raise ValueError('source architecture metadata differs from checkpoint')
        # Restore helper constructs Adam; require actual saved moments, not an
        # empty optimizer that merely happens to have the same hyperparameters.
        if not optimizer.state:raise ValueError('source checkpoint has no Adam state')
        parent_step=progress['step'];old_budget=record['phase_budgets']['channel_render']
        record=copy.deepcopy(record)
        record['phase_budgets']['channel_render']=old_budget+args.additional_steps
        progress['phase_budgets']=dict(record['phase_budgets'])
        progress['phase_complete']=False
        progress.pop('status',None)
        record.update(source_run=str(source),source_step=parent_step,random_initialization=False,source_weights_loaded=True,
                      initialization='joint extension: retained all weights, Adam moments, RNG and phase cursor')
        record['continuation']=dict(parent_run=str(source),parent_checkpoint=str(checkpoint),
            parent_step=parent_step,parent_joint_step=old_budget,additional_steps=args.additional_steps,
            target_step=parent_step+args.additional_steps,target_joint_step=old_budget+args.additional_steps,
            optimizer_preserved=True,rng_preserved=True,validate_every=args.validate_every,save_every=args.save_every)
        if args.dry_run:
            print(json.dumps(record['continuation'],indent=2))
            return record['continuation']
        # Atomic mkdir also prevents two fresh launchers sharing one output.
        out.mkdir(parents=True,exist_ok=False)
        json_write(out/'training.json',record)
        for label in ('initial','latest'):
            save_training_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,rng,scene_record)
        del model,optimizer
    info=record['continuation']
    if args.dry_run:
        print(json.dumps(info,indent=2))
        return info
    print(f'JOINT CONTINUATION: global {info["parent_step"]} -> {info["target_step"]}; '
          f'joint {info["parent_joint_step"]} -> {info["target_joint_step"]}; '
          'Adam/RNG retained; semantic encoder stays frozen; no bootstrap.',flush=True)
    print(f'Learning rates unchanged: {record["module_lrs"]}; validation every {args.validate_every} steps',flush=True)
    result=train(training_options(args,record,out),max_updates=max_updates)
    summary_path=out/'metrics/phase_summary.json'
    if summary_path.exists() and result['completed']['channel_render']==record['phase_budgets']['channel_render']:
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        summary['channel_render'].update(optimizer_reset_at_boundary=False,optimizer_state_inherited=True)
        json_write(summary_path,summary)
    return result


if __name__=='__main__':main(parser().parse_args())
