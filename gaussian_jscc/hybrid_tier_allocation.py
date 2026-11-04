"""Fresh spatial pretraining, then dense calibrated keep + neural positive tiers."""
import argparse
import contextlib
import hashlib
import io
import json
import tarfile
from pathlib import Path

import torch

from . import calibrated_spatial_benefit_allocation as pretraining
from . import structured_evidence_allocation as calibrated
from . import budgeted_allocation as allocator
from .allocation_benefit_replay import cpu_copy
from .dense_rd_allocation import atomic_save
from .hybrid_tier_benefit import FAMILY, HybridPositiveProvider
from .modular_experiment import json_write


def parser():
    p=pretraining.parser(); p.description=__doc__
    p._option_string_actions['--phase'].choices=('auto','pretrain','allocation')
    p.set_defaults(phase='auto',pretrain_windows=700,allocation_steps=2000,beta=.1,
                   keep_lr=.1,tier_lr=.4,candidate_batch=64,aligned_spatial_inputs=True)
    for group in p._mutually_exclusive_groups:
        if any('--beta' in action.option_strings for action in group._group_actions):group.required=False
    p.add_argument('--network-mode',choices=('frozen','online'),default='frozen')
    # The inherited --allocation-batch is an execution control, shared by both modes.
    p.set_defaults(allocation_batch=128)
    return p


def pretrain_options(args,root):
    settings=dict(vars(args),out=str(root/'pretrain'),phase='pretrain',allocation_batch=None,
                  pretrain_until=None)
    settings.pop('allocation_image_objective',None)
    settings['resume']=bool(args.resume and (root/'pretrain/checkpoints/latest/state.pt').is_file())
    return argparse.Namespace(**settings)


def export_network(root,target,*,aligned_spatial_inputs=None):
    path=root/'benefit_network.pt'
    if path.is_file():
        artifact=torch.load(path,map_location='cpu',weights_only=True)
        if artifact.get('version')!=FAMILY or artifact['pretrain_windows']!=target:
            raise ValueError('incompatible exported benefit network')
        if (aligned_spatial_inputs is not None and
                bool(artifact['settings'].get('aligned_spatial_inputs',False))!=aligned_spatial_inputs):
            raise ValueError('spatial input layout mismatch; retrain benefit network and replay in a new output')
        return path
    source=root/'pretrain/checkpoints/latest/state.pt'
    saved=torch.load(source,map_location='cpu',weights_only=True)
    if (aligned_spatial_inputs is not None and
            bool(saved['plan']['settings'].get('aligned_spatial_inputs',False))!=aligned_spatial_inputs):
        raise ValueError('spatial input layout mismatch; retrain benefit network and replay in a new output')
    if (not saved['progress']['pretrain_complete'] or saved['progress']['pretrain_windows_done']!=target
            or saved['progress']['allocation_step']!=0):
        raise ValueError('fresh complete pretraining before any allocation required')
    h=hashlib.sha256();h.update(json.dumps(saved['plan']['codec_config'],sort_keys=True).encode())
    for name,value in sorted(saved['state_dict'].items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    # Same tensor digest protocol as network_digest, without constructing another model.
    n=hashlib.sha256()
    for name,value in sorted(saved['learner']['network'].items()):
        n.update(name.encode());n.update(value.detach().cpu().contiguous().numpy().tobytes())
    artifact=dict(version=FAMILY,pretrain_windows=target,codec_id=h.hexdigest(),
        network_sha256=n.hexdigest(),scene_fingerprint=saved['scene_record']['fingerprint'],
        reference=saved['objective_reference'],settings=saved['plan']['settings'],
        learner=cpu_copy(saved['learner']),replay=saved['replay'])
    atomic_save(path,artifact)
    return path


def allocation_options(args,root,network):
    opts=calibrated.parser().parse_args(['--run',args.run,'--scene',args.scene,
        '--out',str(root/'allocation'),'--beta',str(args.beta)])
    protected={'out','resume','dry_run','phase','allocation_image_objective'}
    for key,value in vars(args).items():
        if key in vars(opts) and key not in protected:setattr(opts,key,value)
    opts.network_mode=args.network_mode;opts.benefit_checkpoint=str(network)
    opts.aligned_spatial_inputs=args.aligned_spatial_inputs
    opts.allocation_batch=args.allocation_batch
    opts.resume=bool(args.resume and (root/'allocation/checkpoints/latest/state.pt').is_file())
    return opts


def review(root,result):
    rows=[json.loads(x) for x in (root/'allocation/metrics/validation.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
    chosen={r['event']:r for r in rows if r['event'] in ('initial','latest','selected','uniform_q2','uniform_q3')}
    q2=chosen['uniform_q2'];comparison={}
    for label in ('latest','selected'):
        r=chosen[label]
        comparison[label]=dict(psnr_difference=r['source_psnr']-q2['source_psnr'],
            cost_saving=1-r['total_channel_uses']/q2['total_channel_uses'],rd_difference=r['rd_score']-q2['rd_score'],
            meets_q2_goal=r['source_psnr']>=q2['source_psnr'] and r['total_channel_uses']<q2['total_channel_uses'])
    json_write(root/'review_summary.json',dict(result=result,rows=chosen,against_q2=comparison))
    temporary=root/'review.tar.gz.tmp'
    with tarfile.open(temporary,'w:gz') as archive:
        for path in sorted(root.rglob('*')):
            rel=path.relative_to(root)
            include=rel.parts[0] in ('plan.json','complete.json','review_summary.json')
            if len(rel.parts)>1:
                include |= (rel.parts[0]=='allocation' and rel.parts[1] in ('metrics','comparison','charts','evaluation','deployment','complete.json','training.json'))
                include |= (rel.parts[0]=='pretrain' and rel.parts[1] in ('metrics','pretrain_complete.json','preflight.json'))
            if path.is_file() and include:archive.add(path,arcname=str(rel),recursive=False)
    temporary.replace(root/'review.tar.gz')
    return comparison


def main(args,max_updates=None,*,provider_factory=HybridPositiveProvider,family=FAMILY,
         table_factory=None,allocation_transform=None,pretrain_factory=None,scheduler_factory=None):
    if args.pretrain_until is not None: raise ValueError('fresh pipeline uses --pretrain-windows; no previous evaluator reuse')
    if args.allocation_batch<1:raise ValueError('positive allocation batch required')
    root=Path(args.out).resolve()
    pre=pretrain_options(args,root)
    train_pre=pretrain_factory or pretraining.main
    if pretrain_factory is None and args.aligned_spatial_inputs:
        from .aligned_spatial_inputs import pretrain
        train_pre=pretrain
    if args.dry_run:
        with contextlib.redirect_stdout(io.StringIO()):plan=train_pre(pre)
        plan.update(version=family,automatic_transition=True,network_mode=args.network_mode,
                    pretrain_windows=args.pretrain_windows,allocation_steps=args.allocation_steps)
        print(json.dumps(plan,ensure_ascii=False,indent=2));return plan
    identity={k:v for k,v in vars(args).items() if k not in ('out','resume','dry_run','phase','device')}
    if root.exists() and not args.resume:raise FileExistsError('new output or --resume required')
    if (root/'plan.json').is_file():
        if json.loads((root/'plan.json').read_text(encoding='utf8'))['settings']!=identity:
            raise ValueError('pipeline settings mismatch')
    else:json_write(root/'plan.json',dict(version=family,settings=identity))
    if (root/'complete.json').is_file() and (root/'review.tar.gz').is_file():
        return json.loads((root/'complete.json').read_text(encoding='utf8'))
    marker=root/'pretrain/pretrain_complete.json'
    done=marker.is_file() and json.loads(marker.read_text(encoding='utf8')).get('step')==args.pretrain_windows
    consumed=0
    if not done:
        if args.phase=='allocation':raise ValueError('finish fresh pretraining first')
        before=0
        if max_updates is not None and pre.resume:
            before=torch.load(root/'pretrain/checkpoints/latest/state.pt',map_location='cpu',weights_only=True)['progress']['pretrain_windows_done']
        status=train_pre(pre,max_updates=max_updates)
        if status['status']=='interrupted':return status
        consumed=args.pretrain_windows-before
    network=export_network(root,args.pretrain_windows,aligned_spatial_inputs=args.aligned_spatial_inputs)
    if args.phase=='pretrain':return dict(status='pretrain_complete',network_checkpoint=str(network))
    if max_updates is not None and consumed>=max_updates:
        return dict(status='interrupted',phase='between_stages')
    options=allocation_options(args,root,network)
    if allocation_transform is not None:allocation_transform(options,args)
    result=allocator.main(options,max_updates=None if max_updates is None else max_updates-consumed,
                          positive_factory=provider_factory,family_override=family,table_factory=table_factory,
                          scheduler_factory=scheduler_factory)
    if result.get('status')=='interrupted':return result
    result=dict(result,pipeline_family=family,pretrain_windows=args.pretrain_windows,
                allocation_steps=args.allocation_steps)
    json_write(root/'complete.json',result)
    comparison=review(root,result)
    print(json.dumps(dict(status='complete',network_mode=args.network_mode,against_q2=comparison),ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':main(parser().parse_args())
