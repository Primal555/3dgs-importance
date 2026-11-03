"""Fork a frozen hybrid allocation snapshot with independent positive-tier steps."""
import argparse
import copy
import hashlib
import json
import shutil
from functools import partial
from pathlib import Path

import torch
from . import budgeted_allocation as engine
from . import structured_evidence_allocation as structured
from .dense_rd_allocation import atomic_save
from .hybrid_tier_allocation import review
from .hybrid_tier_benefit import FAMILY,HybridPositiveProvider
from .hybrid_tier_step_table import TierStepTable
from .modular_experiment import json_write

VERSION='hybrid_tier_step_controls_v1'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--tier-max-log-odds',type=float,default=1.)
    mode.add_argument('--unbounded-tier-updates',action='store_true')
    p.add_argument('--target-step',type=int)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--snapshot-only',action='store_true')
    return p


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_path(source):
    p=Path(source).resolve()
    if p.is_file():return p
    for name in ('allocation/checkpoints/latest/state.pt','checkpoints/latest/state.pt','source_state.pt'):
        if (p/name).is_file():return p/name
    raise FileNotFoundError(f'no allocation snapshot: {p}')


def validate_source(state):
    if state.get('family')!=FAMILY or state['plan']['settings'].get('network_mode')!='frozen':
        raise ValueError('source must be a frozen dense-keep hybrid allocation')
    provider=state.get('positive_provider',{})
    if provider.get('network_mode')!='frozen' or provider.get('learner') is not None or 'keep_policy' in provider:
        raise ValueError('frozen positive provider and original dense keep required')
    if int(state['progress']['step'])<1:raise ValueError('source requires a trained allocation checkpoint')


def pin_source(source,out):
    path=source_path(source)
    owner=path.parent
    if path.parent.parent.name=='checkpoints':
        owner=path.parent.parent.parent
        if owner.name=='allocation':owner=owner.parent
    if out.is_relative_to(owner) or out.is_relative_to(Path(source).resolve()):
        raise ValueError('new independent output directory required')
    out.mkdir(parents=True,exist_ok=False)
    pinned=out/'source_state.pt';temporary=out/'source_state.pt.tmp'
    # The source writer replaces files atomically; this open copies one stable inode.
    shutil.copyfile(path,temporary);temporary.replace(pinned)
    state=torch.load(pinned,map_location='cpu',weights_only=True);validate_source(state)
    candidates=[path.parent/'benefit_network.pt',*[(p/'benefit_network.pt') for p in list(path.parents)[:4]],
                Path(state['plan']['settings']['benefit_checkpoint'])]
    artifact=next((p for p in candidates if p.is_file()),None)
    if artifact is None:raise FileNotFoundError('source benefit_network.pt required')
    network=out/'benefit_network.pt';shutil.copyfile(artifact,network)
    saved=torch.load(network,map_location='cpu',weights_only=True)
    if (saved.get('version')!=FAMILY or saved['network_sha256']!=state['positive_provider']['initial_network_sha256']
            or saved['codec_id']!=state['progress']['initial_codec_id']
            or saved['scene_fingerprint']!=state['scene_record']['fingerprint']):
        raise ValueError('benefit snapshot identity mismatch')
    # Verify tensor contents as well as the declared digest, without creating a network.
    h=hashlib.sha256()
    for name,value in sorted(saved['learner']['network'].items()):
        h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
    if h.hexdigest()!=saved['network_sha256']:raise ValueError('benefit tensor digest mismatch')
    record=dict(version=VERSION,source_path=str(path),source_step=state['progress']['step'],
                source_sha256=digest(pinned),benefit_sha256=digest(network))
    json_write(out/'source.json',record)
    return state,record


def allocation_options(state,args,out):
    original=state['plan'];settings=original['settings']
    options=structured.parser().parse_args(['--run',settings['run'],'--scene',settings['scene'],
        '--out',str(out/'allocation'),'--beta',str(settings['beta'])])
    vars(options).update(copy.deepcopy(settings))
    options.out=str(out/'allocation');options.resume=True;options.dry_run=False;options.device=args.device
    options.benefit_checkpoint=str(out/'benefit_network.pt')
    options.tier_max_log_odds=None if args.unbounded_tier_updates else args.tier_max_log_odds
    options.allocation_steps=args.target_step if args.target_step is not None else settings['allocation_steps']
    if options.allocation_steps<=state['progress']['step']:raise ValueError('target step must exceed fork source step')
    # Reject implicit default changes: only this explicit whitelist may differ.
    candidate=engine.make_plan(options,FAMILY,original['source'],original['spec'],original['codec_config'])
    excluded={'benefit_checkpoint','allocation_steps','tier_max_log_odds'}
    clean=lambda d:{k:v for k,v in d.items() if k not in excluded}
    if clean(candidate['settings'])!=clean(original['settings']):raise ValueError('unapproved source settings change')
    return options,candidate


def main(args,max_updates=None):
    import math
    if not math.isfinite(args.tier_max_log_odds) or args.tier_max_log_odds<=0:raise ValueError('finite positive tier cap required')
    if args.snapshot_only and args.resume:raise ValueError('snapshot-only requires a new directory')
    out=Path(args.out).resolve()
    if args.resume:
        document=json.loads((out/'plan.json').read_text(encoding='utf8'))
        identity=dict(source=str(Path(args.source).resolve()),tier_max_log_odds=None if args.unbounded_tier_updates else args.tier_max_log_odds,
                      target_step=args.target_step)
        if document['settings']!=identity:raise ValueError('fork resume settings mismatch')
        state=torch.load(out/'source_state.pt',map_location='cpu',weights_only=True)
        record=json.loads((out/'source.json').read_text(encoding='utf8'))
        if digest(out/'source_state.pt')!=record['source_sha256'] or digest(out/'benefit_network.pt')!=record['benefit_sha256']:
            raise ValueError('pinned source snapshot changed')
        validate_source(state)
    else:
        if out.exists():raise FileExistsError('new fork output required')
        state,record=pin_source(args.source,out)
        if args.snapshot_only:
            print(json.dumps(record),flush=True);return record
    options,plan=allocation_options(state,args,out)
    document=dict(version=VERSION,source_snapshot=record,settings=dict(source=str(Path(args.source).resolve()),
        tier_max_log_odds=options.tier_max_log_odds,target_step=args.target_step),allocation_plan=plan)
    if not args.resume:
        json_write(out/'plan.json',document)
        fork=copy.deepcopy(state);fork['plan']=plan
        fork['progress'].update(best=None,lowest=None,initial_q=fork['latest_q'].clone(),
            pending=list(fork['progress']['pending'])+['fork_initial'])
        allocation=out/'allocation';json_write(allocation/'plan.json',plan)
        json_write(allocation/'training.json',dict(plan,**fork['scene_record'],objective_reference=fork['objective_reference'],
            fork_source=record,trainable_modules=[],optimizer='mirror descent; independent positive-tier limit',
            keep_max_log_odds=options.max_log_odds,tier_max_log_odds=options.tier_max_log_odds))
        atomic_save(allocation/'checkpoints/latest/state.pt',fork)
    if (out/'complete.json').is_file() and (out/'review.tar.gz').is_file():
        return json.loads((out/'complete.json').read_text(encoding='utf8'))
    print(json.dumps(dict(status='allocation_fork',fork_start_step=record['source_step'],
        target_step=options.allocation_steps,keep_max_log_odds=options.max_log_odds,
        tier_max_log_odds=options.tier_max_log_odds,source_sha256=record['source_sha256'])),flush=True)
    factory=partial(TierStepTable,tier_max_log_odds=options.tier_max_log_odds)
    result=engine.main(options,max_updates=max_updates,positive_factory=HybridPositiveProvider,
                       family_override=FAMILY,table_factory=factory)
    if result.get('status')=='interrupted':return result
    result=dict(result,pipeline_family=VERSION,fork_start_step=record['source_step'],
                continued_updates=result['step']-record['source_step'],tier_max_log_odds=options.tier_max_log_odds)
    json_write(out/'complete.json',result);review(out,result)
    return result


if __name__=='__main__':main(parser().parse_args())
