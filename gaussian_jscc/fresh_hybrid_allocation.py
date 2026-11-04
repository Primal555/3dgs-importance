"""Frozen spatial evaluator with a freshly initialized dense-keep allocator."""
import argparse
import copy
import hashlib
import json
import math
import shutil
from functools import partial
from pathlib import Path

import torch
from . import budgeted_allocation as engine
from . import structured_evidence_allocation as structured
from .hybrid_tier_allocation import review
from .hybrid_tier_benefit import FAMILY,HybridPositiveProvider
from .hybrid_tier_step_controls import digest
from .hybrid_tier_step_table import TierStepTable
from .modular_experiment import json_write

VERSION='fresh_frozen_hybrid_allocation_v1'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='Original hybrid run or its allocation directory')
    p.add_argument('--out',required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--tier-max-log-odds',type=float,default=1.)
    mode.add_argument('--unbounded-tier-updates',action='store_true')
    p.add_argument('--allocation-steps',type=int,default=2000)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    return p


def source_info(path):
    path=Path(path).resolve()
    if (path/'allocation/plan.json').is_file():owner,allocation=path,path/'allocation'
    elif path.name=='allocation' and (path/'plan.json').is_file():owner,allocation=path.parent,path
    else:raise FileNotFoundError(f'hybrid allocation/plan.json required: {path}')
    plan=json.loads((allocation/'plan.json').read_text(encoding='utf8'))
    if plan.get('version')!=FAMILY or plan['settings'].get('network_mode')!='frozen':
        raise ValueError('source must be an original frozen dense-keep hybrid run')
    if not plan['settings'].get('uniform_controls',False):raise ValueError('source must enable uniform q2/q3 controls')
    artifact=owner/'benefit_network.pt'
    if not artifact.is_file():raise FileNotFoundError(artifact)
    network=torch.load(artifact,map_location='cpu',weights_only=True)
    if bool(plan['settings'].get('aligned_spatial_inputs',False))!=bool(network['settings'].get('aligned_spatial_inputs',False)):
        raise ValueError('spatial input layout mismatch between source plan and benefit network')
    return owner,plan,artifact


def pin_network(source,target):
    temporary=target.with_suffix('.pt.tmp');shutil.copyfile(source,temporary);temporary.replace(target)
    saved=torch.load(target,map_location='cpu',weights_only=True)
    if saved.get('version')!=FAMILY:raise ValueError('unsupported benefit network artifact')
    h=hashlib.sha256()
    for name,value in sorted(saved['learner']['network'].items()):
        h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
    if h.hexdigest()!=saved['network_sha256']:raise ValueError('benefit tensor digest mismatch')
    return dict(benefit_sha256=digest(target),network_sha256=saved['network_sha256'],
                pretrain_windows=saved['pretrain_windows'])


def allocation_options(template,args,out):
    settings=template['settings']
    options=structured.parser().parse_args(['--run',settings['run'],'--scene',settings['scene'],
        '--out',str(out/'allocation'),'--beta',str(settings['beta'])])
    vars(options).update(copy.deepcopy(settings))
    options.out=str(out/'allocation');options.resume=args.resume;options.dry_run=False;options.device=args.device
    options.benefit_checkpoint=str(out/'benefit_network.pt');options.network_mode='frozen'
    options.tier_max_log_odds=None if args.unbounded_tier_updates else args.tier_max_log_odds
    options.allocation_steps=args.allocation_steps;options.allocation_initialization='fresh'
    return options


def main(args,max_updates=None):
    if args.allocation_steps<1 or not math.isfinite(args.tier_max_log_odds) or args.tier_max_log_odds<=0:
        raise ValueError('positive allocation budget and finite tier limit required')
    out=Path(args.out).resolve()
    identity=dict(source=str(Path(args.source).resolve()),allocation_steps=args.allocation_steps,
                  tier_max_log_odds=None if args.unbounded_tier_updates else args.tier_max_log_odds)
    if args.resume:
        document=json.loads((out/'plan.json').read_text(encoding='utf8'))
        if document.get('version')!=VERSION or document['settings']!=identity:raise ValueError('fresh resume settings mismatch')
        if digest(out/'benefit_network.pt')!=document['network']['benefit_sha256']:raise ValueError('fixed benefit network changed')
    else:
        if out.exists():raise FileExistsError('new independent output directory required')
        owner,template,artifact=source_info(args.source)
        if out.is_relative_to(owner):raise ValueError('output must be independent of source run')
        out.mkdir(parents=True)
        print('Copying compact frozen evaluator for a new allocation run.',flush=True)
        network=pin_network(artifact,out/'benefit_network.pt')
        document=dict(version=VERSION,settings=identity,network=network,allocation_template=template,
            initialization=dict(keep_probability=.99,conditional_tiers=[1/3]*3,history='empty',
                                corrections='empty',coverage='zero',views='new seeded schedule',step=0))
        json_write(out/'plan.json',document)
    if (out/'complete.json').is_file() and (out/'review.tar.gz').is_file():
        return json.loads((out/'complete.json').read_text(encoding='utf8'))
    options=allocation_options(document['allocation_template'],args,out)
    print(json.dumps(dict(status='fresh_allocation',start_step=0,allocation_steps=args.allocation_steps,
        pretrain_windows_reused=document['network']['pretrain_windows'],keep_max_log_odds=options.max_log_odds,
        tier_max_log_odds=options.tier_max_log_odds,network_mode='frozen',resuming=args.resume)),flush=True)
    result=engine.main(options,max_updates=max_updates,positive_factory=HybridPositiveProvider,
        family_override=FAMILY,table_factory=partial(TierStepTable,tier_max_log_odds=options.tier_max_log_odds))
    if result.get('status')=='interrupted':return result
    result=dict(result,pipeline_family=VERSION,allocation_initialization='fresh',
        pretrain_windows_reused=document['network']['pretrain_windows'],tier_max_log_odds=options.tier_max_log_odds)
    json_write(out/'complete.json',result);review(out,result)
    return result


if __name__=='__main__':main(parser().parse_args())
