"""Train an isolated positive branch on existing immutable candidate observations."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

import torch

from .benefit_footprint_controls import evaluate
from .benefit_footprint_collect import atomic_save
from .benefit_isolated_positive_model import make_learner
from .benefit_isolated_positive_report import compare_choices
from .benefit_rd_choice_report import stratified_report
from .benefit_stage_one import make_learner as make_original
from .benefit_stage_one_data import read_groups,compact_group
from .budgeted_allocation_state import load_checkpoint
from .cli import seed_all
from .modular_experiment import json_write
from .spatial_benefit_replay import SpatialReplay


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='completed footprint_split experiment')
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--evaluate-every',type=int,default=250)
    p.add_argument('--evaluation-batch',type=int,default=8)
    p.add_argument('--training-probe',type=int,default=128)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--allocation-run',help='optional completed online run for read-only feedback audit')
    p.add_argument('--resume',action='store_true')
    return p


def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_data(source):
    data=source/'data';previous=source/'training'
    if not (data/'complete.json').is_file() or not (previous/'complete.json').is_file():
        raise ValueError('completed candidate dataset and previous paired training required')
    metadata=json.loads((data/'metadata.json').read_text(encoding='utf8'))
    manifest=json.loads((data/'data_manifest.json').read_text(encoding='utf8'))
    settings=json.loads((previous/'settings.json').read_text(encoding='utf8'))
    if settings['metadata']!=metadata or settings.get('version')!='footprint_split_controls_v1':
        raise ValueError('reference data identity mismatch')
    replay_root=Path(manifest['replay_root'])
    if not replay_root.is_dir():raise FileNotFoundError(f'missing replay: {replay_root}')
    replay=SpatialReplay(replay_root,max_groups=1,validation_groups=len(manifest['validation']),cache_mib=256)
    train,val=manifest['training'],manifest['validation']
    if not train or not val or set(e['id'] for e in train)&set(e['id'] for e in val):
        raise ValueError('training/validation identity overlap')
    groups=[];sources=set()
    for g in read_groups(replay,val):
        if not g.get('validation'):raise ValueError('validation role mismatch')
        groups.append(compact_group(g));sources.add(g['source'])
    for g in read_groups(replay,train):
        if g.get('validation') or g['source'] in sources:raise ValueError('training/validation source overlap')
        if g.get('supervision')=='action_gap' or any('footprints' not in v['inputs'] for v in g['views']):
            raise ValueError('complete candidate-region teachers required')
    records=[json.loads(s) for s in (previous/'metrics/final_predictions.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    if len(records)!=len(val):raise ValueError('reference labels count differs')
    references={k:[] for k in ('original','upgraded')}
    for entry,g,r in zip(val,groups,records):
        if r['id']!=entry['id'] or r['source']!=g['source'] or r['step']!=g['step']:
            raise ValueError('reference identity differs')
        if not torch.equal(torch.tensor(r['rates'],dtype=torch.float64),torch.as_tensor(g['rates']).double()):
            raise ValueError('reference rates differ')
        truth=torch.stack([v['values'].double() for v in g['views']])
        if not torch.equal(torch.tensor(r['truth'],dtype=torch.float64),truth):
            raise ValueError('reference labels differ')
        for k in references:
            prediction=torch.tensor(r[k],dtype=torch.float64)
            if prediction.shape!=truth.shape or not torch.isfinite(prediction).all():
                raise ValueError('invalid reference prediction')
            references[k].append(prediction)
    initial=load_checkpoint(data/'initial.pt')
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (data/'metadata.json',data/'data_manifest.json',data/'initial.pt',previous/'settings.json',
         previous/'summary.json',previous/'metrics/final_predictions.jsonl')}
    return data,replay,train,val,metadata,initial,settings,groups,references,hashes


def main(args):
    if min(args.updates,args.evaluate_every,args.evaluation_batch,args.training_probe,args.cpu_threads)<1:
        raise ValueError('positive training and evaluation budgets required')
    source,out=Path(args.source).resolve(),Path(args.out).resolve()
    if source.is_relative_to(out) or out.is_relative_to(source):raise ValueError('source/output overlap')
    if out.exists() and not args.resume:raise FileExistsError('new output or --resume required')
    torch.set_num_threads(args.cpu_threads)
    data,replay,training,validation,metadata,initial,previous,groups,references,hashes=load_data(source)
    if out.is_relative_to(replay.root.resolve()) or replay.root.resolve().is_relative_to(out):
        raise ValueError('output/replay overlap')
    config=metadata['config'];seed=metadata['seed'];seed_all(seed)
    learner=make_learner(initial,config,args.device)
    anchor_digest=tensor_digest(learner.network.anchor.state_dict())
    threshold=previous['choice_targets']['large_gap_threshold']
    baselines={k:stratified_report(groups,p,threshold) for k,p in references.items()}
    old_summary=json.loads((source/'training/summary.json').read_text(encoding='utf8'))
    for k in references:
        if abs(baselines[k]['all']['three_tier']['regret_mean']-
               old_summary['after'][k]['validation']['all']['three_tier']['regret_mean'])>1e-12:
            raise ValueError('reference metrics cannot be reconciled')
    probe_order=torch.randperm(len(training),generator=torch.Generator().manual_seed(seed+2))
    probe=[training[i] for i in probe_order[:min(args.training_probe,len(training))].tolist()]
    allocation=str(Path(args.allocation_run).resolve()) if args.allocation_run else None
    settings=dict(version='isolated_positive_controls_v1',source=str(source),data=str(data),source_hashes=hashes,
        updates=args.updates,reference_updates=previous['updates_per_arm'],device=args.device,
        config=config,seed=seed,evaluate_every=args.evaluate_every,evaluation_batch=args.evaluation_batch,
        training_probe_ids=[e['id'] for e in probe],allocation_run=allocation,
        loss='positive centered components + three positive pairs + calibrated positive direction; original RD scale',
        frozen='original common network and all normalizers',selection='final update only',
        trainable_parameters=sum(p.numel() for p in learner.network.parameters() if p.requires_grad))
    checkpoint=None
    if args.resume:
        if json.loads((out/'settings.json').read_text(encoding='utf8'))!=settings:raise ValueError('resume settings mismatch')
        if (out/'complete.json').is_file() and (out/'review.tar.gz').is_file():
            return json.loads((out/'summary.json').read_text(encoding='utf8'))
        checkpoint=load_checkpoint(out/'latest.pt')
    else:
        out.mkdir(parents=True);(out/'metrics').mkdir();json_write(out/'settings.json',settings)
        if allocation:
            from .benefit_feedback_audit import audit_feedback
            json_write(out/'feedback_audit.json',audit_feedback(allocation))
    generator=torch.Generator().manual_seed(seed+1)
    visits=[0]*len(training);order=[];cursor=0;completed=0;history=[]
    if checkpoint:
        learner.load_state_dict(checkpoint['learner']);completed=checkpoint['completed']
        visits,order,cursor=checkpoint['visits'],checkpoint['order'],checkpoint['cursor']
        generator.set_state(checkpoint['generator']);difference=checkpoint['initial_difference']
        history=[json.loads(s) for s in (out/'metrics/history.jsonl').read_text(encoding='utf8').splitlines()
                 if s.strip() and json.loads(s)['additional_updates']<=completed]
        (out/'metrics/history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf8')
    else:
        original=make_original(dict(settings=config,learner=initial['learner']),args.device)
        _,a=evaluate(original,replay,validation,args.evaluation_batch)
        _,b=evaluate(learner,replay,validation,args.evaluation_batch)
        difference=max(float((x-y).abs().max()) for x,y in zip(a,b))
        if difference>1e-6:raise RuntimeError('initial effective functions differ')
        del original
    started=time.perf_counter();last_predictions=None
    def measure(update,append=True):
        nonlocal last_predictions
        _,last_predictions=evaluate(learner,replay,validation,args.evaluation_batch)
        tg,tp=evaluate(learner,replay,probe,args.evaluation_batch)
        row=dict(additional_updates=update,validation=stratified_report(groups,last_predictions,threshold),
            training_probe=stratified_report(tg,tp,threshold),
            **{'against_'+k:compare_choices(groups,p,last_predictions) for k,p in references.items()})
        if tensor_digest(learner.network.anchor.state_dict())!=anchor_digest:raise RuntimeError('frozen common network changed')
        if append:
            history.append(row)
            with (out/'metrics/history.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(f'\n更新 {update}/{args.updates}；独立正档分支；共同网络校验通过。',flush=True)
        for scope in ('all','ordinary','large_gap'):
            r=row['validation'][scope]
            if r['three_tier'] is None:continue
            print(f"  {scope}: regret={r['three_tier']['regret_mean']:.7g}, hit={r['three_tier']['best_tier_hit_fraction']:.2%}, pair_RMSE={r['positive_window']['rd_rmse']:.7g}",flush=True)
        for k in references:
            c=row['against_'+k];e=c['excluding_largest_improvement']
            print(f"  对比{k}末步：改善{c['beneficial_changes']}，恶化{c['harmful_changes']}；排除最大收益后regret={e['reference_regret']}->{e['candidate_regret']}",flush=True)
        active=row['against_original']['active']
        print(f"  非零变化样本={active['records']}，regret={active['reference_regret']}->{active['candidate_regret']}，命中率={active['reference_best_hit_fraction']}->{active['candidate_best_hit_fraction']}",flush=True)
        train_report=row['training_probe']['all']
        print(f"  固定训练探针：regret={train_report['three_tier']['regret_mean']:.7g}, pair_RMSE={train_report['positive_window']['rd_rmse']:.7g}",flush=True)
        return row
    def save():
        atomic_save(dict(learner=learner.state_dict(),completed=completed,visits=visits,order=order,cursor=cursor,
            generator=generator.get_state(),initial_difference=difference),out/'latest.pt')
    if not checkpoint:measure(0);save()
    for update in range(completed+1,args.updates+1):
        chosen=[]
        while len(chosen)<config['batch_groups']:
            if cursor==len(order):order=torch.randperm(len(training),generator=generator).tolist();cursor=0
            take=min(config['batch_groups']-len(chosen),len(order)-cursor)
            chosen.extend(order[cursor:cursor+take]);cursor+=take
        records=list(read_groups(replay,[training[i] for i in chosen]))
        torch.manual_seed(seed+100000+update)
        if torch.device(args.device).type=='cuda':torch.cuda.manual_seed_all(seed+100000+update)
        learner.network.train();losses=learner._batch_losses(records)
        weight=losses['total'].new_tensor([g.get('importance_weight',1.) for g in records])
        if not torch.isfinite(weight).all() or (weight<=0).any():raise ValueError('invalid source importance')
        loss=(losses['total']*weight).mean();learner.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in learner.network.parameters() if p.requires_grad],10.,error_if_nonfinite=True)
        learner.optimizer.step();learner.updates+=1;completed=update
        for i in chosen:visits[i]+=1
        if update%50==0:print(f'独立分支 {update}/{args.updates}: loss={float(loss):.6g}, grad={float(norm):.5g}, sec={time.perf_counter()-started:.1f}',flush=True)
        if update%args.evaluate_every==0 or update==args.updates:measure(update);save()
    if last_predictions is None:measure(completed,append=False)
    final=dict(learner=learner.state_dict(),settings=settings,architecture='isolated_positive_v1')
    atomic_save(final,out/'final.pt')
    with (out/'metrics/final_predictions.jsonl').open('w',encoding='utf8') as f:
        for i,(e,g) in enumerate(zip(validation,groups)):
            r=dict(id=e['id'],source=g['source'],step=g['step'],truth=[v['values'].tolist() for v in g['views']],
                rates=torch.as_tensor(g['rates']).tolist(),isolated=last_predictions[i].tolist(),
                **{k:p[i].tolist() for k,p in references.items()})
            f.write(json.dumps(r,allow_nan=False)+'\n')
    result=dict(status='complete',updates=completed,initial_max_prediction_difference=difference,
        anchor_unchanged=tensor_digest(learner.network.anchor.state_dict())==anchor_digest,
        baselines=baselines,before=history[0],after=history[-1],
        training_coverage=dict(draws=sum(visits),min_visits=min(visits),max_visits=max(visits),never_visited=sum(v==0 for v in visits)))
    json_write(out/'summary.json',result)
    with tarfile.open(out/'review.tmp','w:gz') as archive:
        for rel in ('settings.json','summary.json','metrics/history.jsonl','metrics/final_predictions.jsonl','feedback_audit.json'):
            if (out/rel).is_file():archive.add(out/rel,arcname=rel)
    (out/'review.tmp').replace(out/'review.tar.gz');json_write(out/'complete.json',dict(status='complete',updates=completed))
    print(f'完成。下载 {out/"review.tar.gz"}',flush=True)
    return result


if __name__=='__main__':main(parser().parse_args())
