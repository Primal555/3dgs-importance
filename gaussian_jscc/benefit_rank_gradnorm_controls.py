"""Train RD magnitude and probabilistic ordering with adaptive task gradients."""
import argparse
import json
import math
from pathlib import Path
import tarfile
import time

import torch

from .benefit_footprint_collect import atomic_save
from .benefit_footprint_controls import evaluate
from .benefit_isolated_positive_controls import load_data,tensor_digest
from .benefit_isolated_positive_model import make_learner
from .benefit_isolated_positive_report import compare_choices
from .benefit_rank_gradnorm import RankGradNormLearner,GradNormBalancer,StratifiedCycles,OBJECTIVE
from .benefit_rd_choice_report import stratified_report
from .benefit_stage_one import make_learner as make_original
from .benefit_stage_one_data import read_groups
from .budgeted_allocation_state import load_checkpoint
from .cli import seed_all
from .modular_experiment import json_write


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='completed footprint_split run with immutable data')
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--evaluate-every',type=int,default=250)
    p.add_argument('--evaluation-batch',type=int,default=8)
    p.add_argument('--training-probe',type=int,default=128)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--gradnorm-alpha',type=float,default=1.5)
    p.add_argument('--weight-lr',type=float,default=.025)
    p.add_argument('--rate-ema',type=float,default=.95)
    p.add_argument('--resume',action='store_true')
    return p


def stratum(group,threshold):
    cost=torch.stack([v['values'].double().cpu() for v in group['views']]).mean(0).sum(-1)
    cost+=torch.as_tensor(group['rates']).double().cpu()
    return int(float(cost.max()-cost.min())>threshold)


def importance(records,device):
    weight=torch.tensor([g.get('importance_weight',1.) for g in records],device=device,dtype=torch.float64)
    if not torch.isfinite(weight).all() or (weight<=0).any():raise ValueError('invalid source importance weights')
    return weight


def aggregate_tasks(losses,labels,weights,strata,mean_importance):
    labels=torch.as_tensor(labels,device=weights.device)
    result=[]
    for s in strata:
        selected=labels==s
        if not selected.any():raise ValueError('every enabled task requires batch support')
        for key in ('magnitude','ranking'):
            result.append((losses[key][selected]*weights[selected]).mean()/mean_importance[s])
    return torch.stack(result)


@torch.no_grad()
def training_references(learner,replay,entries,labels,strata,batch):
    learner.network.eval()
    numerators=torch.zeros(len(strata),2,dtype=torch.float64);denominators=torch.zeros(len(strata),dtype=torch.float64)
    for start in range(0,len(entries),batch):
        records=list(read_groups(replay,entries[start:start+batch]))
        if any(g.get('validation') for g in records):raise ValueError('validation cannot fit training references')
        losses=learner._batch_losses(records);weights=importance(records,learner.device)
        selected=torch.tensor(labels[start:start+batch],device=learner.device)
        for i,s in enumerate(strata):
            mask=selected==s
            for j,key in enumerate(('magnitude','ranking')):
                numerators[i,j]+=float((losses[key][mask]*weights[mask]).sum())
            denominators[i]+=float(weights[mask].sum())
        if start==0 or start+batch>=len(entries) or (start+batch)%512==0:
            print(f'训练基准遍历 {min(start+batch,len(entries))}/{len(entries)}；网络未更新。',flush=True)
    if (denominators<=0).any():raise ValueError('empty training reference stratum')
    means={s:float(denominators[i])/labels.count(s) for i,s in enumerate(strata)}
    return (numerators/denominators[:,None]).flatten().tolist(),means


def main(args):
    if min(args.updates,args.evaluate_every,args.evaluation_batch,args.training_probe,args.cpu_threads)<1:
        raise ValueError('positive training and evaluation budgets required')
    if (not math.isfinite(args.gradnorm_alpha) or args.gradnorm_alpha<0 or
            not math.isfinite(args.weight_lr) or args.weight_lr<=0 or not 0<=args.rate_ema<1):
        raise ValueError('invalid gradient balancing configuration')
    source,out=Path(args.source).resolve(),Path(args.out).resolve()
    if source.is_relative_to(out) or out.is_relative_to(source):raise ValueError('source/output overlap')
    if out.exists() and not args.resume:raise FileExistsError('new output or --resume required')
    torch.set_num_threads(args.cpu_threads)
    data,replay,training,validation,metadata,initial,previous,groups,references,hashes=load_data(source)
    if out.is_relative_to(replay.root.resolve()) or replay.root.resolve().is_relative_to(out):raise ValueError('output/replay overlap')
    config=metadata['config'];seed=metadata['seed'];seed_all(seed)
    threshold=previous['choice_targets']['large_gap_threshold']
    temperature=previous['choice_targets']['temperature']
    factor=1+3*config['pair_weight']
    if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(factor) or factor<=0:
        raise ValueError('invalid training-fitted supervision settings')
    labels=[stratum(g,threshold) for g in read_groups(replay,training)]
    sampler=StratifiedCycles(labels,config['batch_groups'],seed+1)
    task_names=[('ordinary' if s==0 else 'large_gap')+'_'+k for s in sampler.strata for k in ('magnitude','ranking')]
    learner=make_learner(initial,config,args.device,learner_class=RankGradNormLearner)
    learner.configure(temperature,factor)
    anchor_digest=tensor_digest(learner.network.anchor.state_dict())
    order=torch.randperm(len(training),generator=torch.Generator().manual_seed(seed+2))
    probe=[training[i] for i in order[:min(args.training_probe,len(training))].tolist()]
    settings=dict(version=OBJECTIVE,source=str(source),data=str(data),source_hashes=hashes,
        updates=args.updates,reference_updates=previous['updates_per_arm'],device=args.device,source_config=config,
        seed=seed,evaluate_every=args.evaluate_every,evaluation_batch=args.evaluation_batch,
        training_probe_ids=[e['id'] for e in probe],tasks=task_names,temperature=temperature,
        temperature_fit_scope='training records only',regression_factor=factor,
        regression_scale=float(learner.scales['rd']),threshold=threshold,
        gradnorm_alpha=args.gradnorm_alpha,weight_lr=args.weight_lr,rate_ema=args.rate_ema,
        task_weight_parameterization='float64 softmax, sum equals task count, no fixed floor',
        gradient_parameters='relative.state_head.2.weight and bias',
        strata_counts={str(s):labels.count(s) for s in sampler.strata},batch_quotas=sampler.quotas,
        sampling='independent shuffled stratum cycles; each enabled stratum present in every batch',
        weighting='source importance divided by fixed full-training stratum mean; GradNorm changes training surrogate',
        loss='one physical centered component regression + window-mean soft RD RankNet KL',
        selection='final update only',trainable_parameters=sum(p.numel() for p in learner.network.parameters() if p.requires_grad))
    # JSON converts integer dictionary keys to strings; use that form for resume identity.
    settings['batch_quotas']={str(k):v for k,v in sampler.quotas.items()}
    stored=None
    if args.resume:
        stored=json.loads((out/'settings.json').read_text(encoding='utf8'))
        reference_losses=stored['initial_reference_losses']
        mean_importance={int(k):v for k,v in stored['stratum_mean_importance'].items()}
    else:
        reference_losses,mean_importance=training_references(learner,replay,training,labels,sampler.strata,args.evaluation_batch)
    settings['initial_reference_losses']=reference_losses
    settings['stratum_mean_importance']={str(k):v for k,v in mean_importance.items()}
    balancer=GradNormBalancer(task_names,reference_losses,device=args.device,
        alpha=args.gradnorm_alpha,lr=args.weight_lr,ema=args.rate_ema)
    checkpoint=None
    if args.resume:
        if stored!=settings:raise ValueError('resume settings mismatch')
        if (out/'complete.json').is_file() and (out/'review.tar.gz').is_file():
            return json.loads((out/'summary.json').read_text(encoding='utf8'))
        checkpoint=load_checkpoint(out/'latest.pt')
    else:
        out.mkdir(parents=True);(out/'metrics').mkdir();json_write(out/'settings.json',settings)
    baselines={k:stratified_report(groups,p,threshold) for k,p in references.items()}
    old=json.loads((source/'training/summary.json').read_text(encoding='utf8'))
    for k in references:
        if abs(baselines[k]['all']['three_tier']['regret_mean']-old['after'][k]['validation']['all']['three_tier']['regret_mean'])>1e-12:
            raise ValueError('reference metrics cannot be reconciled')
    completed=0;history=[];last_predictions=None;last_stats=None
    if checkpoint:
        learner.load_state_dict(checkpoint['learner']);balancer.load_state_dict(checkpoint['balancer'])
        sampler.load_state_dict(checkpoint['sampler']);completed=checkpoint['completed']
        difference=checkpoint['initial_difference'];last_stats=checkpoint['last_stats']
        history=[json.loads(s) for s in (out/'metrics/history.jsonl').read_text(encoding='utf8').splitlines()
                 if s.strip() and json.loads(s)['additional_updates']<=completed]
        for name,key in [('history','additional_updates'),('optimization','update')]:
            path=out/'metrics'/f'{name}.jsonl'
            if path.is_file():
                records=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]
                path.write_text(''.join(json.dumps(r)+'\n' for r in records if r[key]<=completed),encoding='utf8')
    else:
        original=make_original(dict(settings=config,learner=initial['learner']),args.device)
        _,a=evaluate(original,replay,validation,args.evaluation_batch)
        _,b=evaluate(learner,replay,validation,args.evaluation_batch)
        difference=max(float((x-y).abs().max()) for x,y in zip(a,b))
        if difference>1e-6:raise RuntimeError('initial effective functions differ')
        del original
    print(f'任务={task_names}；批次分层={sampler.quotas}；排序温度={temperature:.7g}；数值系数={factor:g}。',flush=True)
    started=time.perf_counter()
    def measure(update,append=True):
        nonlocal last_predictions
        _,last_predictions=evaluate(learner,replay,validation,args.evaluation_batch)
        tg,tp=evaluate(learner,replay,probe,args.evaluation_batch)
        row=dict(additional_updates=update,validation=stratified_report(groups,last_predictions,threshold),
            training_probe=stratified_report(tg,tp,threshold),task_weights=balancer.weights().detach().cpu().tolist(),
            **{'against_'+k:compare_choices(groups,p,last_predictions) for k,p in references.items()})
        if tensor_digest(learner.network.anchor.state_dict())!=anchor_digest:raise RuntimeError('frozen common network changed')
        if append:
            history.append(row)
            with (out/'metrics/history.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(f'\nRankNet＋GradNorm {update}/{args.updates}',flush=True)
        for scope in ('all','ordinary','large_gap'):
            r=row['validation'][scope]
            if r['three_tier'] is None:continue
            print(f"  {scope}: regret={r['three_tier']['regret_mean']:.7g}, hit={r['three_tier']['best_tier_hit_fraction']:.2%}, pair_RMSE={r['positive_window']['rd_rmse']:.7g}",flush=True)
        c=row['against_original'];excluded=c['excluding_largest_improvement'];active=c['active']
        print(f"  对原结构：改善{c['beneficial_changes']}，恶化{c['harmful_changes']}；排除最大收益regret={excluded['reference_regret']}->{excluded['candidate_regret']}",flush=True)
        print(f"  非零变化样本={active['records']}，命中率={active['candidate_best_hit_fraction']}；任务权重={row['task_weights']}",flush=True)
        print(f"  固定训练探针regret={row['training_probe']['all']['three_tier']['regret_mean']:.7g}",flush=True)
    def save():
        atomic_save(dict(learner=learner.state_dict(),balancer=balancer.state_dict(),sampler=sampler.state_dict(),
            completed=completed,initial_difference=difference,last_stats=last_stats),out/'latest.pt')
    if not checkpoint:measure(0);save()
    shared_parameters=list(learner.network.relative.state_head[2].parameters())
    for update in range(completed+1,args.updates+1):
        chosen=sampler.next();records=list(read_groups(replay,[training[i] for i in chosen]))
        torch.manual_seed(seed+100000+update)
        if torch.device(args.device).type=='cuda':torch.cuda.manual_seed_all(seed+100000+update)
        learner.network.train();learner.optimizer.zero_grad(set_to_none=True)
        losses=learner._batch_losses(records)
        tasks=aggregate_tasks(losses,[labels[i] for i in chosen],importance(records,learner.device),sampler.strata,mean_importance)
        last_stats=balancer.measure(tasks,shared_parameters)
        objective=(tasks*balancer.weights().detach()).sum()
        objective.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in learner.network.parameters() if p.requires_grad],10.,error_if_nonfinite=True)
        learner.optimizer.step();balancer.update_weights();learner.updates+=1;completed=update
        last_stats.update(update=update,network_loss=float(objective.detach()),network_grad_norm=float(norm))
        if update==1 or update%10==0:
            with (out/'metrics/optimization.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(last_stats,allow_nan=False)+'\n')
        if update%50==0:
            print(f"更新 {update}/{args.updates}: loss={float(objective.detach()):.6g}, grad={float(norm):.5g}, elapsed={time.perf_counter()-started:.1f}s",flush=True)
            print(f"  weights={last_stats['weights']}；weighted_grad={last_stats['weighted_gradient_norms']}",flush=True)
        if update%args.evaluate_every==0 or update==args.updates:measure(update);save()
    if last_predictions is None:measure(completed,append=False)
    atomic_save(dict(learner=learner.state_dict(),balancer=balancer.state_dict(),settings=settings,
                     architecture=OBJECTIVE),out/'final.pt')
    with (out/'metrics/final_predictions.jsonl').open('w',encoding='utf8') as f:
        for i,(e,g) in enumerate(zip(validation,groups)):
            r=dict(id=e['id'],source=g['source'],step=g['step'],truth=[v['values'].tolist() for v in g['views']],
                rates=torch.as_tensor(g['rates']).tolist(),rank_gradnorm=last_predictions[i].tolist(),
                **{k:p[i].tolist() for k,p in references.items()})
            f.write(json.dumps(r,allow_nan=False)+'\n')
    result=dict(status='complete',updates=completed,initial_max_prediction_difference=difference,
        anchor_unchanged=tensor_digest(learner.network.anchor.state_dict())==anchor_digest,
        baselines=baselines,before=history[0],after=history[-1],last_optimization=last_stats,
        training_coverage=dict(draws=sum(sampler.visits),min_visits=min(sampler.visits),max_visits=max(sampler.visits),
            never_visited=sum(v==0 for v in sampler.visits),per_stratum={str(s):dict(records=labels.count(s),
                draws=sum(v for v,k in zip(sampler.visits,labels) if k==s)) for s in sampler.strata}))
    json_write(out/'summary.json',result)
    with tarfile.open(out/'review.tmp','w:gz') as archive:
        for rel in ('settings.json','summary.json','metrics/history.jsonl','metrics/optimization.jsonl','metrics/final_predictions.jsonl'):
            if (out/rel).is_file():archive.add(out/rel,arcname=rel)
    (out/'review.tmp').replace(out/'review.tar.gz');json_write(out/'complete.json',dict(status='complete',updates=completed))
    print(f'完成。下载 {out/"review.tar.gz"}',flush=True)
    return result


if __name__=='__main__':main(parser().parse_args())
