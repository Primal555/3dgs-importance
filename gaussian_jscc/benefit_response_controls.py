"""Exact response components: frozen-codec collection and fixed-label fitting."""
import argparse
import copy
import gc
import json
from pathlib import Path
import tarfile
import time

import torch

from .benefit_footprint_collect import atomic_save,collect
from .benefit_footprint_rank_gradnorm_controls import branch_digests
from .benefit_isolated_positive_controls import tensor_digest
from .benefit_rank_gradnorm import GradNormBalancer,StratifiedCycles
from .benefit_rd_choice_losses import fit_choice_targets
from .benefit_rd_choice_report import stratified_report
from .benefit_response_data import bootstrap,load_data,fit_term_scales,compact
from .benefit_response_features import SCHEMA,ResponseTermsWindow
from .benefit_response_model import (VERSION,TASKS,ResponseTermsNetwork,combine_terms,
                                     response_losses,task_weights,assign_gradients)
from .benefit_footprint_model import footprint_pack
from .benefit_stage_one_data import read_groups
from .benefit_three_task import TaskAccumulator,sampling_weights,make_learner
from .benefit_three_task_controls import append_json,trim_records
from .benefit_rank_gradnorm_controls import importance,stratum
from .benefit_view_gap import OBJECTIVE as VIEW_GAP,gap_report
from .budgeted_allocation_state import load_checkpoint
from .cli import seed_all
from .modular_experiment import json_write


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='original completed footprint_split run')
    p.add_argument('--out',required=True,help='new experiment root with data and training subdirectories')
    p.add_argument('--phase',choices=('all','collect','train'),default='all')
    p.add_argument('--codec');p.add_argument('--pipeline-root');p.add_argument('--manifest')
    p.add_argument('--scene',default='truck');p.add_argument('--data')
    p.add_argument('--collect-device',default='cuda');p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--train-records',type=int,default=4096)
    p.add_argument('--validation-records',type=int,default=512)
    p.add_argument('--label-groups',type=int,default=64);p.add_argument('--candidate-batch',type=int,default=64)
    p.add_argument('--max-collect-windows',type=int,default=300)
    p.add_argument('--samples',type=int,default=64000);p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--evaluate-every-samples',type=int,default=2048)
    p.add_argument('--gradient-every-samples',type=int,default=128)
    p.add_argument('--evaluation-batch',type=int,default=8)
    p.add_argument('--training-probe',type=int,default=128)
    p.add_argument('--resume',action='store_true')
    return p


def collect_data(args):
    bundle,metadata=bootstrap(args.source)
    options=copy.copy(args)
    options.out=str(Path(args.out)/'data');options.comparison=args.source
    options.codec=args.codec or metadata['codec_run']
    options.device=args.collect_device
    options.pipeline_root=args.pipeline_root or str(Path(bundle['initial']['settings']['source_checkpoint']).parents[3])
    options.resume=args.resume and Path(options.out).exists()
    return collect(options,window_factory=ResponseTermsWindow,label_version=SCHEMA,initial_bundle=bundle)


def batch_losses(network,records,rd_scale,term_scales,temperature,config):
    sizes=[len(g['views']) for g in records]
    inputs=[v['inputs'] for g in records for v in g['views']]
    prediction=network.forward_terms(*footprint_pack(inputs,next(network.parameters()).device))
    truth=torch.stack([v['response_terms'] for g in records for v in g['views']]).to(prediction)
    rates=torch.stack([torch.as_tensor(g['rates']) for g in records]).to(prediction)
    owner=torch.repeat_interleave(torch.arange(len(records),device=prediction.device),torch.tensor(sizes,device=prediction.device))
    per=response_losses(prediction,truth,rates[owner],rd_scale,term_scales,temperature,
                        config['common_weight'],config['pair_weight'])
    result={k:torch.stack([chunk.mean() for chunk in per[k].split(sizes)]) for k in TASKS[:-1]}
    mean_p=torch.stack([v.mean(0) for v in prediction.split(sizes)])
    mean_t=torch.stack([v.mean(0) for v in truth.split(sizes)])
    result['ranking']=response_losses(mean_p,mean_t,rates,rd_scale,term_scales,temperature,
        config['common_weight'],config['pair_weight'])['ranking']
    return result


@torch.no_grad()
def evaluate(network,replay,entries,batch):
    network.eval();groups=[];terms=[]
    for start in range(0,len(entries),batch):
        records=list(read_groups(replay,entries[start:start+batch]))
        output=network.forward_terms(*footprint_pack([v['inputs'] for g in records for v in g['views']],
                                                    next(network.parameters()).device))
        groups.extend(compact(g) for g in records)
        terms.extend(v.cpu() for v in output.split([len(g['views']) for g in records]))
    return groups,terms,[combine_terms(v) for v in terms]


def response_report(groups,terms,threshold):
    result={}
    for name in ('all','ordinary','large_gap'):
        selected=[(g,p) for g,p in zip(groups,terms) if name=='all' or
                  ('large_gap' if stratum(g,threshold) else 'ordinary')==name]
        if not selected:result[name]=None;continue
        error=torch.stack([(p-torch.stack([v['response_terms'] for v in g['views']])).square().mean((0,1))
                           for g,p in selected])
        result[name]=dict(records=len(selected),per_view_term_rmse=error.mean(0).sqrt().tolist(),
            terms=['cross','energy','weighted_structure'],minimum_predicted_energy=min(float(p[...,1].min()) for _,p in selected))
    return result


def train(args):
    data=Path(args.data or Path(args.out)/'data').resolve();out=Path(args.out).resolve()/'training'
    if data.is_relative_to(out) or out.is_relative_to(data):raise ValueError('dataset/training output overlap')
    replay,training,validation,groups,metadata,initial,hashes=load_data(data)
    if out.is_relative_to(replay.root.resolve()) or replay.root.resolve().is_relative_to(out):
        raise ValueError('output/replay overlap')
    if out.exists() and not args.resume:raise FileExistsError('new output or --resume required')
    config=metadata['config'];seed=metadata['seed'];seed_all(seed)
    targets=fit_choice_targets(groups['training']);threshold=targets['large_gap_threshold'];temperature=targets['temperature']
    term_scales=fit_term_scales(groups['training']).to(args.device)
    base=make_learner(initial,config,args.device,temperature,objective=VIEW_GAP)
    network=ResponseTermsNetwork.from_spatial(base.network.state_dict(),config['dropout'],float(term_scales[1])).to(args.device)
    rd_scale=float(base.scales['rd']); initial_digests=branch_digests(network)
    del base
    initial_digests['energy_head']=tensor_digest(network.energy_head.state_dict())
    buffers=tensor_digest(dict(network.named_buffers()))
    optimizer=torch.optim.AdamW(network.parameters(),lr=config['benefit_lr'],weight_decay=config['weight_decay'])
    labels=[stratum(g,threshold) for g in groups['training']]
    sampler=StratifiedCycles(labels,64,seed+1)
    probe_order=torch.randperm(len(training),generator=torch.Generator().manual_seed(seed+2))
    probe=[training[i] for i in probe_order[:min(args.training_probe,len(training))].tolist()]
    settings=dict(version=VERSION,data=str(data),source_hashes=hashes,seed=seed,config=config,
        samples=args.samples,batch=64,microbatch=args.microbatch,term_scales=term_scales.cpu().tolist(),
        rd_scale=rd_scale,temperature=temperature,threshold=threshold,initial_branch_digests=initial_digests,
        main_value_weight=1.,auxiliary_total_weight=1.,ranking_gain=temperature/rd_scale,
        ranking_requested_weight=1.,ranking_gradient_budget='all-parameter norm <= numeric gradient norm',
        gradnorm_tasks=['cross','energy','structure'],gradnorm_parameters='points.2.weight and bias',
        evaluate_every_samples=args.evaluate_every_samples,gradient_every_samples=args.gradient_every_samples,
        evaluation_batch=args.evaluation_batch,training_probe_ids=[v['id'] for v in probe],
        device=args.device,cpu_threads=args.cpu_threads,selection='final update only')
    settings['stratum_loss_scope']='importance and sampling corrected batch mean, before task weighting'
    stored=json.loads((out/'settings.json').read_text(encoding='utf8')) if (out/'settings.json').is_file() else None
    if stored and {k:v for k,v in stored.items() if k!='auxiliary_reference_losses'}!=settings:
        raise ValueError('response training resume settings mismatch')
    refs=stored['auxiliary_reference_losses'] if stored else None
    if refs is None:
        network.eval();total=torch.zeros(3,dtype=torch.double,device=args.device)
        with torch.no_grad():
            for first in range(0,len(training),args.evaluation_batch):
                records=list(read_groups(replay,training[first:first+args.evaluation_batch]))
                losses=batch_losses(network,records,rd_scale,term_scales,temperature,config)
                weight=importance(records,args.device)
                total+=torch.stack([(losses[k]*weight).sum() for k in TASKS[1:4]])
                if first==0 or first+args.evaluation_batch>=len(training) or (first+args.evaluation_batch)%512==0:
                    print(f'Fit auxiliary references: {min(first+args.evaluation_batch,len(training))}/{len(training)}',flush=True)
        refs=(total/len(training)).cpu().tolist()
    settings['auxiliary_reference_losses']=refs
    balancer=GradNormBalancer(list(TASKS[1:4]),refs,device=args.device)
    completed=0;history=[];last=None;totals=dict(updates=0,clipped=0,rank_limited=0,max_gradient=0.)
    if (out/'latest.pt').is_file():
        state=load_checkpoint(out/'latest.pt')
        network.load_state_dict(state['network']);optimizer.load_state_dict(state['optimizer'])
        balancer.load_state_dict(state['balancer']);sampler.load_state_dict(state['sampler'])
        completed=state['completed'];last=state['last'];totals=state['gradient_totals']
        history=trim_records(out/'metrics/history.jsonl',completed*64)
        trim_records(out/'metrics/optimization.jsonl',completed*64)
    elif args.resume and out.exists():raise FileNotFoundError('resume requires training/latest.pt')
    else:
        out.mkdir(parents=True);(out/'metrics').mkdir();json_write(out/'settings.json',settings)
    if (out/'complete.json').is_file():return json.loads((out/'summary.json').read_text(encoding='utf8'))
    parameters=list(network.parameters());shared={id(p) for p in network.points[2].parameters()}
    shared_indices=[i for i,p in enumerate(parameters) if id(p) in shared]
    latest_predictions=None;latest_terms=None
    def measure():
        nonlocal latest_predictions,latest_terms
        compact_val,latest_terms,latest_predictions=evaluate(network,replay,validation,args.evaluation_batch)
        compact_train,train_terms,train_predictions=evaluate(network,replay,probe,args.evaluation_batch)
        row=dict(sample_visits=completed*64,optimizer_updates=completed,
            validation=stratified_report(compact_val,latest_predictions,threshold),
            training_probe=stratified_report(compact_train,train_predictions,threshold),
            gap_quality=gap_report(compact_val,latest_predictions,threshold),
            training_gap_quality=gap_report(compact_train,train_predictions,threshold),
            response_quality=response_report(compact_val,latest_terms,threshold),
            training_response_quality=response_report(compact_train,train_terms,threshold),
            task_weights=task_weights(balancer.weights().detach()).cpu().tolist(),gradient_totals=dict(totals))
        if tensor_digest(dict(network.named_buffers()))!=buffers:raise RuntimeError('normalizers changed')
        history.append(row);append_json(out/'metrics/history.jsonl',row)
        print(f'\nsamples={completed*64}/{args.samples}, updates={completed}, weights={row["task_weights"]}',flush=True)
        for scope in ('all','ordinary','large_gap'):
            m=row['validation'][scope];g=row['gap_quality'][scope]
            if m['three_tier']:
                print(f'  {scope}: n={m["records"]}, regret={m["three_tier"]["regret_mean"]:.7g}, '
                      f'view_gap_RMSE={g["per_view_pair_rmse"]:.7g}, complete_order={g["complete_order_correct_fraction"]:.2%}, '
                      f'term_RMSE={row["response_quality"][scope]["per_view_term_rmse"]}',flush=True)
        return row
    def save():
        atomic_save(dict(network=network.state_dict(),optimizer=optimizer.state_dict(),balancer=balancer.state_dict(),
            sampler=sampler.state_dict(),completed=completed,last=last,gradient_totals=totals),out/'latest.pt')
    if not history:measure();save()
    started=time.perf_counter()
    for update in range(completed+1,args.samples//64+1):
        chosen=sampler.next();correction=sampling_weights(labels,sampler,chosen).to(args.device)
        records=list(read_groups(replay,[training[i] for i in chosen]))
        weights=importance(records,args.device)*correction/64
        network.train();optimizer.zero_grad(set_to_none=True)
        accumulator=TaskAccumulator(parameters,shared_indices,names=TASKS)
        diagnostics=update==1 or update*64%args.gradient_every_samples==0 or update*64==args.samples
        for first in range(0,64,args.microbatch):
            end=min(first+args.microbatch,64)
            torch.manual_seed(seed+100000+(update-1)*64+first)
            if torch.device(args.device).type=='cuda':torch.cuda.manual_seed_all(seed+100000+(update-1)*64+first)
            losses=batch_losses(network,records[first:end],rd_scale,term_scales,temperature,config)
            accumulator.add(torch.stack([losses[k] for k in TASKS],-1),weights[first:end],
                            [labels[i] for i in chosen[first:end]],diagnostics=diagnostics)
        balance=balancer.measure_gradients(accumulator.losses[1:4],accumulator.shared_norms()[1:4])
        requested=task_weights(balancer.weights().detach())
        balance['raw_weight_sum']=float(sum(balance['weights']))
        balance['scope']='raw auxiliary GradNorm controller; divide weighted/target norms by raw_weight_sum for applied auxiliary units'
        applied,budget=assign_gradients(accumulator,requested)
        norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
        optimizer.step();balancer.update_weights();completed=update
        totals['updates']+=1;totals['clipped']+=int(norm>10);totals['rank_limited']+=int(budget['rank_gradient_limited'])
        totals['max_gradient']=max(totals['max_gradient'],float(norm))
        last=dict(sample_visits=update*64,update=update,names=list(TASKS),losses=accumulator.losses.cpu().tolist(),
                  requested_task_weights=requested.cpu().tolist(),gradnorm=balance,network_grad_norm=float(norm),
                  auxiliary_weighted_gradient_norms=(accumulator.shared_norms()[1:4]*requested[1:4]).cpu().tolist(),
                  auxiliary_target_gradient_norms=[v/balance['raw_weight_sum'] for v in balance['target_gradient_norms']],
                  auxiliary_gradient_scope='shared parameters; actual auxiliary weights sum1; before global clip',
                  clipped=bool(norm>10),applied_weighted_loss=float((accumulator.losses*applied).sum()),**budget,
                  strata=accumulator.stratum_report(applied) if diagnostics else None)
        if diagnostics:
            append_json(out/'metrics/optimization.jsonl',last)
            print(f'update={update}, samples={update*64}, grad={float(norm):.5g}, '
                  f'numeric_grad={budget["numeric_gradient_norm"]:.5g}, '
                  f'rank_grad={budget["requested_rank_gradient_norm"]:.5g}->{budget["applied_rank_gradient_norm"]:.5g}, '
                  f'value_weight={float(applied[0]):.1f}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
            print(f'  value/cross/energy/structure/rank losses={last["losses"]}; '
                  f'applied_weights={last["applied_task_weights"]}',flush=True)
        if update*64%args.evaluate_every_samples==0 or update*64==args.samples:measure();save()
    if latest_predictions is None:
        _,latest_terms,latest_predictions=evaluate(network,replay,validation,args.evaluation_batch)
    with (out/'metrics/final_predictions.jsonl').open('w',encoding='utf8') as stream:
        for e,g,p,t in zip(validation,groups['validation'],latest_predictions,latest_terms):
            row=dict(id=e['id'],source=g['source'],step=g['step'],rates=torch.as_tensor(g['rates']).tolist(),
                truth=[v['values'].tolist() for v in g['views']],truth_terms=[v['response_terms'].tolist() for v in g['views']],
                response=p.tolist(),predicted_terms=t.tolist())
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    final=dict(network=network.state_dict(),settings=settings,optimizer=optimizer.state_dict(),balancer=balancer.state_dict())
    atomic_save(final,out/'final.pt')
    final_digests=branch_digests(network)
    final_digests['energy_head']=tensor_digest(network.energy_head.state_dict())
    result=dict(status='complete',version=VERSION,sample_visits=completed*64,optimizer_updates=completed,
        before=history[0],after=history[-1],gradient_totals=totals,
        branch_changed={k:final_digests[k]!=v for k,v in initial_digests.items()},
        training_coverage=dict(draws=sum(sampler.visits),min_visits=min(sampler.visits),max_visits=max(sampler.visits),
                               never_visited=sum(v==0 for v in sampler.visits)),
        normalizers_unchanged=tensor_digest(dict(network.named_buffers()))==buffers)
    json_write(out/'summary.json',result);json_write(out/'complete.json',dict(status='complete',sample_visits=completed*64))
    return result


def main(args):
    numbers=(args.cpu_threads,args.train_records,args.validation_records,args.label_groups,args.candidate_batch,
        args.max_collect_windows,args.samples,args.microbatch,args.evaluate_every_samples,args.gradient_every_samples,
        args.evaluation_batch,args.training_probe)
    if min(numbers)<1 or any(v%64 for v in (args.samples,args.evaluate_every_samples,args.gradient_every_samples)):
        raise ValueError('positive budgets; sample/report intervals divisible by64 required')
    source,out=Path(args.source).resolve(),Path(args.out).resolve()
    if source.is_relative_to(out) or out.is_relative_to(source):raise ValueError('source/output overlap')
    torch.set_num_threads(args.cpu_threads)
    if args.phase in ('all','collect'):collect_data(args)
    if args.phase=='collect':return
    if args.phase=='all' and torch.device(args.device).type=='cpu' and torch.cuda.is_initialized():
        gc.collect();torch.cuda.empty_cache()
    result=train(args)
    with tarfile.open(out/'review.tmp','w:gz') as archive:
        for n in ('settings.json','summary.json','metrics/history.jsonl','metrics/optimization.jsonl','metrics/final_predictions.jsonl'):
            archive.add(out/'training'/n,arcname='training/'+n)
        data=Path(args.data or out/'data')
        archive.add(data/'metadata.json',arcname='data/metadata.json')
    (out/'review.tmp').replace(out/'review.tar.gz')
    json_write(out/'complete.json',dict(status='complete',version=VERSION,sample_visits=args.samples,optimizer_updates=args.samples//64))
    print(f'Complete. Download {out/"review.tar.gz"}',flush=True)
    return result


if __name__=='__main__':main(parser().parse_args())
