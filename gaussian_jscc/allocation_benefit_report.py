"""Paired action diagnostics; network validation never supplies training labels."""
import csv
from pathlib import Path

import numpy as np
import torch

from .allocation_benefit_learning import pack_groups
from .factored_benefit_report import keep_policy_metrics
from .modular_experiment import json_write


STAGES = dict(first_order='first_order',network='prediction',masked='prediction_masked')


def scalar_cost(record,key):
    value=torch.as_tensor(record[key],dtype=torch.float64)
    rate=torch.as_tensor(record['rates'],dtype=torch.float64)
    conditional=torch.as_tensor(record['conditional'],dtype=torch.float64)
    return float(((value.sum(-1)+rate)*conditional).sum())


def rank(values):
    values=np.asarray(values);order=np.argsort(values,kind='stable');sorted_values=values[order]
    result=np.empty(len(order),dtype=float)
    starts=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1]
    ends=np.r_[starts[1:],len(order)]
    for start,end in zip(starts,ends):result[order[start:end]]=(start+end-1)/2
    return result


def correlation(a,b):
    if len(a)<2:return None
    ar,br=rank(a),rank(b)
    if np.std(ar)==0 or np.std(br)==0:return None
    return float(np.corrcoef(ar,br)[0,1])


def summarize_records(records,*,seed=42):
    result={}
    for branch in ('positive','keep'):
        rows=[r for r in records if r['branch']==branch]
        truth=np.array([scalar_cost(r,'target') for r in rows]);stages={};regrets={};wrong={}
        for name,key in STAGES.items():
            predicted=np.array([scalar_cost(r,key) for r in rows])
            # A tied predicted cost is a missed decision when truth is informative.
            mistakes=(np.abs(truth)>1e-14)&(predicted*truth<=0)
            regret=np.where(mistakes,np.abs(truth),0.);regrets[name]=regret;wrong[name]=mistakes
            if branch=='keep' and rows:
                target=np.stack([(np.asarray(r['target'],dtype=float)*np.asarray(r['conditional'])[:,None]).sum(0) for r in rows])
                values=np.stack([(np.asarray(r[key],dtype=float)*np.asarray(r['conditional'])[:,None]).sum(0) for r in rows])
            else:
                target=np.concatenate([np.asarray(r['target'],dtype=float) for r in rows]) if rows else np.empty((0,2))
                values=np.concatenate([np.asarray(r[key],dtype=float) for r in rows]) if rows else np.empty((0,2))
            error=values-target
            stages[name]=dict(count=len(rows),sign_errors=int(mistakes.sum()),
                informative=int((np.abs(truth)>1e-14).sum()),
                sign_error_fraction=float(mistakes.sum()/max(1,(np.abs(truth)>1e-14).sum())),
                regret_mean=float(regret.mean()) if len(rows) else None,
                rd_mae=float(np.abs(predicted-truth).mean()) if len(rows) else None,
                rd_rmse=float(np.sqrt(np.square(predicted-truth).mean())) if len(rows) else None,
                component_mae=np.abs(error).mean(0).tolist() if len(rows) else None,
                component_rmse=np.sqrt(np.square(error).mean(0)).tolist() if len(rows) else None,
                distortion_mae=float(np.abs(error.sum(-1)).mean()) if len(rows) else None,
                distortion_spearman=correlation(target.sum(-1),values.sum(-1)))
        rescued=wrong['first_order']&~wrong['network'];broken=~wrong['first_order']&wrong['network']
        windows=sorted(set(r['step'] for r in rows));paired=[]
        for step in windows:
            mask=np.array([r['step']==step for r in rows])
            paired.append(float((regrets['network'][mask]-regrets['first_order'][mask]).mean()))
        interval=None
        if len(paired)>1:
            generator=np.random.default_rng(seed)
            # Bootstrap whole windows, keeping all q0 states within each group.
            distribution=[float(np.mean(generator.choice(paired,len(paired),replace=True))) for _ in range(1000)]
            interval=np.quantile(distribution,[.025,.975]).tolist()
        groups={}
        for r in rows:
            key=f'{r["old"]}->{"mixed" if branch=="keep" else r["new"][0]}/{r["group"]}'
            value=groups.setdefault(key,dict(count=0,network_sign_errors=0,network_regret_sum=0.))
            actual=scalar_cost(r,'target');pred=scalar_cost(r,'prediction')
            error=abs(actual)>1e-14 and actual*pred<=0
            value['count']+=1;value['network_sign_errors']+=int(error)
            value['network_regret_sum']+=abs(actual) if error else 0.
        tail=np.argsort(regrets['network'])[-10:][::-1] if rows else []
        part=dict(count=len(rows),stages=stages,rescued_count=int(rescued.sum()),broken_count=int(broken.sum()),
            statistics_unit='conditional mixture per keep group' if branch=='keep' else 'one directed positive action',
            rescued_regret=float(np.abs(truth[rescued]).sum()),broken_regret=float(np.abs(truth[broken]).sum()),
            window_paired_mean=float(np.mean(paired)) if paired else None,window_paired_ci95=interval,
            improved_windows=sum(x<0 for x in paired),evaluated_windows=len(windows),groups=groups,
            highest_regret_actions=[dict(source=rows[i]['source'],step=rows[i]['step'],regret=float(regrets['network'][i]))
                                    for i in tail if regrets['network'][i]>0])
        if branch=='keep' and rows:
            selection={}
            for name,key in STAGES.items():
                values=[]
                for r in rows:
                    actual=np.asarray(r['target']).sum(-1)+np.asarray(r['rates'])
                    choice=int((np.asarray(r[key]).sum(-1)+np.asarray(r['rates'])).argmin())
                    values.append(float(actual[choice]-actual.min()))
                selection[name]=float(np.mean(values))
            part['three_tier_selection_regret']=selection
            part.update(keep_policy_metrics(rows,STAGES))
        result[branch]=part
    result['scope']='selected paired actions; positive pairwise RD and conditional keep RD reported separately'
    return result


@torch.no_grad()
def validate_network(learner,step):
    records=learner.validation.records()
    for start in range(0,len(records),128):
        batch=records[start:start+128]
        features,_,_,_,groups,branches=pack_groups(batch,learner.device)
        full=learner.predict(features,branches[groups]);masked=learner.predict(features,branches[groups],masked=True)
        offset=0
        for r in batch:
            count=len(r['target'])
            r['prediction']=full[offset:offset+count].cpu();r['prediction_masked']=masked[offset:offset+count].cpu()
            offset+=count
    result=summarize_records(records)
    result.update(step=step,event='benefit_validation',network_updates=learner.updates,
                  network_ready=bool(learner.network.ready),split='heldout Morton blocks; labels excluded from fitting')
    parts=[]
    improved=True
    for branch in ('positive','keep'):
        stages=result[branch]['stages'];base=stages['first_order']['regret_mean'];new=stages['network']['regret_mean']
        if base is None or new is None:
            improved=False;continue
        parts.append(new/max(base,1e-14));improved=improved and new<base
    result['selection_score']=float(np.mean(parts)) if len(parts)==2 else None
    result['both_branches_lower_regret']=improved and len(parts)==2
    return result


def action_log(record):
    return {k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in record.items() if k!='features'}


def write_benefit_report(out,report):
    out=Path(out);json_write(out/'comparison/benefit_validation.json',report)
    path=out/'comparison/benefit_stages.csv';path.parent.mkdir(parents=True,exist_ok=True)
    fields=('branch','stage','count','sign_errors','sign_error_fraction','regret_mean','rd_mae','rd_rmse','distortion_mae','distortion_spearman')
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for branch in ('positive','keep'):
            for name,values in report[branch]['stages'].items():
                writer.writerow(dict(branch=branch,stage=name,**{k:values.get(k) for k in fields[2:]}))


def write_timing_report(out,rows):
    phases={}
    for phase in ('pretrain','allocation'):
        values=[r for r in rows if r['phase']==phase]
        stable=values[10:] if len(values)>10 else values
        keys=sorted({k for r in stable for k in r if k.endswith('_seconds')})
        phases[phase]={k:dict(median=float(np.median([r[k] for r in stable if k in r])),
                             p95=float(np.percentile([r[k] for r in stable if k in r],95))) for k in keys}
        phases[phase]['peak_allocated_mib']=max((r['peak_allocated_mib'] or 0 for r in values),default=0)
    json_write(Path(out)/'comparison/benefit_timing.json',phases)
