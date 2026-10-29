"""Held-out numeric calibration and decision costs of a unified value model."""
import numpy as np
import torch

from .allocation_benefit_learning import pack_groups
from .allocation_benefit_report import scalar_cost
from .benefit_bias_audit import _common_mode


def summarize_response(records):
    result={}
    for branch in ('positive','keep'):
        rows=[r for r in records if r['branch']==branch]
        if not rows:
            result[branch]=dict(count=0,stages={});continue
        truth=np.array([scalar_cost(r,'target') for r in rows])
        predicted=np.array([scalar_cost(r,'prediction') for r in rows])
        wrong=(np.abs(truth)>1e-14)&(predicted*truth<=0)
        error=np.concatenate([np.asarray(r['prediction'])-np.asarray(r['target']) for r in rows])
        metric=dict(count=len(rows),sign_errors=int(wrong.sum()),sign_error_fraction=float(wrong.mean()),
            regret_mean=float(np.where(wrong,np.abs(truth),0).mean()),rd_mae=float(np.abs(predicted-truth).mean()),
            rd_rmse=float(np.sqrt(np.square(predicted-truth).mean())),
            component_mae=np.abs(error).mean(0).tolist(),component_rmse=np.sqrt(np.square(error).mean(0)).tolist())
        result[branch]=dict(count=len(rows),stages=dict(network=metric))
        regret=np.where(wrong,np.abs(truth),0)
        worst=np.argsort(regret)[-10:][::-1]
        result[branch]['highest_regret_actions']=[dict(source=rows[i]['source'],window=rows[i]['step'],
            old=rows[i]['old'],new=rows[i]['new'],true_cost=float(truth[i]),predicted_cost=float(predicted[i]),
            regret=float(regret[i])) for i in worst if regret[i]>0]
        result[branch]['top10_regret_fraction']=float(regret[worst].sum()/regret.sum()) if regret.sum() else 0.
        if branch=='keep':
            actual=np.stack([np.asarray(r['target']).sum(-1)+np.asarray(r['rates']) for r in rows])
            estimated=np.stack([np.asarray(r['prediction']).sum(-1)+np.asarray(r['rates']) for r in rows])
            choice=estimated.argmin(-1);minimum=actual.min(-1)
            full_actual=np.c_[np.zeros(len(rows)),actual];full_estimated=np.c_[np.zeros(len(rows)),estimated]
            full_choice=full_estimated.argmin(-1)
            values=np.stack([np.asarray(r['prediction']) for r in rows])
            targets=np.stack([np.asarray(r['target']) for r in rows])
            result[branch].update(three_tier_baselines={f'fixed_q{k+1}':float((actual[:,k]-minimum).mean()) for k in range(3)},
                three_tier_selection_regret=dict(network=float((actual[np.arange(len(rows)),choice]-minimum).mean())),
                tier_counts=np.bincount(choice,minlength=3).tolist(),true_best_tier_counts=np.bincount(actual.argmin(-1),minlength=3).tolist(),
                tie_aware_best_accuracy=float(np.isclose(actual[np.arange(len(rows)),choice],minimum,rtol=0,atol=1e-14).mean()),
                four_state_selection_regret=float((full_actual[np.arange(len(rows)),full_choice]-full_actual.min(-1)).mean()),
                factor_errors=dict(common_component_rmse=np.sqrt(np.square((values-targets).mean(1)).mean(0)).tolist(),
                    increment_component_rmse=np.sqrt(np.square(np.diff(values,axis=1)-np.diff(targets,axis=1)).mean((0,1))).tolist()),
                keep_common_mode=_common_mode(rows))
    result['scope']='complete paired four-state groups; calibrated distortion values; no first-order input or baseline'
    return result


def value_selection_score(report,scale):
    keep=report['keep'];positive=report['positive']['stages']['network']
    rmse=np.mean(keep['stages']['network']['component_rmse'])
    return float((rmse+keep['stages']['network']['regret_mean']+positive['regret_mean']+
        keep['three_tier_selection_regret']['network']+keep['four_state_selection_regret'])/max(scale,1e-12))


@torch.no_grad()
def validate_response(learner,step):
    records=learner.validation.records()
    for start in range(0,len(records),128):
        batch=records[start:start+128]
        x,_,_,_,groups,branches=pack_groups(batch,learner.device)
        prediction=learner.predict(x,branches[groups]);offset=0
        for r in batch:
            count=len(r['target']);r['prediction']=prediction[offset:offset+count].cpu();offset+=count
    result=summarize_response(records)
    complete=all(result[b]['count'] for b in ('positive','keep'))
    scale=max(float(learner.network.scales.max()),1e-12)
    score=None
    if complete:
        score=value_selection_score(result,scale)
    result.update(step=step,event='benefit_validation',network_updates=learner.updates,
        network_ready=bool(learner.network.ready),selection_score=score,
        split='heldout Morton blocks; labels excluded from training')
    if hasattr(learner.network,'pair_scale'):
        result['supervision_scales']={key:getattr(learner.network,key+'_scale').detach().cpu().tolist()
                                      for key in ('common','centered','pair')}
    return result


def write_response_report(out,report):
    from .modular_experiment import json_write
    json_write(out/'comparison/benefit_validation.json',report)
