"""Decision-boundary checks on fixed, paired action labels."""
import numpy as np
import torch

from .allocation_benefit_replay import cpu_copy
from .allocation_benefit_report import STAGES, scalar_cost


def fit_offsets(records):
    if not records or any(r['validation'] for r in records):
        raise ValueError('offset fitting requires training records; excludes validation')
    return {branch:float(np.mean([scalar_cost(r,'prediction')-scalar_cost(r,'target')
                                 for r in records if r['branch']==branch]))
            for branch in ('positive','keep') if any(r['branch']==branch for r in records)}


def correct_offsets(records, offsets):
    result=cpu_copy(records)
    for record in result:
        # One shared distortion offset leaves the analytic rate and tier ranking intact.
        record['prediction'][:,0]-=offsets[record['branch']]
    return result


def fit_thresholds(records):
    if not records or any(r['validation'] for r in records):
        raise ValueError('magnitude thresholds require training records')
    return {branch:np.quantile([abs(scalar_cost(r,'target')) for r in records if r['branch']==branch],
                              [.25,.75]).tolist() for branch in ('positive','keep')}


def _metrics(truth, predicted):
    n=len(truth)
    if not n:return dict(count=0,signed_bias=None,rd_mae=None,regret_mean=None,
                         false_beneficial=0,false_harmful=0,sign_error_fraction=None)
    error=predicted-truth
    if not np.isfinite(np.r_[truth,predicted]).all():raise ValueError('nonfinite diagnostic cost')
    informative=np.abs(truth)>1e-14
    wrong=informative&(predicted*truth<=0)
    return dict(count=n,informative=int(informative.sum()),signed_bias=float(error.mean()),
        error_std=float(error.std()),rd_mae=float(np.abs(error).mean()),rd_rmse=float(np.sqrt(np.square(error).mean())),
        true_mean=float(truth.mean()),predicted_mean=float(predicted.mean()),
        true_std=float(truth.std()),predicted_std=float(predicted.std()),
        signed_error_quantiles=np.quantile(error,[.1,.5,.9]).tolist(),
        sign_error_fraction=float(wrong.sum()/max(1,informative.sum())),
        regret_mean=float(np.where(wrong,np.abs(truth),0.).mean()),
        false_beneficial=int(((truth>1e-14)&(predicted<=0)).sum()),
        false_harmful=int(((truth < -1e-14)&(predicted>=0)).sum()))


def _common_mode(records):
    if not records:return dict(count=0)
    actual=np.stack([np.asarray(r['target']).sum(-1)+np.asarray(r['rates']) for r in records])
    predicted=np.stack([np.asarray(r['prediction']).sum(-1)+np.asarray(r['rates']) for r in records])
    error=predicted-actual;common=error.mean(-1,keepdims=True);centered=error-common
    energy=float(np.square(error).sum())
    right=predicted.argmin(-1)==actual.argmin(-1)
    true_keep=np.array([scalar_cost(r,'target') for r in records])
    pred_keep=np.array([scalar_cost(r,'prediction') for r in records])
    wrong=(np.abs(true_keep)>1e-14)&(true_keep*pred_keep<=0)
    return dict(count=len(records),common_bias_mean=float(common.mean()),
        common_error_rmse=float(np.sqrt(np.square(common).mean())),
        centered_error_rmse=float(np.sqrt(np.square(centered).mean())),
        common_mode_error_energy_fraction=float(3*np.square(common).sum()/energy) if energy else None,
        best_tier_accuracy=float(right.mean()),correct_tier_wrong_keep_decision=int((right&wrong).sum()))


def cost_audit(records, thresholds):
    result={}
    for branch in ('positive','keep'):
        rows=[r for r in records if r['branch']==branch]
        truth=np.array([scalar_cost(r,'target') for r in rows])
        values={name:np.array([scalar_cost(r,key) for r in rows]) for name,key in STAGES.items()}
        def subset(mask):return {name:_metrics(truth[mask],v[mask]) for name,v in values.items()}
        low,high=thresholds[branch];size=np.abs(truth)
        transitions={}
        names=[f'{r["old"]}->{r["new"][0]}' if branch=='positive' else f'{r["old"]}->keep_vs_drop' for r in rows]
        for name in sorted(set(names)):transitions[name]=subset(np.array([v==name for v in names]))
        part=dict(stages={name:_metrics(truth,v) for name,v in values.items()},transitions=transitions,
            magnitude_thresholds=[low,high],magnitude=dict(
                small=subset(size<=low),medium=subset((size>low)&(size<=high)),large=subset(size>high)),
            truth_direction=dict(beneficial=subset(truth < -1e-14),harmful=subset(truth>1e-14)))
        if branch=='keep':part['keep_common_mode']=_common_mode(rows)
        wrong=(np.abs(truth)>1e-14)&(truth*values['network']<=0)
        order=np.argsort(np.where(wrong,np.abs(truth),0.))[::-1][:10]
        part['highest_regret_actions']=[dict(source=rows[i]['source'],window=rows[i]['step'],
            old=rows[i]['old'],new=rows[i]['new'],true_cost=float(truth[i]),
            predicted_cost=float(values['network'][i]),regret=float(abs(truth[i]))) for i in order if wrong[i]]
        result[branch]=part
    return result
