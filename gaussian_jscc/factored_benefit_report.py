"""Complete keep-group policy references and identified value/increment errors."""
import numpy as np


def keep_policy_metrics(records,stages):
    targets=np.stack([np.asarray(r['target'],dtype=float) for r in records])
    rates=np.stack([np.asarray(r['rates'],dtype=float) for r in records])
    actual=targets.sum(-1)+rates;minimum=actual.min(-1)
    optimal=np.isclose(actual,minimum[:,None],rtol=0,atol=1e-14)
    canonical=actual.argmin(-1)
    baselines={f'fixed_q{k+1}':float((actual[:,k]-minimum).mean()) for k in range(3)}
    predictions={};errors={}
    for name,key in stages.items():
        values=np.stack([np.asarray(r[key],dtype=float) for r in records])
        choice=(values.sum(-1)+rates).argmin(-1)
        predictions[name]=dict(tier_counts=np.bincount(choice,minlength=3).tolist(),
            tie_aware_best_accuracy=float(optimal[np.arange(len(records)),choice].mean()))
        common=values.mean(1)-targets.mean(1)
        increments=np.diff(values,axis=1)-np.diff(targets,axis=1)
        errors[name]=dict(common_component_mae=np.abs(common).mean(0).tolist(),
            common_component_rmse=np.sqrt(np.square(common).mean(0)).tolist(),
            increment_component_mae=np.abs(increments).mean((0,1)).tolist(),
            increment_component_rmse=np.sqrt(np.square(increments).mean((0,1))).tolist())
    first=np.stack([np.asarray(r['first_order'],dtype=float) for r in records])
    specific=not bool(np.allclose(first,first[:,:1],rtol=0,atol=1e-14))
    return dict(three_tier_baselines=baselines,three_tier_predictions=predictions,
        true_best_tier_counts=np.bincount(canonical,minlength=3).tolist(),
        true_optimal_tier_counts=optimal.sum(0).tolist(),
        best_tier_count_scope='canonical argmin; true_optimal counts every tied optimum',
        first_order_tier_specific=specific,
        first_order_selection_scope='tier-specific distortion' if specific else 'shared gate distortion; differences come from analytic rate',
        factor_errors=errors)
