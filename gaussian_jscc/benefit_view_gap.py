"""View-wise RD magnitude learning with separately scaled ordering gradients."""
import math

import torch

from .benefit_three_task import ThreeTaskLearner, three_losses

OBJECTIVE = 'footprint_view_rd_gap_scaled_rank_v1'
RD_DIRECTION_WEIGHT = 2.
RANK_NUMERIC_RATIO = 1.


def view_gap_losses(prediction, truth, rates, scale, temperature, common_weight, pair_weight):
    losses = three_losses(prediction, truth, rates, scale, temperature, common_weight, pair_weight)
    error = (prediction-truth)/scale
    centered = error-error.mean(-2, keepdim=True)
    rd_error = centered.sum(-1)
    contrast_error = centered[..., 0]-centered[..., 1]
    # Orthogonal directions in the original two-component target space.
    # RD weight=1 recovers the previous relative regression exactly.
    losses['relative'] = (1+2*pair_weight)/4*(
        RD_DIRECTION_WEIGHT*rd_error.square()+contrast_error.square()).mean(-1)
    # dL/d(predicted_gap/scale) is bounded independently of temperature.
    losses['ranking'] = losses['ranking']*(temperature/scale)
    return losses


class ViewGapLearner(ThreeTaskLearner):
    objective = OBJECTIVE
    loss_function = staticmethod(view_gap_losses)


@torch.no_grad()
def assign_with_rank_budget(accumulator, weights, ratio=RANK_NUMERIC_RATIO):
    """Limit the ordering contribution at this update using all parameters."""
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('positive finite rank/numeric gradient ratio required')
    numeric_square = weights.new_zeros((), dtype=torch.double)
    rank_square = numeric_square.clone()
    for common, relative, rank in zip(*accumulator.gradients):
        numeric = weights[0]*common.double()+weights[1]*relative.double()
        ranking = weights[2]*rank.double()
        numeric_square += numeric.square().sum()
        rank_square += ranking.square().sum()
    numeric_norm, rank_norm = numeric_square.sqrt(), rank_square.sqrt()
    if not torch.isfinite(numeric_norm+rank_norm):
        raise ValueError('nonfinite accumulated task gradients')
    factor = min(1., ratio*float(numeric_norm)/float(rank_norm)) if rank_norm > 0 else 1.
    applied = weights.clone(); applied[2] *= factor
    accumulator.assign(applied)
    return applied, dict(numeric_gradient_norm=float(numeric_norm),
        requested_rank_gradient_norm=float(rank_norm), applied_rank_gradient_norm=float(rank_norm)*factor,
        rank_numeric_ratio_limit=ratio, rank_gradient_factor=factor, rank_gradient_limited=factor < 1.,
        applied_task_weights=applied.cpu().tolist())


@torch.no_grad()
def gap_report(groups, predictions, threshold):
    """Record-uniform metrics; pair errors retain the original RD units."""
    if len(groups) != len(predictions): raise ValueError('group/prediction length mismatch')
    buckets = {k: [] for k in ('all', 'ordinary', 'large_gap')}
    i, j = torch.triu_indices(3, 3, 1)
    for group, prediction in zip(groups, predictions):
        truth = torch.stack([v['values'].double().cpu() for v in group['views']])
        prediction = torch.as_tensor(prediction).double().cpu()
        if prediction.shape != truth.shape: raise ValueError('view/tier shape mismatch')
        rates = torch.as_tensor(group['rates']).double().cpu()
        true_cost = truth.sum(-1)+rates
        predicted_cost = prediction.sum(-1)+rates
        t, p = true_cost.mean(0), predicted_cost.mean(0)
        tg, pg = t[j]-t[i], p[j]-p[i]
        view_error = (predicted_cost[:, j]-predicted_cost[:, i])-(true_cost[:, j]-true_cost[:, i])
        correct = bool(t[p.argmin()]-t.min() <= 1e-12)
        item = dict(view_mse=float(view_error.square().mean()),
            window_mse=float((pg-tg).square().mean()), correct=correct,
            order=bool(((tg.abs() <= 1e-12) | (pg*tg > 0)).all()))
        buckets['all'].append(item)
        buckets['large_gap' if float(t.max()-t.min()) > threshold else 'ordinary'].append(item)
    result = dict(scope='record uniform; views averaged within record; RD units; tie tolerance 1e-12')
    for name, rows in buckets.items():
        correct = [r for r in rows if r['correct']]
        result[name] = None if not rows else dict(records=len(rows),
            per_view_pair_rmse=math.sqrt(sum(r['view_mse'] for r in rows)/len(rows)),
            window_pair_rmse=math.sqrt(sum(r['window_mse'] for r in rows)/len(rows)),
            best_tier_correct_fraction=sum(r['correct'] for r in rows)/len(rows),
            complete_order_correct_fraction=sum(r['order'] for r in rows)/len(rows),
            correct_choice_records=len(correct),
            correct_choice_pair_rmse=math.sqrt(sum(r['window_mse'] for r in correct)/len(correct)) if correct else None)
    return result
