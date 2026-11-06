"""Paired record-level decision accounting in the original normalized RD units."""
from collections import Counter

import torch


TOL = 1e-14


def compare_record(entry, group, original, repaired):
    truth = torch.stack([v['values'].double().cpu() for v in group['views']])
    predictions = [torch.as_tensor(p).detach().cpu().double() for p in (original, repaired)]
    rates = torch.as_tensor(group['rates']).double().cpu()
    if truth.ndim != 3 or truth.shape[1:] != (3, 2) or rates.shape != (3,):
        raise ValueError('complete positive-tier teacher records required')
    if any(p.shape != truth.shape for p in predictions):
        raise ValueError('prediction and teacher shapes differ')
    if not all(torch.isfinite(v).all() for v in (truth, rates, *predictions)):
        raise ValueError('all predictions, rates and labels must be finite')
    mean = truth.mean(0)
    actual = mean.sum(-1)+rates
    minimum = actual.min()
    indices = []
    arms = {}
    for name, prediction in zip(('original', 'repaired'), predictions):
        p = prediction.mean(0)
        cost = p.sum(-1)+rates
        choice = int(cost.argmin())
        indices.append(choice)
        regret = float(actual[choice]-minimum)
        four = torch.cat((cost.new_zeros(1), cost))
        four_actual = torch.cat((actual.new_zeros(1), actual))
        common_error = (p-mean).mean(0)
        arms[name] = dict(choice=choice+1, regret=regret, correct=regret <= TOL,
            predicted_components=p.tolist(), predicted_costs=cost.tolist(),
            prediction_per_view=prediction.tolist(), component_errors=(p-mean).tolist(),
            common_component_errors=common_error.tolist(),
            centered_component_errors=((p-mean)-common_error).tolist(),
            predicted_decision_margin=float(cost.sort().values[1]-cost.min()),
            auxiliary_drop=bool(four.argmin() == 0),
            auxiliary_four_regret=float(four_actual[four.argmin()]-four_actual.min()))
    a, b = indices
    delta = arms['repaired']['regret']-arms['original']['regret']
    ac, bc = arms['original']['correct'], arms['repaired']['correct']
    category = ('both_correct' if ac and bc else 'new_wrong_choice' if ac else
                'corrected_choice' if bc else 'both_wrong')
    # Difference of regrets equals the actual RD change of selecting B instead of A.
    components = torch.cat((mean[b]-mean[a], (rates[b]-rates[a]).reshape(1)))
    return dict(id=entry['id'], source=int(group['source']), collection_step=int(group['step']),
        views=len(truth), category=category, changed=a != b,
        truth_components=mean.tolist(), truth_per_view=truth.tolist(), rates=rates.tolist(),
        true_costs=actual.tolist(), true_best_tier=int(actual.argmin())+1,
        true_best_margin=float(actual.sort().values[1]-minimum),
        original=arms['original'], repaired=arms['repaired'], regret_change=delta,
        true_switch_components=components.tolist(),
        original_predicted_switch_gap=float(sum(arms['original']['predicted_components'][b])-
            sum(arms['original']['predicted_components'][a])+rates[b]-rates[a]),
        repaired_predicted_switch_gap=float(sum(arms['repaired']['predicted_components'][b])-
            sum(arms['repaired']['predicted_components'][a])+rates[b]-rates[a]))


def summarize_records(rows, top_k):
    if not rows or top_k < 1:
        raise ValueError('nonempty records and positive top_k required')
    n = len(rows)
    harm = sorted((r for r in rows if r['regret_change'] > TOL),
                  key=lambda r: r['regret_change'], reverse=True)
    gain = sorted((r for r in rows if r['regret_change'] < -TOL),
                  key=lambda r: r['regret_change'])
    harm_total = sum(r['regret_change'] for r in harm)
    gain_total = -sum(r['regret_change'] for r in gain)
    switch_components = torch.tensor([r['true_switch_components'] for r in rows], dtype=torch.float64)
    transitions = Counter(f"q{r['original']['choice']}->q{r['repaired']['choice']}" for r in rows)
    result = dict(records=n, weighting='record uniform; camera average before tier selection',
        component_order=['MSE/d_ref', 'lambda_s*structure/d_ref', 'beta*C/C_ref'],
        categories=dict(Counter(r['category'] for r in rows)), transitions=dict(transitions),
        paired=dict(changed_records=sum(r['changed'] for r in rows),
            harmful_switches=len(harm), beneficial_switches=len(gain),
            harm_total=harm_total, gain_total=gain_total,
            mean_regret_change=(harm_total-gain_total)/n,
            mean_switch_components=switch_components.mean(0).tolist(),
            harmful_switch_component_totals=[sum(r['true_switch_components'][k] for r in harm) for k in range(3)],
            top10_harm_fraction=sum(r['regret_change'] for r in harm[:10])/harm_total if harm_total else 0.),
        worst_added_harm=harm[:top_k], largest_improvements=gain[:top_k])
    for name in ('original', 'repaired'):
        result[name] = dict(three_tier_regret=sum(r[name]['regret'] for r in rows)/n,
            correct_fraction=sum(r[name]['correct'] for r in rows)/n,
            predicted_counts=[sum(r[name]['choice'] == k for r in rows) for k in (1, 2, 3)],
            auxiliary_four_regret=sum(r[name]['auxiliary_four_regret'] for r in rows)/n,
            auxiliary_drop_fraction=sum(r[name]['auxiliary_drop'] for r in rows)/n)
    return result
