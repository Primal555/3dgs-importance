"""Global and training-threshold-fixed decision and numerical metrics."""
import torch

from .benefit_stage_one_report import summarize_predictions


def stratified_report(groups, predictions, threshold):
    full = [(g, p) for g, p in zip(groups, predictions) if g.get('supervision') != 'action_gap']
    masks = {'all': [], 'ordinary': [], 'large_gap': []}
    for i, (g, p) in enumerate(full):
        truth = torch.stack([v['values'].double().cpu() for v in g['views']]).mean(0)
        cost = truth.sum(-1)+torch.as_tensor(g['rates']).double().cpu()
        masks['all'].append(i)
        masks['large_gap' if float(cost.max()-cost.min()) > threshold else 'ordinary'].append(i)
    result = dict(threshold=threshold, split_scope='fixed training Q90; no validation fitting')
    for scope, indices in masks.items():
        if not indices:
            result[scope] = dict(records=0, three_tier=None, selection_tail=None, value_error=None)
            continue
        selected = [full[i] for i in indices]
        report = summarize_predictions([g for g, _ in selected], [p for _, p in selected])
        errors = torch.cat([torch.as_tensor(p).double().cpu()-
            torch.stack([v['values'].double().cpu() for v in g['views']]) for g, p in selected])
        report['value_error'].update(per_view_component_rmse=errors.square().mean((0, 1)).sqrt().tolist(),
                                     per_view_component_mae=errors.abs().mean((0, 1)).tolist())
        regret = []
        for g, p in selected:
            cost = torch.stack([v['values'].double().cpu() for v in g['views']]).mean(0).sum(-1)
            rates = torch.as_tensor(g['rates']).double().cpu()
            actual = cost+rates
            choice = (torch.as_tensor(p).double().cpu().mean(0).sum(-1)+rates).argmin()
            regret.append(actual[choice]-actual.min())
        regret = torch.stack(regret)
        total = float(regret.sum())
        report['selection_tail'] = dict(total_regret=total, max_regret=float(regret.max()),
            top10_regret_fraction=float(regret.topk(min(10, len(regret))).values.sum())/total if total else 0.)
        result[scope] = report
    return result
