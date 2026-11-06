"""Explicit numeric and decision metrics for cached evaluator labels."""
import torch


def action_report(actual, estimate):
    actual, estimate = actual.double().flatten(), estimate.double().flatten()
    informative = actual.abs() > 1e-14
    wrong = informative & (actual*estimate <= 0)
    regret = torch.where(wrong, actual.abs(), 0.)
    error = estimate-actual
    total = float(regret.sum())
    return dict(comparisons=len(actual), informative_comparisons=int(informative.sum()),
        sign_errors=int(wrong.sum()),
        sign_error_fraction=float(wrong.sum()/informative.sum()) if informative.any() else 0.,
        rd_mae=float(error.abs().mean()), rd_rmse=float(error.square().mean().sqrt()),
        direction_regret_mean=float(regret.mean()),
        false_beneficial=int(((actual > 1e-14) & (estimate <= 0)).sum()),
        missed_beneficial=int(((actual < -1e-14) & (estimate >= 0)).sum()),
        top10_regret_fraction=float(regret.topk(min(10, len(regret))).values.sum())/total if total else 0.)


def summarize_predictions(groups, predictions):
    if len(groups) != len(predictions) or not groups:
        raise ValueError('matching nonempty groups and predictions required')
    full, gaps = [], []
    for g, value in zip(groups, predictions):
        p = torch.as_tensor(value).detach().cpu().double()
        if not torch.isfinite(p).all():
            raise ValueError('nonfinite evaluator prediction')
        if g.get('supervision') == 'action_gap':
            mean = p.mean(0)
            estimate = (mean[g['new_tier']-1]-mean[g['old_tier']-1]).sum()+g['rate_delta']
            actual = torch.as_tensor(g['delta_values']).double().sum()+g['rate_delta']
            gaps.append((actual, estimate))
        else:
            target = torch.stack([v['values'].double().cpu() for v in g['views']])
            if target.shape != p.shape:
                raise ValueError('prediction and teacher view shapes differ')
            full.append((p, target, torch.as_tensor(g['rates']).double().cpu()))
    result = dict(records=len(groups), full_state_groups=len(full), action_gap_groups=len(gaps),
                  weighting='record uniform', positive_window=None, positive_per_view=None,
                  three_tier=None, q0_auxiliary=None, value_error=None, observed_actions=None)
    if gaps:
        result['observed_actions'] = action_report(torch.stack([a for a, _ in gaps]),
                                                  torch.stack([p for _, p in gaps]))
    if not full:
        return result
    p = torch.stack([a.mean(0) for a, _, _ in full])
    t = torch.stack([b.mean(0) for _, b, _ in full])
    r = torch.stack([c for _, _, c in full])
    actual, estimate = t.sum(-1)+r, p.sum(-1)+r
    i, j = torch.triu_indices(3, 3, offset=1)
    result['positive_window'] = action_report(actual[:, j]-actual[:, i], estimate[:, j]-estimate[:, i])
    view_p = torch.cat([a.sum(-1)+c for a, _, c in full])
    view_t = torch.cat([b.sum(-1)+c for _, b, c in full])
    result['positive_per_view'] = action_report(view_t[:, j]-view_t[:, i], view_p[:, j]-view_p[:, i])
    ids = torch.arange(len(full))
    minimum = actual.min(-1).values
    choice = estimate.argmin(-1)
    regret = actual[ids, choice]-minimum
    result['three_tier'] = dict(regret_mean=float(regret.mean()),
        best_tier_hit_fraction=float((regret.abs() <= 1e-14).double().mean()),
        fixed_regret={f'q{k+1}': float((actual[:, k]-minimum).mean()) for k in range(3)},
        predicted_counts=torch.bincount(choice, minlength=3).tolist(),
        teacher_best_counts=torch.bincount(actual.argmin(-1), minlength=3).tolist())
    error = p-t
    common = error.sum(-1).mean(-1)
    four_actual = torch.cat((actual.new_zeros(len(full), 1), actual), -1)
    four_estimate = torch.cat((estimate.new_zeros(len(full), 1), estimate), -1)
    four_choice = four_estimate.argmin(-1)
    result['q0_auxiliary'] = dict(common_rd_bias=float(common.mean()),
        common_rd_rmse=float(common.square().mean().sqrt()),
        best_positive_vs_drop=action_report(actual.min(-1).values, estimate.min(-1).values),
        four_state_regret=float((four_actual[ids, four_choice]-four_actual.min(-1).values).mean()),
        predicted_drop_fraction=float((four_choice == 0).double().mean()),
        teacher_optimal_drop_fraction=float((four_actual.argmin(-1) == 0).double().mean()),
        scope='auxiliary prediction only; no mask or deployment update')
    per_view_error = torch.cat([a-b for a, b, _ in full])
    result['value_error'] = dict(
        per_view_rd_rmse=float(per_view_error.sum(-1).square().mean().sqrt()),
        window_rd_rmse=float(error.sum(-1).square().mean().sqrt()),
        component_rmse=error.square().mean((0, 1)).sqrt().tolist(),
        components=['MSE/d_ref', 'lambda_s*structure/d_ref'])
    return result


def print_report(scope, report):
    title = {'fixed_validation': '固定留出集（独立于训练，早期上下文）',
             'recent_training': '近期训练回放（已参与训练）'}.get(scope, scope)
    print(f'  【{title}】完整教师={report["full_state_groups"]}，动作标签={report["action_gap_groups"]}', flush=True)
    p = report['positive_window']
    if p is not None:
        print(f'    正档位：方向误判={p["sign_errors"]}/{p["informative_comparisons"]}'
              f'（{p["sign_error_fraction"]:.2%}），方向误判代价={p["direction_regret_mean"]:.6g}，'
              f'RD差值 MAE={p["rd_mae"]:.6g}，RMSE={p["rd_rmse"]:.6g}', flush=True)
        s = report['three_tier']
        fixed = '，'.join(f'{key}={value:.6g}' for key, value in s['fixed_regret'].items())
        print(f'    三档选优代价={s["regret_mean"]:.6g}，最优档位命中={s["best_tier_hit_fraction"]:.2%}；'
              f'固定档位代价：{fixed}', flush=True)
        q, v = report['q0_auxiliary'], report['value_error']
        keep = q['best_positive_vs_drop']
        print(f'    q0辅助：共同偏差={q["common_rd_bias"]:+.6g}，共同RMSE={q["common_rd_rmse"]:.6g}，'
              f'最佳正档对删除的方向误判={keep["sign_errors"]}/{keep["informative_comparisons"]}'
              f'（{keep["sign_error_fraction"]:.2%}），'
              f'四档选优代价={q["four_state_regret"]:.6g}', flush=True)
        print(f'    价值RD误差：逐视角RMSE={v["per_view_rd_rmse"]:.6g}，'
              f'窗口平均RMSE={v["window_rd_rmse"]:.6g}', flush=True)
    a = report['observed_actions']
    if a is not None:
        print(f'    已记录真实换挡：方向误判={a["sign_errors"]}/{a["informative_comparisons"]}'
              f'（{a["sign_error_fraction"]:.2%}），方向误判代价={a["direction_regret_mean"]:.6g}，'
              f'RD差值RMSE={a["rd_rmse"]:.6g}', flush=True)
