import pytest
import torch

from gaussian_jscc.benefit_stage_one_report import summarize_predictions, print_report


def record(values, views=1):
    return dict(source=1, step=1, rates=torch.zeros(3),
                views=[dict(values=torch.tensor(values, dtype=torch.float64)) for _ in range(views)])


def test_report_distinguishes_exact_sorting_from_wrong_common_keep_value(capsys):
    g = record([[-1., 0.], [-2., 0.], [-3., 0.]])
    p = torch.tensor([[[3., 0.], [2., 0.], [1., 0.]]])
    result = summarize_predictions([g], [p])
    assert result['positive_window']['sign_errors'] == 0
    assert result['three_tier']['regret_mean'] == 0
    assert result['q0_auxiliary']['common_rd_rmse'] == 4
    assert result['q0_auxiliary']['best_positive_vs_drop']['sign_errors'] == 1
    assert result['q0_auxiliary']['four_state_regret'] == 3
    print_report('fixed_validation', result)
    text = capsys.readouterr().out
    assert '正档位' in text and '辅助' in text and '固定' in text


def test_per_view_error_is_not_hidden_by_window_averaging():
    g = record([[-1., 0.], [-2., 0.], [-3., 0.]], views=2)
    truth = torch.stack([v['values'] for v in g['views']])
    pred = truth+torch.tensor([[[2., 0.]], [[-2., 0.]]])
    report = summarize_predictions([g], [pred])
    assert report['value_error']['window_rd_rmse'] == 0
    assert report['value_error']['per_view_rd_rmse'] == 2


def test_positive_fixed_baselines_and_direction_cost_are_interpretable():
    g = record([[-1., 0.], [-2., 0.], [-3., 0.]])
    p = torch.tensor([[[-3., 0.], [-2., 0.], [-1., 0.]]])
    report = summarize_predictions([g], [p])
    assert report['positive_window']['sign_errors'] == 3
    assert report['positive_window']['comparisons'] == 3
    assert report['three_tier']['regret_mean'] == 2
    assert report['three_tier']['fixed_regret'] == {'q1': 2., 'q2': 1., 'q3': 0.}
    assert report['three_tier']['best_tier_hit_fraction'] == 0


def test_actual_gap_records_are_separate_from_three_tier_validation():
    g = dict(source=1, step=1, supervision='action_gap', old_tier=1, new_tier=2,
             delta_values=torch.tensor([-1., 0.]), rate_delta=0., views=[{}])
    p = torch.tensor([[[0., 0.], [1., 0.], [2., 0.]]])
    report = summarize_predictions([g], [p])
    assert report['full_state_groups'] == 0
    assert report['observed_actions']['sign_errors'] == 1
    assert report['three_tier'] is None
