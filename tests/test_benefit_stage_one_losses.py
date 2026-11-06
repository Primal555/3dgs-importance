import pytest
import torch

from gaussian_jscc.benefit_stage_one_losses import fit_positive_scales, positive_losses


def group(values, rates=(0., 0., 0.)):
    return dict(source=1, step=1, importance_weight=1., rates=torch.tensor(rates),
                views=[dict(values=torch.tensor(values, dtype=torch.float64))])


def test_pair_scale_ignores_large_common_offset_and_never_fits_validation():
    a = group([[-100., 0.], [-100.5, 0.], [-101., 0.]])
    b = group([[-1., 0.], [-1.5, 0.], [-2., 0.]])
    sa = fit_positive_scales([a])
    sb = fit_positive_scales([b])
    torch.testing.assert_close(sa['components'], sb['components'])
    assert sa['rd'] == pytest.approx((.5)**.5)
    with pytest.raises(ValueError, match='validation'):
        fit_positive_scales([dict(a, validation=True)])


def test_reversed_positive_order_has_direction_gradient_even_with_large_q0_scale():
    y = torch.tensor([[[-100., 0.], [-100.5, 0.], [-101., 0.]]], dtype=torch.float64)
    p = y.flip(1).clone().requires_grad_()
    scales = fit_positive_scales([group(y[0].tolist())])
    loss = positive_losses(p, y, torch.zeros(1, 3), scales, 100.)
    assert loss['positive_direction'].item() > 0
    loss['total'].sum().backward()
    assert p.grad.abs().max() > 0
    exact = positive_losses(y, y, torch.zeros(1, 3), scales, 100.)
    assert exact['total'].abs().max() < 1e-12


def test_common_offset_does_not_change_positive_loss_but_retains_q0_auxiliary():
    y = torch.tensor([[[-2., -.1], [-4., -.2], [-8., -.3]]], dtype=torch.float64)
    scales = fit_positive_scales([group(y[0].tolist())])
    p = y+torch.tensor([2., .4])
    loss = positive_losses(p, y, torch.zeros(1, 3), scales, 2.)
    assert loss['positive_regression'].abs().max() < 1e-12
    assert loss['positive_direction'].abs().max() < 1e-12
    assert loss['q0_auxiliary'].item() > 0


def test_gap_records_fit_their_actual_positive_differences():
    g = dict(source=1, step=1, supervision='action_gap', old_tier=1, new_tier=3,
             delta_values=torch.tensor([2., .5]), rate_delta=.1, importance_weight=1., views=[])
    scales = fit_positive_scales([g])
    torch.testing.assert_close(scales['components'], torch.tensor([2., .5], dtype=torch.float64))
    assert float(scales['rd']) == pytest.approx(2.6)


def test_direction_scale_matches_window_average_and_reports_old_ignore_band():
    g = group([[0., 0.], [1., 0.], [2., 0.]], rates=(0., .1, .2))
    g['views'].append(dict(values=-g['views'][0]['values']))
    s = fit_positive_scales([g], original_rd_scale=100.)
    assert float(s['rd']) == pytest.approx((.02)**.5)
    assert s['original_ignored_comparisons'] == s['positive_comparisons'] == 3
