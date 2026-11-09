import pytest
import torch

from gaussian_jscc import benefit_view_gap as model


@pytest.mark.parametrize('tau', [1e-8, 6.7345e-7, 1e-4])
def test_rank_gradient_in_normalized_rd_units_is_bounded(tau):
    scale = 9.5e-5
    z = torch.tensor([[0., 1., -1.], [.1, -.1, 0.]], dtype=torch.double, requires_grad=True)
    p = torch.stack((z*scale, z*0), -1)
    truth = -p.detach()
    rates = torch.tensor([1e-8, 2e-8, 4e-8])
    loss = model.view_gap_losses(p, truth, rates, scale, tau, 1., 1.)
    grad = torch.autograd.grad(loss['ranking'].sum(), z)[0]
    assert grad.abs().max() <= 2/3+1e-12
    zero = model.view_gap_losses(truth, truth, rates, scale, tau, 1., 1.)
    assert all(v.abs().max() < 1e-12 for v in zero.values())


def test_rd_direction_strengthened_with_component_contrast_preserved():
    from gaussian_jscc.benefit_three_task import three_losses
    t = torch.zeros(1, 3, 2, dtype=torch.double)
    axis = torch.tensor([-1., 0., 1.], dtype=torch.double)
    for sign, factor in [(1, 2), (-1, 1)]:
        p = torch.stack((axis, axis*sign), -1)[None]
        new = model.view_gap_losses(p, t, torch.zeros(3), 1., .1, 1., 1.)
        old = three_losses(p, t, torch.zeros(3), 1., .1, 1., 1.)
        torch.testing.assert_close(new['relative'], old['relative']*factor)
        torch.testing.assert_close(new['common'], old['common'])


def test_saturated_order_still_has_gap_learning_and_camera_errors_do_not_cancel():
    t = torch.tensor([[[0., 0.], [1e-4, 0.], [3e-4, 0.]]], dtype=torch.double)
    p = (t*.1).requires_grad_()
    losses = model.view_gap_losses(p, t, torch.zeros(3), 1e-4, 1e-8, 1., 1.)
    assert losses['ranking'].abs().max() < 1e-12
    grad = torch.autograd.grad(losses['relative'].sum(), p)[0]
    assert grad[0, 2, 0] < 0
    perturb = t*.3
    views = torch.cat((t+perturb, t-perturb))
    true_views = t.expand_as(views)
    losses = model.view_gap_losses(views, true_views, torch.zeros(3), 1e-4, 1e-8, 1., 1.)
    assert losses['relative'].mean() > 0
    assert model.view_gap_losses(views.mean(0), t[0], torch.zeros(3), 1e-4, 1e-8, 1., 1.)['relative'] < 1e-12


def test_rank_budget_preserves_numeric_gradient_and_limits_actual_all_parameter_norm():
    from gaussian_jscc.benefit_three_task import TaskAccumulator
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.double))
    a = TaskAccumulator([p], [0])
    a.gradients = [[p.new_tensor([3., 0.])], [p.new_tensor([0., 4.])], [p.new_tensor([60., 80.])]]
    weights, report = model.assign_with_rank_budget(a, torch.ones(3, dtype=torch.double), 1.)
    torch.testing.assert_close(weights, torch.tensor([1., 1., .05], dtype=torch.double))
    torch.testing.assert_close(p.grad, torch.tensor([6., 8.], dtype=torch.double))
    assert report['numeric_gradient_norm'] == pytest.approx(5)
    assert report['applied_rank_gradient_norm'] == pytest.approx(5)
    a.gradients[0][0].zero_(); a.gradients[1][0].zero_()
    model.assign_with_rank_budget(a, torch.ones(3, dtype=torch.double), 1.)
    assert p.grad.abs().sum() == 0


def test_report_distinguishes_best_tier_from_complete_order_and_gap_fit():
    values = torch.tensor([[[0., 0.], [2., 0.], [3., 0.]], [[0., 0.], [2., 0.], [3., 0.]]])
    groups = [dict(views=[dict(values=v) for v in values], rates=[0., 0., 0.])]
    prediction = [torch.tensor([[[0., 0.], [.3, 0.], [.2, 0.]], [[0., 0.], [.3, 0.], [.2, 0.]]])]
    report = model.gap_report(groups, prediction, 2.)
    assert report['large_gap']['best_tier_correct_fraction'] == 1
    assert report['large_gap']['complete_order_correct_fraction'] == 0
    assert report['large_gap']['correct_choice_pair_rmse'] > 1
    assert report['ordinary'] is None


def test_rank_budget_includes_nonshared_head_parameters():
    from gaussian_jscc.benefit_three_task import TaskAccumulator
    shared = torch.nn.Parameter(torch.zeros(1, dtype=torch.double))
    head = torch.nn.Parameter(torch.zeros(1, dtype=torch.double))
    a = TaskAccumulator([shared, head], [0])
    a.gradients = [[shared.new_tensor([3.]), head.new_tensor([0.])],
                   [shared.new_tensor([0.]), head.new_tensor([4.])],
                   [shared.new_tensor([0.]), head.new_tensor([100.])]]
    _, report = model.assign_with_rank_budget(a, torch.ones(3, dtype=torch.double))
    assert report['rank_gradient_factor'] == pytest.approx(.05)
    torch.testing.assert_close(shared.grad, shared.new_tensor([3.]))
    torch.testing.assert_close(head.grad, head.new_tensor([9.]))
