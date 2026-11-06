import copy

import pytest
import torch

from gaussian_jscc.benefit_rd_choice_losses import choice_losses, fit_choice_targets, RDChoiceLearner


def test_exact_values_are_stationary_and_rates_only_enter_once():
    y = torch.tensor([[[-.1, .02], [-.2, .04], [-.3, .06]],
                      [[-.2, .04], [-.1, .02], [-.4, .08]]], dtype=torch.float64)
    rates = torch.tensor([0., .05, .1], dtype=torch.float64)
    p = y.clone().requires_grad_()
    losses = choice_losses(p, y, rates, .3, .02)
    assert losses['total'].item() == pytest.approx(0., abs=1e-12)
    losses['total'].backward()
    assert p.grad.abs().max() < 1e-10
    shifted = choice_losses(y+1., y, rates, .3, .02)
    assert shifted['rd_regression'] == pytest.approx(0., abs=1e-12)
    assert shifted['joint_choice'] == pytest.approx(0., abs=1e-12)
    assert shifted['component_regression'] > 0


def test_ordinary_errors_and_dangerous_wrong_choices_both_have_gradients():
    y = torch.tensor([[[0., 0.], [-.02, 0.], [-.03, 0.]]], dtype=torch.float64)
    for magnitude in (1e-4, .5):
        p = (y+torch.tensor([[[magnitude, 0.], [0., 0.], [0., 0.]]])).requires_grad_()
        loss = choice_losses(p, y, torch.zeros(3), 1., .01)
        loss['total'].backward()
        assert p.grad.abs().sum() > 0
    p = y.flip(1).clone().requires_grad_()
    losses = choice_losses(p, y, torch.zeros(3), 1., .01)
    assert losses['joint_choice'] > 0
    losses['joint_choice'].backward()
    assert p.grad[0, 2].sum() > 0  # Gradient descent lowers the true best q3 cost.


def test_single_shared_scale_preserves_component_units_and_view_auxiliary():
    target = torch.zeros(2, 3, 2, dtype=torch.float64)
    a = target.clone(); a[0, 0, 0] = .1; a[1, 0, 0] = -.1
    b = a.flip(-1)
    la = choice_losses(a, target, torch.zeros(3), 2., .1)
    lb = choice_losses(b, target, torch.zeros(3), 2., .1)
    assert la['rd_regression'] == pytest.approx(0., abs=1e-12)
    assert la['component_regression'] > 0
    assert la['total'] == pytest.approx(lb['total'])


def test_cold_temperature_does_not_overwhelm_numeric_supervision():
    target = torch.tensor([[[0., 0.], [-.1, 0.], [-.2, 0.]]], dtype=torch.float64)
    gradients = []
    for temperature in (.1, 1e-6):
        p = target.flip(1).clone().requires_grad_()
        loss = choice_losses(p, target, torch.zeros(3), 1., temperature)
        loss['joint_choice'].backward()
        gradients.append(float(p.grad.abs().max()))
    assert gradients[1] < gradients[0]


def test_train_only_temperature_quantile_and_gap_records():
    full = dict(source=1, step=1, views=[dict(values=torch.tensor([[0., 0.], [1., 0.], [2., 0.]]))],
                rates=torch.tensor([0., .1, .2]))
    gap = dict(source=2, step=2, supervision='action_gap', views=[],
        delta_values=torch.tensor([.2, .3]), rate_delta=.1)
    fitted = fit_choice_targets([full, gap])
    assert fitted['temperature'] == pytest.approx(1.1)
    assert fitted['large_gap_threshold'] == pytest.approx(2.2)
    with pytest.raises(ValueError, match='validation'):
        fit_choice_targets([dict(full, validation=True)])


def test_mixed_records_preserve_forward_order_and_partial_supervision(tmp_path):
    from test_benefit_stage_one import source
    from gaussian_jscc.benefit_stage_one_data import snapshot, read_groups
    from gaussian_jscc.benefit_stage_one import make_learner
    root, _ = source(tmp_path)
    saved = snapshot(root)
    records = list(read_groups(saved['replay'], saved['replay'].training))
    s = saved['settings']
    learner = RDChoiceLearner(None, 'cpu', hidden=s['hidden'], dropout=s['dropout'])
    learner.load_state_dict(copy.deepcopy(saved['learner']))
    learner.configure_objective(fit_choice_targets(records))
    original = make_learner(saved, 'cpu')
    mixed = [records[-1], records[0]]
    outputs = [[], []]
    hooks = [x.network.register_forward_hook(lambda m, i, o, k=k: outputs[k].append(o.detach().clone()))
             for k, x in enumerate((original, learner))]
    for x in (original, learner):
        torch.manual_seed(333)
        x.network.train()
        loss = x._batch_losses(mixed)
        loss['total'].mean().backward()
    for hook in hooks:
        hook.remove()
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(torch.isfinite(v).all() for v in loss.values())
    assert learner.state_dict()['choice_objective']['fit_scope'] == 'fixed training records only'
