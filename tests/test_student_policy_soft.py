import json
import numpy as np
import pytest
import torch
from gaussian_jscc.allocation import deployment_probabilities, deployment_tiers
from gaussian_jscc.student_policy_model import teacher_loss
from gaussian_jscc.student_policy_soft import load_soft_teacher, soft_teacher_loss


def fixture_teacher(tmp_path):
    root = tmp_path/'teacher'; (root/'deployment').mkdir(parents=True)
    p = torch.tensor([[1., 0., 0., 0.], [.3, .1, .2, .4], [0., .4, .4, .2], [0., 0., 0., 1.]], dtype=torch.float64)
    np.save(root/'deployment/selected_probabilities.npy', p.numpy())
    (root/'training.json').write_text(json.dumps(dict(deployment_draws=10, deployment_seed=42)))
    (root/'plan.json').write_text(json.dumps(dict(settings=dict(tier_max_log_odds=1.))))
    for name in ('complete.json', 'deployment/selection.json'):
        (root/name).write_text(json.dumps(dict(selected_step=1600)))
    return root, p, deployment_tiers(p, 42)


def test_soft_teacher_exact_distribution_and_cached_identity(tmp_path):
    root, p, q = fixture_teacher(tmp_path)
    target, identity = load_soft_teacher(root, q, tmp_path/'out', seed=42, required_cap=1., chunk_size=2)
    torch.testing.assert_close(target.double(), deployment_probabilities(p), atol=1e-7, rtol=1e-6)
    assert target[1, 0].item() == pytest.approx(.3**10)
    assert identity['selected_step'] == 1600 and identity['teacher_tier_cap'] == 1.
    rng = torch.get_rng_state().clone()
    again, record = load_soft_teacher(root, q, tmp_path/'out', seed=42, required_cap=1., chunk_size=3)
    assert torch.equal(target, again) and record == identity
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match='map'):
        load_soft_teacher(root, q.roll(1), tmp_path/'other', seed=42, required_cap=1.)
    with pytest.raises(ValueError, match='cap'):
        load_soft_teacher(root, q, tmp_path/'other', seed=42, required_cap=.25)
    np.save(root/'deployment/selected_probabilities.npy', p.flip(0).numpy())
    with pytest.raises(ValueError):
        load_soft_teacher(root, q.flip(0), tmp_path/'out', seed=42, required_cap=1.)


def test_soft_one_hot_loss_and_gradients_equal_hard_hierarchical_loss():
    q = torch.tensor([0, 1, 2, 3, 0, 1])
    keep = torch.randn(6, 2, dtype=torch.double, requires_grad=True)
    tier = torch.randn(6, 3, dtype=torch.double, requires_grad=True)
    hard, _ = teacher_loss(keep, tier, q)
    soft, stats = soft_teacher_loss(keep, tier, torch.nn.functional.one_hot(q, 4).double())
    torch.testing.assert_close(hard, soft)
    for a, b in zip(torch.autograd.grad(hard, (keep, tier)), torch.autograd.grad(soft, (keep, tier))):
        torch.testing.assert_close(a, b)
    assert stats['teacher_keep_kl'] >= 0 and stats['teacher_tier_kl'] >= 0


def test_soft_teacher_stationary_point_and_all_drop_boundary():
    target = torch.tensor([[.2, .1, .3, .4], [.8, .1, .05, .05]], dtype=torch.double)
    keep = torch.stack((target[:, 0], target[:, 1:].sum(1)), 1).log().requires_grad_()
    tier = (target[:, 1:]/target[:, 1:].sum(1, keepdim=True)).log().requires_grad_()
    loss, stats = soft_teacher_loss(keep, tier, target)
    gradients = torch.autograd.grad(loss, (keep, tier))
    assert all(g.abs().max() < 1e-12 for g in gradients)
    assert abs(stats['teacher_keep_kl']) < 1e-12 and abs(stats['teacher_tier_kl']) < 1e-12
    target[:] = torch.tensor([1., 0., 0., 0.])
    loss, stats = soft_teacher_loss(keep, tier, target)
    loss.backward()
    assert torch.isfinite(keep.grad).all() and torch.equal(tier.grad, torch.zeros_like(tier))
    assert stats['tier_ce'] == 0.


def test_soft_tier_uses_expected_retention_normalized_weights():
    target = torch.tensor([[.9, .1, 0., 0.], [.1, 0., .9, 0.]], dtype=torch.double)
    keep = torch.zeros(2, 2, dtype=torch.double, requires_grad=True)
    tier = torch.tensor([[2., 0., 0.], [2., 0., 0.]], dtype=torch.double, requires_grad=True)
    _, stats = soft_teacher_loss(keep, tier, target)
    logp = tier.log_softmax(-1)
    assert stats['tier_ce'] == pytest.approx(float((-.1*logp[0, 0]-.9*logp[1, 1]).detach()))
