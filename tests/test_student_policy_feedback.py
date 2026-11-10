import pytest
import torch
from gaussian_jscc.student_policy_feedback import CurrentPolicy, UniformFeedback, policy_surrogate
from gaussian_jscc.student_policy_model import hard_tiers


def test_policy_adapter_uses_same_hard_decision_without_rng():
    keep = torch.tensor([[0., 1.], [1., 0.], [0., 0.]])
    tier = torch.tensor([[0., 2., 0.], [0., 0., 2.], [2., 0., 0.]])
    adapter = CurrentPolicy(keep, tier)
    state = torch.get_rng_state().clone()
    assert torch.equal(adapter.sample(torch.arange(3), 10.)[0], hard_tiers(keep, tier))
    assert torch.equal(state, torch.get_rng_state())


def test_uniform_sampler_empirical_inclusion_and_resume():
    ids = [torch.arange(9).reshape(1, 9)]
    sampler = UniformFeedback(ids, rounds=2, seed=17)
    q = torch.tensor([0, 1, 2, 3, 1, 0, 1, 2, 0]); counts = torch.zeros(9)
    for step in range(1, 1001):
        result = sampler.select_rounds(q, None, None, step)
        selected = torch.cat([ids[0][0, r[0]] for r in result])
        counts[selected] += 1
        assert len(selected.unique()) == 2
        assert torch.equal(sampler.inverse_inclusion[selected], torch.full((2,), 3., dtype=torch.float64))
    assert float((counts[q > 0]-1000/3).abs().max()) < 50
    resumed = UniformFeedback(ids, rounds=2, seed=17); resumed.load_state_dict(sampler.state_dict())
    a = sampler.select_rounds(q, None, None, 1001); b = resumed.select_rounds(q, None, None, 1001)
    assert all(torch.equal(x[0], y[0]) for x, y in zip(a, b))


def test_surrogate_gradient_has_dense_keep_and_weighted_positive_without_double_gate():
    keep = torch.tensor([[0., 0.], [0., 0.], [0., 0.]], requires_grad=True)
    tier = torch.zeros(3, 3, requires_grad=True)
    signal = torch.tensor([[0., 2.], [0., -3.], [0., 4.]])
    costs = torch.tensor([[0., 1., 2.], [3., 0., 1.]])
    loss, stats = policy_surrogate(keep, tier, signal, torch.tensor([0, 2]), costs,
                                   torch.tensor([2., 3.]))
    loss.backward()
    torch.testing.assert_close(keep.grad[:, 1], torch.tensor([.5, -.75, 1.]))
    assert tier.grad[1].abs().sum() == 0
    torch.testing.assert_close(tier.grad[0], torch.tensor([-2/3, 0., 2/3]))
    torch.testing.assert_close(tier.grad[2], torch.tensor([5/3, -4/3, -1/3]))
    assert stats['positive_sample_count'] == 2


def test_all_closed_sampling_and_empty_tier_loss_remain_finite():
    sampler = UniformFeedback([torch.arange(6).reshape(2, 3)], rounds=2, seed=7)
    rounds = sampler.select_rounds(torch.zeros(6).long(), None, None, 1)
    assert all((batch[0] == -1).all() for batch in rounds)
    keep = torch.randn(6, 2, requires_grad=True); tier = torch.randn(6, 3, requires_grad=True)
    loss, _ = policy_surrogate(keep, tier, torch.randn(6, 2), torch.empty(0).long(),
                               torch.empty(0, 3), torch.empty(0))
    loss.backward(); assert torch.isfinite(keep.grad).all() and tier.grad.abs().sum() == 0
