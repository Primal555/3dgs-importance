import pytest
import torch
from gaussian_jscc.structured_evidence import StructuredEvidence, BoundedMirrorTable, risk_choices


def scheduler(**kw):
    return StructuredEvidence([torch.arange(4).reshape(1, 4)], half_life=10., **kw)


def test_drift_keeps_history_and_decay_counts_only_new_observations():
    s = scheduler(); ids = torch.tensor([0])
    x = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
    a = s.accumulate_components(ids, x, torch.zeros(1, 3), 1, .1)
    p = torch.full((4, 4), .25); q = torch.ones(4, dtype=torch.long)
    s.select_rounds(q, torch.ones(4), p, 1)
    p[:, 0] = .99; p[:, 1:] = .01/3
    s.select_rounds(q, torch.ones(4), p, 11)
    assert s.weight[0] == 1 and s.observations[0] == 1
    assert s.refresh[0] and s.invalidations == 0
    b = s.accumulate_components(ids, x*2, torch.zeros(1, 3), 11, .1)
    assert s.weight[0] == 1.5 and s.observations[0] == 2
    torch.testing.assert_close(b, a*5/3)
    with pytest.raises(ValueError): s.accumulate_components(ids, x, torch.zeros(1, 3), 11, .1)


def test_keep_tier_readiness_and_staleness_do_not_fill_missing_with_zero():
    s = scheduler(); conditional = torch.full((4, 3), 1/3)
    parts = torch.tensor([[2., 0.], [4., 0.], [8., 0.], [16., 0.]], dtype=torch.float64)
    got = s.accumulate_keep(torch.ones(4, dtype=torch.long), parts, conditional, 1, 0.)
    torch.testing.assert_close(got, parts[:, 0])
    assert not s.keep_ready(1).any()
    s.accumulate_keep(torch.full((4,), 2), parts*2, conditional, 2, 0.)
    got = s.accumulate_keep(torch.full((4,), 3), parts*3, conditional, 3, 0.)
    assert s.keep_ready(3).all()
    torch.testing.assert_close(got, parts[:, 0]*2)
    assert not s.keep_ready(14).any()
    got = s.accumulate_keep(torch.ones(4, dtype=torch.long), parts, conditional, 14, 0.)
    torch.testing.assert_close(got, parts[:, 0])
    assert s.keep_observations.sum() == 16


def test_bounded_mirror_limits_odds_without_freezing_or_forcing_tiers():
    table = BoundedMirrorTable(4, max_log_odds=.25)
    before = table.logits.clone(); keep = table.keep_logits.clone()
    table.update(torch.tensor([[0., 100.]]).expand(4, -1), torch.arange(4),
        torch.tensor([[0., 10., -10.]]).expand(4, -1), .1, .4, 1000)
    delta = table.logits-before; kd = table.keep_logits-keep
    assert torch.all(delta.amax(-1)-delta.amin(-1) <= .250000001)
    assert torch.all(kd.amax(-1)-kd.amin(-1) <= .250000001)
    assert torch.all(delta[:, 2] > delta[:, 0])
    assert torch.all(kd[:, 1] < kd[:, 0])


def test_history_roundtrip_and_risk_selection_are_deterministic():
    s = scheduler(); q = torch.ones(4, dtype=torch.long)
    ids = torch.arange(4); image = torch.tensor([[0., .1, -.2]]).expand(4, -1)
    s.accumulate_components(ids, torch.stack((image, image), -1), image*0, 1, .1)
    scores = torch.tensor([[0., .01, .9], [0., .3, .2], [0., .8, .7], [0., .6, .5]])
    s.finite_counts[:3] = 1
    selected = risk_choices(ids, scores, q, s, 3, 1)
    assert len({i for i, action, group in selected}) == 3
    assert {group for i, action, group in selected} == {'boundary', 'uncertainty', 'coverage'}
    other = scheduler(); other.load_state_dict(s.state_dict())
    assert selected == risk_choices(ids, scores, q, other, 3, 1)
    for name in ('mean', 'weight', 'keep_mean', 'keep_weight'):
        assert torch.equal(getattr(s, name), getattr(other, name))


def test_exact_replaces_one_action_without_double_counting_or_marking_others():
    s = scheduler(); ids = torch.tensor([0]); approximate = torch.zeros(1, 3, 2)
    resolved = approximate.clone(); resolved[0, 1] = torch.tensor([3., 6.])
    exact = torch.tensor([[False, True, False]])
    s.accumulate_components(ids, resolved, torch.zeros(1, 3), 1, .1,
        approximate=approximate, exact=exact)
    assert s.weight[0] == 1 and s.observations[0] == 1
    assert s.positive_exact_count[0].tolist() == [0, 1, 0]
    assert s.positive_approx_count[0].tolist() == [1, 1, 1]
    assert s.mean[0, 1].tolist() == [2., 4.]
