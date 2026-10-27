import copy
import pytest
import torch

from gaussian_jscc.calibrated_evidence import CalibratedEvidence


def make():
    s = CalibratedEvidence([torch.arange(4).reshape(1, 4)], half_life=1000.)
    s.configure_views(16, 4)
    return s


def begin(s, step, current=1, views=(0, 1, 2, 3)):
    s.set_view_window(views)
    s.select_rounds(torch.full((4,), current), torch.ones(4), torch.full((4, 4), .25), step)


def positive(s, step, checked=False, current=1, views=(0, 1, 2, 3)):
    begin(s, step, current, views)
    approx = torch.zeros(1, 3, 2, dtype=torch.float64)
    parts = approx.clone()
    exact = torch.zeros(1, 3, dtype=torch.bool)
    if checked:
        parts[0, 1, 0] = 3.
        exact[0, 1] = True
    return s.accumulate_components(torch.tensor([0]), parts, torch.zeros(1, 3), step, 0.,
                                  approximate=approx, exact=exact)


def test_exact_residual_survives_unchecked_proxy_history():
    s = make()
    first = positive(s, 1, checked=True)
    assert float(first[0, 1]-first[0, 0]) == pytest.approx(3.)
    for step in range(2, 22):
        got = positive(s, step)
    assert float(got[0, 1]-got[0, 0]) > 2.9
    assert s.positive_exact_count[0].tolist() == [0, 1, 0]
    assert s.observations[0] == 21


def test_correction_isolated_by_current_action_and_missing_action_stays_uncorrected():
    s = make(); positive(s, 1, checked=True)
    got = positive(s, 2, current=3)
    torch.testing.assert_close(got, torch.zeros_like(got))
    got = positive(s, 3)
    assert float(got[0, 2]-got[0, 0]) == 0.
    assert float(got[0, 1]-got[0, 0]) > 2.9


def test_current_exact_value_overrides_smoothed_prediction():
    s = make(); positive(s, 1, checked=True)
    begin(s, 2)
    parts = torch.zeros(1, 3, 2, dtype=torch.float64); parts[0, 1, 0] = -2.
    got = s.accumulate_components(torch.tensor([0]), parts, torch.zeros(1, 3), 2, 0.,
        approximate=torch.zeros_like(parts), exact=torch.tensor([[False, True, False]]))
    assert float(got[0, 1]-got[0, 0]) == pytest.approx(-2.)


def test_repeated_view_window_does_not_inflate_effective_samples():
    s = make()
    for step in range(1, 9): positive(s, step)
    assert float(s.positive_effective_samples[0]) == pytest.approx(1.)
    for step, views in enumerate(((4,5,6,7), (8,9,10,11), (12,13,14,15)), 9):
        positive(s, step, views=views)
    assert float(s.positive_effective_samples[0]) == pytest.approx(4.)


def test_mean_uncertainty_rewards_distinct_consistent_evidence():
    s = make(); ids = torch.tensor([0]); rates = torch.tensor([[0., 1., 2.]])
    trusts = []
    for step in range(1, 9):
        begin(s, step, views=tuple(range(4*((step-1)%4), 4*((step-1)%4)+4)))
        value = 1. if step % 2 else 3.
        parts = torch.tensor([[[0.,0.], [value,0.], [2*value,0.]]])
        s.accumulate_components(ids, parts, rates, step, 0.)
        trusts.append(float(s.tier_trust[0]))
    assert trusts[-1] > trusts[1]


def test_keep_correction_persists_and_drop_restore_histories_are_separate():
    s = make(); tiers = torch.ones(4, dtype=torch.long); p = torch.full((4,3), 1/3)
    for step in range(1, 21):
        begin(s, step)
        approx = torch.zeros(4,2, dtype=torch.float64); parts = approx.clone()
        exact = torch.zeros(4, dtype=torch.bool)
        if step == 1: parts[0,0] = -4.; exact[0] = True
        got = s.accumulate_keep(tiers, parts, p, step, 0., approximate=approx, exact=exact)
    assert float(got[0]) < -3.9
    begin(s, 21, current=0)
    got = s.accumulate_keep(tiers, torch.zeros(4,2), p, 21, 0.)
    assert float(got[0]) == 0.


def test_calibrated_history_roundtrip_and_missing_state_rejection():
    s = make(); positive(s, 1, checked=True)
    state = s.state_dict(); other = make(); other.load_state_dict(state)
    a = positive(s, 2, views=(4,5,6,7)); b = positive(other, 2, views=(4,5,6,7))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(s.tier_trust, other.tier_trust, rtol=0, atol=0)
    broken = copy.deepcopy(state); del broken['calibrated']
    with pytest.raises(ValueError, match='calibrated'): make().load_state_dict(broken)


def test_correction_fades_with_age_and_changed_block_context():
    s = make(); positive(s,1,checked=True)
    keys = torch.tensor([1])  # point 0, q1 -> q2
    _, fresh, _ = s.positive_corrections.query(keys,s.action_context,1)
    _, aged, _ = s.positive_corrections.query(keys,s.action_context,1001)
    assert float(fresh[0,0]) == pytest.approx(3.)
    assert float(aged[0,0]) == pytest.approx(1.5)
    changed = s.action_context.clone(); changed[0] = torch.tensor([0.,0.,0.,1.])
    _, drifted, _ = s.positive_corrections.query(keys,changed,1)
    assert float(drifted[0,0]) < .001


def test_repeated_exact_view_window_keeps_correction_uncertainty():
    s = make()
    for step in range(1,11): positive(s,step,checked=True)
    _,_,error = s.positive_corrections.query(torch.tensor([1]),s.action_context,10)
    assert float(error[0,0]) > 2.


def test_no_memory_keep_uncertainty_uses_current_slot():
    s = CalibratedEvidence([torch.arange(4).reshape(1,4)],half_life=1000.,memory=False)
    s.configure_views(16,4)
    conditional = torch.tensor([[.01,.98,.01]]).expand(4,-1)
    for step,(tier,value) in enumerate(((1,0.),(2,-100.),(3,0.),(2,100.),(1,1.)),1):
        begin(s,step,current=0)
        parts = torch.tensor([[value,0.]]).expand(4,-1)
        result = s.accumulate_keep(torch.full((4,),tier),parts,conditional,step,0.)
    assert float(result[0]) == 1.
    assert float(s.keep_standard_error[0]) < 1.
