import torch
import pytest
from gaussian_jscc.structured_evidence import BoundedMirrorTable
from gaussian_jscc.hybrid_tier_step_table import TierStepTable


def inputs():
    torch.manual_seed(13)
    keep=torch.randn(12,2,dtype=torch.float64)
    ids=torch.tensor([0,3,6,8])
    costs=torch.randn(4,3,dtype=torch.float64)
    trust=torch.tensor([.1,.3,.7,1.],dtype=torch.float64)
    return keep,ids,costs,trust


def test_original_cap_bitwise_equivalence_and_keep_isolation():
    keep,ids,costs,trust=inputs()
    old=BoundedMirrorTable(12);old.tier_trust=trust
    same=TierStepTable(12,tier_max_log_odds=.25);same.tier_trust=trust
    wide=TierStepTable(12,tier_max_log_odds=1.);wide.tier_trust=trust
    for table in (old,same,wide):table.update(keep,ids,costs,.1,.4,711691)
    assert torch.equal(old.logits,same.logits)
    assert torch.equal(old.keep_logits,same.keep_logits)
    assert torch.equal(old.keep_logits,wide.keep_logits)
    assert not torch.equal(old.logits,wide.logits)
    assert wide.update_stats['tier_raw_log_odds_step_mean']>wide.update_stats['tier_applied_log_odds_step_mean']
    assert wide.update_stats['tier_step_limited_fraction']==1.
    assert len(wide.update_stats['tier_raw_log_odds_step_quantiles'])==3


def test_unbounded_matches_raw_mirror_and_reports_real_step():
    keep,ids,costs,trust=inputs()
    table=TierStepTable(12,tier_max_log_odds=None);table.tier_trust=trust
    before=table.logits[ids].clone();expected=before-.4*12*(costs-costs.amin(-1,keepdim=True))
    expected-=expected.amax(-1,keepdim=True)
    table.update(keep,ids,costs,.1,.4,12)
    assert torch.equal(table.logits[ids],expected)
    stats=table.update_stats
    assert stats['tier_step_limited_fraction']==0.
    assert stats['tier_update_scale_mean']==1.
    assert stats['tier_actual_to_raw_step_ratio']==1.


def test_empty_and_numeric_support_floor_are_logged():
    table=TierStepTable(3,tier_max_log_odds=None)
    table.update(torch.zeros(3,2),torch.tensor([],dtype=torch.long),torch.empty(0,3),.1,.4,3)
    assert table.update_stats['tier_update_rows']==0
    table.update(torch.zeros(3,2),torch.tensor([0]),torch.tensor([[0.,1e6,1e6]]),.1,.4,3)
    assert table.logits.min()==-700
    assert table.update_stats['tier_numeric_floor_fraction']==1.
    assert table.update_stats['tier_applied_log_odds_step_mean']==700.


@pytest.mark.parametrize('cap',[0.,-1.,float('inf'),float('nan')])
def test_invalid_caps_are_rejected(cap):
    with pytest.raises(ValueError):TierStepTable(3,tier_max_log_odds=cap)
