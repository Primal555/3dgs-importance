import pytest
import torch
from gaussian_jscc import fast_first_order_allocation as entry
from gaussian_jscc.structured_evidence import BoundedMirrorTable


def options(*extra):
    return entry.parser().parse_args(['--run', 'source', '--scene', 'truck', '--out', 'new', '--beta', '.5', *extra])


def test_default_preserves_old_plan_and_factory():
    args = options()
    assert 'tier_max_log_odds' not in vars(args)
    assert entry.table_factory_for(args) is None


def test_independent_tier_cap_preserves_keep_updates():
    old = BoundedMirrorTable(4, max_log_odds=.25)
    new = entry.table_factory_for(options('--tier-max-log-odds', '1.0'))(4, max_log_odds=.25)
    new.load_state_dict(old.state_dict())
    keep = torch.tensor([[0., 2.]]).repeat(4, 1)
    costs = torch.tensor([[0., 1., 3.], [2., 1., 0.]])
    ids = torch.tensor([1, 3])
    for table in (old, new):
        table.keep_trust = torch.tensor([.4, .5, .8, 1.])
        table.tier_trust = torch.tensor([.4, .8])
        table.update(keep, ids, costs, .1, .4, 100.)
    torch.testing.assert_close(old.keep_logits, new.keep_logits, rtol=0, atol=0)
    torch.testing.assert_close(new.logits[ids], 4*old.logits[ids], rtol=0, atol=1e-14)
    assert new.update_stats['max_log_odds'] == .25
    assert new.update_stats['tier_max_log_odds'] == 1.


@pytest.mark.parametrize('cap', ['0', '-1', 'nan', 'inf'])
def test_invalid_tier_cap_rejected(cap):
    with pytest.raises(ValueError): entry.table_factory_for(options('--tier-max-log-odds', cap))
