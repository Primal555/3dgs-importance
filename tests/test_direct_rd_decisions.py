import importlib.util

import pytest
import torch


def api():
    assert importlib.util.find_spec('gaussian_jscc.direct_rd_decisions'), 'direct RD decisions missing'
    from gaussian_jscc.direct_rd_decisions import HardTierMap, ProposalBank, audit_proposals
    return HardTierMap, ProposalBank, audit_proposals


def test_hard_map_delivers_q0_and_sampling_is_the_same_map():
    HardTierMap, _, _ = api()
    table = HardTierMap(torch.tensor([0, 2, 1, 3]))
    q, presence, _, _ = table.sample(torch.tensor([[0, 3, 2]]), 10.)
    assert q.tolist() == [[0, 3, 1]]
    assert presence.tolist() == [[0., 1., 1.]]
    assert table.probabilities(torch.arange(4), 10.).argmax(-1).tolist() == [0, 2, 1, 3]


def test_proposals_average_visits_and_allow_only_one_change_per_block():
    _, Bank, _ = api()
    bank = Bank(4)
    bank.add(torch.tensor([0, 1, 2]), torch.tensor([[2., 3., 4.], [-3., -1., 2.], [2., 1., 3.]]))
    bank.add(torch.tensor([2]), torch.tensor([[2., 3., 3.]]))
    restored = Bank(4); restored.load_state_dict(bank.state_dict())
    proposal = restored.propose(torch.full((4,), 2), torch.tensor([0, 0, 1, 1]))
    # Point 0 deletes with gain 3; point 1 would choose q1 with gain 2 in same block.
    assert proposal['ids'].tolist() == [0, 2]
    assert proposal['tiers'].tolist() == [0, 0]
    assert proposal['gains'].tolist() == [3., 2.]


def test_rd_audit_backtracks_and_rejection_preserves_original():
    _, _, audit = api()
    q = torch.full((4,), 2)
    proposal = dict(ids=torch.arange(4), tiers=torch.ones(4, dtype=torch.long), gains=torch.ones(4))
    calls = []
    def measure(candidate):
        calls.append(candidate.clone())
        return {'rd_score': 2. if int((candidate == 1).sum()) > 2 else .8}
    result = audit(q, proposal, {'rd_score': 1.}, measure, max_backtracks=2, min_gain=1e-6)
    assert result['accepted'] and result['changed'] == 2
    assert len(calls) == 2 and q.tolist() == [2]*4
    denied = audit(q, proposal, {'rd_score': 1.}, lambda _: {'rd_score': 1.1}, max_backtracks=2)
    assert not denied['accepted'] and torch.equal(denied['q'], q)
    assert denied['measurement']['rd_score'] == 1.
    empty = audit(q, dict(ids=torch.empty(0, dtype=torch.long), tiers=torch.empty(0, dtype=torch.long), gains=torch.empty(0)),
                  {'rd_score': 1.}, lambda _: pytest.fail('empty proposal rendered'))
    assert not empty['accepted'] and empty['attempts'] == []


def test_empty_bank_and_tied_costs_do_not_propose_churn():
    _, Bank, _ = api()
    bank = Bank(2); q = torch.tensor([2, 0]); blocks = torch.tensor([0, 1])
    assert len(bank.propose(q, blocks)['ids']) == 0
    bank.add(torch.arange(2), torch.zeros(2, 3))
    assert len(bank.propose(q, blocks)['ids']) == 0


def test_audit_rejects_nonfinite_measurements():
    _, _, audit = api()
    proposal = dict(ids=torch.tensor([0]), tiers=torch.tensor([0]), gains=torch.tensor([1.]))
    with pytest.raises(ValueError, match='nonfinite'):
        audit(torch.tensor([2]), proposal, {'rd_score':1.}, lambda _: {'rd_score':float('nan')})
