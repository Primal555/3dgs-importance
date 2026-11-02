import importlib.util

import torch
from types import SimpleNamespace


def module():
    assert importlib.util.find_spec('gaussian_jscc.matched_rd_decisions'), 'matched decisions missing'
    from gaussian_jscc import matched_rd_decisions
    return matched_rd_decisions


def test_coverage_shortlist_is_bounded_and_includes_tail():
    m = module()
    p = dict(ids=torch.arange(100), tiers=torch.zeros(100,dtype=torch.long), gains=torch.arange(100,0,-1).double())
    ids = m.shortlist(p, 16)
    assert len(ids)==16 and len(ids.unique())==16
    assert torch.equal(ids[:12], torch.arange(12))
    assert int(ids.max())>=95


def test_disjoint_batches_can_commit_when_highest_prefix_fails():
    m = module()
    q = torch.full((8,),2,dtype=torch.long)
    p = dict(ids=torch.arange(8),tiers=torch.ones(8,dtype=torch.long),gains=torch.arange(8,0,-1).double())
    weights = torch.tensor([20.,-4.,-3.,-2.,-4.,-3.,-2.,-1.])
    def measure(value):
        return dict(rd_score=100.+float(weights[value!=2].sum()))
    result=m.audit_chunks(q,p,dict(rd_score=100.),measure,chunk_size=2,max_checks=8,min_gain=1e-6)
    assert result['accepted'] and result['measurement']['rd_score']<100.
    assert result['q'][0]==2 and (result['q'][1:]==1).all()
    assert len(result['attempts'])<=8
    assert all(row['after_rd']<=row['before_rd'] for row in result['commits'])
    assert torch.equal(q,torch.full((8,),2))


def test_budget_is_strict_even_when_every_batch_fails():
    m=module(); q=torch.full((64,),2,dtype=torch.long)
    p=dict(ids=torch.arange(64),tiers=torch.ones(64,dtype=torch.long),gains=torch.ones(64))
    calls=[]
    def measure(v): calls.append(v.clone()); return dict(rd_score=2.)
    r=m.audit_chunks(q,p,dict(rd_score=1.),measure,chunk_size=16,max_checks=5,min_gain=0.)
    assert len(calls)==5 and not r['accepted'] and r['changed']==0
    assert [len((v!=q).nonzero()) for v in calls[:4]]==[16]*4


def test_common_condition_costs_can_reverse_initial_ranking():
    m=module(); q=torch.tensor([2,2,2]); ids=torch.arange(3)
    # Recomputed four-state values: first source is harmful in all alternatives.
    costs=torch.tensor([[3.,0.,2.],[-2.,0.,1.],[.5,0.,-.3]],dtype=torch.float64)
    p=m.ranked_actions(ids,costs,q)
    assert p['ids'].tolist()==[1,2]
    assert p['tiers'].tolist()==[1,3]
    assert torch.allclose(p['gains'],torch.tensor([2.,.3],dtype=torch.float64))


def test_unlabeled_candidates_share_budget_without_polluting_verified_batches():
    m=module()
    from gaussian_jscc.direct_rd_decisions import ProposalBank
    q=torch.full((512,),2,dtype=torch.long);ids=torch.arange(512);bank=ProposalBank(512)
    bank.add(ids,torch.tensor([[2.,1.,3.]]).repeat(512,1))
    args=SimpleNamespace(rescore_groups=512,exact_groups=64,audit_chunk=16,max_local_checks=12,min_rd_gain=1e-6)
    class Window:
        def score(self,ids,*a):return torch.tensor([[2.,1.,3.]]).repeat(len(ids),1)
        def exact_costs(self,ids,*a):return torch.tensor([[-1.,0.,1.]]).repeat(len(ids),1)
        def measure(self,v,*a):
            return dict(rd_score=float(10*(v==0).sum()-(v==1).sum()))
    got=m.matched_audit(Window(),bank,q,ids,None,[],[],None,dict(rd_score=0.),args)
    assert got['exact_groups']==64 and got['execution_candidates']==512
    assert got['changed']==64 and (got['q']==1).sum()==64
    assert len(got['attempts'])==12
    assert all(row['verified'] for row in got['attempts'][:4])
    # Beneficial network-only actions can also be committed within the same budget.
    class Good(Window):
        def measure(self,v,*a):return dict(rd_score=-float((v!=2).sum()))
    got=m.matched_audit(Good(),bank,q,ids,None,[],[],None,dict(rd_score=0.),args)
    assert got['changed']==512 and len(got['attempts'])<=12
