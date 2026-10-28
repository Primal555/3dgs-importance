import copy
import importlib.util
from types import SimpleNamespace

import pytest
import torch


def api():
    assert importlib.util.find_spec('gaussian_jscc.allocation_benefit_features') is not None, 'missing benefit features'
    from gaussian_jscc.allocation_benefit_features import view_features, keep_rates, FeatureProtocol
    return view_features,keep_rates,FeatureProtocol


def test_projection_samples_current_residual_and_masks_behind_camera():
    view,_,_=api()
    camera=SimpleNamespace(world_view_transform=torch.eye(4),full_proj_transform=torch.eye(4))
    base=torch.full((3,5,5),.4);target=torch.full_like(base,.1)
    got=view(torch.tensor([[0.,0.,1.],[0.,0.,-1.],[3.,0.,1.]]),[camera],[(base,target,None)],.02,.03)
    assert got.shape==(3,1,12)
    assert got[0,0,3]==1 and got[1,0,3]==got[2,0,3]==0
    torch.testing.assert_close(got[0,0,4:10],torch.full((6,),.3),rtol=1e-6,atol=1e-6)
    assert got[1,0,4:10].count_nonzero()==0
    assert got[0,0,10]==pytest.approx(.02)


def test_keep_scale_cost_counts_only_newly_opened_segments():
    _,rates,_=api()
    q=torch.tensor([[1,3,0],[1,0,0]])
    # 16 bits at 2 bit/use = 8 uses per newly required block segment.
    got=rates(q,torch.ones_like(q,dtype=torch.bool),[0,8,24,48],xyz=5.,bits=2.)
    torch.testing.assert_close(got[0,0],torch.tensor([13.,29.,53.],dtype=torch.float64))
    torch.testing.assert_close(got[0,1],torch.tensor([13.,37.,69.],dtype=torch.float64))
    torch.testing.assert_close(got[1,0],torch.tensor([21.,45.,77.],dtype=torch.float64))
    torch.testing.assert_close(got[1,1],torch.tensor([13.,37.,69.],dtype=torch.float64))


def test_protocol_keeps_markers_unstandardized():
    _,_,Protocol=api();p=Protocol(56,48)
    assert p.point_dim==222 and p.action_dim==89
    assert not p.continuous['point'][-3:].any()
    assert not p.continuous['action'][:9].any()
    assert not p.continuous['views'][3]


def test_scheduler_has_stable_block_holdout_and_covers_without_history():
    assert importlib.util.find_spec('gaussian_jscc.allocation_benefit_schedule') is not None, 'missing benefit scheduler'
    from gaussian_jscc.allocation_benefit_schedule import BenefitScheduler
    ids=[torch.arange(24).reshape(6,4)]
    s=BenefitScheduler(ids,seed=4);other=BenefitScheduler(ids,seed=4)
    torch.testing.assert_close(s.validation,other.validation)
    assert s.validation.any() and (~s.validation).any()
    assert all(len(s.validation[block].unique())==1 for block in ids[0])
    q=torch.ones(24,dtype=torch.long);p=torch.full((24,4),.25)
    rounds=s.select_rounds(q,torch.ones(24),p,1)
    chosen=torch.cat([ids[0][torch.arange(6),cols[0]] for cols in rounds])
    assert len(chosen.unique())==12
    s.record(chosen,1)
    restored=BenefitScheduler(ids,seed=4);restored.load_state_dict(copy.deepcopy(s.state_dict()))
    a=s.select_rounds(q,torch.ones(24),p,2);b=restored.select_rounds(q,torch.ones(24),p,2)
    assert all(torch.equal(x[0],y[0]) for x,y in zip(a,b))
    empty=s.select_rounds(torch.zeros_like(q),torch.ones(24),p,3)
    assert all((x[0]==-1).all() for x in empty)
