import copy
import math

import pytest
import torch


def example(dtype=torch.float64):
    truth=torch.tensor([[[-.4,-.02],[-.5,-.03],[-.6,-.04]]],dtype=dtype)
    rates=torch.tensor([[.1,.3,.5]],dtype=dtype)
    return truth,rates


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_exact_values_have_zero_direction_loss_and_gradient(dtype):
    from gaussian_jscc.calibrated_spatial_benefit import calibrated_direction_loss
    from gaussian_jscc.physical_benefit import decision_loss
    truth,rates=example(dtype)
    pred=truth.clone().requires_grad_()
    old=decision_loss(pred,truth,rates,1.)
    assert torch.autograd.grad(old.sum(),pred,retain_graph=True)[0].abs().max()>.01
    loss=calibrated_direction_loss(pred,truth,rates,1.)
    assert torch.equal(loss,torch.zeros_like(loss))
    assert torch.count_nonzero(torch.autograd.grad(loss.sum(),pred)[0])==0


def test_soft_direction_matches_manual_six_pair_kl():
    from gaussian_jscc.calibrated_spatial_benefit import calibrated_direction_loss
    truth=torch.tensor([[[-.4,0],[-.5,0],[-.6,0]]],dtype=torch.float64)
    rates=torch.tensor([[.1,.3,.5]],dtype=torch.float64)
    actual=[-.3,-.2,-.1,.1,.2,.1]
    predicted=[.1,.3,.5,.2,.4,.2]
    expected=0.
    for a,e in zip(actual,predicted):
        p=1/(1+math.exp(a)); q=1/(1+math.exp(e))
        expected+=abs(a)*(p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q)))/6
    loss=calibrated_direction_loss(torch.zeros_like(truth),truth,rates,1.)
    assert abs(float(loss)-expected)<1e-14
    single=calibrated_direction_loss(torch.zeros_like(truth[0]),truth[0],rates[0],1.)
    torch.testing.assert_close(single,loss[0])


def test_extremes_ties_and_gradient_toward_true_numeric_value():
    from gaussian_jscc.calibrated_spatial_benefit import calibrated_direction_loss
    truth,rates=example()
    pred=torch.zeros_like(truth,requires_grad=True)
    optimizer=torch.optim.Adam([pred],lr=.03)
    initial=calibrated_direction_loss(pred,truth,rates,1.)
    for _ in range(150):
        loss=calibrated_direction_loss(pred,truth,rates,1.)
        optimizer.zero_grad(); loss.sum().backward(); optimizer.step()
    assert calibrated_direction_loss(pred,truth,rates,1.)<initial*.001
    extreme=(truth*1e6).requires_grad_(False)
    wrong=(-extreme).clone().requires_grad_()
    loss=calibrated_direction_loss(wrong,extreme,rates,1.)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(torch.autograd.grad(loss.sum(),wrong)[0]).all()
    zero=torch.zeros_like(truth)
    assert calibrated_direction_loss(pred,zero,torch.zeros_like(rates),1.).item()==0
    exact=extreme.clone().requires_grad_()
    loss=calibrated_direction_loss(exact,extreme,rates,1.)
    assert loss.item()==0 and torch.count_nonzero(torch.autograd.grad(loss.sum(),exact)[0])==0


def test_new_learner_only_changes_direction_and_preserves_initial_weights(tmp_path):
    from gaussian_jscc.calibrated_spatial_benefit import CalibratedSpatialLearner,DIRECTION_OBJECTIVE
    from gaussian_jscc.spatial_benefit import SpatialLearner
    from gaussian_jscc.spatial_benefit_replay import SpatialReplay
    from test_physical_benefit import record
    from test_spatial_benefit import collision_inputs
    inputs=collision_inputs()[0][0]
    replay=SpatialReplay(tmp_path,max_groups=8,validation_groups=1)
    for i in range(3):
        g=record(i,.001,158); g['views'][0]['inputs']=inputs
        replay.add(g)
    torch.manual_seed(12); old=SpatialLearner(replay,'cpu',hidden=16,dropout=0)
    torch.manual_seed(12); new=CalibratedSpatialLearner(replay,'cpu',hidden=16,dropout=0)
    for key,value in old.network.state_dict().items():
        assert torch.equal(value,new.network.state_dict()[key])
    old.fit_scales(); new.fit_scales()
    groups=list(replay.training_records())
    a,b=old._batch_losses(groups),new._batch_losses(groups)
    for key in ('absolute','common','pair','regret'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert not torch.equal(a['decision'],b['decision'])
    new.train_updates(2,2)
    state=copy.deepcopy(new.state_dict())
    assert state['direction_objective']==DIRECTION_OBJECTIVE
    restored=CalibratedSpatialLearner(replay,'cpu',hidden=16,dropout=0)
    restored.load_state_dict(state)
    torch.testing.assert_close(new.predict(inputs,16,48,.01),restored.predict(inputs,16,48,.01))
    with pytest.raises(ValueError,match='direction objective'):
        restored.load_state_dict(old.state_dict())
