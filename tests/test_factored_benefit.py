import copy
import importlib.util

import pytest
import torch
from torch.nn import functional as F

from test_allocation_benefit import features,record


def api():
    assert importlib.util.find_spec('gaussian_jscc.factored_benefit') is not None,'missing factored benefit model'
    from gaussian_jscc.factored_benefit import FactoredBenefitNetwork,DecisionBenefitLearner
    from gaussian_jscc.factored_benefit_loss import decision_loss_parts
    return FactoredBenefitNetwork,DecisionBenefitLearner,decision_loss_parts


def inputs():
    x=features(1)
    x['action']=torch.randn(1,20)
    x={key:value.repeat(3,*([1]*(value.ndim-1))) for key,value in x.items()}
    x['action'][:,6:9]=torch.eye(3)
    x['action'][:,10]=torch.tensor([8.,24.,48.]);x['action'][:,11]=x['action'][:,10]-8
    return x


def training_records():
    result=[]
    for branch in ('positive','keep'):
        r=record(branch);x=inputs();r['features']={k:v[:len(r['target'])].clone() for k,v in x.items()}
        if branch=='keep':r['target']=torch.tensor([[.1,.02],[.2,.01],[.4,.03]])
        result.append(r)
    return result


def test_keep_common_value_is_target_invariant_and_increments_reconstruct_exactly():
    Network,_,_=api();net=Network(5,20,3,2)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
        net.common_scale.fill_(1.);net.increment_scale.fill_(1.)
        net.keep_value_head.bias.copy_(torch.tensor([2.,3.]))
        net.keep_increment_head.bias.copy_(torch.tensor([1.,-1.,2.,1.]))
    x=inputs();base,gaps=net.keep_factors(x)
    torch.testing.assert_close(base,torch.tensor([[2.,3.]]).expand(3,-1))
    values=net.keep_values(x)
    torch.testing.assert_close(values.mean(1),base)
    torch.testing.assert_close(values.diff(dim=1),gaps)
    expected=torch.tensor([[2/3,10/3],[5/3,7/3],[11/3,10/3]])
    torch.testing.assert_close(net(x,'keep'),expected)


def test_keep_value_can_have_negative_increment_and_mask_ignores_auxiliary_input():
    Network,_,_=api();torch.manual_seed(9);net=Network(5,20,3,2)
    x=inputs();y={k:v.clone() for k,v in x.items()};y['aux']+=10000
    torch.testing.assert_close(net(x,'keep',masked=True),net(y,'keep',masked=True),rtol=0,atol=0)
    assert not torch.equal(net(x,'keep'),net(y,'keep'))
    with torch.no_grad():
        net.keep_increment_head.weight.zero_();net.keep_increment_head.bias.fill_(-1.)
    assert (net.keep_values(x).diff(dim=1)<0).all()


def test_random_normalized_keep_context_removes_target_identity_but_positive_actions_differ():
    Network,_,_=api();torch.manual_seed(72);net=Network(5,20,3,2)
    with torch.no_grad():
        net.action_mean.copy_(torch.randn(20)*10)
        net.action_std.copy_(torch.linspace(.2,2.,20))
    x=inputs();base,gaps=net.keep_factors(x)
    y={key:value.clone() for key,value in x.items()}
    y['action'][:,6:9]+=1e6;y['action'][:,10:12]-=1e6
    changed_base,changed_gaps=net.keep_factors(y)
    torch.testing.assert_close(base,changed_base,rtol=0,atol=0)
    torch.testing.assert_close(gaps,changed_gaps,rtol=0,atol=0)
    torch.testing.assert_close(base,base[:1].expand_as(base),rtol=1e-5,atol=1e-12)
    torch.testing.assert_close(gaps,gaps[:1].expand_as(gaps),rtol=1e-5,atol=1e-12)
    positive=net(x,'positive')
    assert not torch.equal(positive[0],positive[1])


def test_factored_scales_fit_only_training_true_common_value_and_increments():
    Network,_,_=api();net=Network(5,20,3,2);rows=training_records();net.fit(rows)
    assert float(net.common_scale)==pytest.approx(.7/3)
    assert float(net.increment_scale)==pytest.approx(.19)
    bad=training_records();bad[1]['validation']=True
    with pytest.raises(ValueError,match='validation|training'):Network(5,20,3,2).fit(bad)


def loss_parts(pred,target,rates,conditional,branches,scale=1.):
    _,_,parts=api();sizes=[1 if b==0 else 3 for b in branches]
    groups=torch.repeat_interleave(torch.arange(len(sizes)),torch.tensor(sizes))
    return parts(pred,target,rates,conditional,groups,torch.full((len(sizes),),scale),
                 torch.tensor(branches),torch.tensor(scale),torch.tensor(scale))


def test_decision_loss_prefers_correct_side_when_numeric_errors_are_equal():
    target=torch.tensor([[.001,0.]])
    wrong=torch.tensor([[.0008,0.]]);right=torch.tensor([[.0012,0.]])
    rates=torch.tensor([-.0009]);conditional=torch.ones(1)
    a=loss_parts(wrong,target,rates,conditional,[0],.001)
    b=loss_parts(right,target,rates,conditional,[0],.001)
    assert float(a['numeric'])==pytest.approx(float(b['numeric']),rel=1e-5)
    assert float(a['total'])>float(b['total'])


def test_q0_direction_gradient_follows_runtime_conditional_mixture():
    target=torch.tensor([[.1,0.],[-.2,0.],[-.3,0.]])
    for conditional,index,sign in [(torch.tensor([1.,0.,0.]),0,-1),(torch.tensor([0.,1.,0.]),1,1)]:
        pred=torch.zeros(3,2,requires_grad=True)
        parts=loss_parts(pred,target,torch.zeros(3),conditional,[1]);parts['direction'].sum().backward()
        assert pred.grad[index,0]*sign>0
        assert pred.grad[torch.arange(3)!=index].count_nonzero()==0


def test_q0_ranking_penalizes_reversed_order_and_handles_exact_ties():
    target=torch.tensor([[0.,0.],[1.,0.],[2.,0.]])
    kwargs=dict(rates=torch.zeros(3),conditional=torch.full((3,),1/3),branches=[1])
    a=loss_parts(target.clone(),target,**kwargs)
    b=loss_parts(target.flip(0),target,**kwargs)
    assert float(b['ranking'])>float(a['ranking'])
    zero=loss_parts(torch.zeros(3,2),torch.zeros(3,2),**kwargs)
    assert all(torch.isfinite(v).all() for v in zero.values())
    assert float(zero['direction'])==0. and float(zero['ranking'])==0.


def test_new_learner_preserves_next_adam_step_and_rejects_legacy_objective():
    Network,Learner,_=api();torch.manual_seed(52);torch.set_num_threads(1)
    learner=Learner(Network(5,20,3,2),capacity=20,fit_windows=1,batch_size=4)
    learner.observe(training_records(),1);learner.train_step(2);state=copy.deepcopy(learner.state_dict())
    other=Learner(Network(5,20,3,2),capacity=20,fit_windows=1,batch_size=4);other.load_state_dict(state)
    learner.train_step(2);other.train_step(2)
    for key,value in learner.network.state_dict().items():torch.testing.assert_close(value,other.network.state_dict()[key],rtol=0,atol=0)
    state.pop('objective_variant')
    with pytest.raises(ValueError,match='objective'):other.load_state_dict(state)


def test_factored_mse_model_has_no_structure_output():
    Network,_,_=api();net=Network(5,20,3,2,mse_only=True);x=inputs()
    assert net(x,'keep')[:,1].count_nonzero()==0
    assert net(x,'positive')[:,1].count_nonzero()==0
