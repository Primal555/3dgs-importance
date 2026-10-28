import copy
import importlib.util

import pytest
import torch


def api():
    assert importlib.util.find_spec('gaussian_jscc.allocation_benefit') is not None, 'missing benefit evaluator'
    from gaussian_jscc.allocation_benefit import BenefitNetwork, benefit_loss
    return BenefitNetwork, benefit_loss


def features(count=3):
    return dict(point=torch.arange(count*5).reshape(count,5).float()/10,
                action=torch.randn(count,4), views=torch.randn(count,2,3),
                aux=torch.randn(count,2), available=torch.ones(count,1))


def record(branch='positive',step=1,source=0,validation=False):
    n=1 if branch=='positive' else 3
    return dict(branch=branch,step=step,source=source,validation=validation,
                features=features(n),target=torch.full((n,2),.02),
                rates=torch.linspace(.001,.003,n),conditional=torch.tensor([.2,.3,.5]) if n==3 else torch.ones(1),
                first_order=torch.full((n,2),.01),old=1,new=[2] if n==1 else [1,2,3],group='coverage')


def test_masked_prediction_is_independent_of_every_auxiliary_value():
    Network,_=api();torch.manual_seed(4)
    net=Network(5,4,3,2);x=features();y={k:v.clone() for k,v in x.items()}
    y['aux'].mul_(10000).add_(20)
    a=net(x,'positive',masked=True);b=net(y,'positive',masked=True)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.equal(net(x,'positive'),net(y,'positive'))
    a.sum().backward()
    assert all(p.grad is None or p.grad.count_nonzero()==0 for p in net.aux_encoder.parameters())


def test_normalization_uses_true_training_labels_and_mse_head_is_exactly_zero():
    Network,_=api();net=Network(5,4,3,2,mse_only=True)
    data=[record(),record('keep')]
    net.fit(data)
    assert bool(net.ready)
    assert torch.all(net.scales>0)
    assert net(features(),'keep')[:,1].count_nonzero()==0
    bad=record(validation=True)
    with pytest.raises(ValueError,match='validation'): net.fit([bad])


def test_loss_uses_conditional_q0_cost_and_keeps_group_weight_balanced():
    _,loss=api()
    target=torch.tensor([[1.,0.],[2.,0.],[3.,0.]])
    pred=target.clone().requires_grad_(True)
    rate=torch.tensor([.1,.2,.3]);pi=torch.tensor([.2,.3,.5])
    value=loss(pred,target,rate,pi,1.,'keep')
    assert torch.isfinite(value)
    value.backward();assert torch.isfinite(pred.grad).all()
    # Regression and summed regression vanish at the exact target. Remaining
    # direction term is tau * softplus(-2.53/tau) * capped importance * 0.1.
    expected=.1*2.53*.1*torch.nn.functional.softplus(torch.tensor(-25.3))
    assert float(value.detach())==pytest.approx(float(expected),abs=1e-10)
    zero=loss(torch.zeros(1,2),torch.zeros(1,2),torch.zeros(1),torch.ones(1),1.,'positive')
    assert float(zero)==0.


def test_replay_copies_old_context_excludes_validation_and_resumes_sampling():
    assert importlib.util.find_spec('gaussian_jscc.allocation_benefit_replay') is not None, 'missing benefit replay'
    from gaussian_jscc.allocation_benefit_replay import BenefitReplay
    replay=BenefitReplay(10,42)
    r=record();before=r['features']['point'].clone();replay.add(r)
    r['features']['point'].fill_(100)
    torch.testing.assert_close(replay.groups[0]['features']['point'],before)
    with pytest.raises(ValueError,match='validation'):replay.add(record(validation=True))
    for i in range(1,20): replay.add(record('keep' if i%2 else 'positive',i,i))
    saved=copy.deepcopy(replay.state_dict());other=BenefitReplay(10,42);other.load_state_dict(saved)
    a=replay.sample(8,20);b=other.sample(8,20)
    assert [r['source'] for r in a]==[r['source'] for r in b]
    assert sum(r['branch']=='keep' for r in a)==4


def test_static_source_is_referenced_while_dynamic_input_is_frozen():
    assert importlib.util.find_spec('gaussian_jscc.allocation_benefit_replay') is not None
    from gaussian_jscc.allocation_benefit_replay import BenefitReplay
    source=torch.tensor([[.1,.2],[.3,.4]])
    replay=BenefitReplay(4,1,static_source=source)
    r=record();r['features']['point'][:,:2]=source[0]
    replay.add(r)
    assert replay.groups[0]['features']['point'].shape[-1]==3
    got=replay.sample(1,1)[0]
    torch.testing.assert_close(got['features']['point'][:,:2],source[0:1])


def test_real_network_learns_a_nonlinear_full_benefit_without_first_order():
    Network,loss=api();torch.set_num_threads(1);torch.manual_seed(10)
    net=Network(5,4,3,2);records=[]
    for i in range(20):
        r=record(source=i);r['features']['point'][:,0]=i/10-1
        r['target']=torch.tensor([[((i/10-1)**2+.1)*.001,0.]])
        records.append(r)
    net.fit(records+[record('keep')]);opt=torch.optim.Adam(net.parameters(),lr=.003)
    def total():
        return torch.stack([loss(net(r['features'],'positive',masked=True),r['target'],
                    r['rates'],r['conditional'],net.scales[0],'positive') for r in records]).mean()
    before=float(total().detach())
    for _ in range(60):
        opt.zero_grad();v=total();v.backward();opt.step()
    assert float(total().detach())<before*.2


def test_learner_fits_training_only_and_restores_next_adam_update_exactly():
    Network,_=api()
    assert importlib.util.find_spec('gaussian_jscc.allocation_benefit_learning') is not None, 'missing learner'
    from gaussian_jscc.allocation_benefit_learning import BenefitLearner
    torch.manual_seed(55);torch.set_num_threads(1)
    learner=BenefitLearner(Network(5,4,3,2),capacity=20,seed=42,fit_windows=1,
                           updates_per_window=1,batch_size=4)
    valid=record(validation=True);valid['target'].fill_(10000)
    learner.observe([record(),record('keep'),valid],1)
    assert learner.network.scales.max()<1.
    assert len(learner.replay.groups)==2 and len(learner.validation.groups)==1
    assert learner.updates==0
    state=copy.deepcopy(learner.state_dict())
    other=BenefitLearner(Network(5,4,3,2),capacity=20,seed=42,fit_windows=1,
                         updates_per_window=1,batch_size=4)
    other.load_state_dict(state)
    learner.train_step(2);other.train_step(2)
    for k,v in learner.network.state_dict().items():
        torch.testing.assert_close(v,other.network.state_dict()[k],rtol=0,atol=0)


def test_constant_training_feature_does_not_get_an_extreme_unseen_gain():
    Network,_=api();net=Network(5,4,3,2)
    data=[record(),record('keep')]
    for r in data:r['features']['point'][:,0]=.1
    net.fit(data)
    assert float(net.point_std[0])==1.


def test_masked_supervision_is_added_without_reducing_full_input_loss():
    Network,loss=api()
    from gaussian_jscc.allocation_benefit_learning import BenefitLearner
    torch.manual_seed(12);torch.set_num_threads(1)
    learner=BenefitLearner(Network(5,4,3,2),capacity=20,fit_windows=1,batch_size=4,
                           updates_per_window=1,mask_fraction=1.,mask_weight=.5)
    learner.observe([record(),record('keep')],1)
    rng=learner.replay.generator.get_state();batch=learner.replay.sample(4,2);learner.replay.generator.set_state(rng)
    values=[]
    for r in batch:
        branch=r['branch'];scale=learner.network.scales[int(branch=='keep')]
        full=learner.network(r['features'],branch);masked=learner.network(r['features'],branch,masked=True)
        values.append(loss(full,r['target'],r['rates'],r['conditional'],scale,branch)+
                      .5*loss(masked,r['target'],r['rates'],r['conditional'],scale,branch))
    expected=float(torch.stack(values).mean().detach())
    assert learner.train_step(2)['benefit_training_loss']==pytest.approx(expected,rel=1e-5)


def test_normalization_keeps_warmup_labels_when_replay_reservoir_is_tiny():
    Network,_=api()
    from gaussian_jscc.allocation_benefit_learning import BenefitLearner
    torch.manual_seed(2)
    learner=BenefitLearner(Network(5,4,3,2),capacity=2,seed=42,fit_windows=2,batch_size=4)
    learner.observe([record(),record('keep')],1)
    learner.observe([record(step=2,source=i) for i in range(20)],2)
    assert bool(learner.network.ready)
    assert learner.normalization_records==[]
