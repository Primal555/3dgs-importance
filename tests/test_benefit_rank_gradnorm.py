import torch

from gaussian_jscc.benefit_rank_gradnorm import rd_losses,GradNormBalancer,StratifiedCycles
from gaussian_jscc.benefit_isolated_positive_model import positive_losses


def test_magnitude_preserves_old_coefficient_and_rank_is_distinct():
    p=torch.tensor([[[.4,0.],[-.2,.1],[.1,-.1]]],dtype=torch.float64)
    t=torch.tensor([[[.1,0.],[.2,-.1],[-.3,.2]]],dtype=torch.float64)
    rates=torch.tensor([[.01,.03,.09]],dtype=torch.float64)
    old=positive_losses(p,t,rates,.5)
    new=rd_losses(p,t,rates,.5,.1,4.)
    torch.testing.assert_close(new['magnitude'],old['centered']+old['pair'])
    larger=rd_losses(p*8,t*8,rates,.5,.1,4.)
    torch.testing.assert_close(larger['magnitude'],new['magnitude']*64)
    assert not torch.allclose(larger['ranking'],new['ranking']*64)


def test_soft_near_ties_and_extreme_rank_errors_are_finite():
    p=torch.tensor([[[1e-7,0.],[0.,0.],[-1e-7,0.]]],dtype=torch.float64,requires_grad=True)
    t=-p.detach();rates=torch.zeros(1,3,dtype=torch.float64)
    loss=rd_losses(p,t,rates,1e-4,1e-7,4.)['ranking']
    loss.sum().backward()
    assert loss.item()>0 and p.grad.abs().sum()>0
    big=rd_losses(p.detach()*1e10,t*1e10,rates,1e-4,1e-7,4.)
    assert all(torch.isfinite(v).all() for v in big.values())
    exact=rd_losses(t,t,rates,1e-4,1e-7,4.)
    assert exact['ranking'].abs().max()<1e-12


def test_gradnorm_changes_task_weights_without_changing_model_gradients():
    parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
    balancer=GradNormBalancer(['ordinary_magnitude','tail_ranking'],[1.,1.],device='cpu')
    losses=torch.stack((parameter.square().sum()*100,parameter.square().sum()*.01))
    stats=balancer.measure(losses,[parameter])
    assert parameter.grad is None
    assert stats['base_gradient_norms'][0]>stats['base_gradient_norms'][1]*1000
    initial=balancer.weights().detach().clone()
    balancer.update_weights()
    assert not torch.equal(initial,balancer.weights().detach())
    assert (balancer.weights()>0).all()
    torch.testing.assert_close(balancer.weights().sum(),torch.tensor(2.,dtype=torch.float64))
    losses.sum().backward()
    assert torch.isfinite(parameter.grad).all()


def test_stratified_cycles_cover_each_stratum_and_resume_exactly():
    sampler=StratifiedCycles([0,0,0,0,1],batch_size=4,seed=42)
    first=sampler.next()
    assert sum(i==4 for i in first)==1
    state=sampler.state_dict()
    expected=[sampler.next() for _ in range(4)]
    restored=StratifiedCycles([0,0,0,0,1],batch_size=4,seed=42)
    restored.load_state_dict(state)
    assert [restored.next() for _ in range(4)]==expected
    single=StratifiedCycles([0,0,0],batch_size=2,seed=42)
    assert len(single.next())==2


def test_sampler_snapshot_is_not_mutated_by_later_draws():
    sampler=StratifiedCycles([0,0,0,0,1],4,17)
    sampler.next();state=sampler.state_dict();visits=list(state['visits'])
    sampler.next()
    assert state['visits']==visits


def test_task_weights_can_balance_large_unit_differences_without_a_fixed_floor():
    balance=GradNormBalancer(['numeric','ranking'],[1.,1.],device='cpu')
    with torch.no_grad():balance.logits.copy_(torch.tensor([0.,-20.],dtype=torch.float64))
    assert 0<float(balance.weights()[1].detach())<1e-6
