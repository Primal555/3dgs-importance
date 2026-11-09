import copy
import hashlib
import json

import pytest
import torch


def network_args():
    from gaussian_jscc.spatial_benefit import CONTEXT_DIM, STATE_DIM, FEATURE_DIM
    return (torch.randn(2,4,FEATURE_DIM),torch.randn(2,5,CONTEXT_DIM),
            torch.randn(2,4,5,STATE_DIM),torch.ones(2,5),torch.ones(2,5,dtype=torch.bool),
            torch.randn(2,4,2,15,9,9))


def test_independent_energy_changes_final_value_and_receives_main_gradient():
    from gaussian_jscc.benefit_independent_response import IndependentResponseNetwork
    from gaussian_jscc.benefit_footprint_model import FootprintSplitNetwork
    from gaussian_jscc.benefit_response_model import combine_terms, response_losses
    torch.manual_seed(4)
    base=FootprintSplitNetwork(16,0)
    network=IndependentResponseNetwork.from_footprint(base.state_dict(),0,torch.tensor([.2,.1,.03]),.04)
    torch.testing.assert_close(network.points[0].weight,base.points[0].weight,rtol=0,atol=0)
    assert not torch.equal(network.common_head[-1].weight,base.common_head[-1].weight)
    args=network_args();network.eval()
    before=network.forward_terms(*args)
    torch.testing.assert_close(before[...,1],torch.full_like(before[...,1],.04))
    with torch.no_grad():network.energy_head[-1].bias.add_(.5)
    after=network.forward_terms(*args)
    torch.testing.assert_close(after[...,0],before[...,0],rtol=0,atol=0)
    torch.testing.assert_close(after[...,2],before[...,2],rtol=0,atol=0)
    assert (after[...,1]>before[...,1]).all()
    assert (combine_terms(after)[...,0]>combine_terms(before)[...,0]).all()
    truth=torch.zeros_like(after)
    loss=response_losses(after,truth,torch.zeros(2,3),.1,torch.tensor([.2,.1,.03]),.01,1,1)['value'].sum()
    loss.backward()
    assert network.energy_head[-1].bias.grad.abs().sum()>0
    assert network.common_head[-1].weight.grad.abs().sum()>0
    identical=list(args)
    for k in (0,2,5):identical[k][:,1:]=identical[k][:,:1]
    assert network.forward_terms(*identical).abs().max()==0


def test_energy_initialization_uses_positive_training_labels_only():
    from gaussian_jscc.benefit_independent_response import positive_energy_mean
    groups=[dict(views=[dict(response_terms=torch.tensor([[1.,0.,2.],[1.,2.,3.],[1.,4.,3.]]))])]
    assert positive_energy_mean(groups)==pytest.approx(3)
    assert positive_energy_mean([dict(views=[dict(response_terms=torch.zeros(3,3))])])==0
    with pytest.raises(ValueError,match='validation'):
        positive_energy_mean([dict(validation=True,views=groups[0]['views'])])


def test_group_accumulation_matches_original_full_gradients():
    from gaussian_jscc.benefit_independent_response import GroupTaskAccumulator
    from gaussian_jscc.benefit_three_task import TaskAccumulator
    p=torch.nn.Parameter(torch.tensor([.3,-.4],dtype=torch.double))
    labels=[0,1,0,1];weights=torch.tensor([.1,.2,.3,.4],dtype=torch.double)
    z=torch.arange(40,dtype=torch.double).reshape(4,5,2)/20
    per=(z*p).square().sum(-1)
    a=GroupTaskAccumulator([p],[0]);a.add(per,weights,labels,diagnostics=True)
    b=TaskAccumulator([p],[0],names=a.names)
    b.add((z*p).square().sum(-1),weights,labels,diagnostics=True)
    torch.testing.assert_close(a.losses,b.losses)
    for ga,gb in zip(a.gradients,b.gradients):torch.testing.assert_close(ga[0],gb[0])
    for s in (0,1):
        for k in range(5):
            torch.testing.assert_close(a.group_gradients[s][k][0],b.strata[s]['gradients'][k][0])


def test_budget_preserves_ordinary_and_limits_tail_before_rank():
    from gaussian_jscc.benefit_independent_response import GroupTaskAccumulator, OrdinaryGradientBudget
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.double))
    a=GroupTaskAccumulator([p],[0]);a.strata[0]['records']=1;a.strata[1]['records']=1
    a.group_gradients[0][0][0].copy_(torch.tensor([2.,0.]))
    a.group_gradients[1][0][0].copy_(torch.tensor([0.,200.]))
    a.group_gradients[0][4][0].copy_(torch.tensor([100.,0.]))
    requested=torch.tensor([1.,1/3,1/3,1/3,1.],dtype=torch.double)
    budget=OrdinaryGradientBudget(.95)
    applied,r=budget.assign(a,requested)
    assert r['tail_gradient_factor']==pytest.approx(.01)
    assert r['ordinary_numeric_gradient_norm']==pytest.approx(2.)
    assert r['applied_tail_numeric_gradient_norm']==pytest.approx(2.)
    assert r['applied_rank_gradient_norm']<=r['numeric_gradient_norm']+1e-12
    torch.testing.assert_close(p.grad,torch.tensor([2.+r['applied_rank_gradient_norm'],2.],dtype=torch.double))
    torch.testing.assert_close(a.budgeted_shared_norms()[0],torch.tensor(8.,dtype=torch.double).sqrt())
    state=copy.deepcopy(budget.state_dict());new=OrdinaryGradientBudget(.95);new.load_state_dict(state)
    assert new.state_dict()==state
    a.group_gradients[0][0][0].zero_();_,r=new.assign(a,requested)
    assert r['ordinary_gradient_reference']>0 and r['tail_gradient_factor']>0
    new=OrdinaryGradientBudget(.95);_,r=new.assign(a,requested)
    assert r['tail_gradient_factor']==1 and r['ordinary_reference_fallback']
    previous=copy.deepcopy(new.state_dict())
    a.group_gradients[1][0][0].fill_(float('nan'))
    with pytest.raises(ValueError,match='nonfinite'):new.assign(a,requested)
    assert new.state_dict()==previous


def test_group_microbatches_preserve_weighted_gradients_and_budgeted_loss():
    from gaussian_jscc.benefit_independent_response import GroupTaskAccumulator, OrdinaryGradientBudget
    p=torch.nn.Parameter(torch.tensor([.3,-.4],dtype=torch.double))
    z=torch.arange(40,dtype=torch.double).reshape(4,5,2)/20
    weights=torch.tensor([.1,.2,.3,.4],dtype=torch.double);labels=[0,0,1,1]
    full=GroupTaskAccumulator([p],[0]);full.add((z*p).square().sum(-1),weights,labels)
    micro=GroupTaskAccumulator([p],[0])
    for start in (0,2):micro.add((z[start:start+2]*p).square().sum(-1),weights[start:start+2],labels[start:start+2])
    for s in (0,1):
        for a,b in zip(full.group_gradients[s],micro.group_gradients[s]):torch.testing.assert_close(a[0],b[0])
    requested=torch.tensor([1.,1/3,1/3,1/3,1.],dtype=torch.double)
    applied,r=OrdinaryGradientBudget().assign(full,requested)
    assert r['tail_gradient_factor']<1
    factors=torch.tensor([r['tail_gradient_factor']]*4+[1.],dtype=torch.double)
    expected=((full.strata[0]['losses']+factors*full.strata[1]['losses'])*applied).sum()
    assert full.applied_loss(applied)==pytest.approx(float(expected))
    empty=GroupTaskAccumulator([p],[0])
    empty.add((z[2:]*p).square().sum(-1),weights[2:],[1,1])
    _,r=OrdinaryGradientBudget().assign(empty,requested)
    assert r['tail_gradient_factor']==1 and r['ordinary_reference_fallback']


def test_independent_training_exact_resume_and_source_read_only(tmp_path, monkeypatch):
    from test_benefit_response_terms import response_dataset
    from gaussian_jscc import benefit_independent_controls as controls
    data,_=response_dataset(tmp_path)
    source=tmp_path/'response_run';source.mkdir()
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
    def args(out):
        return controls.parser().parse_args(['--source',str(source),'--data',str(data),'--out',str(out),
            '--samples','192','--evaluate-every-samples','64','--cpu-threads','1'])
    original=torch.optim.AdamW.step;calls=0
    def stop(self,*a,**kw):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('interrupted')
        return original(self,*a,**kw)
    interrupted=tmp_path/'interrupted'
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW,'step',stop)
        with pytest.raises(RuntimeError,match='interrupted'):controls.main(args(interrupted))
    resumed=args(interrupted);resumed.resume=True
    result=controls.main(resumed)
    controls.main(args(tmp_path/'straight'))
    left=torch.load(interrupted/'training/latest.pt',weights_only=True)
    right=torch.load(tmp_path/'straight/training/latest.pt',weights_only=True)
    for k,v in left['network'].items():torch.testing.assert_close(v,right['network'][k],atol=0,rtol=0)
    assert left['gradient_budget']==right['gradient_budget']
    assert left['gradient_budget']['updates']==3
    assert left['sampler']['visits']==right['sampler']['visits']
    torch.testing.assert_close(left['balancer']['logits'],right['balancer']['logits'],atol=0,rtol=0)
    assert result['normalizers_unchanged']
    assert result['version']=='independent_response_ordinary_budget_v1'
    settings=json.loads((interrupted/'training/settings.json').read_text())
    assert settings['architecture']=='independent_A_B_S_common_relative_v1'
    rows=[json.loads(l) for l in (interrupted/'training/metrics/optimization.jsonl').read_text().splitlines()]
    assert all(r['group_budget_scope'].startswith('all model parameters') for r in rows)
    assert all(r['requested_task_weights'][0]==1 for r in rows)
    assert all('effective_task_weights' in r['strata']['large_gap'] for r in rows)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert (interrupted/'review.tar.gz').is_file()
    assert controls.main(resumed)['status']=='complete'
    resumed.ordinary_gradient_ema=.9
    with pytest.raises(ValueError,match='resume settings mismatch'):controls.main(resumed)
