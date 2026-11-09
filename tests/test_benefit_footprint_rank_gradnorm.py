import copy
import hashlib
import json

import pytest
import torch

from gaussian_jscc import benefit_footprint_rank_gradnorm as model


def test_numeric_loss_and_gradients_equal_original_six_pair_supervision():
    from gaussian_jscc.spatial_benefit import value_losses
    p=torch.randn(3,3,2,dtype=torch.float64,requires_grad=True)
    t=torch.randn_like(p);rates=torch.rand(3,3,dtype=torch.float64)
    old=value_losses(p,t,rates,.7)
    expected=old['absolute']+.8*old['common']+1.3*old['pair']
    new=model.split_losses(p,t,rates,.7,.2,.8,1.3)
    torch.testing.assert_close(new['magnitude'],expected)
    a=torch.autograd.grad(expected.sum(),p,retain_graph=True)[0]
    b=torch.autograd.grad(new['magnitude'].sum(),p)[0]
    torch.testing.assert_close(a,b)


def test_ranking_includes_q0_and_is_zero_at_teacher():
    p=torch.ones(1,3,2,dtype=torch.float64,requires_grad=True)
    t=-p.detach();rates=torch.zeros(1,3,dtype=torch.float64)
    loss=model.split_losses(p,t,rates,1.,.2,1.,1.)['ranking']
    loss.sum().backward()
    assert loss.item()>0 and p.grad.sum().abs()>0
    assert model.split_losses(t,t,rates,1.,.2,1.,1.)['ranking'].abs().max()<1e-12


def test_uniform_cycles_match_original_draws_and_resume():
    sampler=model.UniformCycles(11,8,42);generator=torch.Generator().manual_seed(42)
    order=[];cursor=0
    for _ in range(8):
        expected=[]
        while len(expected)<8:
            if cursor==len(order):order=torch.randperm(11,generator=generator).tolist();cursor=0
            take=min(8-len(expected),len(order)-cursor)
            expected.extend(order[cursor:cursor+take]);cursor+=take
        assert sampler.next()==expected
    state=sampler.state_dict();expected=sampler.next()
    restored=model.UniformCycles(11,8,42);restored.load_state_dict(state)
    assert restored.next()==expected


def test_initialization_matches_original_split_weights_and_all_branches_train(tmp_path):
    from test_benefit_isolated_positive_controls import prepared
    from gaussian_jscc.benefit_isolated_positive_controls import load_data
    from gaussian_jscc.benefit_stage_one import make_learner as original
    from gaussian_jscc.benefit_footprint_model import FootprintLearner,FootprintSplitNetwork
    from gaussian_jscc.benefit_stage_one_data import read_groups
    from gaussian_jscc.cli import seed_all
    source=prepared(tmp_path)
    _,replay,train,_,metadata,initial,previous,*_=load_data(source)
    cfg=metadata['config'];seed_all(metadata['seed'])
    original(dict(settings=cfg,learner=initial['learner']),'cpu')
    reference=FootprintLearner(None,'cpu',hidden=cfg['hidden'],dropout=cfg['dropout'])
    reference.load_state_dict(copy.deepcopy(initial['learner']))
    reference.network=FootprintSplitNetwork.from_spatial(initial['learner']['network'],cfg['dropout'])
    seed_all(metadata['seed']);learner=model.make_learner(initial,cfg,'cpu',previous['choice_targets']['temperature'])
    assert type(learner.network) is FootprintSplitNetwork
    assert all(torch.equal(v,learner.network.state_dict()[k]) for k,v in reference.network.state_dict().items())
    records=list(read_groups(replay,train[:2]));losses=learner._batch_losses(records)
    parameters=list(learner.network.points[2].parameters())
    for name in ('magnitude','ranking'):
        gradients=torch.autograd.grad(losses[name].mean(),parameters,retain_graph=True)
        assert sum(float(g.abs().sum()) for g in gradients)>0
    sum(v.mean() for v in losses.values()).backward()
    for branch in ('points','common_head','relative_head','profile_projection'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(learner.network,branch).parameters())


def test_real_runner_preserves_sources_and_restores_exactly(tmp_path,monkeypatch):
    from gaussian_jscc import benefit_footprint_rank_gradnorm_controls as controls
    from test_benefit_isolated_positive_controls import prepared
    source=prepared(tmp_path)
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    def args(out):
        return controls.parser().parse_args(['--source',str(source),'--out',str(out),
            '--updates','3','--evaluate-every','1','--cpu-threads','1'])
    interrupted=tmp_path/'interrupted';a=args(interrupted)
    original=torch.optim.AdamW.step;calls=0
    def stop(self,*pos,**kw):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('interrupted')
        return original(self,*pos,**kw)
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW,'step',stop)
        with pytest.raises(RuntimeError,match='interrupted'):controls.main(a)
    a.resume=True;result=controls.main(a)
    controls.main(args(tmp_path/'straight'))
    left=torch.load(interrupted/'latest.pt',weights_only=True)
    right=torch.load(tmp_path/'straight/latest.pt',weights_only=True)
    assert all(torch.equal(v,right['learner']['network'][k]) for k,v in left['learner']['network'].items())
    assert torch.equal(left['balancer']['logits'],right['balancer']['logits'])
    assert left['sampler']['visits']==right['sampler']['visits']
    assert result['training_coverage']['draws']==6
    assert result['branch_changed']['common_head'] and result['branch_changed']['relative_head']
    assert result['branch_changed']['points']
    settings=json.loads((interrupted/'settings.json').read_text())
    assert settings['tasks']==['magnitude','ranking']
    assert settings['initialization']=='data/initial.pt; original split construction order'
    assert (interrupted/'review.tar.gz').is_file()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert controls.main(a)['status']=='complete'
    (interrupted/'review.tar.gz').unlink()
    assert controls.main(a)['status']=='complete'
    a.weight_lr=.001
    with pytest.raises(ValueError,match='resume settings'):controls.main(a)
