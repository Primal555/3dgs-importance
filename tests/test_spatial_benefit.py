import copy

import torch


def collision_inputs():
    from gaussian_jscc.pixel_local_renderer import TileScene
    from gaussian_jscc.physical_benefit_features import visibility_statistics
    from gaussian_jscc.spatial_benefit_features import spatial_inputs
    p = torch.tensor([[24,8,1,1,0,1,.2,1,.4,.4,.4,3],
                      [8,8,1,1,0,1,.7,1,.4,.4,.4,3],
                      [40,8,1,1,0,1,.7,1,.4,.4,.4,3]], dtype=torch.float32)
    tile = TileScene.from_projected(p,16,48,backend='cpu')
    a=p[None].repeat(4,1,1); a[0,0,7]=0
    b=a.clone(); a[1:,1,8]=.7; a[1:,2,8]=.1
    b[1:,1,8]=.1; b[1:,2,8]=.7
    target=TileScene.from_projected(a[1],16,48,backend='cpu').image
    stats=visibility_statistics(tile,target)
    raw=torch.zeros(4,3,14); raw[...,7]=1
    inputs=[spatial_inputs(torch.zeros(24),stats,raw,v,0,tile,target,torch.arange(3),.01,.006)
            for v in (a,b)]
    return inputs, tile, target, a, b


def test_spatial_pairing_breaks_old_collision_and_keeps_permutation():
    from gaussian_jscc.spatial_benefit import SpatialBenefitNetwork, pack_inputs
    (a,b),_,_,_,_=collision_inputs()
    assert torch.equal(a['features'],b['features'])
    assert not torch.equal(a['states'],b['states'])
    assert torch.count_nonzero(a['states'][0,0])==0
    torch.manual_seed(3)
    net=SpatialBenefitNetwork(hidden=16,dropout=0).eval()
    pa=net(*pack_inputs([a],'cpu'))
    pb=net(*pack_inputs([b],'cpu'))
    assert not torch.allclose(pa,pb,rtol=0,atol=1e-9)
    order=torch.tensor([2,0,1]); perm=dict(a,context=a['context'][order],
        states=a['states'][:,order],weights=a['weights'][order])
    torch.testing.assert_close(pa,net(*pack_inputs([perm],'cpu')),atol=1e-7,rtol=1e-6)
    same=dict(a,features=a['features'][:1].expand(4,-1),states=a['states'][:1].expand(4,-1,-1))
    assert torch.count_nonzero(net(*pack_inputs([same],'cpu')))==0
    # The set encoder must represent both signed responses, even when global inputs collide.
    optimizer=torch.optim.Adam(net.parameters(),lr=.01)
    packed=pack_inputs([a,b],'cpu'); truth=torch.tensor([[[-.04,0.]]*3,[[.03,0.]]*3])
    for _ in range(120):
        loss=(net(*packed)-truth).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert (net(*packed)-truth).abs().max()<.002


def test_loss_preserves_physical_component_units_and_supervises_pairs():
    from gaussian_jscc.spatial_benefit import value_losses
    truth=torch.tensor([[[-2.,-.1],[-4.,-.2],[-8.,-.3]]],dtype=torch.float64)
    exact=value_losses(truth,truth,torch.zeros(3),1.)
    assert exact['absolute']==0 and exact['common']==0 and exact['pair']==0
    shifted=truth+torch.tensor([1.,0.])
    swapped=truth.flip(1)
    assert value_losses(shifted,truth,torch.zeros(3),1.)['common']>0
    assert value_losses(swapped,truth,torch.zeros(3),1.)['pair']>0
    # Equal errors in calibrated physical units receive equal component penalty.
    a=value_losses(truth+torch.tensor([.2,0.]),truth,torch.zeros(3),1.)
    b=value_losses(truth+torch.tensor([0.,.2]),truth,torch.zeros(3),1.)
    torch.testing.assert_close(a['absolute'],b['absolute'])


def test_spatial_preflight_cpu():
    from gaussian_jscc.spatial_benefit_allocation import verify_spatial_inputs
    assert verify_spatial_inputs('cpu')['spatial_inputs_cpu_device_max_abs']==0


def test_odd_hidden_and_batched_prediction_equal_individual(tmp_path):
    from gaussian_jscc.spatial_benefit import SpatialLearner
    a,b=collision_inputs()[0]
    learner=SpatialLearner(None,'cpu',hidden=15,dropout=0)
    actual=learner.predict_group_views([[a,b],[b]])
    for result,inputs in zip(actual,([a,b],[b])):
        for value,view in zip(result,inputs):
            torch.testing.assert_close(value,learner.predict(view,16,48,.01),rtol=1e-5,atol=1e-7)


def test_replay_probability_visits_and_resume(tmp_path):
    from gaussian_jscc.spatial_benefit_replay import SpatialReplay
    from test_physical_benefit import record
    replay=SpatialReplay(tmp_path,max_groups=3,validation_groups=1,cache_mib=1,seed=7)
    ids=[replay.add(record(i,.01,158)) for i in range(3)]
    replay.mark_updated([ids[0],ids[0],ids[1]])
    replay.update_priority(ids[0],1.); replay.update_priority(ids[1],2.); replay.update_priority(ids[2],3.)
    probabilities=replay.probabilities()
    expected=torch.tensor([.25/3+.25/6,.25/3+.25*2/6,.25/3+.5+.25*3/6],dtype=torch.float64)
    torch.testing.assert_close(probabilities,expected)
    groups=replay.sample(5)
    for g in groups:
        i=ids.index(g['replay_record_id'])
        assert g['replay_probability']==float(expected[i])
        assert abs(g['replay_importance_weight']-1/(3*float(expected[i])))<1e-12
    # Merely drawing samples is not counted as successful optimizer visits.
    assert replay.usage[ids[2]]==0
    replay.mark_updated([ids[2]])
    state=copy.deepcopy(replay.state_dict())
    restored=SpatialReplay(tmp_path,max_groups=3,validation_groups=1,cache_mib=1)
    restored.load_state_dict(state)
    assert [g['replay_record_id'] for g in replay.sample(12)]==[g['replay_record_id'] for g in restored.sample(12)]
    replay.add(record(8,.02,158))
    assert replay.usage[ids[0]]==2 and ids[0] not in replay.priorities
    report=replay.coverage()
    assert report['all_group_draws']==4 and report['all_groups']==4


def test_scales_train_only_and_learner_checkpoint(tmp_path):
    from gaussian_jscc.spatial_benefit import SpatialLearner
    from gaussian_jscc.spatial_benefit_replay import SpatialReplay
    from test_physical_benefit import record
    (a,b),_,_,_,_=collision_inputs()
    replay=SpatialReplay(tmp_path,max_groups=10,validation_groups=2)
    for i,inputs in enumerate((a,b)):
        g=record(i,.001,158); g['views'][0]['inputs']=inputs
        replay.add(g); replay.add(copy.deepcopy(g),validation=True)
    learner=SpatialLearner(replay,'cpu',hidden=16,dropout=0)
    learner.fit_scales(); scales=copy.deepcopy(learner.scales)
    stats=learner.train_updates(2,2)
    assert stats['replay_all_group_draws']==4
    assert stats['pair_loss']>=0 and torch.isfinite(torch.tensor(stats['grad_norm']))
    report=learner.validate()
    assert report['validation_groups']==2 and 'weighted_training' in report
    assert torch.equal(scales['rd'],learner.scales['rd'])
    restored=SpatialLearner(replay,'cpu',hidden=16,dropout=0)
    restored.load_state_dict(copy.deepcopy(learner.state_dict()))
    torch.testing.assert_close(learner.predict(a,16,48,.01),restored.predict(a,16,48,.01))
