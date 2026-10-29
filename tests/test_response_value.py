from types import SimpleNamespace

import torch


def test_projected_footprint_scales_with_distance_and_has_correct_center():
    from gaussian_jscc.response_value_features import project_footprints
    camera=SimpleNamespace(image_width=100,image_height=100,FoVx=1.5707963267948966,
        FoVy=1.5707963267948966,world_view_transform=torch.eye(4),camera_center=torch.zeros(3))
    raw=torch.zeros(2,14);raw[:,2]=torch.tensor([2.,4.]);raw[:,4:7]=torch.tensor(.1).log();raw[:,7]=1
    center,cov,depth,visible=project_footprints(raw,camera)
    torch.testing.assert_close(center,torch.full((2,2),49.5,dtype=torch.float64))
    torch.testing.assert_close(cov[:,0,0],torch.tensor([6.55,1.8625],dtype=torch.float64),atol=1e-5,rtol=1e-5)
    assert visible.all() and depth.tolist()==[2.,4.]


def test_offscreen_jacobian_matches_rasterizer_frustum_clamp():
    from gaussian_jscc.response_value_features import project_footprints
    camera=SimpleNamespace(image_width=100,image_height=100,FoVx=1.5707963267948966,
        FoVy=1.5707963267948966,world_view_transform=torch.eye(4),camera_center=torch.zeros(3))
    raw=torch.zeros(1,14);raw[:,0]=100;raw[:,2]=2;raw[:,7]=1
    center,cov,_,visible=project_footprints(raw,camera)
    assert abs(center[0,0].item()-2549.5)<1e-8
    assert abs(cov[0,0,0].item()-1681.55)<1e-5
    assert not visible.item()


def test_value_network_zero_anchor_shared_differences_and_ignores_auxiliary():
    from gaussian_jscc.response_value import ResponseValueNetwork
    torch.manual_seed(3)
    model=ResponseValueNetwork(6,20,view_dim=10)
    x=dict(point=torch.randn(3,6),action=torch.randn(3,20),views=torch.randn(3,2,10),
           aux=torch.zeros(3,15),available=torch.zeros(3,1))
    x['point'][:]=x['point'][0].clone();x['views'][:]=x['views'][0].clone()
    x['action'][:,12:]=x['action'][0,12:].clone();x['action'][:,:12]=0
    x['action'][:,2:6]=torch.eye(4)[torch.tensor([1,2,1])]
    x['action'][:,6:9]=torch.eye(3)[torch.tensor([1,0,0])]
    values=model.values(x)
    assert torch.count_nonzero(values[:,0])==0
    result=model(x,'positive')
    torch.testing.assert_close(result[0],-result[1])
    torch.testing.assert_close(result[2],torch.zeros(2))
    original=model(x,'keep')
    x['aux'].fill_(1e9);x['available'].fill_(1)
    torch.testing.assert_close(model(x,'keep'),original,atol=0,rtol=0)


def test_regression_loss_prefers_true_magnitude_over_confident_correct_sign():
    from gaussian_jscc.response_value import ResponseValueLearner
    target=torch.tensor([[-1.,-.2],[-2.,-.3],[-3.,-.4]])
    rates=torch.zeros(3);conditional=torch.ones(3)/3;groups=torch.zeros(3,dtype=torch.long)
    learner=object.__new__(ResponseValueLearner)
    exact=learner.losses(target,target,rates,conditional,groups,torch.ones(1),torch.ones(1,dtype=torch.long))
    exaggerated=learner.losses(target*10,target,rates,conditional,groups,torch.ones(1),torch.ones(1,dtype=torch.long))
    assert exact.item()==0 and exaggerated.item()>1


def test_value_normalization_excludes_validation_and_large_scale_corruption():
    import pytest
    from gaussian_jscc.response_value import ResponseValueNetwork
    model=ResponseValueNetwork(6,20,view_dim=10)
    features=dict(point=torch.ones(3,6),action=torch.ones(3,20),views=torch.ones(3,2,10))
    row=dict(features=features,target=torch.tensor([[1.,.1],[2.,.2],[3.,.3]]),branch='keep',validation=True)
    with pytest.raises(ValueError,match='training-only'):model.fit([row])
    assert not model.ready and torch.equal(model.component_scale,torch.ones(2)*1e-6)
    row['validation']=False;model.fit([row])
    torch.testing.assert_close(model.component_scale,torch.tensor([2.8,.28]))


def test_previously_unseen_q0_context_is_not_amplified_by_normalization():
    from gaussian_jscc.response_value import ResponseProtocol,ResponseValueNetwork
    protocol=ResponseProtocol(2)
    model=ResponseValueNetwork(protocol.point_dim,protocol.action_dim,protocol.view_dim,continuous=protocol.continuous)
    x=dict(point=torch.zeros(3,protocol.point_dim),action=torch.zeros(3,protocol.action_dim),
           views=torch.zeros(3,1,protocol.view_dim))
    x['point'][:,-3]=1
    model.fit([dict(features=x,target=torch.tensor([[1.,0.],[2.,0.],[3.,0.]]),branch='keep',validation=False)])
    new=x['point'][0].clone();new[-4:]=torch.tensor([1.,0.,0.,0.])
    normalized=(new-model.point_mean)/model.point_std
    assert normalized[-4:].abs().max()<=1
