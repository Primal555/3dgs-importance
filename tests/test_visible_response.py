import torch
import pytest
from types import SimpleNamespace

from gaussian_jscc.visible_response import supervision_scales, VisibleResponseNetwork, VisibleResponseLearner
from gaussian_jscc.visible_response_features import layer_context, transmittance_bounds


def record(values, validation=False):
    return dict(branch='keep', target=torch.tensor(values,dtype=torch.float64),validation=validation)


def test_scales_separate_common_and_pair_and_reject_validation():
    rows=[record([[100.,1.],[101.,1.01],[102.,1.02]]),record([[200.,2.],[202.,2.02],[204.,2.04]])]
    scales=supervision_scales(rows)
    assert torch.all(scales['common']>scales['pair']*20)
    assert torch.all(scales['pair']>scales['centered'])
    with pytest.raises(ValueError):supervision_scales(rows+[record([[1,1]]*3,True)])


def test_common_offset_cancels_from_centered_supervision():
    net=VisibleResponseNetwork(4,16,view_dim=5)
    net.common_scale.fill_(100);net.centered_scale.fill_(1);net.pair_scale.fill_(1)
    learner=VisibleResponseLearner(net)
    target=torch.zeros(3,2);pred=torch.full((3,2),10.,requires_grad=True)
    groups=torch.zeros(3,dtype=torch.long);branches=torch.ones(1,dtype=torch.long)
    loss=learner.losses(pred,target,torch.zeros(3),torch.ones(3)/3,groups,torch.ones(1),branches)
    assert loss.item()<.02
    loss.sum().backward()
    assert torch.allclose(pred.grad[0],pred.grad[1])
    pred2=torch.tensor([[-1.,0.],[0.,0.],[1.,0.]],requires_grad=True)
    loss2=learner.losses(pred2,target,torch.zeros(3),torch.ones(3)/3,groups,torch.ones(1),branches)
    assert loss2.item()>loss.item()*5


def test_depth_layer_bounds_include_front_occluder_and_closed_query():
    camera=SimpleNamespace(world_view_transform=torch.eye(4),image_width=1,image_height=1)
    scene=torch.zeros(3,15);scene[:,2]=torch.tensor([1.,2.,3.]);scene[:,-1]=torch.tensor([1.,1.,0.])
    def render(raw,camera,degree,white,existence,colors):
        # Literal two-alpha front-to-back compositor, third point is closed.
        return (colors[0]*.5+colors[1]*.25)[:,None,None]
    ctx=layer_context(scene,camera,0,render)
    coordinates=torch.zeros(1,1,1,2)
    bounds=transmittance_bounds(ctx,torch.tensor([[3.]]),coordinates,camera)
    assert bounds.shape==(1,1,2,1,1)
    assert torch.allclose(bounds.flatten(),torch.tensor([.25,.5]))
    assert ctx['render_calls']==4


def test_empty_scene_visibility_is_transparent():
    camera=SimpleNamespace(world_view_transform=torch.eye(4),image_width=1,image_height=1)
    scene=torch.zeros(2,15)
    def render(*args,**kwargs):return torch.zeros(3,1,1)
    ctx=layer_context(scene,camera,0,render)
    got=transmittance_bounds(ctx,torch.ones(1,1),torch.zeros(1,1,1,2),camera)
    assert torch.equal(got,torch.ones_like(got))


def test_block_response_contains_signed_neighbor_change_and_q0_visibility():
    from gaussian_jscc.visible_response_features import visible_block_views,LAYERS
    camera=SimpleNamespace(world_view_transform=torch.eye(4),image_width=9,image_height=9,
                           FoVx=1.,FoVy=1.,camera_center=torch.zeros(3))
    raw=torch.zeros(1,4,2,14);raw[...,2]=2;raw[...,4:7]=-2;raw[...,7]=1
    tiers=torch.ones(1,4,2,dtype=torch.long);tiers[:,0,0]=0
    # Only a neighbor's decoded red changes. Selected Gaussian stays identical.
    raw[:,1,1,11]=1;raw[:,2,1,11]=-1
    ctx=dict(edges=torch.linspace(1.,3.,LAYERS-1),boundary=torch.ones(LAYERS+1,9,9))
    baseline=[(torch.zeros(3,9,9),torch.zeros(3,9,9),None)]
    got=visible_block_views(raw,tiers,torch.tensor([0]),[camera],baseline,[ctx],0)
    assert got.shape==(1,1,272) and torch.isfinite(got).all()
    maps=got[0,0,:162].reshape(3,2,3,3,3)
    # q1/q2 difference isolates the changed neighbor from the shared selected point.
    assert (maps[0,:,0]>maps[1,:,0]).any()
    assert torch.count_nonzero(got[...,216:270])>0
    dark=dict(ctx,boundary=ctx['boundary']*.1)
    shaded=visible_block_views(raw,tiers,torch.tensor([0]),[camera],baseline,[dark],0)
    assert torch.allclose(shaded[...,:162],got[...,:162]*.1,atol=1e-7)
    assert torch.allclose(shaded[...,216:270],got[...,216:270]*.1,atol=1e-7)


def test_closed_and_offscreen_blocks_are_finite():
    from gaussian_jscc.visible_response_features import visible_block_views,LAYERS
    camera=SimpleNamespace(world_view_transform=torch.eye(4),image_width=9,image_height=9,
                           FoVx=1.,FoVy=1.,camera_center=torch.zeros(3))
    raw=torch.zeros(2,4,3,14);raw[...,2]=-2;raw[...,7]=1
    ctx=dict(edges=torch.linspace(1.,3.,LAYERS-1),boundary=torch.ones(LAYERS+1,9,9))
    baseline=[(torch.ones(3,9,9),torch.zeros(3,9,9),None)]
    got=visible_block_views(raw,torch.zeros(2,4,3,dtype=torch.long),torch.tensor([0,1]),[camera],baseline,[ctx],0)
    assert torch.equal(got,torch.zeros_like(got))


def test_visible_features_and_prediction_are_fixed_before_intervention_labels():
    from test_response_value_window import run_response_kernel
    result,first,_=run_response_kernel(closed=True,visible=True)
    _,second,_=run_response_kernel(closed=True,label_scale=2,visible=True)
    assert result['response_q0_candidate_rows']==4 and result['visibility_context_render_calls']==4
    changed=False
    for a,b in zip(first.records,second.records):
        assert a['features']['views'].shape[-1]==949
        for key in a['features']:torch.testing.assert_close(a['features'][key],b['features'][key],rtol=0,atol=0)
        torch.testing.assert_close(a['prediction'],b['prediction'],rtol=0,atol=0)
        torch.testing.assert_close(b['target'],a['target']*2,rtol=0,atol=0)
        changed|=not torch.equal(a['target'],b['target'])
    assert changed
