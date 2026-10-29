from types import SimpleNamespace

import torch


def test_candidate_decoding_changes_one_point_and_reencodes_q0_with_paired_noise():
    from gaussian_jscc.response_value_window import decode_candidates
    from gaussian_jscc.budgeted_allocation_kernel import rng_state,raw_batches
    from gaussian_jscc.data import Geometry
    class Codec:
        attr_mean=torch.zeros(11);attr_std=torch.ones(11)
        def forward_tier_batches(self,f,xyz,selection,snr,kind,**kwargs):
            assert '_semantic_y' not in kwargs
            q=selection.argmax(-1)
            noise=torch.randn_like(f)
            # Presence-dependent neighbor reconstruction exposes stale semantic caches.
            result=f+noise*.01+(q>0).sum(-1)[:,None,None]*.1
            result[...,0]=q.float()
            return result,None
    model=Codec();geometry=Geometry(torch.zeros(3),torch.ones(3),16)
    f=torch.zeros(2,3,14);q=torch.tensor([[0,2,1],[1,0,3]])
    columns=torch.tensor([0,1]);state=rng_state(torch.device('cpu'))
    actual=model.forward_tier_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4),10,'awgn')[0]
    outputs=list(decode_candidates(model,f,q,columns,actual,state,geometry,10,'awgn'))
    assert len(outputs)==4
    torch.testing.assert_close(outputs[0][1],raw_batches(actual,geometry,model),rtol=0,atol=0)
    # Removing and restoring a point changes another point's decoded attributes.
    assert not torch.equal(outputs[0][1][0,1,3:],outputs[1][1][0,1,3:])
    # Under positive tiers the same common noise and context are preserved.
    torch.testing.assert_close(outputs[1][1][0,1,3:],outputs[3][1][0,1,3:],rtol=0,atol=0)
    for tier,(alternative,raw) in enumerate(outputs):
        assert alternative.tolist()==[[tier,2,1],[1,tier,3]]


def test_coverage_scheduler_includes_q0_and_rotates_without_repeat():
    from gaussian_jscc.response_value_window import ResponseScheduler
    ids=[torch.arange(12).reshape(3,4)]
    scheduler=ResponseScheduler(ids,rounds=2)
    rounds=scheduler.response_rounds(1)
    picked=[]
    for columns in rounds:
        picked.extend(ids[0][torch.arange(3),columns[0]].tolist())
    assert len(set(picked))==6
    scheduler.record(torch.tensor(picked),1)
    next_round=scheduler.response_rounds(2)
    next_ids=torch.cat([ids[0][torch.arange(3),cols[0]] for cols in next_round])
    assert set(next_ids.tolist()).isdisjoint(picked)


def test_patch_context_covers_footprint_beyond_center_and_zeros_offscreen():
    from gaussian_jscc.response_value_features import patch_views
    camera=SimpleNamespace(image_width=21,image_height=21,FoVx=1.5707963267948966,
        FoVy=1.5707963267948966,world_view_transform=torch.eye(4),camera_center=torch.zeros(3))
    raw=torch.zeros(2,4,14);raw[:,:,2]=2;raw[:,:,4:7]=torch.tensor(.3).log();raw[:,:,7]=1
    raw[1,:,0]=100
    image=torch.zeros(3,21,21);target=image.clone();target[:,10,13]=1
    values=patch_views(raw,[camera],[(image,target,None)],[torch.zeros(3,21,21)],0,1.)
    assert values.shape==(2,1,677)
    assert values[0,0,3*49:6*49].abs().sum()>0
    assert torch.count_nonzero(values[1])==0


def test_extreme_thin_rotated_and_behind_camera_footprints_remain_finite():
    from gaussian_jscc.response_value_features import patch_views
    camera=SimpleNamespace(image_width=21,image_height=21,FoVx=1.2,FoVy=1.2,
        world_view_transform=torch.eye(4),camera_center=torch.zeros(3))
    raw=torch.zeros(2,4,14);raw[:,:,2]=.21;raw[1,:,2]=-1
    raw[:,:,4:7]=torch.tensor([10.,-20.,-20.]);raw[:,:,7]=.9238795;raw[:,:,10]=.3826834
    image=torch.zeros(3,21,21)
    values=patch_views(raw,[camera],[(image,image,None)],[image],0,1.)
    assert torch.isfinite(values).all()
    assert torch.count_nonzero(values[1])==0


def test_high_resolution_random_thin_rotations_have_finite_response_patches():
    from gaussian_jscc.response_value_features import patch_views
    torch.set_num_threads(1);torch.manual_seed(10)
    camera=SimpleNamespace(image_width=1920,image_height=1080,FoVx=1.2,FoVy=1.2,
        world_view_transform=torch.eye(4),camera_center=torch.zeros(3))
    raw=torch.zeros(1000,4,14);raw[:,:,2]=.21
    raw[:,:,4:7]=torch.tensor([3.,-15.,-15.])
    raw[:,:,7:11]=torch.nn.functional.normalize(torch.randn(1000,4),dim=-1)[:,None]
    image=torch.zeros(3,1080,1920)
    values=patch_views(raw,[camera],[(image,image,None)],[image],0,1.)
    assert torch.isfinite(values).all()


def run_response_kernel(closed=False,label_scale=1):
    from unittest.mock import patch
    from test_successive_allocation import small_successive
    from test_mask_gradient import Reference,toy_mask_render
    from gaussian_jscc.allocation_distortion import ComponentRenderTask,ImageDistortion
    from gaussian_jscc.response_value import ResponseProtocol,ResponseValueNetwork,ResponseValueLearner,ResponseMirrorTable
    from gaussian_jscc.response_value_window import ResponseScheduler,ResponseValueWindow
    from gaussian_jscc.budgeted_allocation_kernel import budgeted_mask_step
    torch.set_num_threads(1);torch.manual_seed(18)
    _,geometry,f,model=small_successive();model.eval().requires_grad_(False)
    ids=[torch.arange(16).reshape(2,8)];table=ResponseMirrorTable(16)
    if closed:table.keep_logits[:,0]=100;table.keep_logits[:,1]=-100
    scheduler=ResponseScheduler(ids)
    protocol=ResponseProtocol(len(model.attr_mean))
    network=ResponseValueNetwork(protocol.point_dim,protocol.action_dim,protocol.view_dim)
    learner=ResponseValueLearner(network,capacity=64,fit_windows=1,mask_fraction=0,mask_weight=0)
    window=ResponseValueWindow(learner,scheduler,finite_samples=6)
    transform=torch.eye(4);transform[3,2]=3
    camera=SimpleNamespace(image_width=4,image_height=4,FoVx=1.2,FoVy=1.2,
        world_view_transform=transform,camera_center=torch.zeros(3),factor=1.,source_count=16,image_name='a')
    task=ComponentRenderTask([camera],Reference(),1,distortion=ImageDistortion('mse',0))
    original_delta=task.delta_components
    if label_scale!=1:task.delta_components=lambda scene:original_delta(scene)*label_scale
    def render(raw,camera,degree,white=False,existence=None,colors=None):
        assert not torch.is_grad_enabled()
        if colors is not None:
            return ((colors*existence[:,None]).mean(0))[:,None,None].expand(3,4,4)
        return toy_mask_render(raw,camera,degree,white,existence)
    with patch('gaussian_jscc.rendering.render',side_effect=render):
        result=budgeted_mask_step(model,table,[f],ids,geometry,task,scheduler,10,'awgn',d_ref=.1,
            price=.1,rate_normalizer=1000,side_proxy=dict(xyz_per_retained=1.,bits_per_use=2.,fixed_uses=8.,step=1),timings=None,
            benefit=window)
    return result,window,table


def test_closed_points_receive_complete_paired_labels_and_only_visited_points_update():
    result,window,table=run_response_kernel(closed=True)
    assert result['response_q0_candidate_rows']==4 and result['benefit_exact_render_calls']==6
    assert result['render_backward_calls']==0
    before=table.keep_logits.clone();tiers=table.logits.clone()
    # Set a deterministic direction to test the sparse update contract.
    costs=result['keep_costs'];costs[result['tier_ids'],1]=-.001
    table.update(costs,result['tier_ids'],result['tier_costs'],.1,.1,16)
    untouched=torch.ones(16,dtype=torch.bool);untouched[result['tier_ids']]=False
    torch.testing.assert_close(before[untouched],table.keep_logits[untouched],atol=0,rtol=0)
    torch.testing.assert_close(tiers[untouched],table.logits[untouched],atol=0,rtol=0)
    assert not torch.equal(before[~untouched],table.keep_logits[~untouched])
    groups={r['source']:r for r in window.records if r['branch']=='keep'}
    assert len(groups)==2
    for record in window.records:
        assert 'first_order' not in record
        # The current sampled state is context, separate from a directed-action
        # selector which may name another origin when deriving paired labels.
        torch.testing.assert_close(record['features']['point'][0,-4:],torch.tensor([1.,0.,0.,0.]))
        if record['branch']=='positive':
            keep=groups[record['source']]['target']
            expected=keep[record['new'][0]-1]-keep[record['old']-1]
            torch.testing.assert_close(record['target'][0],expected,rtol=0,atol=0)


def test_uniform_positive_candidates_have_exact_zero_current_state_difference():
    result,window,_=run_response_kernel()
    assert result['candidate_codec_calls']==6
    for record in window.records:
        if record['branch']=='keep' and record['old']>0:
            # Absolute values are relative to q0. Returning to the current tier
            # is represented by the same complete group, never a second label.
            assert record['target'].shape==(3,2)
            assert torch.isfinite(record['features']['views']).all()


def test_true_intervention_labels_cannot_change_frozen_prediction_inputs():
    _,first,_=run_response_kernel(closed=True)
    _,second,_=run_response_kernel(closed=True,label_scale=2)
    changed=False
    for a,b in zip(first.records,second.records):
        for key in a['features']:
            torch.testing.assert_close(a['features'][key],b['features'][key],rtol=0,atol=0)
        torch.testing.assert_close(a['prediction'],b['prediction'],rtol=0,atol=0)
        torch.testing.assert_close(b['target'],a['target']*2,rtol=0,atol=0)
        changed|=not torch.equal(a['target'],b['target'])
    assert changed
