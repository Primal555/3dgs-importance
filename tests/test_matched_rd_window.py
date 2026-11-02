import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import torch


def fixture():
    assert importlib.util.find_spec('gaussian_jscc.matched_rd_window'), 'matched window missing'
    from gaussian_jscc.matched_rd_window import MatchedWindow
    from gaussian_jscc.direct_rd_allocation import FrozenBenefit,FrozenCandidateScheduler
    from gaussian_jscc.direct_rd_decisions import HardTierMap
    from gaussian_jscc.spatial_benefit import SpatialBenefitNetwork
    from test_successive_allocation import small_successive
    torch.set_num_threads(1);torch.manual_seed(8)
    raw,geometry,features,codec=small_successive();codec.eval().requires_grad_(False)
    transform=torch.eye(4);transform[3,2]=3
    cameras=[SimpleNamespace(image_name=str(i),image_width=16,image_height=16,FoVx=1.2,FoVy=1.2,
        world_view_transform=transform,full_proj_transform=torch.eye(4),camera_center=torch.zeros(3)) for i in range(2)]
    scene=SimpleNamespace(raw=raw,geometry=geometry,groups=[features],group_ids=[torch.arange(16).reshape(2,8)],
        degree=1,white_background=False,reference=SimpleNamespace(get=lambda c,d:torch.zeros(3,16,16,device=d)))
    q=torch.full((16,),2);q[0]=0
    network=SpatialBenefitNetwork(hidden=16)
    learner=FrozenBenefit(dict(network=network.state_dict(),hidden=16,dropout=.05,
        scales={'value':torch.ones(2),'rd':torch.tensor(2.)},updates=0),'cpu')
    reference=dict(lambda_s=.01,d_ref=.01,c_ref=1000.,side_proxy=dict(xyz_per_retained=24.,bits_per_use=2.))
    args=SimpleNamespace(allocation_batch=4,label_groups=0,snr=10.,beta=.1,neighbors=4,cpu_threads=1,accept_trials=2)
    table=HardTierMap(q)
    window=MatchedWindow(codec,scene,table,FrozenCandidateScheduler(scene.group_ids,seed=42),reference,args)
    return window,codec,scene,learner,cameras,q


def test_two_conditions_cache_and_exact_labels_reuse_all_projections():
    window,codec,scene,learner,cameras,q=fixture()
    ids=torch.tensor([0,9]);seeds=[1042,1043]
    costs=window.score(ids,learner,seeds,cameras)
    assert costs.shape==(2,3) and len(window.contexts)==2
    with patch.object(codec,'forward_tier_batches',side_effect=AssertionError('decoded again')), \
         patch.object(window,'candidate_batches',side_effect=AssertionError('candidate reconstructed again')):
        true=window.exact_costs(ids,seeds,cameras)
    assert true.shape==(2,3) and torch.isfinite(true).all()
    window.table.set_map(q.clone());window.set_condition(1042)
    window.prepare_context(cameras)
    assert len(window.contexts)==1 and not window.candidates


def test_cached_screening_matches_existing_window_and_does_not_recapture():
    window,codec,scene,learner,cameras,q=fixture()
    from gaussian_jscc.batched_spatial_window import BatchedSpatialWindow
    from gaussian_jscc.direct_rd_allocation import NoReplay,FrozenCandidateScheduler
    from gaussian_jscc.cli import seed_all
    regular=BatchedSpatialWindow(codec,scene,window.table,FrozenCandidateScheduler(scene.group_ids,seed=42),
                                  window.reference,window.args)
    seed_all(1042);expected=regular.collect(1,cameras,learner,NoReplay(),False)
    window.set_condition(1042);got=window.collect(1,cameras,learner,NoReplay(),False)
    assert torch.equal(got['tier_ids'],expected['tier_ids'])
    torch.testing.assert_close(got['tier_costs'],expected['tier_costs'],atol=1e-12,rtol=0)
    with patch('gaussian_jscc.pixel_benefit_window.capture',side_effect=AssertionError('baseline captured again')):
        window.collect(2,cameras,learner,NoReplay(),False)
    assert window.cache_hits==1 and window.cache_misses==1


def test_joint_tile_rd_matches_full_decode_including_prior_commit_and_q0_restore():
    window,codec,scene,learner,cameras,q=fixture()
    from gaussian_jscc.pixel_local_renderer import capture
    from gaussian_jscc.modular_evaluation import decode_path
    from gaussian_jscc.cli import seed_all
    from gaussian_jscc.allocation_distortion import ImageDistortion
    ids=torch.tensor([0,9]);seeds=[1042,1043]
    window.score(ids,learner,seeds,cameras)
    class Meter:
        def details(self,v): return dict(total_channel_uses=int(v.sum())*10,tier_counts=torch.bincount(v,minlength=4).tolist())
    meter=Meter();base=window.measure(q,seeds,cameras,meter)
    first=q.clone();first[0]=1
    second=first.clone();second[9]=0
    for candidate in [first,second]:
        with patch.object(codec,'forward_tier_batches',side_effect=AssertionError('joint check decoded again')):
            local=window.measure(candidate,seeds,cameras,meter)
        values=[];objective=ImageDistortion('mse_ssim',.01)
        for seed in seeds:
            seed_all(seed)
            received,_=decode_path(codec,scene,[candidate.reshape(2,8)],'awgn',10.)
            for camera in cameras:
                image=capture(received,camera,1,torch.ones(len(received)),backend='cpu').image
                target=scene.reference.get(camera,'cpu')
                values.append(float(objective.combine(objective.components(image,target)))/.01)
        expected=sum(values)/len(values)+.1*meter.details(candidate)['total_channel_uses']/1000.
        assert abs(local['rd_score']-expected)<1e-6
    assert base['tier_counts']==[1,0,15,0]
