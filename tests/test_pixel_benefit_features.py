import torch

from gaussian_jscc.pixel_benefit_features import build_group
from gaussian_jscc.pixel_benefit import integrate_density
from gaussian_jscc.pixel_local_renderer import TileScene
from gaussian_jscc.allocation_distortion import ImageDistortion


def scene_fixture():
    p = torch.tensor([[15.,15.,1.,.13,0.,.13,.8,1.,.8,.1,.2,9.],
                      [18.,17.,2.,.04,0.,.04,.9,1.,.1,.6,.3,16.]])
    return TileScene.from_projected(p, 35, 37, backend='cpu')


def test_halo_integral_matches_full_image_with_edges_and_unocclusion():
    scene = scene_fixture()
    ids = torch.tensor([0]); candidates = scene.projected[ids].repeat(4,1,1)
    candidates[0,:,7] = 0
    candidates[2,:,8] = .2
    candidates[3,:,0] = 29.; candidates[3,:,11] = 16
    target = torch.rand_like(scene.image)
    record = build_group(scene, ids, candidates, target, .01, .04, with_labels=True)
    loss = ImageDistortion('mse_ssim', .01)
    full = []
    for c in candidates:
        p=scene.projected.clone(); p[ids]=c
        im=TileScene.from_projected(p,scene.height,scene.width,backend='cpu').image
        full.append(loss.components(im,target).double())
    truth=(torch.stack(full)[1:]-full[0])*torch.tensor([1.,.01])/.04
    actual=integrate_density(record['density'], record['inputs']['valid'],35,37,.04)
    torch.testing.assert_close(actual,truth,rtol=1e-6,atol=2e-8)


def test_inference_inputs_equal_training_inputs_and_no_teacher_images():
    scene=scene_fixture(); ids=torch.tensor([0])
    candidates=scene.projected[ids].repeat(4,1,1); candidates[0,:,7]=0
    target=torch.zeros_like(scene.image)
    train=build_group(scene,ids,candidates,target,.01,1.,with_labels=True)
    infer=build_group(scene,ids,candidates,target,.01,1.,with_labels=False)
    assert 'density' not in infer
    assert set(train['inputs']) == {'image','context','context_mask','candidate','candidate_mask','valid'}
    for key in train['inputs']: torch.testing.assert_close(train['inputs'][key],infer['inputs'][key])


def test_depth_tokens_are_not_truncated():
    p=scene_fixture().projected[:1].repeat(80,1); p[:,2]=torch.arange(80)+1
    scene=TileScene.from_projected(p,35,37,backend='cpu')
    ids=torch.tensor([0]); c=p[ids].repeat(4,1,1)
    record=build_group(scene,ids,c,torch.zeros_like(scene.image),.01,1.,with_labels=False)
    assert max(int(x.sum()) for x in record['inputs']['context_mask']) == 80


def test_empty_footprint_has_no_tiles_and_zero_integrated_value():
    p=scene_fixture().projected[:1].clone(); p[:,11]=0
    scene=TileScene.from_projected(p,35,37,backend='cpu')
    record=build_group(scene,torch.tensor([0]),p.repeat(4,1,1),torch.zeros_like(scene.image),.01,1.,with_labels=True)
    assert record['inputs']['image'].shape[0] == 0
    assert integrate_density(record['density'],record['inputs']['valid'],35,37,1.).abs().sum()==0


def test_static_tile_context_is_reused_and_target_change_invalidates_cache():
    scene=scene_fixture(); ids=torch.tensor([0]); c=scene.projected[ids].repeat(4,1,1)
    target=torch.zeros_like(scene.image)
    first=build_group(scene,ids,c,target,.01,1.,with_labels=False)
    second=build_group(scene,ids,c,target,.01,1.,with_labels=False)
    assert second['cache_hits'] == len(second['tiles']) and first['cache_hits'] == 0
    target.add_(.1)
    third=build_group(scene,ids,c,target,.01,1.,with_labels=False)
    assert third['cache_hits']==0
    assert not torch.equal(second['inputs']['image'],third['inputs']['image'])


def test_radius_tail_in_halo_retains_the_original_tile_contributor():
    p=torch.tensor([[.6,8.,1.,.1,0.,.1,.99,1.,1.,0.,0.,10.]])
    scene=TileScene.from_projected(p,16,32,backend='cpu')
    c=p.repeat(4,1,1); c[0,:,7]=0
    record=build_group(scene,torch.tensor([0]),c,torch.zeros_like(scene.image),.01,1.,with_labels=True)
    slot=record['tiles'].tolist().index(1)
    assert scene.image[0,8,11]>0
    assert record['inputs']['context_mask'][slot].sum()==1
    assert record['inputs']['candidate_mask'][slot,1,0]


def test_teacher_label_batches_have_bounded_tiles(monkeypatch):
    import gaussian_jscc.pixel_benefit_features as module
    original=module.pixel_density_labels; batches=[]
    def measure(images,target,weight):
        batches.append(len(images)); return original(images,target,weight)
    monkeypatch.setattr(module,'pixel_density_labels',measure)
    p=torch.tensor([[64.,64.,1.,.001,0.,.001,.9,1.,.3,.8,.4,80.]])
    scene=TileScene.from_projected(p,130,130,backend='cpu')
    c=p.repeat(4,1,1); c[0,:,7]=0
    record=build_group(scene,torch.tensor([0]),c,torch.zeros_like(scene.image),.01,1.,with_labels=True)
    assert len(record['tiles'])>32 and max(batches)<=16
    assert len(record['density'])==len(record['tiles'])
