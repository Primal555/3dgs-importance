"""CPU mathematical checks for fixed-context, whole-block relation labels."""
import importlib.util
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def teacher_class():
    assert importlib.util.find_spec('gaussian_jscc.relation_benefit_teacher') is not None, 'relation teacher is missing'
    from gaussian_jscc.relation_benefit_teacher import RelationTeacher
    return RelationTeacher


def fixture():
    from test_successive_allocation import small_successive
    from gaussian_jscc.pixel_local_renderer import capture
    from utils.graphics_utils import getProjectionMatrix
    raw, geometry, features, model = small_successive()
    model.eval().requires_grad_(False)
    # Packet order and group order deliberately differ, with neither equal to PLY order.
    order = torch.tensor([7, 1, 12, 4, 15, 2, 10, 0, 9, 6, 14, 3, 8, 5, 13, 11])
    grouped = order.roll(5)
    class Camera(SimpleNamespace):
        @property
        def original_image(self):
            raise AssertionError('photographs must never enter relation teacher')
    cameras = [Camera(image_name=f'train_{v}', image_height=17, image_width=19,
        FoVx=math.pi/2, FoVy=math.pi/2, world_view_transform=torch.eye(4),
        full_proj_transform=getProjectionMatrix(.01, 100., math.pi/2, math.pi/2).T.contiguous(),
        camera_center=torch.tensor([float(v)*.03, 0., 0.])) for v in range(4)]
    class Reference:
        target = 'source'
        def __init__(self):
            self.calls = 0
        def get(self, camera, device):
            self.calls += 1
            return capture(raw[order].to(device), camera, 1, torch.ones(16, device=device), backend='cpu').image
    scene = SimpleNamespace(raw=raw[order], order=order, groups=[features.reshape(16, -1)[grouped].reshape(2, 8, -1)],
        group_ids=[grouped.reshape(2, 8)], geometry=geometry, degree=1, white_background=False,
        reference=Reference(), fingerprint='fixture-ply', spec={'name': 'fixture'})
    return model, scene, cameras, dict(lambda_s=.02, d_ref=.03)


def test_repeated_labels_keep_fixed_trials_contexts_and_float64_with_original_ply_ids(monkeypatch):
    model, scene, cameras, reference = fixture()
    Teacher = teacher_class()
    before = torch.get_rng_state().clone()
    teacher = Teacher(model, scene, cameras, reference, trials=2, seed=9, candidate_batch=2)
    decode_calls = []
    original = model.forward_tier_batches
    def counted(*args, **kwargs):
        decode_calls.append(args[0].shape[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'forward_tier_batches', counted)
    first = teacher.collect([0, 9], step=4)
    first_decode_calls = len(decode_calls)
    captures = teacher.captures
    counts = (scene.reference.calls, len(teacher.entries))
    second = teacher.collect([9, 0], step=19)
    assert counts == (4, 2)
    assert teacher.captures is captures
    assert scene.reference.calls == 4
    assert len(decode_calls) == first_decode_calls
    assert torch.equal(before, torch.get_rng_state())
    for a, b in zip(first, reversed(second)):
        assert a['source'] == b['source']
        assert a['step'] == 4 and b['step'] == 19
        assert a['noise_trials'] == 2
        assert a['values'].shape == (4, 3, 2)
        assert a['values'].dtype == torch.float64
        torch.testing.assert_close(a['values'], b['values'], atol=0., rtol=0.)
        assert not a['values'].requires_grad
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not model.training
    inverse = torch.argsort(scene.order)
    assert teacher.positions[0, 3] == inverse[0]
    assert teacher.positions[9, 3] == inverse[9]
    # Returned replay data owns its storage.
    first[0]['values'].fill_(100)
    assert not torch.equal(teacher.collect([0])[0]['values'], first[0]['values'])
    # One unseen source needs three alternate block decodes per trial, with all
    # full-q2 scene decodes and camera captures retained from the first call.
    teacher.collect([0, 3])
    assert len(decode_calls) == first_decode_calls + 6
    assert decode_calls[-6:] == [1] * 6
    assert scene.reference.calls == 4
    assert teacher.captures is captures


def test_four_trials_equal_independent_full_batch_codec_and_float64_full_image_components():
    from gaussian_jscc.allocation_distortion import ImageDistortion
    from gaussian_jscc.budgeted_allocation_kernel import raw_batches
    from gaussian_jscc.pixel_local_renderer import capture
    model, scene, cameras, reference = fixture()
    teacher = teacher_class()(model, scene, cameras, reference, trials=4, seed=381, candidate_batch=3)
    item = teacher.collect([0])[0]
    batch, row, col, _ = teacher.positions[0].tolist()
    objective = ImageDistortion('mse_ssim', reference['lambda_s'])
    independent = []
    neighbor_change = []
    for trial in range(4):
        f, ids, q, y, actual, noise = teacher.entries[trial][batch]
        states = []
        alternatives, tiers = teacher.forward_candidates([0], trial)[0][1:]
        for tier in range(4):
            changed_q = q.clone()
            changed_q[row, col] = tier
            decoded = model.forward_tier_batches(f, f[..., :3], F.one_hot(changed_q, 4).to(f),
                10., 'awgn', paired_noise=True, _channel_noise=noise)[0]
            torch.testing.assert_close(alternatives[0, tier], decoded[row], atol=2e-5, rtol=2e-5)
            assert int(tiers[0, tier, col]) == tier
            raw = raw_batches(decoded, scene.geometry, model)
            morton = torch.empty_like(scene.raw)
            masks = torch.empty(len(morton))
            for block in range(len(ids)):
                slots = teacher.positions[ids[block], 3]
                morton[slots] = raw[block]
                masks[slots] = (changed_q[block] > 0).float()
            states.append(torch.stack([objective.components(
                capture(morton, camera, 1, masks, backend='cpu').image,
                teacher.targets[v]).double() for v, camera in enumerate(cameras)]))
            if tier == 0:
                others = torch.arange(q.shape[1]) != col
                neighbor_change.append(float((decoded[row, others] - actual[row, others]).abs().max()))
        components = torch.stack(states, 1)
        independent.append((components[:, 1:] - components[:, :1]) * torch.tensor([1., .02], dtype=torch.float64) / .03)
    expected = torch.stack(independent).mean(0)
    torch.testing.assert_close(item['values'], expected, atol=2e-9, rtol=3e-5)
    assert max(neighbor_change) > 1e-5
    assert not torch.equal(teacher.entries[0][0][-1], teacher.entries[1][0][-1])


def test_identity_tracks_camera_geometry_names_source_and_fixed_noise_protocol():
    model, scene, cameras, reference = fixture()
    teacher = teacher_class()(model, scene, cameras, reference, seed=42)
    identity = teacher.identity()
    json.dumps(identity, allow_nan=False)
    assert identity['noise_protocol'] == 'fixed_pair_trials'
    assert identity['view_names'] == ['train_0', 'train_1', 'train_2', 'train_3']
    assert identity['view_weights'] == [.25] * 4
    assert identity['source_order'] == 'original_ply'
    cameras[0].image_name = 'renamed'
    renamed = teacher_class()(model, scene, cameras, reference).identity()
    assert renamed != identity
    scene.raw = scene.raw.clone()
    scene.raw[0, 3] += .1
    assert teacher_class()(model, scene, cameras, reference).identity() != renamed
    scene.white_background = True
    assert teacher_class()(model, scene, cameras, reference).identity()['white_background'] is True


def test_candidate_batch_boundaries_preserve_all_labels_and_noiseless_bank_independence():
    model, scene, cameras, reference = fixture()
    Teacher = teacher_class()
    first = Teacher(model, scene, cameras, reference, trials=2, seed=87, candidate_batch=1)
    second = Teacher(model, scene, cameras, reference, trials=2, seed=87, candidate_batch=3)
    sources = [0, 3, 7, 9]
    narrow = first.collect(sources)
    wide = second.collect(sources)
    for a, b in zip(narrow, wide):
        torch.testing.assert_close(a['values'], b['values'], atol=2e-9, rtol=2e-5)
    # Changing the teacher seed changes only this supervision experiment;
    # the original scene input features remain untouched.
    features = [f.clone() for f in scene.groups]
    other = Teacher(model, scene, cameras, reference, trials=2, seed=88).collect([0])[0]
    assert not torch.equal(other['values'], narrow[0]['values'])
    for before, after in zip(features, scene.groups):
        assert torch.equal(before, after)


def test_invalid_sources_and_cpu_preflight_fail_explicitly():
    model, scene, cameras, reference = fixture()
    teacher = teacher_class()(model, scene, cameras, reference)
    with pytest.raises(ValueError, match='source'):
        teacher.collect([-1])
    with pytest.raises(ValueError, match='source'):
        teacher.collect([16])
    with pytest.raises(RuntimeError, match='CUDA'):
        teacher.verify()
    assert scene.reference.calls == 0
    with pytest.raises(ValueError, match='frozen'):
        teacher_class()(model.train(), scene, cameras, reference)


@pytest.mark.parametrize('background', ['q1', 'q3', 'mixed'])
def test_context_labels_match_independent_full_render_and_retention_masks(background):
    from gaussian_jscc.allocation_distortion import ImageDistortion
    from gaussian_jscc.budgeted_allocation_kernel import raw_batches
    from gaussian_jscc.pixel_local_renderer import capture
    model, scene, cameras, reference = fixture()
    tiers = (torch.full((16,), 1 if background == 'q1' else 3) if background != 'mixed'
             else torch.arange(16).remainder(4))
    teacher = teacher_class()(model, scene, cameras[:2], reference, trials=2, seed=381,
                              background_tiers=tiers, context_id=3, context_name=background)
    source = 0  # Deleted in mixed context; intervention must restore this point too.
    item = teacher.collect([source])[0]
    assert item['context'] == 3 and item['context_name'] == background
    objective = ImageDistortion('mse_ssim', reference['lambda_s'])
    batch,row,col,_ = teacher.positions[source].tolist()
    independent=[]
    for trial in range(2):
        f,ids,q,_,_,noise = teacher.entries[trial][batch]
        assert torch.equal(q, tiers[ids])
        states=[]
        for tier in range(4):
            changed_q=q.clone();changed_q[row,col]=tier
            decoded=model.forward_tier_batches(f,f[...,:3],F.one_hot(changed_q,4).to(f),
                10.,'awgn',paired_noise=True,_channel_noise=noise)[0]
            raw=raw_batches(decoded,scene.geometry,model)
            morton=torch.empty_like(scene.raw);masks=torch.empty(len(morton))
            for block in range(len(ids)):
                slots=teacher.positions[ids[block],3]
                morton[slots]=raw[block];masks[slots]=(changed_q[block]>0).float()
            states.append(torch.stack([objective.components(
                capture(morton,camera,1,masks,backend='cpu').image,
                teacher.targets[v]).double() for v,camera in enumerate(cameras[:2])]))
        parts=torch.stack(states,1)
        independent.append((parts[:,1:]-parts[:,:1])*torch.tensor([1.,.02],dtype=torch.float64)/.03)
    torch.testing.assert_close(item['values'],torch.stack(independent).mean(0),atol=2e-9,rtol=3e-5)
    expected=(tiers[scene.order]>0).float()
    assert torch.equal(teacher.background_masks.cpu(),expected)


def test_released_capture_reuses_background_decode_and_labels(monkeypatch):
    model,scene,cameras,reference=fixture()
    teacher=teacher_class()(model,scene,cameras[:1],reference,trials=1)
    teacher.collect([0]);entries=teacher.entries;backgrounds=teacher._backgrounds
    def fail(*args,**kwargs):raise AssertionError('cached background must not decode again')
    monkeypatch.setattr(model,'forward_tier_batches',fail)
    teacher.release_device()
    assert not teacher.captures and not teacher.targets and not teacher._prepared
    teacher._prepare()
    assert teacher.entries is entries and teacher._backgrounds is backgrounds
    assert len(teacher.captures)==1 and torch.equal(teacher.collect([0])[0]['values'],teacher._labels[0])


def test_context_manager_keeps_labels_separate_and_one_capture_set():
    from gaussian_jscc.relation_benefit_teacher import RelationContextTeacher
    model,scene,cameras,reference=fixture()
    contexts=[dict(name=f'q{k}',tiers=torch.full((16,),k,dtype=torch.long)) for k in (1,2,3)]
    contexts.append(dict(name='mixed',tiers=torch.arange(16).remainder(4)))
    teacher=RelationContextTeacher(model,scene,cameras[:1],reference,contexts,trials=1)
    a=teacher.collect([0],context=0)[0];b=teacher.collect([0],context=3)[0]
    assert a['context']==0 and b['context']==3
    assert not torch.equal(a['values'],b['values'])
    assert teacher.cached_groups==2
    assert not teacher.teachers[0].captures and teacher.teachers[3].captures
    assert torch.equal(teacher.teachers[0].entries[0][0][-1],teacher.teachers[3].entries[0][0][-1])
    torch.testing.assert_close(teacher.collect([0],context=0)[0]['values'],a['values'])
    assert teacher.teachers[0].captures == [] and teacher.teachers[3].captures == []
    identity=teacher.identity()
    assert len(identity['contexts'])==4
    assert identity['contexts'][0]['background_tiers_sha256']!=identity['contexts'][3]['background_tiers_sha256']
    assert identity['background_protocol']=='four_fixed_contexts_v1'


@pytest.mark.parametrize('tiers', [torch.ones(16),torch.ones(16,dtype=torch.bool),
    torch.full((16,),4),torch.full((15,),2)])
def test_invalid_background_tier_identity_is_rejected(tiers):
    model,scene,cameras,reference=fixture()
    with pytest.raises(ValueError,match='background_tiers'):
        teacher_class()(model,scene,cameras,reference,background_tiers=tiers)


def test_multi_context_identity_rejects_wrong_order_or_nonuniform_named_context():
    from gaussian_jscc.relation_benefit_teacher import RelationContextTeacher
    from gaussian_jscc.relation_benefit_context import background_configurations
    model,scene,cameras,reference=fixture()
    contexts=background_configurations(16,42)
    with pytest.raises(ValueError,match='q1, q2, q3, mixed'):
        RelationContextTeacher(model,scene,cameras,reference,list(reversed(contexts)))
    contexts[0]['tiers'][0]=3
    with pytest.raises(ValueError,match='uniform background'):
        RelationContextTeacher(model,scene,cameras,reference,contexts)
