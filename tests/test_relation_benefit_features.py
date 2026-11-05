"""Fixed-cache row identities, finite candidate protocol and coverage graphs."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from gaussian_jscc.data import Geometry
from gaussian_jscc.relation_benefit_features import (
    build_projection_graph, build_spatial_graph, prepare_bank,
)


def test_spatial_graph_crosses_rows_and_excludes_coincident_self():
    xyz = torch.tensor([[0., 0, 0], [10, 0, 0], [0, 0, 0], [1, 0, 0]])
    graph = build_spatial_graph(xyz, 6, 1)
    assert graph.shape == (4, 6) and graph.dtype == torch.int64
    for row, neighbors in enumerate(graph):
        assert row not in neighbors
        assert set(neighbors[:3].tolist()) == set(range(4)) - {row}
        assert neighbors[3:].tolist() == [-1, -1, -1]
    assert graph[1, 0] == 3


def test_projection_graph_keeps_distant_large_overlap_and_both_depth_sides():
    # [u,v,depth,radius]: the large disk overlaps the root despite its far center.
    points = torch.tensor([[0., 0, 5, 2], [1, 0, 4, 2], [1, 0, 3, 2],
                           [1, 0, 6, 2], [20, 0, 7, 21], [50, 0, 1, 1]])
    graph = build_projection_graph(points, 4, 1)
    assert set(graph[0].tolist()) == {1, 2, 3, 4}
    assert (graph[5] == -1).all()
    assert all(i not in row for i, row in enumerate(graph))


def test_projection_graph_front_back_quota_prevents_near_foreground_collapse():
    points = torch.tensor([[0., 0, 5, 10], [.01, 0, 1, 10], [.02, 0, 2, 10],
                           [.03, 0, 3, 10], [.5, 0, 6, 10], [.6, 0, 7, 10]])
    chosen = build_projection_graph(points, 4, 1)[0].tolist()
    assert len(set(chosen) & {1, 2, 3}) == 2
    assert {4, 5}.issubset(chosen)


@pytest.mark.parametrize('count', [0, 1, 2])
def test_graph_empty_small_and_zero_radius_padding(count):
    xyz = torch.zeros(count, 3)
    graph = build_spatial_graph(xyz, 3, 1)
    assert graph.shape == (count, 3)
    projected = torch.zeros(count, 12)
    assert (build_projection_graph(projected, 3, 1) == -1).all()
    if count:
        projected[:, 11] = 2
        projected[:, 2] = 1
        valid = build_projection_graph(projected, 3, 1)
        assert (valid[:, max(count-1, 0):] == -1).all()


def test_graph_rejects_nonfinite_coordinates():
    with pytest.raises(ValueError, match='finite'):
        build_spatial_graph(torch.tensor([[float('nan'), 0, 0]]), 2, 1)
    with pytest.raises(ValueError, match='finite'):
        build_projection_graph(torch.tensor([[0, 0, 1, float('inf')]]), 2, 1)


class ToyConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self)


class ToyCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer('attr_mean', torch.zeros(11))
        self.register_buffer('attr_std', torch.ones(11))
        self.cfg = ToyConfig(rates=(0, 8, 24, 48), position_delivery='quantized',
                             position_bits=16, block_size=2, sh_degree=0)

    def set_scene_geometry(self, geometry):
        self.geometry = geometry

    def forward_tier_batches(self, features, xyz, tier_weights, snr, kind,
                             paired_noise=False, **kwargs):
        # Any noisy/clean-identity route fails before producing misleading features.
        assert snr == 10. and kind == 'none'
        assert paired_noise is True
        tier = tier_weights.argmax(-1)
        output = features.clone()
        output[..., 11] -= tier.float() * .2
        output[..., 12] += tier.float() * .1
        return output, {}


def toy_inputs():
    model = ToyCodec()
    raw = torch.zeros(3, 14)
    raw[:, 0] = torch.tensor([-1., 0., 2.])
    raw[:, 1] = torch.tensor([0., .1, 0.])
    raw[:, 2] = 2
    raw[:, 3] = 1
    raw[:, 4:7] = -2
    raw[:, 7] = 1
    raw[:, 11:14] = torch.tensor([[-.3, .1, .2], [.2, -.4, .1], [.3, -.2, -.1]])
    geometry = Geometry.fit(raw[:, :3], 16)
    order = torch.tensor([2, 0, 1])
    packet = raw[order]
    feature = torch.cat((geometry.normalize(packet[:, :3]), packet[:, 3:]), -1)
    groups = [pad_sequence(list(feature.split(2)), batch_first=True)]
    group_ids = [pad_sequence(list(order.split(2)), batch_first=True, padding_value=-1)]
    scene = SimpleNamespace(raw=packet, order=order, groups=groups, group_ids=group_ids,
                            geometry=geometry, degree=0, white_background=False,
                            fingerprint='toy-ply', name='toy')
    projection = torch.eye(4)
    projection[2, 3] = 1
    projection[3, 3] = 0
    camera = SimpleNamespace(image_width=16, image_height=16, FoVx=1.5, FoVy=1.5,
                             world_view_transform=torch.eye(4), full_proj_transform=projection,
                             camera_center=torch.zeros(3), image_name='train0',
                             original_image=torch.full((3, 16, 16), float('nan')))
    args = SimpleNamespace(spatial_neighbors=2, projection_neighbors=2, cpu_threads=1,
                           feature_batch=2, snr=10.)
    return model, scene, [camera], args, raw


def test_bank_scatter_q0_source_signed_candidates_and_source_only_view_reference():
    model, scene, cameras, args, raw = toy_inputs()
    bank = prepare_bank(model, scene, cameras, args)
    meta = bank['metadata']
    candidate = bank['candidate']
    source_slice = slice(*meta['candidate_layout']['attributes'])
    difference_slice = slice(*meta['candidate_layout']['difference'])
    existence = meta['candidate_layout']['existence']
    rate = meta['candidate_layout']['rate']
    assert torch.equal(candidate[:, 0, existence], torch.zeros(3))
    assert torch.equal(candidate[:, 1:, existence], torch.ones(3, 3))
    assert torch.equal(candidate[:, 0, difference_slice], torch.zeros(3, 14))
    torch.testing.assert_close(candidate[:, 0, source_slice][:, 11:14], raw[:, 11:14])
    torch.testing.assert_close(candidate[:, 2, difference_slice][:, 11:13],
                               torch.tensor([[-.4, .2]]).expand(3, 2))
    torch.testing.assert_close(candidate[0, :, rate], torch.tensor([0., 8/48, 24/48, 1.]))
    # Equal scaling preserves the 3D Euclidean-distance ratio even for a thin scene.
    torch.testing.assert_close((bank['xyz'][1]-bank['xyz'][0]).norm() /
                               (bank['xyz'][2]-bank['xyz'][0]).norm(),
                               torch.tensor((1.01 ** .5)/3))
    assert bank['view_geom'].shape == (1, 3, 7)
    torch.testing.assert_close(bank['view_geom'][0, :, 0], torch.tensor([3.5, 7.5, 15.5])/16)
    assert bank['view_weights'].tolist() == [1.]
    assert bank['view_index'].shape == (1, 3, 2)
    assert meta['candidate_protocol']['path'] == 'finite_noiseless'
    assert meta['candidate_protocol']['noise_input'] == 'none'
    assert meta['row_order'] == 'original_ply'
    json.dumps(meta)
    for key, value in bank.items():
        if torch.is_tensor(value):
            assert value.device.type == 'cpu' and torch.isfinite(value).all()
            assert value.dtype == (torch.int64 if key.endswith('index') else torch.float32)
    assert bank['node'].shape[0] == 3 and bank['view_node'].shape[:2] == (1, 3)


def test_bank_chunking_and_training_mode_do_not_change_fixed_cache():
    model, scene, cameras, args, _ = toy_inputs()
    model.train()
    first = prepare_bank(model, scene, cameras, args)
    assert model.training
    args.feature_batch = 8192
    second = prepare_bank(model, scene, cameras, args)
    for key in first:
        if torch.is_tensor(first[key]):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


@pytest.mark.parametrize('group_shift', [0, 5])
def test_real_successive_codec_cache_matches_complete_noiseless_decode_and_xyz(group_shift):
    from test_successive_allocation import small_successive
    from gaussian_jscc.data import to_features, to_raw
    from gaussian_jscc.modular_evaluation import decode_path

    torch.set_num_threads(1)
    torch.manual_seed(32)
    raw, geometry, _, model = small_successive()
    order = torch.tensor([7, 2, 10, 0, 15, 3, 1, 12, 9, 5, 14, 11, 8, 6, 13, 4])
    grouped = order.roll(group_shift)
    feature, _ = to_features(raw[grouped], geometry, model)
    scene = SimpleNamespace(raw=raw[order], order=order, groups=[feature.reshape(2, 8, -1)],
        group_ids=[grouped.reshape(2, 8)], geometry=geometry, degree=1,
        white_background=False, fingerprint='real-codec-fixture', name='toy')
    _, _, cameras, args, _ = toy_inputs()
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    bank = prepare_bank(model, scene, cameras, args)
    assert torch.equal(rng, torch.get_rng_state())
    for tier in (1, 2, 3):
        q = [torch.full((2, 8), tier, dtype=torch.long)]
        packet = decode_path(model, scene, q, 'finite_noiseless', 10.)[0]
        normalized = torch.cat(((packet[:, :3]-raw[:, :3].amin(0)) /
            (raw[:, :3].amax(0)-raw[:, :3].amin(0)).amax(),
            (packet[:, 3:]-model.attr_mean)/model.attr_std), -1)
        torch.testing.assert_close(bank['candidate'][grouped, tier, :raw.shape[-1]], normalized)
    expected_xyz = geometry.denormalize((geometry.normalize(raw[:, :3])*65535).round()/65535)
    stored_xyz = bank['candidate'][:, 2, :3]
    reconstructed_xyz = stored_xyz*bank['metadata']['xyz_normalization']['isotropic_scale'] + \
        torch.tensor(bank['metadata']['xyz_normalization']['origin'])
    torch.testing.assert_close(reconstructed_xyz, expected_xyz)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_projection_broadphase_matches_all_pairs_with_large_radius_range():
    torch.manual_seed(21)
    points = torch.cat((torch.randn(61, 2)*20, torch.rand(61, 1)*10+.1,
                        torch.exp(torch.linspace(-4, 5, 61))[:, None]), -1)
    graph = build_projection_graph(points, 61, 1)
    for row in range(len(points)):
        # Literal geometric predicate, independent of bucket/tree implementation.
        expected = {other for other in range(len(points)) if other != row and
            float(torch.linalg.vector_norm(points[row, :2]-points[other, :2])) <=
            float(points[row, 3]+points[other, 3])}
        assert set(graph[row][graph[row] >= 0].tolist()) == expected


def test_bank_rejects_duplicate_and_missing_original_ids_in_codec_groups():
    model, scene, cameras, args, _ = toy_inputs()
    scene.group_ids = [torch.tensor([[0, 1], [1, -1]])]
    with pytest.raises(ValueError, match='original IDs'):
        prepare_bank(model, scene, cameras, args)


def test_bank_dtype_is_explicit_even_when_application_default_is_float64():
    model, scene, cameras, args, _ = toy_inputs()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        bank = prepare_bank(model, scene, cameras, args)
    finally:
        torch.set_default_dtype(previous)
    for key, value in bank.items():
        if torch.is_tensor(value) and not key.endswith('index'):
            assert value.dtype == torch.float32


@pytest.mark.parametrize('last_x', [2., 4.])
def test_bank_projection_graph_includes_new_candidate_coverage_in_original_rows(last_x):
    class ExpandingCodec(ToyCodec):
        def forward_tier_batches(self, features, xyz, tier_weights, *args, **kwargs):
            output, details = super().forward_tier_batches(features, xyz, tier_weights,
                                                         *args, **kwargs)
            tier = tier_weights.argmax(-1)
            output[..., 4:7] += (tier == 1)[..., None].to(features)*2
            return output, details

    model, scene, cameras, args, raw = toy_inputs()
    if last_x == 4.:
        # This source point is wholly off image; q1's large disk enters the image.
        raw[2, 0] = last_x
        scene.geometry = Geometry.fit(raw[:, :3], 16)
        scene.raw = raw[scene.order]
        features = torch.cat((scene.geometry.normalize(scene.raw[:, :3]), scene.raw[:, 3:]), -1)
        scene.groups = [pad_sequence(list(features.split(2)), batch_first=True)]
    source_bank = prepare_bank(model, scene, cameras, args)
    # Source disks at original rows 0 and 2 are disjoint.
    assert 2 not in source_bank['view_index'][0, 0]
    expanded = prepare_bank(ExpandingCodec(), scene, cameras, args)
    assert 2 in expanded['view_index'][0, 0]
    assert 0 in expanded['view_index'][0, 2]
    assert expanded['metadata']['parameters']['projection_overlap'] == \
        'source_candidate_context_union_bounding_disk'
    profile = expanded['metadata']['projection_coverage_profiles'][0]
    assert profile['expanded_nodes'] == 3
    assert profile['union_max_radius'] > profile['source_max_radius']
    assert profile['candidate_only_visible_nodes'] == int(last_x == 4.)


def test_background_configuration_is_original_row_seeded_and_rng_independent():
    from gaussian_jscc.relation_benefit_context import background_configurations
    before = torch.get_rng_state().clone()
    contexts = background_configurations(10000, seed=31)
    assert torch.equal(before, torch.get_rng_state())
    assert [c['name'] for c in contexts] == ['q1', 'q2', 'q3', 'mixed']
    assert [c['tiers'].unique().tolist() for c in contexts[:3]] == [[1], [2], [3]]
    proportions = torch.bincount(contexts[3]['tiers'], minlength=4).float()/10000
    torch.testing.assert_close(proportions, torch.tensor([.1, .3, .3, .3]), atol=.015, rtol=0)
    assert torch.equal(contexts[3]['tiers'][:10], background_configurations(10, seed=31)[3]['tiers'])
    assert not torch.equal(contexts[3]['tiers'], background_configurations(10000, seed=32)[3]['tiers'])


def test_context_features_keep_original_group_histograms_and_reuse_uniform_candidates(monkeypatch):
    model, scene, cameras, args, raw = toy_inputs()
    calls = []
    original = model.forward_tier_batches
    def counted(features, xyz, weights, *pos, **kwargs):
        calls.append(weights.argmax(-1).clone())
        return original(features, xyz, weights, *pos, **kwargs)
    monkeypatch.setattr(model, 'forward_tier_batches', counted)
    bank = prepare_bank(model, scene, cameras, args)
    assert len(calls) == 4*len(scene.groups)
    assert len(bank['contexts']) == 4
    dim = raw.shape[-1]
    for context_id, context in enumerate(bank['contexts']):
        assert context['node'].shape == (3, 2*dim+8)
        assert context['view_node'].shape == (1, 3, 78)
        assert context['view_geom'].shape == (1, 3, 7)
        assert context['tiers'].dtype == torch.long
        assert torch.isfinite(context['node']).all()
        assert torch.isfinite(context['view_node']).all()
        assert torch.isfinite(context['view_geom']).all()
        torch.testing.assert_close(context['node'][:, 2*dim:2*dim+4],
                                   torch.nn.functional.one_hot(context['tiers'], 4).float())
        for ids in scene.group_ids[0]:
            ids = ids[ids >= 0]
            expected = torch.bincount(context['tiers'][ids], minlength=4).float()/len(ids)
            torch.testing.assert_close(context['node'][ids, -4:], expected.expand(len(ids), 4))
        if context_id < 3:
            torch.testing.assert_close(context['node'][:, :dim], bank['candidate'][:, context_id+1, :dim])
    metadata = bank['metadata']['contexts']
    assert metadata['seed'] == 42
    assert len(metadata['tier_sha256']) == 4
    json.dumps(bank['metadata'], allow_nan=False)


def test_mixed_context_attributes_match_complete_block_decode_with_q0_context():
    from test_successive_allocation import small_successive
    from gaussian_jscc.data import to_features, to_raw
    from gaussian_jscc.relation_benefit_context import background_configurations
    torch.set_num_threads(1)
    raw, geometry, _, model = small_successive()
    order = torch.tensor([7, 2, 10, 0, 15, 3, 1, 12, 9, 5, 14, 11, 8, 6, 13, 4])
    grouped = order.roll(5)
    features, _ = to_features(raw[grouped], geometry, model)
    scene = SimpleNamespace(raw=raw[order], order=order, groups=[features.reshape(2, 8, -1)],
        group_ids=[grouped.reshape(2, 8)], geometry=geometry, degree=1,
        white_background=False, fingerprint='mixed-real-fixture', name='toy')
    _, _, cameras, args, _ = toy_inputs()
    args.seed = 42
    bank = prepare_bank(model, scene, cameras, args)
    context = bank['contexts'][3]
    tiers = background_configurations(16, 42)[3]['tiers']
    assert (tiers == 0).any()
    q = tiers[grouped].reshape(2, 8)
    independent = model.forward_tier_batches(features.reshape(2, 8, -1),
        features.reshape(2, 8, -1)[..., :3], torch.nn.functional.one_hot(q, 4).to(features),
        10., 'none', paired_noise=True)[0]
    packet = to_raw(independent.reshape(16, -1), geometry, model)
    origin = raw[:, :3].amin(0)
    scale = (raw[:, :3].amax(0)-origin).amax()
    expected = torch.cat(((packet[:, :3]-origin)/scale,
        (packet[:, 3:]-model.attr_mean)/model.attr_std), -1)
    active = q.reshape(-1)>0
    torch.testing.assert_close(context['node'][grouped[active], :raw.shape[-1]], expected[active])
    assert torch.equal(context['tiers'], tiers)


def test_projection_graph_covers_expansion_present_only_in_mixed_context():
    class ContextExpandingCodec(ToyCodec):
        def forward_tier_batches(self, features, xyz, weights, *args, **kwargs):
            output, extra = super().forward_tier_batches(features, xyz, weights, *args, **kwargs)
            q = weights.argmax(-1)
            positive = q[q>0]
            if positive.numel() and positive.unique().numel()>1:
                output[..., 4:7] += (q>0)[..., None].to(output)*2
            return output, extra
    model, scene, cameras, args, _ = toy_inputs()
    baseline = prepare_bank(model, scene, cameras, args)
    assert 2 not in baseline['view_index'][0, 0]
    expanded = prepare_bank(ContextExpandingCodec(), scene, cameras, args)
    # Uniform q1-q3 candidates stay unchanged; only the extra mixed decode expands.
    torch.testing.assert_close(expanded['candidate'], baseline['candidate'])
    assert 2 in expanded['view_index'][0, 0]
    assert 0 in expanded['view_index'][0, 2]
