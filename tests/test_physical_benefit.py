import copy

import pytest
import torch


def test_shared_scorer_anchor_and_permutation():
    from gaussian_jscc.physical_benefit import PhysicalBenefitNetwork, FEATURE_DIM
    torch.manual_seed(3)
    model = PhysicalBenefitNetwork(hidden=16, dropout=.2)
    x = torch.randn(2, 4, FEATURE_DIM)
    model.train()
    identical = x[:, :1].expand(-1, 4, -1)
    assert torch.equal(model(identical), torch.zeros(2, 3, 2))
    model.eval()
    assert torch.allclose(model(x[:, [0, 3, 1, 2]]), model(x)[:, [2, 0, 1]])


def test_rd_loss_direction_cost_and_ties():
    from gaussian_jscc.physical_benefit import decision_loss
    truth = torch.tensor([[[-.4, 0.], [-.5, 0.], [-.6, 0.]]])
    rates = torch.tensor([[.1, .3, .5]])
    good = truth.clone().requires_grad_()
    bad = -truth.clone().requires_grad_()
    assert decision_loss(good, truth, rates, 1., .02) < decision_loss(bad, truth, rates, 1., .02)
    loss = decision_loss(bad, truth, rates, 1., .02)
    assert torch.isfinite(loss)
    tie = torch.zeros_like(truth)
    assert decision_loss(good, tie, torch.zeros_like(rates), 1., .02).item() == 0
    # Expensive mistakes must carry more weight, even with identical predicted logits.
    zeros = torch.zeros_like(truth)
    assert decision_loss(zeros, truth*2, torch.zeros_like(rates), 1., 0.) > decision_loss(zeros, truth, torch.zeros_like(rates), 1., 0.)


def test_visibility_matches_compositing_and_occlusion():
    from gaussian_jscc.physical_benefit_features import visibility_statistics
    from test_pixel_local_renderer import scene, splat
    tile = scene([splat(x=0, y=0, z=1, opacity=.5), splat(x=0, y=0, z=2, opacity=.5)], height=1, width=1)
    target = torch.ones_like(tile.image)
    stats = visibility_statistics(tile, target)
    assert torch.allclose(stats[:, 0], torch.tensor([.5, .25]))
    assert torch.allclose(stats[:, 1], stats[:, 0])
    assert torch.allclose(stats[:, 5:8], stats[:, :1].expand(-1, 3))
    assert torch.allclose(stats[:, 8:11], stats[:, :1]*(tile.image[:, 0, 0]-1))


def test_unoccluded_alpha_survives_foreground_early_exit():
    from gaussian_jscc.physical_benefit_features import visibility_statistics
    from test_pixel_local_renderer import scene,splat
    tile=scene([splat(x=0,y=0,z=i+1,opacity=a) for i,a in enumerate([.99,.99,.99,.5])],height=1,width=1)
    stats=visibility_statistics(tile,torch.zeros_like(tile.image))
    assert torch.allclose(stats[:,4],torch.tensor([.99,.99,.99,.5]))
    assert stats[-1,0]==0 and stats[-1,4]>.4


def test_compact_labels_identical_to_pixel_teacher():
    from gaussian_jscc.pixel_benefit_features import build_group, exact_values
    from gaussian_jscc.pixel_benefit import integrate_density
    from test_pixel_local_renderer import scene, splat
    tile = scene([splat(x=15), splat(x=16, z=2)])
    candidate = tile.projected[None, :1].repeat(4, 1, 1)
    candidate[0, :, 7] = 0
    candidate[2, :, 8] = .4
    candidate[3, :, 0] = 24
    target = torch.zeros_like(tile.image)
    dense = build_group(tile, [0], candidate, target, .1, .01, with_labels=True)
    values = exact_values(tile, [0], candidate, target, .1, .01)
    expected = integrate_density(dense['density'], dense['inputs']['valid'], tile.height, tile.width, .01)
    assert torch.allclose(values, expected, atol=1e-12, rtol=1e-10)


def test_static_features_duplicate_and_single_points():
    from gaussian_jscc.physical_benefit_features import static_features
    raw = torch.zeros(3, 14); raw[:, 7] = 1
    features = static_features(raw, neighbors=32, workers=1)
    assert features.shape == (3, 24) and torch.isfinite(features).all()
    assert torch.isfinite(static_features(raw[:1], neighbors=32, workers=1)).all()


def test_exact_teacher_reuses_identical_current_state():
    from unittest.mock import patch
    from gaussian_jscc.pixel_benefit_features import exact_values
    from test_pixel_local_renderer import scene, splat
    tile=scene([splat()])
    candidate=tile.projected[None].repeat(4,1,1)
    candidate[0,:,7]=0
    candidate[2,:,8]=.4
    candidate[3,:,8]=.6
    with patch.object(tile,'render_tiles',wraps=tile.render_tiles) as rendering:
        values=exact_values(tile,[0],candidate,torch.zeros_like(tile.image),.01,1.)
    assert values.shape==(3,2)
    assert rendering.call_count==3


def test_features_include_neighbor_signed_changes():
    from gaussian_jscc.physical_benefit_features import candidate_features,FEATURE_DIM
    from test_pixel_local_renderer import splat
    raw=torch.zeros(4,2,14); raw[...,7]=1
    p=torch.tensor([splat(),splat(x=9)])[None].repeat(4,1,1)
    stats=torch.ones(2,12)
    baseline=candidate_features(torch.zeros(24),stats,raw,p,0,16,16,.01,.1)
    p[2,1,8]+=.3
    changed=candidate_features(torch.zeros(24),stats,raw,p,0,16,16,.01,.1)
    assert baseline.shape==(4,FEATURE_DIM)
    assert torch.equal(baseline[[0,1,3]],changed[[0,1,3]])
    assert not torch.equal(baseline[2],changed[2])


def record(source, value, feature_dim):
    x = torch.randn(4, feature_dim)
    return dict(source=source, step=0, rates=torch.tensor([.01,.02,.03]), importance_weight=1.,
        views=[dict(inputs={'features': x}, values=torch.full((3,2), value, dtype=torch.float64),
                    height=16, width=16, d_ref=1.)])


def test_learning_train_only_scaling_and_resume(tmp_path):
    from gaussian_jscc.physical_benefit import PhysicalLearner, FEATURE_DIM
    from gaussian_jscc.pixel_benefit_replay import PixelReplay
    torch.set_num_threads(1); torch.manual_seed(11)
    replay = PixelReplay(tmp_path, seed=3)
    replay.add(record(1, .1, FEATURE_DIM))
    replay.add(record(2, 100., FEATURE_DIM), validation=True)
    replay.freeze_validation()
    a = PhysicalLearner(replay, 'cpu', hidden=16, dropout=.1)
    scales = a.fit_scales()
    assert torch.allclose(scales['value'], torch.tensor([.1,.1], dtype=torch.float64))
    stats = a.train_updates(2, 1)
    assert {'regression_loss', 'decision_loss', 'train_regret'} <= stats.keys()
    saved = copy.deepcopy(a.state_dict()); rs = copy.deepcopy(replay.state_dict()); rng = torch.get_rng_state()
    expected = a.train_updates(1, 1)
    b = PhysicalLearner(replay, 'cpu', hidden=16, dropout=.1)
    b.load_state_dict(saved); replay.load_state_dict(rs); torch.set_rng_state(rng)
    actual = b.train_updates(1, 1)
    assert expected == actual
    for key, value in a.network.state_dict().items():
        assert torch.equal(value, b.network.state_dict()[key])
    assert b.validate()['validation_groups'] == 1
