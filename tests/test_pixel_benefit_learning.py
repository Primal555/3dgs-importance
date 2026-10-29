import copy
import json

import pytest
import torch

from gaussian_jscc.pixel_benefit_learning import PixelLearner, summarize_pixel_values
from gaussian_jscc.pixel_benefit_replay import PixelReplay


def view(n, value, source=1):
    return dict(inputs=dict(image=torch.zeros(n, 6, 26, 26),
        context=torch.randn(n, 2, 13), context_mask=torch.ones(n, 2, dtype=torch.bool),
        candidate=torch.randn(n, 4, 2, 12), candidate_mask=torch.ones(n, 4, 2, dtype=torch.bool),
        valid=torch.ones(n, 1, 16, 16)), density=torch.full((n, 3, 2, 16, 16), value, dtype=torch.float64),
        height=16, width=16*n, d_ref=1., rates=[.1, .2, .3], source=source, view=0,
        importance_weight=1.)


def test_fit_scales_uses_all_valid_pixels_and_groups(tmp_path):
    replay = PixelReplay(tmp_path)
    replay.add(dict(source=1, step=0, views=[view(1, 1.)]))
    replay.add(dict(source=2, step=0, views=[view(2, 3.)]))
    learner = PixelLearner(replay, 'cpu', hidden=8, tile_batch=1)
    scales = learner.fit_scales()
    assert torch.allclose(scales['map'], torch.full((2,), (19/3)**.5, dtype=torch.float64))
    assert torch.allclose(scales['value'], torch.full((2,), 5**.5, dtype=torch.float64))
    assert torch.allclose(learner.network.output_scale.double(), scales['map'], atol=1e-6)


def test_training_batches_all_tiles_and_restores_optimizer(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    replay = PixelReplay(tmp_path, seed=9)
    v = view(2, .1)
    replay.add(dict(source=1, step=0, views=[v]))
    replay.add(dict(source=2, step=0, views=[view(1, .2)]), validation=True)
    replay.freeze_validation()
    learner = PixelLearner(replay, 'cpu', hidden=8, tile_batch=1)
    learner.fit_scales()
    stats = learner.train_updates(1, batch_groups=1)
    assert stats['loss'] > 0 and learner.updates == 1
    replay_state = copy.deepcopy(replay.state_dict())
    saved = copy.deepcopy(learner.state_dict())
    first = learner.train_updates(1, batch_groups=1)
    weights = copy.deepcopy(learner.network.state_dict())
    replay.load_state_dict(replay_state)
    other = PixelLearner(replay, 'cpu', hidden=8, tile_batch=2)
    other.load_state_dict(saved)
    second = other.train_updates(1, batch_groups=1)
    assert abs(first['loss']-second['loss']) < 1e-6
    for key in weights:
        assert torch.allclose(weights[key], other.network.state_dict()[key], atol=1e-7)
    assert learner.predict(v['inputs'], 16, 32, 1.).shape == (3, 2)
    json.dumps(learner.validate())


def test_regret_metrics_use_rates_and_q0():
    report = summarize_pixel_values([dict(source=1, view=0, rates=[.1,.2,.3],
        truth=torch.tensor([[-.5,0.],[-.2,0.],[.1,0.]]),
        prediction=torch.tensor([[.2,0.],[-.4,0.],[-.7,0.]]))])
    assert abs(report['keep']['three_tier_selection_regret']['network'] - .8) < 1e-6
    assert abs(report['keep']['four_state_selection_regret'] - .8) < 1e-6
    assert abs(report['keep']['three_tier_baselines']['fixed_q2'] - .4) < 1e-6
    assert report['positive']['count'] == 6
    assert report['keep']['factor_errors']['common_component_rmse'][0] > 0


def test_validation_decides_after_averaging_views(tmp_path):
    replay = PixelReplay(tmp_path)
    first, second = view(1, -1.), view(1, 3.)
    for item in (first, second):
        item['rates'] = [0., 0., 0.]
        item['inputs']['candidate'] = item['inputs']['candidate'][:, :1].expand(-1, 4, -1, -1).clone()
    replay.add(dict(source=1, step=0, views=[first, second]), validation=True)
    replay.freeze_validation()
    learner = PixelLearner(replay, 'cpu', hidden=8)
    report = learner.validate()
    assert report['validation_groups'] == 1
    assert report['keep']['four_state_selection_regret'] == 0


def test_importance_corrects_sampled_group_loss(tmp_path):
    torch.set_num_threads(1)
    replay = PixelReplay(tmp_path, seed=14)
    for source, value, weight in ((1, 1., 1.), (2, 3., 4.)):
        item = view(1, value, source)
        item['inputs']['candidate'] = item['inputs']['candidate'][:, :1].expand(-1, 4, -1, -1).clone()
        replay.add(dict(source=source, step=0, views=[item], importance_weight=weight))
    learner = PixelLearner(replay, 'cpu', hidden=8)
    with pytest.raises(ValueError, match='scales'):
        learner.train_updates(1)
    learner.fit_scales()
    state = copy.deepcopy(replay.state_dict())
    samples = replay.sample(12)
    map_scale = learner.scales['map'][0].item()
    value_scale = learner.scales['value'][0].item()
    weights = [r['importance_weight']*r['replay_importance_weight'] for r in samples]
    expected = sum(w*(float(r['views'][0]['density'][0, 0, 0, 0, 0])**2)*
                   (1/map_scale**2+1/value_scale**2) for r, w in zip(samples, weights)) / len(weights)
    replay.load_state_dict(state)
    stats = learner.train_updates(1, batch_groups=12)
    assert abs(stats['loss']-expected) < 1e-6


def test_offscreen_group_has_zero_value_and_trains_without_dropping_group(tmp_path):
    replay = PixelReplay(tmp_path)
    replay.add(dict(source=1, step=0, views=[view(1, .1)]))
    learner = PixelLearner(replay, 'cpu', hidden=8)
    learner.fit_scales()
    empty = view(0, 0.)
    empty['width'] = 16
    replay.add(dict(source=2, step=0, views=[empty]))
    assert torch.equal(learner.predict(empty['inputs'], 16, 16, 1.), torch.zeros(3, 2))
    assert learner.train_updates(1, 4)['loss'] >= 0


def test_empty_view_does_not_reduce_value_rms(tmp_path):
    replay = PixelReplay(tmp_path)
    empty = view(0, 0.); empty['width'] = 16
    replay.add(dict(source=1, step=0, views=[view(1, 2.), empty]))
    learner = PixelLearner(replay, 'cpu', hidden=8)
    scales = learner.fit_scales()
    assert torch.allclose(scales['value'], torch.full((2,), 2., dtype=torch.float64))


def test_streaming_gradient_matches_full_group_and_importance_weight(tmp_path):
    from gaussian_jscc.pixel_benefit import density_loss
    torch.set_num_threads(1)
    torch.manual_seed(41)
    replay = PixelReplay(tmp_path)
    item = view(3, .2)
    item['inputs']['valid'][2, :, 8:] = 0
    replay.add(dict(source=1, step=0, views=[item], importance_weight=2.5))
    learner = PixelLearner(replay, 'cpu', hidden=8, tile_batch=1)
    learner.fit_scales()
    pred = learner.network(item['inputs'])
    expected_loss = density_loss(pred, item['density'], item['inputs']['valid'], 16, 48, 1.,
                                 learner.scales['map'], learner.scales['value']) * 2.5
    expected_loss.backward()
    gradients = [p.grad.detach().clone() for p in learner.network.parameters()]
    learner.optimizer.zero_grad(set_to_none=True)
    stats = learner.train_updates(1, 1)
    assert abs(stats['loss']-expected_loss.item()) < 1e-6
    for parameter, expected in zip(learner.network.parameters(), gradients):
        assert torch.allclose(parameter.grad, expected, rtol=2e-5, atol=1e-6)


def test_scale_fitting_uses_source_importance(tmp_path):
    replay = PixelReplay(tmp_path)
    replay.add(dict(source=1, step=0, views=[view(1, 1.)], importance_weight=1.))
    replay.add(dict(source=2, step=0, views=[view(1, 3.)], importance_weight=3.))
    scales = PixelLearner(replay, 'cpu', hidden=8).fit_scales()
    expected = torch.full((2,), 7**.5, dtype=torch.float64)
    assert torch.allclose(scales['map'], expected)
    assert torch.allclose(scales['value'], expected)


def test_all_empty_batch_has_zero_loss_and_preserves_parameters(tmp_path):
    replay = PixelReplay(tmp_path, max_groups=1)
    replay.add(dict(source=1, step=0, views=[view(1, .1)]))
    learner = PixelLearner(replay, 'cpu', hidden=8)
    learner.fit_scales()
    empty = view(0, 0.); empty['width'] = 16
    replay.add(dict(source=2, step=1, views=[empty]))
    before = copy.deepcopy(learner.network.state_dict())
    stats = learner.train_updates(1, 1)
    assert stats['loss'] == 0 and stats['grad_norm'] == 0
    for key, value in before.items():
        assert torch.equal(value, learner.network.state_dict()[key])


def test_variable_length_context_matches_padded_input(tmp_path):
    torch.manual_seed(62)
    data = view(2, .1)
    data['inputs']['context_mask'][0, 1] = False
    variable = copy.deepcopy(data)
    variable['inputs']['context'] = tuple(data['inputs']['context'][i, data['inputs']['context_mask'][i]] for i in range(2))
    variable['inputs']['context_mask'] = tuple(torch.ones(len(x), dtype=torch.bool) for x in variable['inputs']['context'])
    replay = PixelReplay(tmp_path)
    replay.add(dict(source=1, step=0, views=[data]))
    learner = PixelLearner(replay, 'cpu', hidden=8, tile_batch=2)
    learner.fit_scales()
    expected = learner.predict(data['inputs'], 16, 32, 1.)
    actual = learner.predict(variable['inputs'], 16, 32, 1.)
    assert torch.allclose(actual, expected, atol=1e-7)
    replay.add(dict(source=2, step=0, views=[variable]))
    assert learner.train_updates(1, 2)['loss'] >= 0


def test_scene_cache_matches_uncached_candidates_and_clears_on_training(tmp_path):
    torch.manual_seed(81)
    data = view(2, .1)
    replay = PixelReplay(tmp_path)
    replay.add(dict(source=1, step=0, views=[data]))
    learner = PixelLearner(replay, 'cpu', hidden=8, tile_batch=2)
    learner.fit_scales()
    keys = [('camera', 0), ('camera', 1)]
    reference = learner.predict(data['inputs'], 16, 32, 1.)
    learner.begin_window()
    first = learner.predict(data['inputs'], 16, 32, 1., cache_keys=keys)
    assert torch.allclose(first, reference, atol=1e-7)
    changed = copy.deepcopy(data['inputs'])
    changed['candidate'][:, 2, :, 8] += 4
    expected = learner.predict(changed, 16, 32, 1.)
    actual = learner.predict(changed, 16, 32, 1., cache_keys=keys)
    assert torch.allclose(actual, expected, atol=1e-7)
    assert not torch.allclose(actual, first, atol=1e-8)
    assert learner.cache_hits == 2 and learner.cache_misses == 2
    learner.train_updates(1, 1)
    new_reference = learner.predict(changed, 16, 32, 1.)
    new_actual = learner.predict(changed, 16, 32, 1., cache_keys=keys)
    assert torch.allclose(new_actual, new_reference, atol=1e-7)
    assert learner.cache_misses == 4


def test_cache_is_bounded_and_not_restored(tmp_path):
    data = view(1, .1)
    replay = PixelReplay(tmp_path)
    replay.add(dict(source=1, step=0, views=[data]))
    learner = PixelLearner(replay, 'cpu', hidden=4)
    learner.fit_scales()
    for key in range(129):
        learner.predict(data['inputs'], 16, 16, 1., cache_keys=[key])
    assert len(learner._scene_cache) == 128
    learner.predict(data['inputs'], 16, 16, 1., cache_keys=[0])
    assert learner.cache_misses == 130
    state = copy.deepcopy(learner.state_dict())
    learner.load_state_dict(state)
    assert not learner._scene_cache
