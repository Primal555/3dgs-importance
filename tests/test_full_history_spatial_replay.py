import copy
import math

import pytest
import torch

from gaussian_jscc.full_history_spatial_replay import FullHistorySpatialReplay


def record(i):
    return dict(source=i, step=i+1, views=[dict(inputs={}, values=torch.ones(3, 2))])


def test_retains_all_labels_and_gc_protects_them(tmp_path):
    r = FullHistorySpatialReplay(tmp_path, max_groups=2, validation_groups=1)
    ids = [r.add(record(i)) for i in range(9)]
    assert len(r.training) == 9
    assert set(r.priorities) == set(ids)
    assert len(list(r.training_records())) == 9
    assert r.garbage_collect([r.manifest()]) == 0
    assert len(r.entries_by_id) == 9


def test_distribution_and_importance_match_original_mixture(tmp_path):
    r = FullHistorySpatialReplay(tmp_path, max_groups=2)
    ids = [r.add(record(i)) for i in range(4)]
    for key, p in zip(ids, [0., 1., 3., 6.]):
        r.update_priority(key, p)
    r.mark_updated([ids[1], ids[2], ids[3], ids[3]])
    expected = torch.tensor([.5625, .0875, .1375, .2125], dtype=torch.float64)
    torch.testing.assert_close(r.probabilities(), expected)
    draws = r.sample(1000)
    for g in draws:
        i = ids.index(g['replay_record_id'])
        assert g['replay_probability'] == pytest.approx(float(expected[i]))
        assert g['replay_importance_weight'] == pytest.approx(1/(4*float(expected[i])))
    actual = torch.bincount(torch.tensor([g['source'] for g in draws]), minlength=4)/len(draws)
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=.05)


def test_sample_resume_and_validation_isolation(tmp_path):
    r = FullHistorySpatialReplay(tmp_path, seed=12, max_groups=2, validation_groups=1)
    keys = [r.add(record(i)) for i in range(8)]
    r.add(record(99), True)
    r.freeze_validation()
    r.mark_updated(keys[:4])
    state = copy.deepcopy(r.state_dict())
    expected = [(g['source'], g['replay_probability']) for g in r.sample(32)]
    other = FullHistorySpatialReplay(tmp_path, seed=99, max_groups=2, validation_groups=1)
    other.load_state_dict(state)
    assert [(g['source'], g['replay_probability']) for g in other.sample(32)] == expected
    assert [g['source'] for g in other.validation_records()] == [99]
    assert other.coverage()['active_groups'] == 8
    assert other.coverage()['min_visits'] == 0
    with pytest.raises(ValueError, match='frozen'):
        other.add(record(100), True)


def test_priority_median_and_updates_remain_indexed(tmp_path):
    r = FullHistorySpatialReplay(tmp_path, max_groups=2)
    keys = [r.add(record(i)) for i in range(3)]
    for key, value in zip(keys, [2., 5., 9.]):
        r.update_priority(key, value)
    assert r.priorities[r.add(record(4))] == 5.
    r.update_priority(keys[0], 10.)
    # torch.median uses the lower median for an even count.
    assert r.priorities[r.add(record(5))] == 5.
    with pytest.raises(ValueError):
        r.update_priority(keys[0], math.nan)


def test_full_history_state_rejects_bounded_or_damaged_state(tmp_path):
    r = FullHistorySpatialReplay(tmp_path, max_groups=2)
    r.add(record(1))
    state = copy.deepcopy(r.state_dict())
    state.pop('full_history_policy')
    with pytest.raises(ValueError, match='policy'):
        r.load_state_dict(state)


def test_draws_do_not_scan_training_or_priorities(tmp_path):
    r = FullHistorySpatialReplay(tmp_path, max_groups=2)
    keys = [r.add(record(i)) for i in range(40)]
    class IndexedList(list):
        def __iter__(self):
            raise AssertionError('full training scan')
    class IndexedDict(dict):
        def __iter__(self):
            raise AssertionError('full priorities scan')
        def items(self):
            raise AssertionError('full priorities scan')
        def values(self):
            raise AssertionError('full priorities scan')
    r.training = IndexedList(r.training)
    r.priorities = IndexedDict(r.priorities)
    assert len(r.sample(8)) == 8
    r.mark_updated(keys[:3])
    r.update_priority(keys[0], 3.)
    r.add(record(41))


def test_index_matches_reference_after_many_mutations(tmp_path):
    from gaussian_jscc.spatial_benefit_replay import SpatialReplay
    from gaussian_jscc.hybrid_tier_benefit import clone_replay
    r = FullHistorySpatialReplay(tmp_path/'full', max_groups=2)
    generator = torch.Generator().manual_seed(17)
    keys = [r.add(record(i)) for i in range(30)]
    for _ in range(120):
        index = int(torch.randint(len(keys), (), generator=generator))
        value = float(torch.rand((), generator=generator)*10)
        r.update_priority(keys[index], value)
        r.mark_updated([keys[index]])
        actual = r.probabilities()
        reference = SpatialReplay.probabilities(r)
        torch.testing.assert_close(actual, reference, atol=1e-15, rtol=1e-14)
    assert r.coverage()['all_group_draws'] == 120
    assert r.coverage()['p10_visits'] == pytest.approx(float(torch.quantile(
        torch.tensor(list(r.usage.values()), dtype=torch.float64), .1)))
    cloned = clone_replay(r.state_dict(), tmp_path/'clone')
    other = FullHistorySpatialReplay(tmp_path/'clone', max_groups=2)
    other.load_state_dict(cloned)
    assert [g['source'] for g in r.sample(64)] == [g['source'] for g in other.sample(64)]


def test_zero_priority_falls_back_to_uniform_and_corruption_is_rejected(tmp_path):
    r = FullHistorySpatialReplay(tmp_path, max_groups=1)
    ids = [r.add(record(i)) for i in range(5)]
    for key in ids:
        r.update_priority(key, 0.)
    torch.testing.assert_close(r.probabilities(), torch.full((5,), .2, dtype=torch.float64))
    assert len(r.sample(50)) == 50
    state = copy.deepcopy(r.state_dict())
    state['visit_buckets'][0].append(0)
    with pytest.raises(ValueError, match='visit index'):
        r.load_state_dict(state)
