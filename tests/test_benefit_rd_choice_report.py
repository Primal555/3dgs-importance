import pytest
import torch

from gaussian_jscc.benefit_rd_choice_report import stratified_report


def test_fixed_training_threshold_reports_ordinary_and_large_gap_and_all_samples():
    groups = []
    predictions = []
    for gap in (.01, 2.):
        target = torch.tensor([[0., 0.], [-gap, 0.], [-2*gap, 0.]])
        groups.append(dict(source=len(groups), step=1, rates=torch.zeros(3), views=[dict(values=target)]))
        predictions.append(target.flip(0)[None])
    report = stratified_report(groups, predictions, .5)
    assert report['ordinary']['records'] == report['large_gap']['records'] == 1
    assert report['all']['records'] == 2
    assert report['ordinary']['three_tier']['regret_mean'] == pytest.approx(.02)
    assert report['large_gap']['three_tier']['regret_mean'] == pytest.approx(4.)
    assert report['all']['three_tier']['regret_mean'] == pytest.approx(2.01)
    assert report['all']['selection_tail']['top10_regret_fraction'] == 1.


def test_empty_stratum_stays_explicit_without_reassigning_threshold():
    g = dict(source=0, step=1, rates=torch.zeros(3), views=[dict(values=torch.zeros(3, 2))])
    result = stratified_report([g], [torch.zeros(1, 3, 2)], 100.)
    assert result['large_gap']['records'] == 0
    assert result['large_gap']['three_tier'] is None
    assert result['threshold'] == 100.


def test_ordinary_per_view_component_errors_survive_camera_cancellation():
    g = dict(source=0, step=1, rates=torch.zeros(3),
             views=[dict(values=torch.zeros(3, 2)), dict(values=torch.zeros(3, 2))])
    p = torch.ones(2, 3, 2); p[1] *= -1
    r = stratified_report([g], [p], 100.)['ordinary']['value_error']
    assert r['component_rmse'] == [0., 0.]
    assert r['per_view_component_rmse'] == [1., 1.]
    assert r['per_view_component_mae'] == [1., 1.]
