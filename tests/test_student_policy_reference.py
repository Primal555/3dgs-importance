import copy
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import torch
from gaussian_jscc.student_policy_experiment import reusable_reference, prepare_prediction_reference, allocation_timing


def test_same_ply_and_names_with_changed_camera_cannot_reuse_reference():
    record = dict(fingerprint='ply', source_gaussians=4, sh_degree=1, geometry={'bits': 16},
        camera_fingerprints={'train': 'pose0', 'test': 'pose1'}, resolution=1, images='images',
        white_background=False, blocks_per_batch=64)
    scene = SimpleNamespace(fingerprint='ply', record=record, net_bits_per_use=2.)
    plan = dict(scene=copy.deepcopy(record), source={'net_bits_per_use': 2.})
    assert reusable_reference(scene, plan)
    scene.record['camera_fingerprints']['train'] = 'changed-pose-same-name'
    assert not reusable_reference(scene, plan)
    scene.record = copy.deepcopy(plan['scene']); scene.net_bits_per_use = 1.
    assert not reusable_reference(scene, plan)


def test_q3_calibration_and_cost_are_inside_reference_timer(tmp_path):
    args = SimpleNamespace(lpips=True, beta=.5)
    model = Mock(); scene = SimpleNamespace(raw=torch.zeros(4, 23))
    meter = Mock(); meter.details.return_value = {'total_channel_uses': 200}
    clock = Mock(side_effect=[10., 17.])
    with patch('gaussian_jscc.student_policy_experiment.evaluate_map', return_value={
            'source_objective_mse': .01, 'source_structure_error': .1}) as evaluate:
        ref, seconds = prepare_prediction_reference(model, scene, ['calibration'], args, meter, tmp_path, clock=clock)
    assert seconds == 7. and ref['c_ref'] == 200 and ref['beta'] == .5
    assert ref['d_ref'] == pytest.approx(.0125)
    assert evaluate.call_count == 1 and meter.details.call_count == 1
    timing = allocation_timing(12., seconds, 2.)
    assert timing['preparation_seconds_this_run'] == 19.
    assert timing['allocation_seconds'] == 21.
    assert allocation_timing(12., 0., 2.)['allocation_seconds'] == 14.
