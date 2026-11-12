import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gaussian_jscc.student_policy_inputs import load_teacher, split_student_cameras
from gaussian_jscc.budgeted_allocation_evaluation import tier_hash


def teacher(tmp_path):
    root = tmp_path/'teacher'; (root/'deployment').mkdir(parents=True)
    q = torch.tensor([0, 1, 2, 3])
    identity = dict(status='complete', codec_id='abc', tier_sha256=tier_hash(q), beta=.5,
                    d_ref=.01, c_ref=200, lambda_s=.1, frozen_codec_verified=True)
    ref = dict(d_ref=.01, c_ref=200, lambda_s=.1, beta=.5)
    meta = dict(fingerprint='xyz', source_gaussians=4, objective_reference=ref,
                allocation_train_view_names=['a', 'b'], calibration_view_names=['c'],
                test_view_names=['d'], camera_fingerprints={'train': 'camhash'})
    for name, data in [('complete.json', identity), ('deployment/selection.json', identity),
                       ('training.json', meta), ('plan.json', dict(codec_config={'rates': [0, 8, 24, 48]}, settings=dict(seed=42, snr=10., trials=4)))]:
        (root/name).write_text(json.dumps(data))
    np.save(root/'deployment/selected_tiers.npy', q.numpy().astype('uint8'))
    return root


def test_teacher_identity_reference_and_camera_split(tmp_path):
    root = teacher(tmp_path)
    scene = SimpleNamespace(fingerprint='xyz', raw=torch.zeros(4, 59),
        record=dict(test_view_names=['d'], camera_fingerprints={'train': 'camhash'}),
        cameras=[SimpleNamespace(image_name=n) for n in ('a', 'b', 'c')])
    q, reference, record = load_teacher(root, scene, 'abc', .5, {'rates': [0, 8, 24, 48]}, seed=42, snr=10., trials=4)
    assert q.tolist() == [0, 1, 2, 3] and reference['d_ref'] == .01
    training, calibration, preparation = split_student_cameras(scene, record, 8)
    assert [c.image_name for c in training] == ['a', 'b']
    assert [c.image_name for c in calibration] == ['c']
    assert set(c.image_name for c in preparation) == {'a', 'b'}
    with pytest.raises(ValueError): load_teacher(root, scene, 'wrong', .5, {'rates': [0, 8, 24, 48]}, seed=42, snr=10., trials=4)
    with pytest.raises(ValueError): load_teacher(root, scene, 'abc', .1, {'rates': [0, 8, 24, 48]}, seed=42, snr=10., trials=4)
    np.save(root/'deployment/selected_tiers.npy', np.ones(4, dtype='uint8'))
    with pytest.raises(ValueError): load_teacher(root, scene, 'abc', .5, {'rates': [0, 8, 24, 48]}, seed=42, snr=10., trials=4)


def test_expanded_calibration_is_deterministic_and_excluded_from_student_inputs():
    names = [f'{i:06d}' for i in range(215)]
    scene = SimpleNamespace(cameras=[SimpleNamespace(image_name=n) for n in names])
    boundary = dict(train_names=names[4:], calibration_names=names[:4], test_names=['test'])
    training, calibration, preparation = split_student_cameras(scene, boundary, 64, 32)
    name_set = lambda views: {c.image_name for c in views}
    assert len(training) == 183 and len(calibration) == 32 and len(preparation) == 64
    assert not name_set(training) & name_set(calibration)
    assert not name_set(preparation) & name_set(calibration)
    assert name_set(preparation) <= name_set(training)
    assert name_set(training) | name_set(calibration) == set(names)
    assert 'test' not in name_set(training) | name_set(calibration)
    again = split_student_cameras(scene, boundary, 64, 32)
    assert [[c.image_name for c in part] for part in again] == [[c.image_name for c in part] for part in (training, calibration, preparation)]
    inherited = split_student_cameras(scene, boundary, 64, 4)
    assert [c.image_name for c in inherited[1]] == names[:4]
    for invalid in (0, 215):
        with pytest.raises(ValueError, match='leave training'): split_student_cameras(scene, boundary, 64, invalid)
