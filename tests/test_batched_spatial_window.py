import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def scene_fixture(device='cpu'):
    from test_successive_allocation import small_successive
    raw, geometry, f, model = small_successive()
    model.to(device).eval().requires_grad_(False)
    ids = torch.arange(16).reshape(2, 8); ids[1, 5:] = -1
    transform = torch.eye(4, device=device); transform[3, 2] = 3
    cameras = []
    for offset in (0., .1):
        view = transform.clone(); view[3, 0] = offset
        cameras.append(SimpleNamespace(image_width=17, image_height=16, FoVx=1.2, FoVy=1.2,
            world_view_transform=view, full_proj_transform=torch.eye(4, device=device),
            camera_center=torch.zeros(3, device=device)))
    scene = SimpleNamespace(raw=raw[:13], geometry=geometry, groups=[f], group_ids=[ids], degree=1,
        white_background=False, reference=SimpleNamespace(get=lambda c, d: torch.zeros(3, 16, 17, device=d)))
    args = SimpleNamespace(snr=10., beta=.1, label_groups=4, candidate_batch=2,
                           allocation_batch=64, neighbors=4, cpu_threads=1)
    reference = dict(lambda_s=.01, d_ref=.01, c_ref=1000.,
                     side_proxy=dict(xyz_per_retained=24., bits_per_use=2.))
    return model, scene, args, reference, cameras


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_fast_collect_preserves_labels_predictions_and_mirror_update(tmp_path, device):
    from gaussian_jscc.batched_spatial_window import BatchedSpatialWindow
    from gaussian_jscc.spatial_benefit_features import SpatialWindow
    from gaussian_jscc.spatial_benefit import SpatialLearner
    from gaussian_jscc.spatial_benefit_replay import SpatialReplay
    from gaussian_jscc.response_value import ResponseMirrorTable
    from gaussian_jscc.response_value_window import ResponseScheduler
    from gaussian_jscc.pixel_local_renderer import _project
    torch.set_num_threads(1); torch.manual_seed(13)
    model, scene, args, reference, cameras = scene_fixture(device)
    learner_states = None; results = []; replays = []; tables = []; calls = []
    for cls in (SpatialWindow, BatchedSpatialWindow):
        table = ResponseMirrorTable(13).to(device)
        with torch.no_grad():
            table.keep_logits[[0, 8], 0] = 50
            table.keep_logits[[0, 8], 1] = -50
        scheduler = ResponseScheduler(scene.group_ids, seed=7)
        replay = SpatialReplay(tmp_path/cls.__name__, validation_groups=2, cache_mib=1)
        learner = SpatialLearner(replay, device, hidden=16, dropout=0)
        window = cls(model, scene, table, scheduler, reference, args)
        torch.manual_seed(123)
        window.collect(1, cameras, learner, replay, True)
        learner.fit_scales(); replay.freeze_validation()
        if learner_states is None:
            learner_states = copy.deepcopy(learner.state_dict())
        else:
            learner.load_state_dict(learner_states)
        before = len(replay.training)
        torch.manual_seed(919)
        with patch('gaussian_jscc.pixel_local_renderer._project', wraps=_project) as projected:
            result = window.collect(2, cameras, learner, replay, False)
        calls.append(projected.call_count)
        table.update(result['keep_costs'], result['tier_ids'], result['tier_costs'], .01, .01, 13)
        results.append(result); tables.append(table)
        replays.append([replay._read(e) for e in replay.training[before:]])
    old, fast = results
    assert calls[1] < calls[0]
    assert old['tier_ids'].tolist() == fast['tier_ids'].tolist()
    assert old['q'][0] == 0 and old['q'][8] == 0
    assert old['metrics']['evaluated_groups'] == fast['metrics']['evaluated_groups'] == 4
    for name in ('q', 'keep_costs', 'tier_costs'):
        torch.testing.assert_close(old[name], fast[name], rtol=3e-5, atol=2e-6)
    for name, value in tables[0].state_dict().items():
        torch.testing.assert_close(value, tables[1].state_dict()[name], rtol=3e-5, atol=2e-6)
    assert [g['source'] for g in replays[0]] == [g['source'] for g in replays[1]]
    for a, b in zip(*replays):
        torch.testing.assert_close(a['rates'], b['rates'], rtol=0, atol=0)
        for x, y in zip(a['views'], b['views']):
            torch.testing.assert_close(x['values'], y['values'], rtol=3e-5, atol=2e-6)
            for key in x['inputs']:
                torch.testing.assert_close(x['inputs'][key], y['inputs'][key], rtol=3e-5, atol=2e-6)
    assert all(p.grad is None for p in model.parameters())
