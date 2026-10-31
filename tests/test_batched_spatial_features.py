from unittest.mock import patch

import pytest
import torch

from test_spatial_benefit import collision_inputs


def test_batched_features_match_serial_and_reuse_block_baseline():
    from gaussian_jscc.batched_spatial_features import spatial_inputs_batch
    from gaussian_jscc.spatial_benefit_features import spatial_inputs
    from gaussian_jscc.physical_benefit_features import visibility_statistics
    _, tile, target, a, b = collision_inputs()
    raw = torch.zeros(3, 4, 3, 14); raw[..., 7] = 1
    raw[1, :, :, 4:7] = .2
    projected = torch.stack((a, b, a))
    stats = visibility_statistics(tile, target)[None].expand(3, -1, -1)
    changed = torch.arange(3)[None].expand(3, -1)
    static = torch.arange(72).reshape(3, 24).float()/100
    points = torch.tensor([0, 1, 2])
    expected = [spatial_inputs(static[i], stats[i], raw[i], projected[i], int(points[i]),
        tile, target, changed[i], .01, .006) for i in range(3)]
    actual = spatial_inputs_batch(static, stats, raw, projected, points, tile, target,
                                  changed, .01, .006, [(0, 1, 2)]*3)
    for x, y in zip(actual, expected):
        for key in y:
            torch.testing.assert_close(x[key], y[key], rtol=1e-6, atol=1e-7)
    cached = tile._batched_spatial_baselines[(0, 1, 2)]
    spatial_inputs_batch(static, stats, raw, projected, points, tile, target,
                         changed, .01, .006, [(0, 1, 2)]*3)
    assert tile._batched_spatial_baselines[(0, 1, 2)] is cached
    raw[0, 1, 0, 4] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        spatial_inputs_batch(static, stats, raw, projected, points, tile, target,
                             changed, .01, .006, [(0, 1, 2)]*3)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_one_projection_call_preserves_four_states_and_ragged_tail(device):
    from types import SimpleNamespace
    from gaussian_jscc.batched_spatial_features import project_candidates
    from gaussian_jscc.pixel_local_renderer import capture
    transform = torch.eye(4, device=device); transform[3, 2] = 3
    cam = SimpleNamespace(image_width=31, image_height=17, FoVx=1.2, FoVy=1.2,
        world_view_transform=transform, full_proj_transform=torch.eye(4, device=device),
        camera_center=torch.zeros(3, device=device))
    raw = torch.zeros(5, 14, device=device); raw[:, 7] = 1; raw[:, 4:7] = -2
    raw[:, 0] = torch.linspace(-1, 1, 5, device=device)
    tile = capture(raw, cam, 0, torch.ones(5, device=device), backend=device)
    raws = [raw[:3][None].repeat(4, 1, 1), raw[3:][None].repeat(4, 1, 1)]
    raws[0][2, 0, 0] += 2
    tiers = [torch.ones(4, len(r[0]), device=device, dtype=torch.long) for r in raws]
    tiers[0][0, 0] = 0
    expected = [torch.stack([tile.project(r[q], (t[q] > 0).float()) for q in range(4)])
                for r, t in zip(raws, tiers)]
    with patch.object(tile, 'project', wraps=tile.project) as projected:
        actual = project_candidates(tile, raws, tiers)
    assert projected.call_count == 1
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
