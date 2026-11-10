import contextlib
from unittest.mock import patch
import pytest
import torch

from test_successive_allocation import small_successive
from test_structured_allocation_integration import run_kernel
from gaussian_jscc.budgeted_allocation_kernel import mixed_point_candidates, rng_state
from gaussian_jscc.fast_first_order_execution import FastFirstOrderExecution


def test_geometry_reuse_rebuilds_after_keep_or_coordinate_change():
    from gaussian_jscc.pointnet_context import SpatialPointContext, allocation_geometry_reuse
    from gaussian_jscc.data import Geometry
    from test_pointnet_context import PointContextTests
    torch.manual_seed(9); torch.set_num_threads(1)
    net = SpatialPointContext(PointContextTests().cfg()).eval()
    net.set_scene_geometry(Geometry([0, 0, 0], [3, 5, 8], 16))
    xyz = torch.rand(2, 16, 3).repeat(2, 1, 1)
    h = torch.randn(4, 16, 16); active = torch.ones(4, 16, dtype=torch.bool)
    active[:, -2:] = False
    with torch.no_grad():
        expected = net(h, xyz, active)
        with allocation_geometry_reuse(net, 2) as cache:
            torch.testing.assert_close(net(h, xyz, active), expected, rtol=0, atol=0)
            torch.testing.assert_close(net(h, xyz, active), expected, rtol=0, atol=0)
            assert (cache.builds, cache.hits) == (1, 1)
            active[0, 0] = False
            changed_keep = net(h, xyz, active)
            assert cache.builds == 2
            xyz[0, 2] += .3
            changed_position = net(h, xyz, active)
            assert cache.builds == 3
        xyz[0, 2] -= .3
        torch.testing.assert_close(net(h, xyz, active), changed_keep)
        xyz[0, 2] += .3
        torch.testing.assert_close(net(h, xyz, active), changed_position)
    assert not hasattr(net, '_allocation_geometry_cache')


@pytest.mark.parametrize('batch_size', [2, 4, 8])
@pytest.mark.parametrize('context', [False, True])
def test_batched_candidates_match_original_and_preserve_rng(batch_size, context):
    torch.set_num_threads(1); torch.manual_seed(213)
    _, geometry, f, model = small_successive()
    if context:
        from gaussian_jscc.model_factory import build_codec
        model.cfg.context_variant = 'both'
        old = model; model = build_codec(old.cfg)
        model.load_state_dict(old.state_dict(), strict=False)
        model.set_scene_geometry(geometry)
    model.eval().requires_grad_(False)
    q = torch.tensor([[1, 2, 0, 3, 1, 2, 3, 0], [3, 0, 2, 1, 2, 0, 1, 3]])
    columns = [torch.tensor([1, 3]), torch.tensor([3, -1])]
    y = model.semantic_encoder(f, f[..., :3], q > 0)
    state = rng_state(f.device)
    actual = model.forward_tier_batches(f, f[..., :3], torch.nn.functional.one_hot(q, 4).to(f),
                                       10., 'awgn', paired_noise=True, _semantic_y=y)[0]
    expected = [list(mixed_point_candidates(model, f, q, c, 10., 'awgn', geometry,
                semantic_y=y, actual=actual, actual_rng=state, timings=None)) for c in columns]
    before = torch.get_rng_state().clone()
    execution = FastFirstOrderExecution(batch_size)
    got = execution.candidates(model, f, q, columns, 10., 'awgn', geometry,
                               semantic_y=y, actual=actual, actual_rng=state, timings=None)
    assert torch.equal(before, torch.get_rng_state())
    for group, other in zip(expected, got):
        for (tier, delta), (new_tier, new_delta) in zip(group, other):
            assert torch.equal(tier, new_tier)
            torch.testing.assert_close(delta, new_delta, rtol=2e-5, atol=2e-6)
    assert execution.candidate_calls == (4 * len(q) + batch_size - 1) // batch_size
    if context: assert execution.geometry_builds > 0


@pytest.mark.parametrize('closed', [False, True])
def test_fast_kernel_preserves_feedback_sampling_and_dense_keep(closed):
    base, _, rng = run_kernel(8, closed=closed, calibrated=True)
    fast = FastFirstOrderExecution(8)
    with patch('test_structured_allocation_integration.budgeted_mask_step',
               wraps=__import__('gaussian_jscc.budgeted_allocation_kernel', fromlist=['budgeted_mask_step']).budgeted_mask_step) as call:
        original = call._mock_wraps
        call.side_effect = lambda *a, **kw: original(*a, **kw, execution=fast)
        got, _, after = run_kernel(8, closed=closed, calibrated=True)
    assert torch.equal(rng, after)
    assert torch.equal(base['q'], got['q'])
    assert torch.equal(base['tier_ids'], got['tier_ids'])
    for name in ('keep_costs', 'tier_costs'):
        torch.testing.assert_close(base[name], got[name], rtol=2e-4, atol=2e-6)
    assert len(base['audit']) == len(got['audit'])
    for a, b in zip(base['audit'], got['audit']):
        assert (a['source_row'], a['new_tier']) == (b['source_row'], b['new_tier'])
        assert b['measured_delta_rd'] == pytest.approx(a['measured_delta_rd'], abs=2e-6)
