from types import SimpleNamespace
from unittest.mock import patch
import torch
from gaussian_jscc.student_policy_features import retained_latent, summarize_visibility, visibility_from_capture
from test_successive_allocation import small_successive


def test_all_keep_latent_is_aligned_to_original_rows_and_padding_is_masked():
    _, geometry, f, model = small_successive(); model.eval().requires_grad_(False)
    order = torch.tensor([5, 0, 15, 2, 8, 6, 11, 3, 9, 1, 7, 4, 10, 14, 13, 12])
    scene = SimpleNamespace(groups=[f], group_ids=[order.reshape(2, 8)], raw=torch.empty(16, 1))
    expected = model.semantic_encoder(f, f[..., :3], torch.ones(2, 8).bool()).detach()
    result = retained_latent(model, scene)
    torch.testing.assert_close(result[order], expected.reshape(16, -1))


def test_visibility_summary_preserves_missingness_and_camera_order_invariance():
    features = torch.randn(4, 3, 14)
    visible = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]]).bool()
    result = summarize_visibility(features, visible)
    assert result.shape == (3, 43) and torch.isfinite(result).all()
    assert torch.equal(result[2], torch.zeros(43))
    torch.testing.assert_close(result[:, -1], torch.tensor([.75, .5, 0.]))
    torch.testing.assert_close(result, summarize_visibility(features.flip(0), visible.flip(0)))


def test_capture_statistics_return_original_row_order():
    order = torch.tensor([2, 0, 1]); raw = torch.zeros(3, 14); raw[:, 0] = torch.tensor([30., 10., 20.])
    scene = SimpleNamespace(raw=raw, order=order, geometry=SimpleNamespace(span=torch.ones(3)))
    camera = SimpleNamespace(camera_center=torch.zeros(3))
    projected = torch.zeros(3, 12); projected[:, 2] = raw[:, 0]; projected[:, 3] = 1; projected[:, 5] = 1
    stats = torch.zeros(3, 12); stats[:, 0:2] = 1; stats[:, 4] = 2; stats[:, 11] = 1
    tile = SimpleNamespace(projected=projected, height=8, width=8)
    features, seen = visibility_from_capture(scene, camera, tile, stats)
    assert seen.tolist() == [True, True, True]
    torch.testing.assert_close(features[:, 6], torch.tensor([10., 20., 30.]))
