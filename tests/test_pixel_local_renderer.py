"""Exact local compositing, including contributors hidden by early termination."""
import importlib.util
import math
from types import SimpleNamespace

import pytest
import torch


def renderer():
    assert importlib.util.find_spec("gaussian_jscc.pixel_local_renderer") is not None, "local renderer is missing"
    from gaussian_jscc import pixel_local_renderer
    return pixel_local_renderer


def splat(x=8., y=8., z=1., color=(1., 0., 0.), opacity=.8, radius=3., conic=1.):
    return [x, y, z, conic, 0., conic, opacity, 1., *color, radius]


def scene(rows, height=16, width=32, white=False, backend="cpu"):
    return renderer().TileScene.from_projected(torch.tensor(rows, dtype=torch.float32,
        device="cuda" if backend == "cuda" else "cpu").reshape(-1, 12), height, width,
        white_background=white, backend=backend)


def test_removing_foreground_reveals_full_depth_tail():
    s = scene([splat(z=i + 1, opacity=.99) for i in range(8)] +
              [splat(z=20, color=(0., 0., 1.), opacity=.8)])
    assert len(s.tile_ids) == 9
    ids = torch.arange(8)
    replacement = s.projected[ids].clone()
    replacement[:, 7] = 0
    out = s.render_tiles(ids, replacement, torch.tensor([0]))
    torch.testing.assert_close(out[0, :, 8, 8], torch.tensor([0., 0., .8]))
    assert s.image[2, 8, 8] == 0


def test_candidate_expansion_enters_new_tile_and_preserves_background():
    s = scene([splat(x=7., radius=2.), splat(x=24., z=2., color=(0., 0., 1.))])
    candidate = s.projected[:1].clone()
    candidate[:, 11] = 30
    candidate[:, 3] = candidate[:, 5] = .001
    affected = s.affected_tiles(torch.tensor([0]), candidate.expand(4, -1, -1))
    assert affected.tolist() == [0, 1]
    out = s.render_tiles(torch.tensor([0]), candidate, affected)
    alpha = .8 * math.exp(-.5 * .001 * 17**2)
    torch.testing.assert_close(out[1, :, 8, 8], torch.tensor([alpha, 0., (1-alpha)*.8]))


def test_depth_ties_use_original_rows_even_when_candidate_ids_reversed():
    s = scene([splat(opacity=.5), splat(color=(0., 1., 0.), opacity=.5),
               splat(color=(0., 0., 1.), opacity=.5)])
    ids = torch.tensor([2, 0])
    out = s.render_tiles(ids, s.projected[ids], torch.tensor([0]))
    torch.testing.assert_close(out[0, :, 8, 8], torch.tensor([.5, .25, .125]))


def test_offscreen_white_background_partial_tile_and_owned_cache():
    p = torch.tensor([splat(x=-200., radius=2.)])
    s = renderer().TileScene.from_projected(p, 17, 19, white_background=True, backend="cpu")
    p.fill_(9.)
    assert s.projected[0, 0] == -200
    assert len(s.tile_ids) == 0
    assert torch.equal(s.image, torch.ones(3, 17, 19))
    tiles = s.render_tiles(torch.tensor([], dtype=torch.long), s.projected[:0], torch.tensor([3]))
    assert torch.equal(tiles[0, :, 0, :3], torch.ones(3, 3))
    assert tiles[0, :, 1:].count_nonzero() == 0


def camera(height=32, width=48):
    projection = torch.zeros(4, 4)
    projection[0, 0] = projection[1, 1] = projection[2, 2] = projection[2, 3] = 1
    return SimpleNamespace(image_height=height, image_width=width, FoVx=math.pi/2,
        FoVy=math.pi/2, world_view_transform=torch.eye(4), full_proj_transform=projection,
        camera_center=torch.zeros(3))


def raw_scene():
    raw = torch.zeros(5, 14)
    raw[:, :3] = torch.tensor([[0., 0., 2.], [.7, 0., 3.], [0., 0., 4.], [100., 0., 2.], [0., 0., .1]])
    raw[:, 3] = 2
    raw[:, 4:7] = math.log(.12)
    raw[:, 7] = 1
    raw[:, 11:14] = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.], [1., 1., 1.]])
    return raw


def test_capture_uses_perspective_covariance_lowpass_and_near_culling():
    s = renderer().capture(raw_scene(), camera(), 0, torch.ones(5), backend="cpu")
    torch.testing.assert_close(s.projected[0, :3], torch.tensor([23.5, 15.5, 2.]))
    torch.testing.assert_close(s.projected[0, [3, 5]], torch.tensor([1/(1.44**2+.3), 1/(.96**2+.3)]))
    assert s.projected[3:, 11].count_nonzero() == 0


def test_alpha_threshold_precedes_existence_and_crossing_splat_is_excluded():
    small = scene([splat(opacity=.004)])
    candidate = small.projected.clone()
    candidate[:, 7] = .1
    out = small.render_tiles(torch.tensor([0]), candidate, torch.tensor([0]))
    torch.testing.assert_close(out[0, :, 8, 8], torch.tensor([.0004, 0., 0.]))
    opaque = scene([splat(opacity=1.), splat(z=2., color=(0., 1., 0.), opacity=1.)])
    torch.testing.assert_close(opaque.image[:, 8, 8], torch.tensor([.99, 0., 0.]))


def test_halo_and_empty_interventions_keep_exact_tile_geometry():
    s = scene([splat(x=24., y=24., radius=1.)], height=64, width=64)
    affected = s.affected_tiles(torch.tensor([0]), s.projected[None].expand(4, -1, -1), halo_tiles=1)
    assert affected.tolist() == [0, 1, 2, 4, 5, 6, 8, 9, 10]
    empty = torch.empty(0, dtype=torch.long)
    assert s.affected_tiles(empty, torch.empty(4, 0, 12)).numel() == 0
    assert s.render_tiles(empty, torch.empty(0, 12), empty).shape == (0, 3, 16, 16)


def test_candidate_changed_depth_is_reordered_against_unchanged_scene():
    s = scene([splat(opacity=.5), splat(z=2., color=(0., 0., 1.), opacity=.5)])
    p = s.projected[:1].clone()
    p[:, 2] = 3
    out = s.render_tiles(torch.tensor([0]), p, torch.tensor([0]))
    torch.testing.assert_close(out[0, :, 8, 8], torch.tensor([.25, 0., .5]))


def test_capture_owns_camera_projection_for_later_candidates():
    cam = camera()
    raw = raw_scene()
    s = renderer().capture(raw, cam, 0, torch.ones(5), backend="cpu")
    expected = s.projected.clone()
    cam.world_view_transform.zero_()
    cam.full_proj_transform.zero_()
    torch.testing.assert_close(s.project(raw, torch.ones(5)), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and nvcc")
@pytest.mark.parametrize("white", [False, True])
def test_cuda_matches_original_mask_renderer_and_full_interventions(white):
    from gaussian_jscc.rendering import render
    r = renderer()
    raw = raw_scene().cuda()
    exists = torch.tensor([1., .5, 1., 1., 1.], device="cuda")
    cam = camera()
    s = r.capture(raw, cam, 0, exists, white_background=white)
    torch.testing.assert_close(s.image, render(raw, cam, 0, white, exists), atol=3e-5, rtol=3e-5)
    cpu = r.capture(raw.cpu(), cam, 0, exists.cpu(), white_background=white, backend="cpu")
    torch.testing.assert_close(s.projected.cpu(), cpu.projected, atol=3e-5, rtol=3e-5)
    for mode in ("delete", "expand", "depth", "offscreen"):
        changed = raw[:2].clone()
        mask = exists[:2].clone()
        if mode == "delete": mask.zero_()
        if mode == "expand": changed[:, 4:7] += 2
        if mode == "depth": changed[:, 2] = 5
        if mode == "offscreen": changed[:, 0] = -100
        p = s.project(changed, mask)
        ids = torch.arange(2, device="cuda")
        affected = s.affected_tiles(ids, p.expand(4, -1, -1))
        local = s.render_tiles(ids, p, affected)
        full_raw, full_mask = raw.clone(), exists.clone()
        full_raw[:2], full_mask[:2] = changed, mask
        expected = render(full_raw, cam, 0, white, full_mask)
        actual = s.image.clone()
        for k, tid in enumerate(affected.tolist()):
            y, x = divmod(tid, (s.width+15)//16)
            hh, ww = min(16, s.height-y*16), min(16, s.width-x*16)
            actual[:, y*16:y*16+hh, x*16:x*16+ww] = local[k, :, :hh, :ww]
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and nvcc")
def test_cuda_deep_occlusion_and_stable_candidate_merge_match_cpu():
    rows = [splat(z=i+1, opacity=.99) for i in range(700)] + [splat(z=800, color=(0., 0., 1.))]
    cpu, gpu = scene(rows), scene(rows, backend="cuda")
    ids = torch.arange(700)
    candidate = cpu.projected[ids].clone()
    candidate[:, 7] = 0
    expected = cpu.render_tiles(ids, candidate, torch.tensor([0]))
    actual = gpu.render_tiles(ids.cuda(), candidate.cuda(), torch.tensor([0], device="cuda"))
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and nvcc")
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_cuda_projection_rotation_sh_and_nondefault_stream(degree):
    from gaussian_jscc.rendering import render
    generator = torch.Generator().manual_seed(123)
    raw = torch.randn(40, 11+3*(degree+1)**2, generator=generator)*.2
    raw[:, 2] = torch.linspace(.1, 5., len(raw))
    raw[:, 4:7] = torch.randn(40, 3, generator=generator)*.5-2
    raw[:, 7:11] = torch.randn(40, 4, generator=generator)
    cam = camera(height=35, width=49)
    cam.FoVx, cam.FoVy = .93, 1.17
    cam.full_proj_transform[0, 0] = 1/math.tan(cam.FoVx/2)
    cam.full_proj_transform[1, 1] = 1/math.tan(cam.FoVy/2)
    angle = .2
    cam.world_view_transform[:3, :3] = torch.tensor([
        [math.cos(angle), 0., -math.sin(angle)], [0., 1., 0.],
        [math.sin(angle), 0., math.cos(angle)]])
    cam.world_view_transform[3, :3] = torch.tensor([.1, -.2, .3])
    cam.full_proj_transform = cam.world_view_transform @ cam.full_proj_transform
    cam.camera_center = torch.linalg.inv(cam.world_view_transform)[3, :3]
    cpu = renderer().capture(raw, cam, degree, torch.ones(len(raw)), backend="cpu")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gpu = renderer().capture(raw.cuda(), cam, degree, torch.ones(len(raw), device="cuda"))
    stream.synchronize()
    torch.testing.assert_close(gpu.projected.cpu(), cpu.projected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gpu.image.cpu(), cpu.image, atol=1e-4, rtol=1e-4)
    expected = render(raw.cuda(), cam, degree, existence=torch.ones(len(raw), device="cuda"))
    torch.testing.assert_close(gpu.image, expected, atol=3e-5, rtol=3e-5)
