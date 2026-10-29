"""Strict preflight checks must reject errors and integrate complete SSIM halos."""
import importlib.util

import pytest
import torch


def preflight():
    assert importlib.util.find_spec('gaussian_jscc.pixel_benefit_preflight') is not None, 'preflight is missing'
    from gaussian_jscc import pixel_benefit_preflight
    return pixel_benefit_preflight


def test_error_check_reports_real_error_and_rejects_large_or_nonfinite_difference():
    check = preflight().assert_close
    result = check('fixture', torch.tensor([1., 2.]), torch.tensor([1., 2.00001]), atol=2e-5, rtol=0.)
    assert 0 < result['max_absolute_error'] < 2e-5
    assert result['atol'] == 2e-5
    with pytest.raises(RuntimeError, match='fixture'):
        check('fixture', torch.tensor([1., 2.]), torch.tensor([1., 2.01]), atol=2e-5, rtol=0.)
    with pytest.raises(RuntimeError, match='nonfinite'):
        check('fixture', torch.tensor([float('nan')]), torch.tensor([0.]), atol=2e-5, rtol=0.)


def test_full_image_components_match_dense_group_with_expansion_and_image_edges():
    from gaussian_jscc.pixel_local_renderer import TileScene
    module = preflight()
    p = torch.tensor([[2., 3., 1., .06, 0., .06, .8, 1., 1., .1, .2, 13.],
                      [23., 19., 2., .2, 0., .2, .9, 1., .1, .8, .2, 7.],
                      [43., 32., 3., .1, 0., .1, .7, 1., .2, .1, .9, 10.]])
    s = TileScene.from_projected(p, 35, 47, white_background=True, backend='cpu')
    ids = torch.tensor([0, 2])
    candidates = p[ids].expand(4, -1, -1).clone()
    candidates[0, :, 7] = 0
    candidates[1, :, 6] = .4
    candidates[2, 0, 2] = 4
    candidates[3, :, 11] = 30
    candidates[3, :, 3] = candidates[3, :, 5] = .006
    target = torch.rand(3, 35, 47, generator=torch.Generator().manual_seed(4))
    def full(q):
        alternate = p.clone()
        alternate[ids] = candidates[q]
        return TileScene.from_projected(alternate, 35, 47, white_background=True, backend='cpu').image
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report = module.verify_group(s, ids, candidates, target, full, lambda_s=.1, d_ref=.03)
    finally:
        torch.set_num_threads(threads)
    assert len(report['states']) == 4
    assert report['dense_vs_full_physical']['max_absolute_error'] < 1e-12
    assert report['dense_vs_full_normalized']['max_absolute_error'] < 1e-10
    assert report['label_tiles'] == 9


def test_tile_stitch_keeps_pixels_outside_selected_tiles():
    module = preflight()
    image = torch.ones(3, 18, 19)
    replacement = torch.full((1, 3, 16, 16), .2)
    result = module.stitch_tiles(image, torch.tensor([3]), replacement)
    assert torch.equal(result[:, :16], image[:, :16])
    torch.testing.assert_close(result[:, 16:, 16:], torch.full((3, 2, 3), .2))
    assert torch.equal(image, torch.ones_like(image))


def test_dense_check_rejects_small_systematic_error_even_when_pixel_tolerance_passes():
    from gaussian_jscc.pixel_local_renderer import TileScene
    module = preflight()
    p = torch.tensor([[8., 8., 1., 1., 0., 1., 0., 1., 1., 0., 0., 3.]])
    s = TileScene.from_projected(p, 16, 16, backend='cpu')
    def full(q):
        return s.image + (1e-8 if q == 1 else 0.)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with pytest.raises(RuntimeError, match='dense integral'):
            module.verify_group(s, torch.tensor([0]), p[None].expand(4, -1, -1),
                torch.full((3, 16, 16), .5), full, lambda_s=.1, d_ref=.03)
    finally:
        torch.set_num_threads(threads)


def test_real_frozen_codec_reconstruction_has_deterministic_original_noise_slots():
    from types import SimpleNamespace
    from test_successive_allocation import small_successive
    module = preflight()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        _, _, features, model = small_successive()
        model.eval().requires_grad_(False)
        scene = SimpleNamespace(groups=[features], group_ids=[torch.arange(16).reshape(2, 8)])
        before = torch.get_rng_state()
        first = module._decode_batch(model, scene, 0, 19, 10., torch.device('cpu'))
        second = module._decode_batch(model, scene, 0, 19, 10., torch.device('cpu'))
        assert torch.equal(torch.get_rng_state(), before)
        assert torch.equal(first[2], torch.tensor([[3, 0, 1, 2, 3, 0, 1, 2], [3, 0, 1, 2, 3, 0, 1, 2]]))
        torch.testing.assert_close(first[4], second[4], atol=0., rtol=0.)
        torch.testing.assert_close(first[5], second[5], atol=0., rtol=0.)
        assert first[5].shape == (2, 8, 2*model.cfg.rates[-1])
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('white', [False, True])
def test_expansion_probe_covers_new_tiles_and_matches_full_compositing(white):
    from gaussian_jscc.pixel_local_renderer import capture
    module = preflight()
    def full(raw, camera, degree, white_background, existence):
        return capture(raw, camera, degree, existence, white_background, backend='cpu').image
    before = torch.get_rng_state()
    report = module.verify_expansion_probe(torch.device('cpu'), white, full)
    assert report['original_tiles'] == 1
    assert report['new_tiles'] > 0
    assert report['new_tile_max_change'] > .001
    assert report['image']['max_absolute_error'] < 1e-6
    assert report['baseline']['max_absolute_error'] < 1e-6
    assert torch.equal(before, torch.get_rng_state())


def test_expansion_probe_rejects_renderer_that_omits_new_footprint():
    from gaussian_jscc.pixel_local_renderer import capture
    module = preflight()
    initial = None
    def stuck(raw, camera, degree, white_background, existence):
        nonlocal initial
        if initial is None:
            initial = capture(raw, camera, degree, existence, white_background, backend='cpu').image
        return initial
    with pytest.raises(RuntimeError, match='expanded footprint'):
        module.verify_expansion_probe(torch.device('cpu'), False, stuck)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA expansion probe requires GPU')
def test_cuda_expansion_probe_matches_original_renderer():
    from gaussian_jscc.rendering import render
    report = preflight().verify_expansion_probe(torch.device('cuda'), False, render)
    assert report['new_tiles'] > 0
    assert report['new_tile_max_change'] > .001


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA mask requires GPU')
def test_cpu_teacher_density_accepts_cuda_valid_mask():
    from gaussian_jscc.pixel_benefit import integrate_density
    density=torch.ones(2,3,2,16,16,dtype=torch.float64)
    valid=torch.ones(2,1,16,16,device='cuda')
    actual=integrate_density(density,valid,16,32,1.)
    torch.testing.assert_close(actual,torch.ones(3,2,dtype=torch.float64))
