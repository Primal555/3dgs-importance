import pytest
import torch
from gaussian_jscc import allocation_distortion as ad
from utils.loss_utils import ssim


def test_calibration_uses_fixed_reference_share_and_manual_override():
    ref = ad.calibrate(.004, .08, mode='mse_ssim', share=.2)
    assert ref['lambda_s'] == pytest.approx(.0125)
    assert ref['d_ref'] == pytest.approx(.005)
    assert ref['structure_reference_fraction'] == pytest.approx(.2)
    assert ad.calibrate(.004, .08, lambda_s=.1)['lambda_s'] == .1
    assert ad.calibrate(.004, .08, mode='mse')['d_ref'] == .004
    with pytest.raises(ValueError): ad.calibrate(.004, 0.)
    with pytest.raises(ValueError): ad.calibrate(.004, .08, lambda_s=float('nan'))


def test_component_definition_matches_reporting_and_paired_difference():
    torch.set_num_threads(1); torch.manual_seed(4)
    target = torch.rand(3, 19, 23)
    image = (.9*target+.03).requires_grad_()
    obj = ad.ImageDistortion('mse_ssim', .0125)
    mse, structure = obj.components(image, target)
    assert float(mse.detach()) == pytest.approx(float((image.detach()-target).double().square().mean()))
    assert float(structure.detach()) == pytest.approx(1-float(ssim(image.detach(), target)), abs=2e-6)
    changed = image.detach().clone(); changed[:, 9, 11] += 1e-5
    delta = obj.delta(changed, image.detach(), target)
    direct = obj.components(changed.double(), target.double())-obj.components(image.double(), target.double())
    torch.testing.assert_close(delta, direct, atol=1e-14, rtol=1e-6)
    assert torch.equal(obj.delta(image, image, target), torch.zeros(2, dtype=torch.float64))
    grad, = torch.autograd.grad(obj.combine(torch.stack((mse, structure))), image)
    assert torch.isfinite(grad).all() and grad.norm() > 0


def test_mse_mode_does_not_evaluate_structure(monkeypatch):
    def forbidden(*args): raise AssertionError('SSIM in MSE-only path')
    monkeypatch.setattr(ad, 'ssim_map', forbidden)
    obj = ad.ImageDistortion('mse', 0.)
    x, y = torch.ones(3, 5, 5), torch.zeros(3, 5, 5)
    assert obj.components(x, y).tolist() == [1., 0.]
    assert obj.delta(x, y, y).tolist() == [1., 0.]


def test_streamed_component_gradient_matches_direct_multiview(monkeypatch):
    from types import SimpleNamespace
    from test_mask_gradient import toy_mask_render, Reference
    monkeypatch.setattr('gaussian_jscc.rendering.render', toy_mask_render)
    torch.set_num_threads(1); torch.manual_seed(17)
    scene = torch.rand(5, 15, requires_grad=True)
    cameras = [SimpleNamespace(factor=f, source_count=5) for f in (1., 2.)]
    obj = ad.ImageDistortion('mse_ssim', .0125)
    task = ad.ComponentRenderTask(cameras, Reference(), 1, distortion=obj)
    direct = torch.stack([obj.combine(obj.components(toy_mask_render(scene[:, :-1], c, 1,
        existence=scene[:, -1]), Reference().get(c, 'cpu'))) for c in cameras]).mean()
    expected, = torch.autograd.grad(direct, scene)
    loss, gradient = task.component_gradient(scene)
    assert loss == pytest.approx(float(direct.detach()))
    torch.testing.assert_close(gradient, expected)
    assert torch.equal(task.delta_components(scene.detach()), torch.zeros(2, dtype=torch.float64))
