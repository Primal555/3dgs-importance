import torch
import torch.nn.functional as F

from gaussian_jscc.pixel_benefit import (
    PixelBenefitNetwork, pixel_density_labels, integrate_density, density_loss,
    fit_density_scales,
)
from gaussian_jscc.allocation_distortion import ssim_map


def inputs(length=4):
    return dict(image=torch.randn(1, 6, 26, 26),
                context=torch.randn(1, length, 13),
                context_mask=torch.ones(1, length, dtype=torch.bool),
                candidate=torch.randn(1, 4, 3, 12),
                candidate_mask=torch.ones(1, 4, 3, dtype=torch.bool),
                valid=torch.ones(1, 1, 16, 16))


def test_labels_match_whole_image_with_zero_padding_and_clamp():
    torch.manual_seed(3)
    target = torch.rand(3, 16, 16, dtype=torch.float64)
    images = torch.rand(4, 3, 16, 16, dtype=torch.float64) * 1.4 - .2
    padded = F.pad(images, (5, 5, 5, 5))
    labels = pixel_density_labels(padded, F.pad(target, (5, 5, 5, 5)), .3)
    for q in range(1, 4):
        mse = (images[q] - target).square().mean() - (images[0] - target).square().mean()
        structure = (ssim_map(images[0], target) - ssim_map(images[q], target)).mean() * .3
        assert torch.allclose(labels[q-1, 0].mean(), mse, atol=1e-10)
        assert torch.allclose(labels[q-1, 1].mean(), structure, atol=1e-10)
    batch = pixel_density_labels(padded[None], F.pad(target, (5, 5, 5, 5))[None], .3)
    assert torch.equal(batch[0], labels)


def test_integral_and_full_group_loss_penalize_common_error():
    pred = torch.ones(2, 3, 2, 16, 16, requires_grad=True)
    target = torch.zeros_like(pred)
    valid = torch.ones(2, 1, 16, 16)
    valid[1, :, 8:] = 0
    value = integrate_density(pred, valid, 32, 16, 2.)
    assert torch.allclose(value, torch.full((3, 2), .375,dtype=torch.float64))
    loss = density_loss(pred, target, valid, 32, 16, 2., torch.ones(2), torch.ones(2))
    loss.backward()
    assert pred.grad[0].abs().sum() > 0
    assert pred.grad[1, :, :, :8].abs().sum() > 0
    assert pred.grad[1, :, :, 8:].abs().sum() == 0
    scales = fit_density_scales(target + 2, valid, 32, 16, 2.)
    assert torch.allclose(scales[0], torch.full((2,), 2.))
    assert torch.allclose(scales[1], torch.full((2,), .75,dtype=torch.float64))


def test_all_context_tokens_influence_output_and_identical_candidates_anchor_zero():
    torch.manual_seed(9)
    net = PixelBenefitNetwork(hidden=8)
    data = inputs(70)
    data['context'].requires_grad_()
    out = net(data)
    assert out.shape == (1, 3, 2, 16, 16)
    out.square().sum().backward()
    assert data['context'].grad[0, -1].abs().sum() > 0
    data['candidate'] = data['candidate'][:, :1].expand(-1, 4, -1, -1).clone()
    assert torch.equal(net(data), torch.zeros_like(out))


def test_network_fits_small_dense_target():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    net = PixelBenefitNetwork(hidden=8)
    data = inputs(3)
    wanted = torch.zeros(1, 3, 2, 16, 16)
    wanted[:, :, 0] = .2
    wanted[:, :, 1] = -.1
    opt = torch.optim.Adam(net.parameters(), lr=.02)
    for _ in range(80):
        opt.zero_grad()
        loss = (net(data) - wanted).square().mean()
        loss.backward()
        opt.step()
    assert loss.item() < .002


def test_output_scale_changes_physical_units_and_survives_state_dict():
    torch.manual_seed(12)
    net = PixelBenefitNetwork(hidden=8)
    data = inputs(2)
    initial = net(data).detach()
    net.set_output_scale(torch.tensor([.01, .02]))
    actual = net(data).detach()
    assert torch.allclose(actual, initial * torch.tensor([.01, .02])[None, None, :, None, None], atol=1e-8)
    restored = PixelBenefitNetwork(hidden=8)
    restored.load_state_dict(net.state_dict())
    assert torch.equal(restored(data), net(data))


def test_outer_halo_changes_edge_predictions():
    torch.manual_seed(20)
    net = PixelBenefitNetwork(hidden=8)
    data = inputs(3)
    data['image'].requires_grad_()
    net(data)[0, :, :, 0, 8].square().sum().backward()
    assert data['image'].grad[0, :, 0, :].abs().sum() > 0


def test_batch_matches_single_tiles_with_empty_and_long_sequences():
    torch.manual_seed(21)
    net = PixelBenefitNetwork(hidden=8).eval()
    a = inputs(70)
    data = {key: torch.cat((value, value.clone(), value.clone()), dim=0) for key, value in a.items()}
    data['context_mask'][1] = False
    data['context_mask'][2, 20:] = False
    data['candidate_mask'][1, 0] = False
    data['candidate_mask'][2, 1, 1:] = False
    combined = net(data)
    separate = torch.cat([net({key: value[i:i+1] for key, value in data.items()}) for i in range(3)])
    assert torch.allclose(combined, separate, rtol=2e-5, atol=1e-6)
