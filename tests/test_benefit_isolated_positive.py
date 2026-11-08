import copy

import torch

from gaussian_jscc.benefit_isolated_positive_model import IsolatedPositiveNetwork, positive_losses
from gaussian_jscc.benefit_footprint_model import footprint_pack
from gaussian_jscc.spatial_benefit import SpatialBenefitNetwork, pack_inputs


def inputs():
    from test_spatial_benefit import collision_inputs
    value = copy.deepcopy(collision_inputs()[0][0])
    value['footprints'] = torch.randn(4, 2, 15, 9, 9)
    return value


def test_initial_function_and_partial_identical_candidates_preserved():
    torch.manual_seed(73)
    old = SpatialBenefitNetwork(16, 0).eval()
    net = IsolatedPositiveNetwork.from_spatial(old.state_dict(), 0).eval()
    for identical_count in (0,1,3):
        value = inputs()
        for i in range(1,identical_count+1):
            value['features'][i] = value['features'][0]
            value['states'][i] = value['states'][0]
            value['footprints'][i] = value['footprints'][0]
        expected = old(*pack_inputs([value], 'cpu'))
        actual = net(*footprint_pack([value], 'cpu'))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        if identical_count:
            assert torch.equal(actual[:, :identical_count], torch.zeros_like(actual[:, :identical_count]))


def test_positive_updates_cannot_change_common_values_or_anchor_parameters():
    net = IsolatedPositiveNetwork.from_spatial(SpatialBenefitNetwork(16, 0).state_dict(), 0)
    value = inputs()
    value['features'][1] = value['features'][0]
    value['states'][1] = value['states'][0]
    value['footprints'][1] = value['footprints'][0]
    packed = footprint_pack([value], 'cpu')
    before = net(*packed).mean(1).detach()
    frozen = copy.deepcopy(net.anchor.state_dict())
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        out = net(*packed)
        ((out[:, 2]-out[:, 1])-1).square().sum().backward()
        assert all(p.grad is None for p in net.anchor.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.relative.points.parameters())
        optimizer.step()
    torch.testing.assert_close(net(*packed).mean(1), before, atol=1e-6, rtol=1e-5)
    assert all(torch.equal(v, net.anchor.state_dict()[k]) for k, v in frozen.items())
    net.train()
    assert not net.anchor.training


def test_direct_positive_loss_ignores_common_shift_and_uses_physical_scale():
    p = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(p)
    rates = torch.tensor([[.1,.2,.4],[.2,.4,.8]], dtype=torch.float64)
    a = positive_losses(p, t, rates, .5)
    b = positive_losses(p+3., t-7., rates, .5)
    for k in a:
        torch.testing.assert_close(a[k], b[k])
    a['total'].sum().backward()
    torch.testing.assert_close(p.grad.sum(1), torch.zeros(2,2,dtype=p.dtype), atol=1e-10, rtol=0)
    c = positive_losses(p.detach()*2, t*2, rates*2, 1.)
    torch.testing.assert_close(a['total'], c['total'])
