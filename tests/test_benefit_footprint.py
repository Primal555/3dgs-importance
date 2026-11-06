import copy

import torch

from gaussian_jscc.benefit_footprint_features import footprint_inputs
from gaussian_jscc.benefit_footprint_model import FootprintSplitNetwork, footprint_pack
from gaussian_jscc.spatial_benefit import SpatialBenefitNetwork, pack_inputs


def test_new_footprints_see_candidate_expansion_outside_old_sampling():
    from test_spatial_benefit import collision_inputs
    inputs, tile, target, a, _ = collision_inputs()
    expanded = a.clone(); expanded[3, 0, 11] = 10.; expanded[3, 0, 3] = .02; expanded[3, 0, 5] = .02
    x = footprint_inputs(tile, target, expanded, torch.arange(3), 0)
    assert x.shape == (4, 2, 15, 9, 9)
    assert torch.isfinite(x).all()
    # Alter only the reference outside the old narrow central footprint.
    other = target.clone(); other[:, 0:3, 24:30] += .5
    y = footprint_inputs(tile, other, expanded, torch.arange(3), 0)
    assert not torch.equal(x, y)
    perm = torch.tensor([2, 0, 1])
    z = footprint_inputs(tile, target, expanded[:, perm], perm, 1)
    torch.testing.assert_close(x, z, rtol=1e-6, atol=1e-6)


def test_split_initialization_matches_and_preserves_padded_batch():
    from test_spatial_benefit import collision_inputs
    inputs, tile, target, a, _ = collision_inputs()
    value = dict(inputs[0], footprints=footprint_inputs(tile, target, a, torch.arange(3), 0))
    torch.manual_seed(19)
    baseline = SpatialBenefitNetwork(hidden=16, dropout=0).eval()
    net = FootprintSplitNetwork.from_spatial(baseline.state_dict(), dropout=0).eval()
    with torch.no_grad():
        expected = baseline(*pack_inputs([value], 'cpu'))
        actual = net(*footprint_pack([value], 'cpu'))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # After enabling the profile branch, its different background can affect a prediction.
    torch.nn.init.normal_(net.profile_projection.weight, std=.05)
    other = copy.deepcopy(value); other['footprints'][:, :, :3] += 1.
    assert not torch.allclose(net(*footprint_pack([value], 'cpu')), net(*footprint_pack([other], 'cpu')), atol=1e-9, rtol=0)


def test_common_and_relative_final_heads_receive_separate_supervision():
    from test_spatial_benefit import collision_inputs
    inputs, tile, target, a, _ = collision_inputs()
    # Keep every positive candidate distinguishable from q0 for this routing test.
    value = dict(inputs[0], footprints=footprint_inputs(tile, target, a, torch.arange(3), 0))
    net = FootprintSplitNetwork.from_spatial(SpatialBenefitNetwork(16, 0).state_dict(), dropout=0)
    out = net(*footprint_pack([value], 'cpu'))
    out.mean(1).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in net.common_head.parameters()) > 0
    assert sum(float(p.grad.abs().sum()) for p in net.relative_head.parameters()) < 1e-5
    net.zero_grad(set_to_none=True)
    out = net(*footprint_pack([value], 'cpu'))
    (out[:, 2]-out[:, 0]).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in net.relative_head.parameters()) > 0
    assert sum(float(p.grad.abs().sum()) for p in net.common_head.parameters()) < 1e-5
