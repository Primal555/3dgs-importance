import importlib.util
from types import SimpleNamespace

import torch


def kernel():
    assert importlib.util.find_spec('gaussian_jscc.direct_rd_diagnostic_kernel'), 'diagnostic kernel missing'
    from gaussian_jscc import direct_rd_diagnostic_kernel
    return direct_rd_diagnostic_kernel


def test_stratification_keeps_ranks_and_independent_blocks():
    k = kernel()
    from gaussian_jscc.direct_rd_decisions import ProposalBank
    bank = ProposalBank(32)
    ids = torch.arange(32)
    costs = torch.stack((torch.arange(32).double()+1, torch.zeros(32), torch.zeros(32)), -1)
    # Current q1; every source has a positive q1->q0 predicted gain.
    bank.add(ids, costs)
    q = torch.ones(32, dtype=torch.long); blocks = torch.arange(32)//2
    proposal = bank.propose(q, blocks)
    chosen = k.select_actions(bank, proposal, q, blocks, per_stratum=2, seed=7)
    assert {r['stratum'] for r in chosen} == {'top', 'middle', 'bottom', 'ordinary'}
    assert len({r['source'] for r in chosen}) == len(chosen)
    assert all(r['old_tier'] == 1 and r['new_tier'] != 1 for r in chosen)
    assert [r['rank'] for r in chosen if r['stratum'] == 'top'] == [0, 1]


def test_exact_window_reproduces_candidate_prediction_and_full_render():
    k = kernel()
    from gaussian_jscc.direct_rd_allocation import FrozenBenefit
    from gaussian_jscc.direct_rd_decisions import HardTierMap
    from gaussian_jscc.spatial_benefit import SpatialBenefitNetwork
    from gaussian_jscc.pixel_local_renderer import capture
    from test_successive_allocation import small_successive
    torch.set_num_threads(1); torch.manual_seed(8)
    raw, geometry, features, codec = small_successive(); codec.eval().requires_grad_(False)
    transform = torch.eye(4); transform[3,2] = 3
    camera = SimpleNamespace(image_width=16, image_height=16, FoVx=1.2, FoVy=1.2,
        world_view_transform=transform, full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
    scene = SimpleNamespace(raw=raw, geometry=geometry, groups=[features], group_ids=[torch.arange(16).reshape(2,8)],
        degree=1, white_background=False, reference=SimpleNamespace(get=lambda c,d:torch.zeros(3,16,16,device=d)))
    q = torch.full((16,),2); q[0] = 0
    network = SpatialBenefitNetwork(hidden=16)
    learner = FrozenBenefit(dict(network=network.state_dict(), hidden=16, dropout=.05,
        scales={'value':torch.ones(2),'rd':torch.tensor(2.)}, updates=0), 'cpu')
    reference = dict(lambda_s=.01,d_ref=.01,c_ref=1000.,side_proxy=dict(xyz_per_retained=24.,bits_per_use=2.))
    args = SimpleNamespace(allocation_batch=4,label_groups=0,snr=10.,beta=.1,neighbors=4,cpu_threads=1)
    actions = [{'source':0,'new_tier':1},{'source':9,'new_tier':0}]
    records, window = k.evaluate_actions(codec,scene,q,actions,[camera],learner,reference,args,seed=123)
    assert set(records) == {0,9}
    assert records[0]['old_tier'] == 0 and records[9]['new_tier'] == 0
    from gaussian_jscc.allocation_distortion import ImageDistortion
    objective = ImageDistortion('mse_ssim',.01)
    before = capture(window.base_raw, camera, 1, (q[window.render_sources]>0).float(),backend='cpu').image
    before_d = float(objective.components(before,torch.zeros_like(before)).mul(torch.tensor([1.,.01])).sum())/.01
    for action in actions:
        modified, mask = window.intervention_raw(action['source'],action['new_tier'])
        after = capture(modified,camera,1,mask,backend='cpu').image
        after_d = float(objective.components(after,torch.zeros_like(after)).mul(torch.tensor([1.,.01])).sum())/.01
        assert abs(records[action['source']]['true_image_delta']-(after_d-before_d)) < 2e-5
    assert all(p.grad is None for p in codec.parameters())
    from unittest.mock import patch
    with patch.object(codec,'forward_tier_batches',side_effect=AssertionError('view-only check decoded again')):
        other=k.evaluate_other_views(window,[camera])
    for source,row in records.items():
        for key,value in row.items():
            assert abs(other[source][key]-value)<1e-10
    from gaussian_jscc.cli import seed_all
    from gaussian_jscc.modular_evaluation import decode_path
    seed_all(123)
    received,_=decode_path(codec,scene,[q.reshape(2,8)],'awgn',10.)
    torch.testing.assert_close(received,window.base_raw[q>0],rtol=3e-5,atol=1e-6)
    from gaussian_jscc.direct_rd_diagnostic import full_single_deltas
    def cpu_render(raw,cam,degree,white_background=False):
        return capture(raw,cam,degree,torch.ones(len(raw)),white_background,backend='cpu').image
    with patch('gaussian_jscc.rendering.render',side_effect=cpu_render):
        native=full_single_deltas(window,actions,[camera],objective,.01)
    for source,value in native.items():
        assert abs(value-records[source]['true_image_delta'])<1e-6
