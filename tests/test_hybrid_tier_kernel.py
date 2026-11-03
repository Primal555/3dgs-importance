import copy
from types import SimpleNamespace
from unittest.mock import patch
import torch
from gaussian_jscc.allocation_distortion import ComponentRenderTask, ImageDistortion
from gaussian_jscc.budgeted_allocation_kernel import budgeted_mask_step
from gaussian_jscc.calibrated_evidence import CalibratedEvidence
from gaussian_jscc.structured_evidence import BoundedMirrorTable
from test_successive_allocation import small_successive
from test_mask_gradient import Reference, toy_mask_render


class Provider:
    def predict(self, task, cache, scene, q, tier_ids, locations, **kwargs):
        self.q, self.ids = q.clone(), tier_ids.clone()
        parts = scene.new_zeros(len(tier_ids), 3, 2, dtype=torch.float64)
        parts[:, :, 0] = scene.new_tensor([.003, -.002, -.004])
        parts -= parts[torch.arange(len(tier_ids)), q[tier_ids]-1][:, None].clone()
        return parts, dict(neural_candidates=len(tier_ids))


def run(provider=None,closed=False):
    torch.set_num_threads(1); torch.manual_seed(213)
    raw, geometry, features, model = small_successive(); model.eval().requires_grad_(False)
    ids = [torch.arange(16).reshape(2, 8)]
    table = BoundedMirrorTable(16, .6)
    if closed:
        table.keep_logits[:,0]=100.;table.keep_logits[:,1]=-100.
    scheduler = CalibratedEvidence(ids, rounds=2, finite_samples=0)
    scheduler.configure_views(1,1); scheduler.set_view_window([0])
    task = ComponentRenderTask([SimpleNamespace(factor=1.,source_count=16)], Reference(), 1,
                              distortion=ImageDistortion('mse_ssim',.0125))
    with patch('gaussian_jscc.rendering.render',side_effect=toy_mask_render):
        args = dict(d_ref=.1,price=.1,rate_normalizer=1000,
                    side_proxy=dict(xyz_per_retained=1.,fixed_uses=8.,step=1),timings=None)
        if provider is not None: args['positive_provider']=provider
        result = budgeted_mask_step(model,table,[features],ids,geometry,task,scheduler,10.,'awgn',**args)
    return result,scheduler


def test_network_only_replaces_positive_estimate_and_preserves_dense_keep_and_rng():
    old, a = run(); old_rng=torch.get_rng_state().clone()
    provider=Provider()
    with patch('gaussian_jscc.budgeted_allocation_kernel.mixed_point_candidates',
               side_effect=AssertionError('redundant first-order candidates')):
        new,b = run(provider)
    torch.testing.assert_close(new['keep_image_costs'],old['keep_image_costs'],rtol=0,atol=0)
    torch.testing.assert_close(new['keep_costs'],old['keep_costs'],rtol=0,atol=0)
    torch.testing.assert_close(new['q'],old['q'],rtol=0,atol=0)
    assert torch.equal(old_rng,torch.get_rng_state())
    assert new['positive_estimator']=='spatial_network'
    assert not torch.equal(new['tier_costs'],old['tier_costs'])
    assert b.keep_observations.sum(1).tolist()==[1]*16
    assert int((new['keep_costs'][:,1]!=0).sum())==16
    assert new['neural_candidates']==4


def test_no_positive_candidates_still_updates_every_closed_keep_row():
    result,scheduler=run(Provider(),closed=True)
    assert result['neural_candidates']==0 and result['q0_feedback_rows']==16
    assert result['tier_ids'].numel()==0
    assert scheduler.keep_observations.sum(1).tolist()==[1]*16
    assert int((result['keep_costs'][:,1]!=0).sum())==16
