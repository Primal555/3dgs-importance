import copy
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc.allocation_benefit import BenefitNetwork
from gaussian_jscc.allocation_benefit_features import FeatureProtocol
from gaussian_jscc.allocation_benefit_learning import BenefitLearner
from gaussian_jscc.allocation_benefit_schedule import BenefitScheduler
from gaussian_jscc.allocation_distortion import ComponentRenderTask,ImageDistortion
from gaussian_jscc.budgeted_allocation_kernel import budgeted_mask_step
from gaussian_jscc.structured_evidence import BoundedMirrorTable
from test_mask_gradient import Reference,toy_mask_render
from test_successive_allocation import small_successive


def run(finite=6,closed=False,mode='mse_ssim',label_scale=1.):
    assert importlib.util.find_spec('gaussian_jscc.allocation_benefit_window') is not None, 'missing learned window'
    from gaussian_jscc.allocation_benefit_window import BenefitWindow
    torch.manual_seed(701);torch.set_num_threads(1)
    _,geometry,f,model=small_successive();model.eval().requires_grad_(False)
    ids=[torch.arange(16).reshape(2,8)];table=BoundedMirrorTable(16,.6)
    if closed is True:table.keep_logits[:,0]=100.;table.keep_logits[:,1]=-100.
    if closed=='open':table.keep_logits[:,0]=-100.;table.keep_logits[:,1]=100.
    scheduler=BenefitScheduler(ids,seed=42)
    protocol=FeatureProtocol(len(model.attr_mean),model.cfg.latent_dim)
    net=BenefitNetwork(protocol.point_dim,protocol.action_dim,continuous=protocol.continuous,mse_only=mode=='mse')
    learner=BenefitLearner(net,capacity=100,fit_windows=1,batch_size=4,updates_per_window=1)
    camera=SimpleNamespace(factor=1.,source_count=16,image_name='train0',
        world_view_transform=torch.eye(4),full_proj_transform=torch.eye(4))
    task=ComponentRenderTask([camera],Reference(),1,distortion=ImageDistortion(mode,.01 if mode!='mse' else 0.))
    original=task.delta_components
    task.delta_components=lambda scene:original(scene)*label_scale
    window=BenefitWindow(learner,scheduler,finite_samples=finite,chunk_size=8)
    before=copy.deepcopy(model.state_dict());before_table=copy.deepcopy(table.state_dict())
    with patch('gaussian_jscc.rendering.render',side_effect=toy_mask_render):
        result=budgeted_mask_step(model,table,[f],ids,geometry,task,scheduler,10.,'awgn',d_ref=.1,
            price=.1,rate_normalizer=1000,side_proxy=dict(step=1,xyz_per_retained=1.,fixed_uses=8.,bits_per_use=2.),
            timings=None,benefit=window)
    for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    for k,v in before_table.items():torch.testing.assert_close(v,table.state_dict()[k],rtol=0,atol=0)
    return result,window,torch.get_rng_state()


def test_full_q0_labels_match_costs_and_validation_never_overrides_prediction():
    result,window,_=run()
    assert window.render_calls<=6 and window.records
    for record in window.records:
        source=record['source']
        assert record['applied_to_training']==(not record['validation'])
        expected=record['prediction'] if record['validation'] else record['target']
        if record['branch']=='keep':
            want=((expected.sum(-1)+record['rates'])*record['conditional']).sum()
            assert float(result['keep_costs'][source,1])==pytest.approx(float(want),abs=1e-10)
        else:
            position=(result['tier_ids']==source).nonzero().item();old,new=record['old']-1,record['new'][0]-1
            want=float(expected.sum()+record['rates'].sum())
            assert float(result['tier_costs'][position,new]-result['tier_costs'][position,old])==pytest.approx(want,abs=1e-10)


def test_labels_cannot_change_predictions_and_finite_checks_preserve_rng():
    a,wa,ra=run();b,wb,rb=run(label_scale=10.)
    assert [r['source'] for r in wa.records]==[r['source'] for r in wb.records]
    for x,y in zip(wa.records,wb.records):
        torch.testing.assert_close(x['prediction'],y['prediction'],rtol=0,atol=0)
        torch.testing.assert_close(x['prediction_masked'],y['prediction_masked'],rtol=0,atol=0)
        if x['validation']:
            torch.testing.assert_close(a['keep_costs'][x['source']],b['keep_costs'][x['source']],rtol=0,atol=0)
    _,_,without=run(finite=0)
    torch.testing.assert_close(ra,rb,rtol=0,atol=0)
    torch.testing.assert_close(ra,without,rtol=0,atol=0)


@pytest.mark.parametrize('closed,mode',[(True,'mse_ssim'),(False,'mse'),('open','mse_ssim')])
def test_every_closed_point_has_keep_prediction_and_mse_structure_is_zero(closed,mode):
    result,window,_=run(closed=closed,mode=mode)
    assert result['benefit_keep_predicted_rows']==16
    assert torch.isfinite(result['keep_costs']).all()
    if closed is True:
        assert result['q0_feedback_rows']==16 and len(result['tier_ids'])==0
        assert window.render_calls==6
    if closed=='open':
        assert result['q0_feedback_rows']==0 and len(result['tier_ids'])>0
    if mode=='mse':
        for r in window.records:
            assert r['target'][:,1].count_nonzero()==0
            assert r['prediction'][:,1].count_nonzero()==0
