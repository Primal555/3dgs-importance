import copy
import hashlib
import importlib.util
import json

import pytest
import torch

from test_allocation_benefit import record
from test_structured_allocation_integration import source


def api():
    assert importlib.util.find_spec('gaussian_jscc.benefit_diagnostics') is not None, 'missing benefit diagnostic checks'
    from gaussian_jscc import benefit_diagnostics
    return benefit_diagnostics


def example(branch='positive', validation=False, source=0, bias=.2):
    r=record(branch,source=source,validation=validation)
    r['target']=torch.tensor([[.1,0.]] if branch=='positive' else [[-.3,0.],[-.1,0.],[.2,0.]],dtype=torch.float64)
    r['rates'].zero_()
    r['conditional']=torch.ones(1) if branch=='positive' else torch.full((3,),1/3,dtype=torch.float64)
    r['prediction']=r['target'].clone();r['prediction'][:,0]+=bias
    r['prediction_masked']=r['prediction'].clone();r['first_order']=r['target'].clone()
    return r


def test_bias_audit_distinguishes_common_offset_from_correct_tier_ranking():
    module=api();r=example('keep',validation=True)
    result=module.cost_audit([r],{'positive':[.01,.1],'keep':[.01,.1]})['keep']
    assert result['stages']['network']['signed_bias']==pytest.approx(.2)
    assert result['stages']['network']['false_harmful']==1
    assert result['keep_common_mode']['common_mode_error_energy_fraction']==pytest.approx(1.)
    assert result['keep_common_mode']['centered_error_rmse']==pytest.approx(0.,abs=1e-14)
    assert result['keep_common_mode']['correct_tier_wrong_keep_decision']==1
    assert result['keep_common_mode']['best_tier_accuracy']==1.


def test_bias_correction_uses_training_labels_and_leaves_inputs_unchanged():
    module=api();training=[example(),example('keep')]
    validation=[example(validation=True,source=2),example('keep',validation=True,source=3)]
    offsets=module.fit_offsets(training)
    corrected=module.correct_offsets(validation,offsets)
    for old,new in zip(validation,corrected):
        torch.testing.assert_close(old['prediction'][:,0]-new['prediction'][:,0],torch.full_like(old['prediction'][:,0],.2))
        torch.testing.assert_close(new['prediction'],new['target'],rtol=0,atol=1e-14)
        assert float(old['prediction'][0,0])==pytest.approx(.3 if old['branch']=='positive' else -.1)
    with pytest.raises(ValueError,match='train|validation'):module.fit_offsets(validation)


def test_audit_splits_directed_transitions_and_real_cost_magnitude():
    module=api();a=example(bias=-.2);b=example(bias=.2,source=1)
    a['old']=1;a['new']=[3]
    b['old']=3;b['new']=[1];b['target']*=3;b['prediction'][:,0]=.5
    result=module.cost_audit([a,b],{'positive':[.15,.25],'keep':[.1,.2]})['positive']
    assert result['stages']['network']['false_beneficial']==1
    assert result['transitions']['1->3']['network']['count']==1
    assert result['magnitude']['small']['network']['count']==1
    assert result['magnitude']['large']['network']['count']==1


def checkpoint(tmp_path):
    from gaussian_jscc.allocation_benefit import BenefitNetwork
    from gaussian_jscc.allocation_benefit_learning import BenefitLearner
    torch.manual_seed(42);torch.set_num_threads(1)
    learner=BenefitLearner(BenefitNetwork(5,4,3,2),capacity=32,fit_windows=1,batch_size=4)
    rows=[example(source=0),example('keep',source=1),example(validation=True,source=2),example('keep',validation=True,source=3)]
    learner.observe(rows,1);learner.train_step(2)
    state=dict(family='learned_benefit_allocation_v1',learner=learner.state_dict(),
        plan=dict(settings=dict(distortion='mse_ssim',seed=42)),
        progress=dict(step=200,phase='pretrain',pretrain_complete=True),
        table={'keep_logits':torch.randn(4,2),'logits':torch.randn(4,3)},state_dict={})
    root=tmp_path/'source';path=root/'checkpoints/latest/state.pt';path.parent.mkdir(parents=True)
    torch.save(state,path)
    return root,state


def test_offline_check_keeps_source_labels_normalization_and_validation_fixed(tmp_path):
    module=api();root,state=checkpoint(tmp_path);path=root/'checkpoints/latest/state.pt'
    original=hashlib.sha256(path.read_bytes()).hexdigest();out=tmp_path/'checks'
    args=module.parser().parse_args(['--run',str(root),'--out',str(out),'--updates','2','--evaluate-every','1','--device','cpu'])
    result=module.main(args)
    assert result['status']=='complete' and result['additional_updates']==2
    assert result['source_checkpoint_unchanged']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original
    history=[json.loads(line) for line in (out/'history.jsonl').read_text().splitlines()]
    assert [r['additional_updates'] for r in history]==[0,1,2]
    for row in history:
        for split in ('training','validation'):
            assert row[split]['positive']['count']==1 and row[split]['keep']['count']==1
        assert row['fixed_records_sha256']==history[0]['fixed_records_sha256']
    final=torch.load(out/'offline_final.pt',map_location='cpu',weights_only=True)
    for key,value in state['learner']['network'].items():
        if key.endswith(('_mean','_std','_continuous')) or key in ('scales','ready'):
            torch.testing.assert_close(final['network'][key],value,rtol=0,atol=0)
    assert any(not torch.equal(value,final['network'][key]) for key,value in state['learner']['network'].items() if key.endswith('weight'))
    assert final['probability_table_updated'] is False
    assert (out/'initial_bias.json').is_file() and (out/'final_bias.json').is_file()


@pytest.mark.parametrize('updates',[0,-1])
def test_zero_update_audit_and_invalid_budget(tmp_path,updates):
    module=api();root,_=checkpoint(tmp_path);out=tmp_path/'checks'
    args=module.parser().parse_args(['--run',str(root),'--out',str(out),'--updates',str(updates)])
    if updates<0:
        with pytest.raises(ValueError):module.main(args)
        assert not out.exists()
    else:
        result=module.main(args)
        assert result['additional_updates']==0
        assert len((out/'history.jsonl').read_text().splitlines())==1


def test_diagnostic_refuses_source_output_overlap(tmp_path):
    module=api();root,_=checkpoint(tmp_path)
    args=module.parser().parse_args(['--run',str(root),'--out',str(root/'checks')])
    with pytest.raises(ValueError,match='separate|overlap'):module.main(args)
    assert not (root/'checks').exists()


def test_diagnostic_rejects_overlapping_training_and_validation_points(tmp_path):
    module=api();root,state=checkpoint(tmp_path)
    state['learner']['validation']['groups'][0]['source']=0
    torch.save(state,root/'checkpoints/latest/state.pt')
    with pytest.raises(ValueError,match='overlap'):module.restore_learner(state,'cpu')


def test_static_features_match_saved_geometry_statistics_and_ply_identity(tmp_path):
    module=api()
    from gaussian_jscc.data import read_ply,write_ply
    from gaussian_jscc.allocation import scene_fingerprint
    raw=torch.zeros(2,14);raw[:,:3]=torch.tensor([[1.,2.,4.],[2.,4.,8.]])
    raw[:,7]=1.;raw[:,11:]=.6
    path=tmp_path/'source.ply';write_ply(path,raw,0);raw,_=read_ply(path)
    state=dict(learner={'replay':{'static_width':14,'static_count':2}},
        plan={'spec':{'ply':str(path)}},config={'sh_degree':0},
        state_dict={'attr_mean':torch.zeros(11),'attr_std':torch.full((11,),2.)},
        scene_record={'source_gaussians':2,'fingerprint':scene_fingerprint(raw),
                      'geometry':{'lower':[0.,0.,0.],'span':[2.,4.,8.],'bits':16}})
    features=module.static_features(state)
    assert features.shape==(2,14)
    torch.testing.assert_close(features[0,:3],torch.full((3,),.5))
    assert float(features[0,7])==.5 and float(features[0,11])==pytest.approx(.3)
    bad=copy.deepcopy(state);bad['scene_record']['fingerprint']='invalid'
    with pytest.raises(ValueError,match='identity|fingerprint'):module.static_features(bad)


def test_real_pretraining_replay_restores_identical_baseline_without_rendering(source):
    import contextlib
    import io
    from unittest.mock import patch
    from test_learned_benefit_allocation import entry,cameras,options
    import test_budgeted_allocation as fixture
    module=api();parent,root=source;out=root/'diagnostic_source'
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        entry().main(options(parent,out))
    state=torch.load(out/'checkpoints/latest/state.pt',map_location='cpu',weights_only=True)
    learner=module.restore_learner(state,'cpu')
    rows,_=module.predict_records(learner,learner.validation)
    actual=module.summarize_records(rows)
    original=json.loads((out/'comparison/benefit_pretrain.json').read_text())['validation']
    for branch in ('positive','keep'):
        for stage in ('first_order','network','masked'):
            assert actual[branch]['stages'][stage]['regret_mean']==pytest.approx(original[branch]['stages'][stage]['regret_mean'],abs=1e-12)
            assert actual[branch]['stages'][stage]['rd_mae']==pytest.approx(original[branch]['stages'][stage]['rd_mae'],abs=1e-12)
