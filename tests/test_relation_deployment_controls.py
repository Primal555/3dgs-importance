import copy
import importlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_relation_benefit_experiment import runtime,renderer


def entry():return importlib.import_module('gaussian_jscc.relation_deployment_controls')


def source_run(root,model,scene):
    from gaussian_jscc.budgeted_allocation_evaluation import ExactDeploymentCost,evaluate_map,tier_hash
    from gaussian_jscc.relation_benefit_experiment import FAMILY,unified_cost
    from gaussian_jscc.allocation_distortion import ImageDistortion
    from gaussian_jscc.modular_experiment import json_write
    from gaussian_jscc.transport import model_id
    root.mkdir();(root/'deployment').mkdir()
    model.set_scene_geometry(scene.geometry);cid=model_id(model)
    q=torch.arange(len(scene.raw))%4
    np.save(root/'deployment/selected_tiers.npy',q.numpy().astype(np.uint8))
    settings=dict(run=str(root.parent/'codec'),scene='truck',snr=10.,beta=.1,trials=2,
        seed=42,lpips=False,window_views=2,test_views=2,calibration_views=1)
    meter=ExactDeploymentCost(model,scene,cid,10.,2.)
    ref=dict(beta=.1,d_ref=.03,lambda_s=.02,c_ref=unified_cost(meter,torch.full_like(q,3))['total_channel_uses'])
    training=dict(version=FAMILY,codec_id=cid,codec_config=model.cfg.to_dict(),spec=scene.spec,
        settings=settings,objective_reference=ref,preparation_view_names=[c.image_name for c in scene.cameras[:2]],
        **scene.record)
    plan=dict(version=FAMILY,codec_id=cid,settings=settings)
    complete=dict(status='complete',family=FAMILY,step=700,network_updates=41600,
        codec_id=cid,tier_sha256=tier_hash(q),frozen_codec_verified=True)
    opts=SimpleNamespace(snr=10.,trials=2,seed=800045,lpips=False,image_views=2,
        allocation_image_objective=ImageDistortion('mse_ssim',.02))
    rows={}
    with patch('gaussian_jscc.rendering.render',side_effect=renderer):
        for label,tiers in [('selected',q),('uniform_q1',torch.ones_like(q)),('uniform_q2',torch.full_like(q,2))]:
            metrics=evaluate_map(model,scene,tiers,scene.test_cameras,opts,
                folder=root/'evaluation'/label,label=label)
            rows[label]=dict(metrics,**unified_cost(meter,tiers),
                rd_score=metrics['source_distortion']/.03+.1*unified_cost(meter,tiers)['total_channel_uses']/ref['c_ref'])
    for name,data in [('training',training),('plan',plan),('complete',complete),
                      ('review_summary',dict(result=complete,rows=rows))]:json_write(root/(name+'.json'),data)
    return q


def options(source,out):return entry().parser().parse_args([
    '--source',str(source),'--out',str(out),'--device','cpu','--cpu-threads','1'])


def test_maps_preserve_original_mask_and_input():
    q=torch.tensor([0,1,2,3,0]);before=q.clone()
    maps=entry().controlled_maps(q)
    assert maps['restore_q0_to_q1'].tolist()==[1,1,2,3,1]
    assert maps['kept_uniform_q2'].tolist()==[0,2,2,2,0]
    assert torch.equal(q,before)
    with pytest.raises(ValueError):entry().controlled_maps(torch.tensor([0.,1.]))


def test_real_cpu_evaluation_reuses_baselines_combines_views_and_keeps_sources_readonly(tmp_path):
    torch.set_num_threads(1);model,scene=runtime();source=tmp_path/'source'
    q=source_run(source,model,scene);record=entry().read_source(source)
    before=copy.deepcopy(model.state_dict())
    source_bytes={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    calls=[];original=entry().evaluate_map
    def counted(model,scene,tiers,cameras,args,**kwargs):
        calls.append((kwargs['label'],[c.image_name for c in cameras]))
        return original(model,scene,tiers,cameras,args,**kwargs)
    with patch('gaussian_jscc.rendering.render',side_effect=renderer), \
         patch.object(entry(),'evaluate_map',side_effect=counted), \
         patch('gaussian_jscc.relation_benefit_features.prepare_bank',side_effect=AssertionError('bank')), \
         patch('torch.optim.AdamW',side_effect=AssertionError('training')):
        result=entry().run_loaded(options(source,tmp_path/'controls'),model,scene,record)
    assert result['status']=='complete' and result['optimizer_updates']==0
    assert len(calls)==5
    assert all(len(names)==4 for label,names in calls if label in ('restore_q0_to_q1','kept_uniform_q2'))
    assert all(len(names)==2 for label,names in calls if label in ('selected','uniform_q1','uniform_q2'))
    report=json.loads((tmp_path/'controls/review_summary.json').read_text())
    assert set(report['scopes'])=={'preparation','test'}
    for scope in report['scopes'].values():
        assert set(scope['rows'])=={'selected','uniform_q1','uniform_q2','restore_q0_to_q1','kept_uniform_q2'}
        for row in scope['rows'].values():
            assert row['views']==2 and row['trials']==2
            assert row['total_channel_uses']==row['comparison_accounting']['total_channel_uses']
            assert row['rd_score']==pytest.approx(row['source_distortion']/.03+.1*row['total_channel_uses']/record['reference']['c_ref'])
    assert report['scopes']['test']['reused_source_baselines'] is True
    for key,value in before.items():assert torch.equal(value,model.state_dict()[key])
    assert source_bytes=={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert np.array_equal(np.load(tmp_path/'controls/maps/kept_uniform_q2.npy')==0,q.numpy()==0)
    with tarfile.open(tmp_path/'controls/review.tar.gz') as t:
        names=t.getnames();assert 'review_summary.json' in names
        assert not any('cache/' in n or 'checkpoints/' in n or n.endswith('.pt') for n in names)
    resumed=options(source,tmp_path/'controls');resumed.resume=True
    with patch('gaussian_jscc.rendering.render',side_effect=AssertionError('rerendered cache')):
        assert entry().run_loaded(resumed,model,scene,record)['status']=='complete'


@pytest.mark.parametrize('fault',['tiers','codec','preparation','white_background','cached_views','cached_seed','cached_objective'])
def test_incompatible_source_is_rejected(tmp_path,fault):
    torch.set_num_threads(1);model,scene=runtime();source=tmp_path/'source'
    source_run(source,model,scene)
    if fault=='tiers':np.save(source/'deployment/selected_tiers.npy',np.zeros(16,dtype=np.uint8))
    if fault.startswith('cached_'):
        p=source/'evaluation/selected/metrics.json';d=json.loads(p.read_text())
        if fault=='cached_views':d['identity']['views']=list(reversed(d['identity']['views']))
        if fault=='cached_seed':d['identity']['seed']+=1
        if fault=='cached_objective':d['identity']['image_objective']['lambda_s']+=.1
        p.write_text(json.dumps(d))
    if fault=='codec':
        with torch.no_grad():next(model.parameters()).add_(1)
    if fault=='preparation':scene.cameras=scene.cameras[2:]
    if fault=='white_background':scene.white_background=True
    with patch('gaussian_jscc.rendering.render',side_effect=renderer),pytest.raises(ValueError):
        record=entry().read_source(source)
        entry().run_loaded(options(source,tmp_path/'bad'),model,scene,record)


def test_main_uses_original_codec_and_temporary_scene_without_network_checkpoints(tmp_path):
    torch.set_num_threads(1);model,scene=runtime();source=tmp_path/'source'
    source_run(source,model,scene);args=options(source,tmp_path/'main')
    saved=dict(state_dict=copy.deepcopy(model.state_dict()))
    def inputs(opts):
        assert opts.run==str(tmp_path/'codec') and opts.scene=='truck'
        assert opts.out==args.out and opts.test_views==2
        return saved,model.cfg,scene.spec,scene.record,SimpleNamespace(),{}
    with patch.object(entry(),'load_inputs',side_effect=inputs), \
         patch.object(entry(),'read_ply',return_value=(scene.raw,scene.degree)), \
         patch.object(entry(),'make_scene',return_value=scene), \
         patch('gaussian_jscc.rendering.render',side_effect=renderer):
        assert entry().main(args)['status']=='complete'


def test_partition_averages_the_same_observations_without_duplicate_view_trials():
    metrics=dict(trials=1,tier_counts=[0,1,1,1],tier_shares=[0.,1/3,1/3,1/3],
        codec_id='codec',tier_sha256='tier',protocol={},lambda_s=.02,
        per_view_trial=[dict(view=v,trial=0,source_psnr=x,source_lpips=None)
                        for v,x in [('train',10.),('test0',20.),('test1',30.)]])
    subset=entry().subset_metrics(metrics,['test0','test1'])
    assert subset['source_psnr']==25 and subset['source_lpips'] is None
    metrics['per_view_trial'].append(dict(metrics['per_view_trial'][-1]))
    with pytest.raises(ValueError,match='duplicate'):entry().subset_metrics(metrics,['test0','test1'])
