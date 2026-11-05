"""Zero-update inference on real CPU codec inputs and an unseen scene."""
import copy
import importlib
import json
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_relation_benefit_experiment import runtime, renderer


def entry():
    return importlib.import_module('gaussian_jscc.relation_benefit_predict')


def options(out):
    return entry().parser().parse_args(['--run',str(out.parent/'codec'),
        '--network',str(out.parent/'trained'),'--manifest','unused.json',
        '--scene','unseen','--out',str(out),'--device','cpu','--views','2',
        '--test-views','2','--trials','1','--inference-batch','5',
        '--cpu-threads','1','--no-lpips'])


def checkpoint(model, scene):
    from gaussian_jscc.relation_benefit_features import prepare_bank
    from gaussian_jscc.relation_benefit_model import RelationBenefitNetwork
    from gaussian_jscc.transport import model_id
    bank=prepare_bank(model,scene,scene.cameras[:2],SimpleNamespace(
        snr=10.,spatial_neighbors=2,projection_neighbors=2,cpu_threads=1,feature_batch=16,context_seed=42))
    net=RelationBenefitNetwork(bank['node'].shape[-1],bank['candidate'].shape[-1],
        bank['view_node'].shape[-1],16,4,
        context_node_dim=bank['contexts'][0]['node'].shape[-1],
        context_view_dim=bank['contexts'][0]['view_node'].shape[-1])
    net.fit_normalizers(bank,torch.arange(len(scene.raw)))
    net.output_scale.copy_(torch.tensor([.17,.39]))
    from gaussian_jscc.relation_benefit_teacher import RelationContextTeacher
    teacher=RelationContextTeacher(model,scene,scene.cameras[:2],dict(lambda_s=.02,d_ref=.03),
        bank['contexts'],trials=2).identity()
    return dict(plan=dict(version='relation_benefit_context_v2',codec_id=model_id(model),codec_config=model.cfg.to_dict(),settings=dict(
        width=16,attention_groups=4,spatial_neighbors=2,projection_neighbors=2,feature_batch=16,seed=42)),
        learner=dict(network=copy.deepcopy(net.state_dict()),updates=11,scales=torch.tensor([.17,.39],dtype=torch.float64)),
        reference=dict(d_ref=.03,lambda_s=.02,c_ref=123.,cost_proxy=dict(costs=[0.,9.,25.,49.])),
        teacher_identity=teacher,
        progress=dict(window=7))


def unseen_scene(model, scene, count):
    from gaussian_jscc.data import to_features
    original=scene.raw[torch.argsort(scene.order)][:count]
    scene.order=torch.arange(count-1,-1,-1)
    scene.raw=original[scene.order]
    features,_=to_features(scene.raw,scene.geometry,model)
    from torch.nn.utils.rnn import pad_sequence
    scene.groups=[pad_sequence(list(features.split(model.cfg.block_size)),batch_first=True)]
    scene.group_ids=[pad_sequence(list(scene.order.split(model.cfg.block_size)),
        batch_first=True,padding_value=-1)]
    scene.name='unseen';scene.fingerprint='unseen-ply'
    scene.spec=dict(name='unseen',role='heldout',source='newphotos',ply='new.ply')
    scene.record=dict(scene.record,source_gaussians=count,fingerprint=scene.fingerprint)
    # Preparation can use poses while every RGB access is prohibited.
    class Pose(SimpleNamespace):
        @property
        def original_image(self):
            raise AssertionError('preparation read a photograph')
    scene.cameras=[Pose(**{k:v for k,v in vars(c).items() if k!='original_image'}) for c in scene.cameras]
    from gaussian_jscc.rendering import RenderReference
    scene.reference=RenderReference(scene.raw,scene.degree,False,'source')
    return scene


@pytest.mark.parametrize('count',[9,16])
def test_unseen_scene_inference_preserves_weights_scales_and_has_zero_updates(tmp_path,count):
    torch.set_num_threads(1)
    model,scene=runtime();saved=checkpoint(model,scene)
    before=copy.deepcopy(model.state_dict());saved_before=copy.deepcopy(saved)
    scene=unseen_scene(model,scene,count)
    from gaussian_jscc.relation_benefit_model import RelationBenefitNetwork
    original=RelationBenefitNetwork.predict_all
    seen=[]
    def checked_predict(net,bank,batch,**kwargs):
        assert not net.training and all(not p.requires_grad for p in net.parameters())
        for key,value in saved_before['learner']['network'].items():
            torch.testing.assert_close(net.state_dict()[key].cpu(),value,atol=0,rtol=0)
        assert bank['metadata']['candidate_protocol']['image_reference']=='source_ply_capture'
        assert [c['name'] for c in bank['contexts']]==['q1','q2','q3','mixed']
        values=original(net,bank,batch,**kwargs)
        if kwargs.get('context') is None:
            assert net.last_inference_profile['context_passes']==4
            seen.append(values)
        return values
    with patch('gaussian_jscc.rendering.render',side_effect=renderer), \
         patch.object(RelationBenefitNetwork,'fit_normalizers',side_effect=AssertionError('refit')), \
         patch('gaussian_jscc.relation_benefit_teacher.RelationTeacher.__init__',side_effect=AssertionError('teacher')), \
         patch('gaussian_jscc.relation_benefit_learning.RelationLearner.__init__',side_effect=AssertionError('learner')), \
         patch('gaussian_jscc.relation_benefit_learning.RelationLearner.fit_scales',side_effect=AssertionError('scale fit')), \
         patch('torch.optim.Adam',side_effect=AssertionError('optimizer')), \
         patch('torch.optim.AdamW',side_effect=AssertionError('optimizer')), \
         patch.object(RelationBenefitNetwork,'predict_all',checked_predict):
        result=entry().predict_loaded(options(tmp_path/'prediction'),model,scene,saved,dict(source='fixture'))
    assert result['status']=='complete' and result['scene_optimizer_updates']==0
    assert result['frozen_codec_verified'] and result['frozen_network_verified']
    assert len(seen)==1 and seen[0].shape==(count,3,2)
    assert np.load(tmp_path/'prediction/deployment/selected_tiers.npy').shape==(count,)
    report=json.loads((tmp_path/'prediction/review_summary.json').read_text(encoding='utf-8'))
    assert report['reference']['d_ref']==.03 and report['reference']['lambda_s']==.02
    assert report['reference']['c_ref']!=123.
    assert report['reference']['c_ref']==report['rows']['uniform_q3']['total_channel_uses']
    assert report['reference_policy']['distortion']=='frozen_training_reference'
    assert report['reference_policy']['communication']=='new_scene_uniform_q3_digital_side_file_v1'
    plan=json.loads((tmp_path/'prediction/plan.json').read_text(encoding='utf-8'))
    assert plan['spec']['role']=='heldout'
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    for k,v in saved_before['learner']['network'].items():
        assert torch.equal(v,saved['learner']['network'][k])
    with tarfile.open(tmp_path/'prediction/review.tar.gz') as archive:
        names=archive.getnames()
        assert 'review_summary.json' in names and 'complete.json' in names
        assert not any('cache/' in n or 'checkpoints/' in n or n.endswith('.pt') for n in names)


@pytest.mark.parametrize('fault',['codec','family','dimension','test_overlap','unused_pose_overlap',
    'out_network','out_run','updates_zero','scales_none','scales_nan','scales_mismatch',
    'teacher_codec','teacher_background','teacher_tier','teacher_missing','teacher_protocol',
    'context_normalizer','context_dimension','codec_config'])
def test_invalid_identity_or_input_never_creates_prediction_output(tmp_path,fault):
    torch.set_num_threads(1)
    model,scene=runtime();saved=checkpoint(model,scene);args=options(tmp_path/'out')
    if fault=='codec':saved['plan']['codec_id']='wrong'
    if fault=='family':saved['plan']['version']='other'
    if fault=='dimension':saved['learner']['network']['node_mean']=torch.zeros(3)
    if fault=='test_overlap':scene.test_cameras=[scene.cameras[0]]
    if fault=='unused_pose_overlap':scene.test_cameras=[scene.cameras[1]]
    if fault=='out_network':args.out=str(tmp_path/'trained'/'prediction')
    if fault=='out_run':args.out=str(tmp_path/'codec'/'prediction')
    if fault=='updates_zero':saved['learner']['updates']=0
    if fault=='scales_none':saved['learner']['scales']=None
    if fault=='scales_nan':saved['learner']['scales'][0]=float('nan')
    if fault=='scales_mismatch':saved['learner']['scales'][0]=1.
    if fault=='teacher_codec':saved['teacher_identity']['contexts'][0]['codec_sha256']='wrong'
    if fault=='teacher_background':saved['teacher_identity']['contexts'][1]['context_name']='q3'
    if fault=='teacher_tier':saved['teacher_identity']['contexts'][1]['background_tier']=3
    if fault=='teacher_missing':saved['teacher_identity']['contexts'].pop()
    if fault=='teacher_protocol':saved['teacher_identity']['background_protocol']='other'
    if fault=='context_normalizer':saved['learner']['network']['context_node_std'][0]=float('nan')
    if fault=='context_dimension':saved['learner']['network']['context_view_mean']=torch.zeros(1)
    if fault=='codec_config':saved['plan']['codec_config']['sh_degree']=2
    with patch('gaussian_jscc.rendering.render',side_effect=renderer),pytest.raises(ValueError):
        entry().predict_loaded(args,model,scene,saved,{})
    from pathlib import Path
    assert not Path(args.out).exists()


@pytest.mark.parametrize('checkpoint_label',['final','latest','newer_latest'])
def test_main_loads_unseen_heldout_manifest_without_changing_role_or_using_photo_inputs(tmp_path,checkpoint_label):
    torch.set_num_threads(1)
    model,scene=runtime();model.set_training_snrs([10.]);saved=checkpoint(model,scene)
    unseen_scene(model,scene,9)
    source=tmp_path/'photos';source.mkdir()
    ply=tmp_path/'new.ply';ply.write_text('source fixture',encoding='utf-8')
    manifest_path=tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(dict(scenes=[dict(name='unseen',role='heldout',
        source='photos',ply='new.ply')])),encoding='utf-8')
    args=options(tmp_path/'prediction');args.manifest=str(manifest_path)
    from pathlib import Path
    if checkpoint_label=='newer_latest':
        stale=Path(args.network)/'checkpoints/final/state.pt'
        stale.parent.mkdir(parents=True);torch.save(saved,stale)
        saved['progress']['window']=8
    network_path=Path(args.network)/'checkpoints'/('latest' if checkpoint_label=='newer_latest' else checkpoint_label)/'state.pt'
    network_path.parent.mkdir(parents=True);torch.save(saved,network_path)
    raw=scene.raw[torch.argsort(scene.order)]
    source_bundle=(dict(state_dict=model.state_dict()),dict(settings={}),model.cfg,
        dict(source_checkpoint='codec/final.pt',source_sha256='fixture'))
    def load_cameras(root,resolution,white_background,images,split):
        assert root==str(source.resolve())
        return scene.cameras if split=='train' else scene.test_cameras
    with patch.object(entry(),'load_source',return_value=source_bundle), \
         patch.object(entry(),'read_ply',return_value=(raw,scene.degree)), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=load_cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=renderer), \
         patch('gaussian_jscc.relation_benefit_model.RelationBenefitNetwork.fit_normalizers',
               side_effect=AssertionError('refit')):
        result=entry().main(args)
    plan=json.loads((tmp_path/'prediction/plan.json').read_text(encoding='utf-8'))
    assert result['scene_optimizer_updates']==0
    assert result['training_step']==(8 if checkpoint_label=='newer_latest' else 7)
    assert plan['spec']['role']=='heldout'
    assert plan['scene_record']['train_view_names']==[]
    assert plan['preparation_view_names']==['train_0','train_3']
    assert plan['source']['network_checkpoint']==str(network_path.resolve())
