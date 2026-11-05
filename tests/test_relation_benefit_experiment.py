import contextlib
import copy
import importlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def entry():
    return importlib.import_module('gaussian_jscc.relation_benefit_experiment')


def runtime():
    from test_relation_benefit_teacher import fixture
    model,scene,cameras,_=fixture()
    def clone(camera,name):
        data=dict(vars(camera));data['image_name']=name
        data['original_image']=torch.full((3,camera.image_height,camera.image_width),.5)
        return SimpleNamespace(**data)
    scene.cameras=[clone(c,f'train_{i}') for i,c in enumerate(cameras)]
    scene.val_cameras=[clone(cameras[0],'cal_0')]
    scene.test_cameras=[clone(cameras[1],'test_0'),clone(cameras[2],'test_1')]
    scene.name='truck';scene.spec=dict(name='truck',role='train',source='photos',ply='source.ply')
    scene.record=dict(fingerprint=scene.fingerprint,source_gaussians=len(scene.raw),resolution=2,
        images='images',white_background=False,test_view_names=['test_0','test_1'])
    return model,scene


def renderer(raw,camera,degree,white_background=False,masks=None,colors=None):
    from gaussian_jscc.pixel_local_renderer import capture
    from utils.sh_utils import RGB2SH
    if not len(raw):return raw.new_full((3,camera.image_height,camera.image_width),float(white_background))
    if colors is not None:
        raw=raw.clone();raw[:,11:14]=RGB2SH(colors);raw[:,14:]=0
    if masks is None:masks=raw.new_ones(len(raw))
    return capture(raw,camera,degree,masks,white_background,backend='cpu').image


def args(out):
    return entry().parser().parse_args(['--run','unused','--scene','truck','--out',str(out),
        '--device','cpu','--windows','3','--fit-windows','1','--label-groups','3',
        '--updates-per-window','2','--batch-groups','2','--teacher-trials','1',
        '--width','16','--attention-groups','4','--spatial-neighbors','2',
        '--projection-neighbors','2','--validation-groups','2','--window-views','2',
        '--calibration-views','1','--trials','1','--test-views','2','--image-views','1',
        '--validate-every','1',
        '--save-every','1','--inference-batch','5','--cpu-threads','1','--no-lpips'])


def test_independent_full_pipeline_exact_resume_and_compact_review(tmp_path):
    torch.set_num_threads(1)
    model,scene=runtime()
    plan=dict(version='relation_benefit_v1',source=dict(codec='fixture'),spec=scene.spec)
    initial=copy.deepcopy(model.state_dict())
    with patch('gaussian_jscc.rendering.render',side_effect=renderer):
        full=entry().run_loaded(args(tmp_path/'full'),model,scene,plan)
        assert full['frozen_codec_verified']
        model2,scene2=runtime();model2.load_state_dict(initial)
        interrupted=entry().run_loaded(args(tmp_path/'split'),model2,scene2,plan,max_windows=2)
        assert interrupted['status']=='interrupted'
        resume=args(tmp_path/'split');resume.resume=True
        split=entry().run_loaded(resume,model2,scene2,plan)
    from gaussian_jscc.budgeted_allocation_state import load_checkpoint
    a=load_checkpoint(tmp_path/'full/checkpoints/latest/state.pt')
    b=load_checkpoint(tmp_path/'split/checkpoints/latest/state.pt')
    for key,value in a['learner']['network'].items():
        torch.testing.assert_close(value,b['learner']['network'][key],atol=0,rtol=0)
    assert a['progress']==b['progress']
    assert split['network_updates']==full['network_updates']==4
    losses=[json.loads(s) for s in (tmp_path/'full/metrics/loss.jsonl').read_text().splitlines()]
    assert [r['context'] for r in losses]==[0,1,2]
    inference=json.loads((tmp_path/'full/deployment/inference.json').read_text())
    assert inference['profile']['context_passes']==4
    validation=json.loads((tmp_path/'full/metrics/benefit_validation.jsonl').read_text().splitlines()[-1])
    assert set(validation['by_context'])=={'0','1','2','3'}
    assert 'deployment_context_mean' in validation
    assert full['value_units']=='delta_distortion_over_d_ref'
    review=json.loads((tmp_path/'full/review_summary.json').read_text(encoding='utf-8'))
    for row in review['rows'].values():
        assert row['total_channel_uses']==row['comparison_accounting']['total_channel_uses']
    with tarfile.open(tmp_path/'full/review.tar.gz') as archive:
        names=archive.getnames()
        assert 'review_summary.json' in names
        assert not any('cache/' in n or 'checkpoints/' in n for n in names)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in initial.items())


def test_validation_does_not_share_training_or_reference_test_views(tmp_path):
    options=args(tmp_path/'dry')
    assert options.beta==.1
    options.windows=options.fit_windows
    with pytest.raises(ValueError):entry().validate(options)


def test_completed_run_extending_train_only_invalidates_old_deployment(tmp_path):
    model,scene=runtime();out=tmp_path/'extend'
    plan=dict(version='relation_benefit_v1',source=dict(codec='fixture'),spec=scene.spec)
    with patch('gaussian_jscc.rendering.render',side_effect=renderer):
        entry().run_loaded(args(out),model,scene,plan)
        continued=args(out);continued.resume=True;continued.windows=4;continued.phase='train'
        result=entry().run_loaded(continued,model,scene,plan)
    assert result['step']==4 and not (out/'complete.json').exists()
    from gaussian_jscc.budgeted_allocation_state import load_checkpoint
    assert load_checkpoint(out/'checkpoints/final/state.pt')['progress']['window']==4
    with tarfile.open(out/'review.tar.gz') as archive:
        names=archive.getnames()
        assert 'complete.json' not in names and 'review_summary.json' not in names
        assert not any(n.startswith(('deployment/','evaluation/')) for n in names)


def test_main_accepts_its_own_scene_metadata_directory_and_loads_codec(tmp_path):
    model,scene=runtime();options=args(tmp_path/'entry')
    saved={'state_dict':copy.deepcopy(model.state_dict())}
    def create_scene(spec,raw,degree,codec,source_options,folder,old):
        folder.mkdir(parents=True)
        assert codec.training is False
        assert not any(p.requires_grad for p in codec.parameters())
        return scene
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(entry(),'load_inputs',return_value=(
            saved,model.cfg,scene.spec,{},SimpleNamespace(),{'codec':'fixture'})))
        stack.enter_context(patch.object(entry(),'read_ply',return_value=(scene.raw,scene.degree)))
        stack.enter_context(patch.object(entry(),'make_scene',side_effect=create_scene))
        stack.enter_context(patch('gaussian_jscc.rendering.render',side_effect=renderer))
        result=entry().main(options,max_windows=1)
    assert result=={'status':'interrupted','step':1}
    assert (Path(options.out)/'checkpoints/latest/state.pt').is_file()


def test_untrimmed_legacy_checkpoint_migrates_and_reuses_expensive_input_bank(tmp_path):
    from gaussian_jscc.budgeted_allocation_state import load_checkpoint
    from gaussian_jscc.dense_rd_allocation import atomic_save
    model,scene=runtime();out=tmp_path/'migrate'
    plan=dict(source=dict(codec='fixture'),spec=scene.spec)
    with patch('gaussian_jscc.rendering.render',side_effect=renderer):
        entry().run_loaded(args(out),model,scene,plan,max_windows=2)
    state=load_checkpoint(out/'checkpoints/latest/state.pt')
    rows=state['replay']['records'];visits=state['replay']['visits'];rng=state['replay']['rng']
    state['replay']=dict(version=2,capacity=8192,anchor_capacity=2048,
        validation_ids=state['validation_ids'].tolist(),anchor=rows[:2],recent=rows[2:],
        visits=visits,total_added=len(rows),rng=rng)
    old_settings=state['plan']['settings']
    old_settings.pop('replay_policy');old_settings.update(replay_capacity=8192,anchor_capacity=2048)
    atomic_save(out/'checkpoints/latest/state.pt',state)
    (out/'plan.json').write_text(json.dumps(state['plan']),encoding='utf-8')
    continued=args(out);continued.resume=True
    with patch('gaussian_jscc.rendering.render',side_effect=renderer), \
         patch.object(entry(),'prepare_bank',side_effect=AssertionError('recomputed bank')):
        result=entry().run_loaded(continued,model,scene,plan)
    restored=load_checkpoint(out/'checkpoints/latest/state.pt')
    assert result['status']=='complete' and restored['replay']['version']==3
    assert len(restored['replay']['records'])>=len(rows)
    assert restored['plan']['settings']['replay_policy']=='full_history_context_uniform_v1'
    assert (out/'replay_migration.json').is_file()
    altered=json.loads(json.dumps(restored['plan']));altered['settings']['beta']=.5
    assert not entry().same_plan(altered,restored['plan'],legacy=True)
