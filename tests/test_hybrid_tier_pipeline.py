import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc import hybrid_tier_allocation as entry
from gaussian_jscc.hybrid_tier_benefit import FAMILY, network_digest
from test_structured_allocation_integration import source
from test_learned_benefit_allocation import read,assert_nested_equal
from test_spatial_benefit_integration import Window
import test_budgeted_allocation as fixture


def cameras(*args,**kwargs):
    result=fixture.cameras(*args,**kwargs)
    for c in result:
        c.image_width=c.image_height=8;c.FoVx=c.FoVy=1.2
        c.world_view_transform=torch.eye(4);c.world_view_transform[3,2]=3.
        c.full_proj_transform=torch.eye(4)
    return result


def options(parent,out,mode='frozen'):
    return entry.parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(out),
        '--beta','.1','--device','cpu','--renderer-backend','cpu','--network-mode',mode,
        '--pretrain-windows','4','--fit-windows','1','--allocation-steps','2',
        '--calibration-views','1','--window-views','4','--test-views','2','--trials','1',
        '--image-views','1','--cpu-threads','1','--save-every','1','--control-every','1',
        '--validate-every','1','--profile-every','0','--benefit-validate-every','1',
        '--net-updates','1','--batch-groups','1','--label-groups','2','--hidden','16',
        '--replay-cache-mib','1','--replay-capacity','8','--validation-capacity','2',
        '--allocation-batch','4','--lambda-s','.006','--no-lpips'])


@contextlib.contextmanager
def environment():
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch.object(entry.pretraining.spatial,'SpatialWindow',Window), \
         patch('gaussian_jscc.aligned_spatial_inputs.pretrain',side_effect=entry.pretraining.main), \
         patch.object(entry.pretraining.spatial,'preflight',return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):yield


def test_default_fresh_700_automatic_and_frozen(source):
    parent,root=source
    args=entry.parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(root/'dry_hybrid')])
    assert args.beta==.1
    assert args.pretrain_windows==700 and args.phase=='auto' and args.network_mode=='frozen'
    assert args.allocation_steps==2000 and args.tier_lr==.4
    assert args.aligned_spatial_inputs is True
    args.calibration_views=1;args.dry_run=True
    with environment():plan=entry.main(args)
    assert plan['automatic_transition'] and plan['pretrain_windows']==700
    assert not (root/'dry_hybrid').exists()
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if bash.exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
        run=subprocess.run([str(bash),'scripts/test_hybrid_tier_allocation.sh',parent.as_posix(),'truck',
            str(root/'shell_hybrid'),'--calibration-views','1','--dry-run'],env=env,
            capture_output=True,text=True,encoding='utf8',timeout=40)
        assert run.returncode==0,run.stdout+run.stderr
        printed=json.loads(run.stdout[run.stdout.index('{'):])
        assert printed['version']==FAMILY and printed['pretrain_windows']==700
        assert not (root/'shell_hybrid').exists()


@pytest.mark.parametrize('mode',['frozen','online'])
def test_real_spatial_provider_pipeline_and_resume(source,mode):
    parent,root=source;out=root/('hybrid_'+mode);split=root/('hybrid_split_'+mode)
    with environment():
        result=entry.main(options(parent,out,mode))
        paused=entry.main(options(parent,split,mode),max_updates=5)
        assert paused['status']=='interrupted'
        args=options(parent,split,mode);args.resume=True
        resumed=entry.main(args)
    a=read(out/'allocation/checkpoints/latest/state.pt');b=read(split/'allocation/checkpoints/latest/state.pt')
    initial=read(out/'pretrain/checkpoints/initial/state.pt')
    pretrained=read(out/'pretrain/checkpoints/latest/state.pt')
    assert_nested_equal(initial['table'],pretrained['table'])
    assert a['family']==FAMILY and b['family']==FAMILY
    assert_nested_equal(a['table'],b['table'])
    assert_nested_equal(a['blocks'],b['blocks'])
    assert_nested_equal(a['positive_provider'],b['positive_provider'])
    assert_nested_equal(a['rng'],b['rng'])
    assert result['frozen_codec_verified'] and resumed['frozen_codec_verified']
    artifact=read(out/'benefit_network.pt')
    assert artifact['settings']['aligned_spatial_inputs'] is True
    assert a['plan']['settings']['aligned_spatial_inputs'] is True
    assert pretrained['learner']['updates']==3
    if mode=='frozen':
        assert result['frozen_network_verified'] is True
        assert a['positive_provider']['network_updates']==3
        assert a['positive_provider']['learner'] is None
        assert 'replay' not in a
    else:
        assert a['positive_provider']['network_updates']==5
        assert a['replay']['root']!=pretrained['replay']['root']
        assert a['replay']['training']!=pretrained['replay']['training']
        # Source replay remains fully readable after allocation GC.
        for part in ('training','validation'):
            for record in pretrained['replay'][part]:assert (out/'pretrain/replay'/record['path']).is_file()
    keep=a['blocks']['structured']['tensors']['keep_observations'].sum(1)
    assert torch.equal(keep,torch.full_like(keep,2))
    losses=[json.loads(x) for x in (out/'allocation/metrics/loss.jsonl').read_text().splitlines()]
    assert all(r['keep_estimator']=='dense_first_order' and r['positive_estimator']=='spatial_network' for r in losses)
    assert all(r['q0_feedback_rows']==r['sampled_tier_counts'][0] for r in losses)
    assert all(r['candidate_codec_calls']==r['neural_candidate_decode_calls'] for r in losses)
    assert all(r['candidate_codec_calls']%3==0 for r in losses)
    if mode=='online':assert all(r['neural_labels']>0 and r['neural_teacher_seconds']>0 for r in losses)
    else:assert all(r['neural_labels']==0 and r['neural_teacher_seconds']==0 for r in losses)
    assert result['allocation_steps']==2 and result['pretrain_windows']==4
    with tarfile.open(out/'review.tar.gz') as archive:
        names=archive.getnames()
    assert 'review_summary.json' in names
    assert not any('checkpoint' in p or 'replay' in p for p in names)


def test_pretraining_interruption_then_automatic_allocation(source):
    parent,root=source;out=root/'hybrid_pre_interrupt'
    with environment():
        args=options(parent,out)
        result=entry.main(args,max_updates=2)
        assert result['status']=='interrupted' and not (out/'allocation').exists()
        args.resume=True;entry.main(args)
    assert (out/'complete.json').is_file()


def test_export_rejects_network_with_legacy_feature_layout(source):
    parent,root=source;out=root/'legacy_artifact'
    with environment():entry.main(options(parent,out))
    artifact=read(out/'benefit_network.pt')
    artifact['settings'].pop('aligned_spatial_inputs',None)
    torch.save(artifact,out/'benefit_network.pt')
    with pytest.raises(ValueError,match='spatial input layout'):
        entry.export_network(out,4,aligned_spatial_inputs=True)


def test_actual_aligned_windows_in_both_stages(source):
    from gaussian_jscc.aligned_spatial_inputs import AlignedSpatialWindow
    from gaussian_jscc.physical_benefit_features import static_features
    parent,root=source;out=root/'actual_aligned';windows=[]
    def construct(*args,**kwargs):
        window=AlignedSpatialWindow(*args,**kwargs);windows.append(window);return window
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch('gaussian_jscc.aligned_spatial_inputs.AlignedSpatialWindow',side_effect=construct), \
         patch.object(entry.pretraining.spatial,'preflight',return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):
        entry.main(options(parent,out))
    assert len(windows)==2
    for window in windows:
        assert not torch.equal(window.scene.order,torch.arange(len(window.scene.raw)))
        expected=static_features(window.scene.raw,window.args.neighbors,window.args.cpu_threads)
        torch.testing.assert_close(window.static[window.scene.order],expected,rtol=0,atol=0)
        mixed=torch.cat((window.scene.raw+100,torch.zeros(len(expected),1)),1)
        torch.testing.assert_close(window.capture_inputs(mixed)[:,:-1],window.scene.raw,rtol=0,atol=0)
