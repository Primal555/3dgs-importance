import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_structured_allocation_integration import source
from test_learned_benefit_allocation import cameras, read, assert_nested_equal
from test_pixel_benefit_integration import options as pixel_options
from test_physical_benefit import record
from test_spatial_benefit import collision_inputs
import test_budgeted_allocation as fixture


def options(parent, out):
    from gaussian_jscc.spatial_benefit_allocation import parser
    args = parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(out),'--beta','.1'])
    vars(args).update(vars(pixel_options(parent,out)))
    args.hidden=16; args.replay_cache_mib=1
    return args


class Window:
    def __init__(self, model, scene, table, scheduler, reference, args):
        self.table=table; self.difficulty=torch.zeros(1)
        self.inputs=collision_inputs()[0][0]
    def state_dict(self): return {'difficulty':self.difficulty}
    def load_state_dict(self, state): self.difficulty=state['difficulty']
    def collect(self, step, cameras, learner, replay, pretrain):
        g=record(1,.001,158); g['views'][0]['inputs']=self.inputs
        replay.add(g)
        if not replay.validation_frozen:
            replay.add(dict(g,source=2),validation=True)
        n=len(self.table.logits)
        return dict(keep_costs=torch.zeros(n,2),tier_ids=torch.tensor([0]),
            tier_costs=torch.tensor([[.1,.2,.3]]),q=torch.ones(n,dtype=torch.long),
            metrics=dict(source_mse=.1,source_distortion=.1))


def test_resume_allocation_keeps_codec_table_and_visits_exact(source):
    from gaussian_jscc import spatial_benefit_allocation as module
    parent,root=source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch.object(module,'SpatialWindow',Window), patch.object(module,'preflight',return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):
        full=options(parent,root/'spatial_full'); module.main(full)
        part=options(parent,root/'spatial_part'); module.main(part,max_updates=2)
        part.resume=True; module.main(part)
        a=read(root/'spatial_full/checkpoints/latest/state.pt')
        b=read(root/'spatial_part/checkpoints/latest/state.pt')
        assert_nested_equal(a['learner'],b['learner'])
        assert sorted(a['replay']['usage'].values())==sorted(b['replay']['usage'].values())
        initial=read(root/'spatial_part/checkpoints/initial/state.pt')
        assert_nested_equal(initial['table'],b['table'])
        assert_nested_equal(initial['state_dict'],b['state_dict'])
        assert sum(b['replay']['usage'].values())==2
        part.phase='allocation'; result=module.main(part)
        final=read(root/'spatial_part/checkpoints/final/state.pt')
        assert result['allocation_step']==2 and result['frozen_codec_verified']
        assert final['learner']['updates']==4
        assert sum(final['replay']['usage'].values())==4
        assert_nested_equal(initial['state_dict'],final['state_dict'])
        assert (root/'spatial_part/deployment/latest_tiers.npy').is_file()
        part.common_weight=2
        with pytest.raises(ValueError,match='mismatch'): module.main(part)


def test_actual_codec_window_reuses_profiles_preserves_labels_and_fits(tmp_path):
    from test_successive_allocation import small_successive
    from gaussian_jscc.spatial_benefit import SpatialLearner
    from gaussian_jscc.spatial_benefit_features import SpatialWindow,CONTEXT_DIM,STATE_DIM
    from gaussian_jscc.spatial_benefit_replay import SpatialReplay
    from gaussian_jscc.response_value import ResponseMirrorTable
    from gaussian_jscc.response_value_window import ResponseScheduler
    from gaussian_jscc.physical_benefit_features import PhysicalWindow
    from gaussian_jscc.physical_benefit import PhysicalLearner
    from gaussian_jscc.pixel_benefit_replay import PixelReplay
    torch.set_num_threads(1); torch.manual_seed(7)
    raw,geometry,f,model=small_successive(); model.eval().requires_grad_(False)
    ids=torch.arange(16).reshape(2,8)
    transform=torch.eye(4); transform[3,2]=3
    cam=SimpleNamespace(image_width=16,image_height=16,FoVx=1.2,FoVy=1.2,
        world_view_transform=transform,full_proj_transform=torch.eye(4),camera_center=torch.zeros(3))
    scene=SimpleNamespace(raw=raw,geometry=geometry,groups=[f],group_ids=[ids],degree=1,
        white_background=False,reference=SimpleNamespace(get=lambda c,d:torch.zeros(3,16,16,device=d)))
    ref=dict(lambda_s=.01,d_ref=.01,c_ref=1000.,side_proxy=dict(xyz_per_retained=24.,bits_per_use=2.))
    args=SimpleNamespace(snr=10.,beta=.1,label_groups=4,candidate_batch=2,neighbors=4,cpu_threads=1)
    spatial_replay=SpatialReplay(tmp_path/'new',validation_groups=2)
    old_replay=PixelReplay(tmp_path/'old',validation_groups=2)
    spatial_learner=SpatialLearner(spatial_replay,'cpu',hidden=16)
    old_learner=PhysicalLearner(old_replay,'cpu',hidden=16)
    windows=[]
    for cls,learner,replay in ((PhysicalWindow,old_learner,old_replay),(SpatialWindow,spatial_learner,spatial_replay)):
        table=ResponseMirrorTable(16); scheduler=ResponseScheduler([ids],seed=3)
        window=cls(model,scene,table,scheduler,ref,args); windows.append(window)
        torch.manual_seed(919)
        window.collect(1,[cam],learner,replay,True)
    old=list(old_replay.training_records()); new=list(spatial_replay.training_records())
    assert [g['source'] for g in old]==[g['source'] for g in new]
    for a,b in zip(old,new):
        va,vb=a['views'][0],b['views'][0]
        torch.testing.assert_close(va['values'],vb['values'],rtol=0,atol=0)
        assert vb['inputs']['context'].shape==(8,CONTEXT_DIM)
        assert vb['inputs']['states'].shape==(4,8,STATE_DIM)
        assert vb['inputs']['weights'].shape==(8,)
    spatial_learner.fit_scales(); stats=spatial_learner.train_updates(3,2)
    assert stats['replay_all_group_draws']==6
    with patch.object(spatial_learner,'predict',side_effect=AssertionError('single-view path used')), \
         patch.object(spatial_learner,'predict_group_views',wraps=spatial_learner.predict_group_views) as batch:
        result=windows[-1].collect(2,[cam],spatial_learner,spatial_replay,False)
    assert batch.call_count>0
    assert max(len(call.args[0]) for call in batch.call_args_list)>1
    assert len(result['tier_ids'])>0 and torch.isfinite(result['tier_costs']).all()
    assert all(p.grad is None for p in model.parameters())


def test_profile_cache_and_cpu_cuda_samples():
    from gaussian_jscc.spatial_benefit_features import spatial_inputs
    from gaussian_jscc.physical_benefit_features import visibility_statistics
    inputs,tile,target,a,b=collision_inputs()
    stats=visibility_statistics(tile,target)
    raw=torch.zeros(4,3,14); raw[...,7]=1
    first=spatial_inputs(torch.zeros(24),stats,raw,a,0,tile,target,torch.arange(3),.01,.006,cache_key='block')
    profile=tile._spatial_profiles['block']
    again=spatial_inputs(torch.zeros(24),stats,raw,b,1,tile,target,torch.arange(3),.01,.006,cache_key='block')
    assert tile._spatial_profiles['block'] is profile
    torch.testing.assert_close(first['context'][:,-54:],again['context'][:,-54:])
    if torch.cuda.is_available():
        from gaussian_jscc.pixel_local_renderer import TileScene
        gpu=TileScene.from_projected(tile.projected.cuda(),16,48,backend='cuda')
        actual=spatial_inputs(torch.zeros(24),stats.cuda(),raw.cuda(),a.cuda(),0,
                             gpu,target.cuda(),torch.arange(3,device='cuda'),.01,.006)
        for key in first:
            torch.testing.assert_close(first[key],actual[key].cpu(),rtol=3e-5,atol=3e-5)


def test_launch_dry_run_and_defaults(source):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from gaussian_jscc.spatial_benefit_allocation import parser
    parent,root=source
    args=parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(root/'defaults'),'--beta','.1'])
    assert args.net_updates*args.batch_groups==512
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists(): pytest.skip('Git Bash unavailable')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
    run=subprocess.run([str(bash),'scripts/test_spatial_benefit_allocation.sh',parent.as_posix(),'truck',
        (root/'launch').as_posix(),'--beta','.1','--calibration-views','1','--dry-run'],
        env=env,capture_output=True,text=True,encoding='utf-8',timeout=40)
    assert run.returncode==0,run.stdout+run.stderr
    assert json.loads(run.stdout[run.stdout.index('{'):])['version']=='spatial_benefit_allocation_v1'
