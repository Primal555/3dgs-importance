import contextlib
import io
import json
from unittest.mock import patch

import pytest
import torch

from test_structured_allocation_integration import source
from test_learned_benefit_allocation import cameras, read, assert_nested_equal
from test_physical_benefit import record
from test_pixel_benefit_integration import options as pixel_options
import test_budgeted_allocation as fixture


def options(parent,out):
    from gaussian_jscc.physical_benefit_allocation import parser
    args=pixel_options(parent,out)
    new=parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(out),'--beta','.1'])
    vars(new).update(vars(args)); new.hidden=16
    return new


class Window:
    def __init__(self,model,scene,table,scheduler,reference,args):
        self.table=table; self.difficulty=torch.zeros(1)
    def state_dict(self): return {'difficulty':self.difficulty}
    def load_state_dict(self,state): self.difficulty=state['difficulty']
    def collect(self,step,cameras,learner,replay,pretrain):
        from gaussian_jscc.physical_benefit import FEATURE_DIM
        replay.add(record(1,.001,FEATURE_DIM))
        if not replay.validation_frozen: replay.add(record(2,.002,FEATURE_DIM),validation=True)
        n=len(self.table.logits)
        return dict(keep_costs=torch.zeros(n,2),tier_ids=torch.tensor([0]),tier_costs=torch.tensor([[.1,.2,.3]]),
                    q=torch.ones(n,dtype=torch.long),metrics=dict(source_mse=.1,source_distortion=.1))


def test_dry_run_family_isolation(source):
    from gaussian_jscc import physical_benefit_allocation as module
    parent,root=source; args=options(parent,root/'dry'); args.dry_run=True
    with contextlib.redirect_stdout(io.StringIO()): plan=module.main(args)
    assert plan['version']=='physical_benefit_allocation_v1'
    assert plan['settings']['decision_weight']==1.
    assert not (root/'dry').exists()


def test_exact_resume_and_allocation(source):
    from gaussian_jscc import physical_benefit_allocation as module
    parent,root=source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch.object(module,'PhysicalWindow',Window),patch.object(module,'preflight',return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):
        full=options(parent,root/'full'); module.main(full)
        partial=options(parent,root/'partial'); module.main(partial,max_updates=2)
        partial.resume=True; module.main(partial)
        a=read(root/'full/checkpoints/latest/state.pt'); b=read(root/'partial/checkpoints/latest/state.pt')
        assert_nested_equal(a['learner'],b['learner'])
        initial=read(root/'partial/checkpoints/initial/state.pt')
        assert_nested_equal(initial['state_dict'],b['state_dict'])
        assert_nested_equal(initial['table'],b['table'])
        partial.phase='allocation'; result=module.main(partial)
        assert result['allocation_step']==2 and result['frozen_codec_verified']
        final=read(root/'partial/checkpoints/final/state.pt')
        assert final['learner']['updates']==4
        assert_nested_equal(initial['state_dict'],final['state_dict'])
        assert (root/'partial/deployment/latest_tiers.npy').is_file()
        partial.decision_weight=2
        with pytest.raises(ValueError,match='mismatch'): module.main(partial)


def test_real_cpu_window_complete_labels_and_cache(tmp_path):
    from types import SimpleNamespace
    from test_successive_allocation import small_successive
    from gaussian_jscc.physical_benefit_features import PhysicalWindow,FEATURE_DIM
    from gaussian_jscc.physical_benefit import PhysicalLearner
    from gaussian_jscc.pixel_benefit_replay import PixelReplay
    from gaussian_jscc.response_value import ResponseMirrorTable
    from gaussian_jscc.response_value_window import ResponseScheduler
    torch.set_num_threads(1); torch.manual_seed(7)
    raw,geometry,f,model=small_successive(); model.eval().requires_grad_(False)
    ids=torch.arange(16).reshape(2,8)
    transform=torch.eye(4); transform[3,2]=3
    cam=SimpleNamespace(image_width=16,image_height=16,FoVx=1.2,FoVy=1.2,
        world_view_transform=transform,full_proj_transform=torch.eye(4),camera_center=torch.zeros(3))
    scene=SimpleNamespace(raw=raw,geometry=geometry,groups=[f],group_ids=[ids],degree=1,
        white_background=False,reference=SimpleNamespace(get=lambda c,d:torch.zeros(3,16,16,device=d)))
    table=ResponseMirrorTable(16); scheduler=ResponseScheduler([ids],seed=3)
    ref=dict(lambda_s=.01,d_ref=.01,c_ref=1000.,side_proxy=dict(xyz_per_retained=24.,bits_per_use=2.))
    args=SimpleNamespace(snr=10.,beta=.1,label_groups=4,candidate_batch=2,neighbors=4,cpu_threads=1)
    replay=PixelReplay(tmp_path,validation_groups=2)
    learner=PhysicalLearner(replay,'cpu',hidden=16)
    window=PhysicalWindow(model,scene,table,scheduler,ref,args)
    result=window.collect(1,[cam],learner,replay,True)
    assert result['metrics']['labeled_groups']>0
    groups=list(replay.training_records()); v=groups[0]['views'][0]
    assert v['inputs']['features'].shape==(4,FEATURE_DIM)
    assert v['values'].shape==(3,2) and 'density' not in v
    static=window.static
    learner.fit_scales(); learner.train_updates(1,1)
    result=window.collect(2,[cam],learner,replay,False)
    assert window.static is static
    assert len(result['tier_ids'])>0 and torch.isfinite(result['tier_costs']).all()
    assert all(p.grad is None for p in model.parameters())
    restored=PhysicalWindow(model,scene,table,scheduler,ref,args)
    restored.load_state_dict(window.state_dict())
    assert torch.equal(restored.static,window.static)


def test_visibility_preflight_cpu():
    from gaussian_jscc.physical_benefit_allocation import verify_visibility
    assert verify_visibility('cpu')['visibility_cpu_cuda_max_abs']==0


def test_q0_visibility_probe_uses_source_geometry_and_preserves_image():
    from types import SimpleNamespace
    from gaussian_jscc.physical_benefit_features import PhysicalWindow
    from gaussian_jscc.pixel_local_renderer import capture
    from gaussian_jscc.response_value import ResponseMirrorTable
    from gaussian_jscc.response_value_window import ResponseScheduler
    from test_pixel_local_renderer import raw_scene,camera
    raw=raw_scene()
    ids=torch.tensor([[0,1,2],[3,4,-1]])
    source=SimpleNamespace(raw=raw,group_ids=[ids])
    window=PhysicalWindow(None,source,ResponseMirrorTable(len(raw)),ResponseScheduler([ids]),{},SimpleNamespace())
    mixed=torch.cat((raw.clone(),torch.ones(len(raw),1)),1)
    mixed[0,-1]=0; mixed[0,:3]=100
    result=window.capture_inputs(mixed)
    assert torch.equal(result[0,:-1],raw[0]) and result[0,-1]==0
    assert torch.equal(result[1:],mixed[1:])
    assert torch.equal(mixed[0,:3],torch.full((3,),100.))
    cam=camera()
    old=capture(mixed[:,:-1],cam,0,mixed[:,-1],False,backend='cpu')
    new=capture(result[:,:-1],cam,0,result[:,-1],False,backend='cpu')
    assert torch.allclose(old.image,new.image,atol=1e-7,rtol=1e-7)
    from gaussian_jscc.pixel_benefit_features import exact_values
    candidate=raw[None,:1].repeat(4,1,1)
    candidate[2,:,11]+=.3; candidate[3,:,4]+=.2
    masks=torch.tensor([[0.],[1.],[1.],[1.]])
    projected=torch.stack([new.project(candidate[q],masks[q]) for q in range(4)])
    target=torch.zeros_like(new.image)
    before=exact_values(old,[0],projected,target,.01,1.)
    after=exact_values(new,[0],projected,target,.01,1.)
    assert torch.allclose(before,after,atol=1e-12,rtol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_visibility_preflight_cuda():
    from gaussian_jscc.physical_benefit_allocation import verify_visibility
    assert verify_visibility('cuda')['visibility_backend']=='cuda'


def test_launcher_dry_run(source):
    import os
    from pathlib import Path
    import subprocess
    import sys
    parent,root=source
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists(): pytest.skip('Git Bash unavailable')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
    command=[str(bash),'scripts/test_physical_benefit_allocation.sh',parent.as_posix(),'truck',(root/'launch').as_posix(),
             '--beta','.1','--calibration-views','1','--dry-run']
    run=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf-8',timeout=40)
    assert run.returncode==0,run.stdout+run.stderr
    assert json.loads(run.stdout[run.stdout.index('{'):])['version']=='physical_benefit_allocation_v1'
