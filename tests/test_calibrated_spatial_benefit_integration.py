import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from test_structured_allocation_integration import source
from test_learned_benefit_allocation import cameras, read, assert_nested_equal
from test_pixel_benefit_integration import options as pixel_options
from test_spatial_benefit_integration import Window
import test_budgeted_allocation as fixture


def options(parent, out):
    from gaussian_jscc.calibrated_spatial_benefit_allocation import parser
    args=parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(out),'--beta','.1'])
    vars(args).update(vars(pixel_options(parent,out)))
    args.hidden=16; args.replay_cache_mib=1
    return args


def test_fresh_pretrain_freezes_codec_and_table_and_resume_is_exact(source):
    from gaussian_jscc import calibrated_spatial_benefit_allocation as module
    from gaussian_jscc.calibrated_spatial_benefit import DIRECTION_OBJECTIVE
    parent,root=source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch.object(module.spatial,'SpatialWindow',Window), \
         patch.object(module.spatial,'preflight',return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):
        full=options(parent,root/'soft_full'); module.main(full)
        part=options(parent,root/'soft_part'); module.main(part,max_updates=2)
        part.resume=True; module.main(part)
        a=read(root/'soft_full/checkpoints/latest/state.pt')
        b=read(root/'soft_part/checkpoints/latest/state.pt')
        initial=read(root/'soft_part/checkpoints/initial/state.pt')
        assert a['family']==module.FAMILY
        assert initial['learner']['updates']==0 and initial['learner']['scales'] is None
        assert not initial['learner']['optimizer']['state']
        assert initial['learner']['direction_objective']==DIRECTION_OBJECTIVE
        assert b['learner']['updates']==2
        assert_nested_equal(a['learner'],b['learner'])
        assert_nested_equal(initial['state_dict'],b['state_dict'])
        assert_nested_equal(initial['table'],b['table'])
        assert sorted(a['replay']['usage'].values())==sorted(b['replay']['usage'].values())
        assert b['progress']['phase']=='pretrain' and b['progress']['allocation_step']==0
        assert not (root/'soft_part/complete.json').exists()
        marker=json.loads((root/'soft_part/pretrain_complete.json').read_text())
        assert marker['frozen_codec_verified'] and marker['status']=='pretrain_complete'
        rows=[json.loads(line) for line in (root/'soft_part/metrics/benefit_validation.jsonl').read_text().splitlines()]
        assert all(r['direction_objective']==DIRECTION_OBJECTIVE for r in rows)
        assert {'positive','keep','weighted_validation','weighted_training'}<=rows[-1].keys()
        assert b['plan']['settings']['direction_objective']==DIRECTION_OBJECTIVE
        part.benefit_lr/=10
        with pytest.raises(ValueError,match='mismatch'):
            module.main(part)


def test_launcher_starts_one_fresh_200_window_experiment(source):
    from gaussian_jscc.calibrated_spatial_benefit_allocation import FAMILY
    from gaussian_jscc.calibrated_spatial_benefit import DIRECTION_OBJECTIVE
    parent,root=source
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():
        pytest.skip('Git Bash unavailable')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',RESUME='0',
             PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
    out=root/'soft_launch'
    run=subprocess.run([str(bash),'scripts/test_calibrated_spatial_benefit.sh',parent.as_posix(),'truck',
        out.as_posix(),'--beta','.1','--calibration-views','1','--dry-run'],env=env,
        capture_output=True,text=True,encoding='utf-8',timeout=40)
    assert run.returncode==0,run.stdout+run.stderr
    plan=json.loads(run.stdout[run.stdout.index('{'):]); settings=plan['settings']
    assert plan['version']==FAMILY and settings['direction_objective']==DIRECTION_OBJECTIVE
    assert settings['pretrain_windows']==200 and settings['fit_windows']==50
    assert settings['net_updates']==64 and settings['batch_groups']==8
    assert settings['label_groups']==64 and settings['decision_weight']==.25
    assert not out.exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_calibrated_direction_cuda_agrees_and_is_stationary():
    from gaussian_jscc.calibrated_spatial_benefit import calibrated_direction_loss
    from test_calibrated_spatial_benefit import example
    truth,rates=example()
    cpu_pred=(-truth).clone().requires_grad_()
    cpu_loss=calibrated_direction_loss(cpu_pred,truth,rates,1.)
    cpu_grad=torch.autograd.grad(cpu_loss.sum(),cpu_pred)[0]
    gpu_pred=(-truth).cuda().requires_grad_()
    gpu_loss=calibrated_direction_loss(gpu_pred,truth.cuda(),rates.cuda(),1.)
    gpu_grad=torch.autograd.grad(gpu_loss.sum(),gpu_pred)[0]
    torch.testing.assert_close(cpu_loss,gpu_loss.cpu(),rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(cpu_grad,gpu_grad.cpu(),rtol=1e-10,atol=1e-12)
    exact=truth.cuda().requires_grad_()
    loss=calibrated_direction_loss(exact,truth.cuda(),rates.cuda(),1.)
    assert loss.item()==0 and torch.count_nonzero(torch.autograd.grad(loss.sum(),exact)[0])==0
