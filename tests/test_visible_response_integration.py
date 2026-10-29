import contextlib
import io
import json
from unittest.mock import patch
import torch
from test_structured_allocation_integration import source
from test_learned_benefit_allocation import entry,options,read,assert_nested_equal
from test_response_value_integration import response_cameras
import test_budgeted_allocation as fixture


def test_visible_pretrain_resume_and_frozen_source(source):
    parent,root=source;module=entry()
    args=options(parent,root/'visible');args.benefit_variant='response_visible'
    other=options(parent,root/'resumed');other.benefit_variant='response_visible'
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=response_cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch('gaussian_jscc.allocation_distortion.ComponentRenderTask.component_gradient',side_effect=AssertionError), \
         contextlib.redirect_stdout(io.StringIO()):
        module.main(args)
        before=read(root/'visible/checkpoints/latest/state.pt')
        initial=read(root/'visible/checkpoints/initial/state.pt')
        assert_nested_equal(before['table'],initial['table'])
        args.resume=True;args.phase='allocation';module.main(args)
        module.main(other,max_updates=1);other.resume=True;module.main(other)
        other.phase='allocation';module.main(other,max_updates=1);module.main(other)
    final=read(root/'visible/checkpoints/latest/state.pt');resumed=read(root/'resumed/checkpoints/latest/state.pt')
    assert final['family']=='visible_response_value_v1'
    for key in ('learner','table','blocks','latest_q'):assert_nested_equal(final[key],resumed[key])
    assert_nested_equal(final['state_dict'],initial['state_dict'])
    assert final['learner']['network']['views_mean'].numel()==949
    log=[json.loads(s) for s in (root/'visible/metrics/loss.jsonl').read_text().splitlines()]
    assert all(r['visibility_context_render_calls']==4*r['views_per_step'] for r in log)
    assert all(r['render_backward_calls']==0 and r['benefit_exact_render_calls']<=32 for r in log)
    for name in ('replay','validation'):
        for r in final['learner'][name]['groups']:
            assert r['feature_protocol']=='visible_response_patch_values_v1'
            assert torch.count_nonzero(r['features']['aux'])==0


def test_existing_replay_loss_control_is_paired_and_readonly(source):
    from gaussian_jscc.response_supervision_audit import main,parser,restore_arm
    from gaussian_jscc.benefit_diagnostics import static_features,_record_digest
    parent,root=source;module=entry()
    args=options(parent,root/'old_response');args.benefit_variant='response'
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=response_cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        module.main(args)
    state=read(root/'old_response/checkpoints/latest/state.pt');static=static_features(state)
    a=restore_arm(state,static,False);b=restore_arm(state,static,True)
    assert not a.optimizer.state and not b.optimizer.state
    for key,value in a.network.state_dict().items():assert torch.equal(value,b.network.state_dict()[key])
    assert _record_digest(a)==_record_digest(b)
    assert_nested_equal(a.replay.sample(8,3),b.replay.sample(8,3))
    audit=parser().parse_args(['--run',str(root/'old_response'),'--out',str(root/'audit'),
                              '--updates','2','--evaluate-every','1','--cpu-threads','1'])
    with patch('gaussian_jscc.rendering.render',side_effect=AssertionError('CPU audit rendered')),contextlib.redirect_stdout(io.StringIO()):
        result=main(audit)
    assert result['status']=='complete' and result['source_checkpoint_unchanged']
    before=result['before']['arms']
    assert before['original']['validation']==before['separate_scales']['validation']
    assert (root/'audit/label_scales.json').exists()
    after=result['after']['arms']
    assert after['original']['validation']!=after['separate_scales']['validation']


def test_visible_launcher_forces_protocol_and_dry_run(source):
    import os,sys,subprocess
    from pathlib import Path
    import pytest
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():pytest.skip('Git Bash unavailable')
    parent,root=source;out=root/'dry'
    script=Path(__file__).resolve().parents[1]/'scripts/test_visible_response_allocation.sh'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
    run=subprocess.run([str(bash),str(script),parent.as_posix(),'truck',out.as_posix(),
        '--beta','.1','--calibration-views','1','--benefit-variant','response','--dry-run'],env=env,
        capture_output=True,text=True,encoding='utf-8',timeout=40)
    assert run.returncode==0,run.stdout+run.stderr
    plan=json.loads(run.stdout[run.stdout.index('{'):])
    assert plan['version']=='visible_response_value_v1' and not out.exists()
    assert plan['feature_protocol']['view_dim']==949
