import contextlib
import io
import json
from unittest.mock import patch

import torch

from test_structured_allocation_integration import source
from test_learned_benefit_allocation import entry,options,cameras,read,assert_nested_equal
import test_budgeted_allocation as fixture


def response_cameras(*args,**kwargs):
    result=cameras(*args,**kwargs)
    for c in result:
        c.image_width=c.image_height=8;c.FoVx=c.FoVy=1.2
    return result


def test_response_pretrain_allocation_exact_resume_and_no_gradient_teacher(source):
    parent,root=source;module=entry()
    full=options(parent,root/'response_full');full.benefit_variant='response'
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=response_cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch('gaussian_jscc.allocation_distortion.ComponentRenderTask.component_gradient',
               side_effect=AssertionError('response route requested first-order gradients')), \
         contextlib.redirect_stdout(io.StringIO()):
        module.main(full)
        before=read(root/'response_full/checkpoints/latest/state.pt')
        initial=read(root/'response_full/checkpoints/initial/state.pt')
        assert_nested_equal(before['table'],initial['table'])
        full.phase='allocation';full.resume=True;result=module.main(full)
        resumed=options(parent,root/'response_resumed');resumed.benefit_variant='response'
        module.main(resumed,max_updates=1);resumed.resume=True;module.main(resumed)
        resumed.phase='allocation';module.main(resumed,max_updates=1);module.main(resumed)
    a=read(root/'response_full/checkpoints/latest/state.pt');b=read(root/'response_resumed/checkpoints/latest/state.pt')
    assert a['family']=='render_response_value_v1' and result['frozen_codec_verified']
    for k in ('learner','table','blocks','latest_q'):assert_nested_equal(a[k],b[k])
    assert_nested_equal(initial['state_dict'],a['state_dict'])
    rows=[json.loads(s) for s in (root/'response_full/metrics/loss.jsonl').read_text().splitlines()]
    assert all(r['render_backward_calls']==0 and not r['first_order_input'] for r in rows)
    assert all(r['benefit_exact_render_calls']<=32 for r in rows)
    assert any(r['response_candidate_rows'] for r in rows)
    for replay in ('replay','validation'):
        for r in a['learner'][replay]['groups']:
            assert 'first_order' not in r
            assert torch.count_nonzero(r['features']['aux'])==0


def test_response_cli_has_separate_protocol_and_rejects_first_order_masking(source):
    parent,root=source;module=entry()
    args=options(parent,root/'response_dry');args.benefit_variant='response';args.dry_run=True
    with contextlib.redirect_stdout(io.StringIO()):plan=module.main(args)
    assert plan['version']=='render_response_value_v1'
    assert plan['feature_protocol']['first_order_input'] is False
    assert not (root/'response_dry').exists()


def test_reference_pretraining_checkpoint_resumes_optimized_allocation_exactly(source):
    from test_response_value_reuse import reference_candidates
    parent,root=source;module=entry()
    args=options(parent,root/'response_reference_resume');args.benefit_variant='response'
    reference=options(parent,root/'response_reference');reference.benefit_variant='response'
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=response_cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         contextlib.redirect_stdout(io.StringIO()):
        with patch('gaussian_jscc.response_value_window.decode_candidates',side_effect=reference_candidates):
            module.main(args)
            module.main(reference)
            old=read(root/'response_reference_resume/checkpoints/latest/state.pt')
            reference.resume=True;reference.phase='allocation';module.main(reference)
        args.resume=True;args.phase='allocation';module.main(args)
    expected=read(root/'response_reference/checkpoints/latest/state.pt')
    actual=read(root/'response_reference_resume/checkpoints/latest/state.pt')
    assert old['family']==actual['family']=='render_response_value_v1'
    for key in ('learner','table','blocks','latest_q','state_dict'):
        assert_nested_equal(expected[key],actual[key])
    assert_nested_equal(old['state_dict'],actual['state_dict'])


def test_response_launcher_forces_new_family_and_is_readonly_on_dry_run(source):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import pytest
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():pytest.skip('Git Bash unavailable')
    parent,root=source;out=root/'response_script_dry'
    script=Path(__file__).resolve().parents[1]/'scripts/test_response_value_allocation.sh'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
    command=[str(bash),str(script),parent.as_posix(),'truck',out.as_posix(),
        '--beta','.1','--calibration-views','1','--benefit-variant','legacy','--dry-run']
    run=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf-8',timeout=40)
    assert run.returncode==0,run.stdout+run.stderr
    plan=json.loads(run.stdout[run.stdout.index('{'):])
    assert plan['version']=='render_response_value_v1' and not out.exists()
    assert plan['settings']['mask_fraction']==plan['settings']['mask_weight']==0
