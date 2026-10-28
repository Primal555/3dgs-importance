import contextlib
import hashlib
import io
import json
from unittest.mock import patch

import pytest
import torch

from test_structured_allocation_integration import source
from test_learned_benefit_allocation import entry,options,cameras,read,assert_nested_equal
import test_budgeted_allocation as fixture


def test_factored_online_pretrain_allocation_resume_and_legacy_plan_identity(source):
    parent,root=source;module=entry()
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        old=options(parent,root/'old_dry');old.dry_run=True;plan=module.main(old)
        assert plan['version']=='learned_benefit_allocation_v1'
        assert 'benefit_variant' not in plan['settings']
        # Exercise the CLI choice, not a test-only field assignment.
        args=module.parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(root/'factored'),
                                        '--beta','.1','--benefit-variant','factored'])
        small=options(parent,root/'factored');small.benefit_variant=args.benefit_variant
        module.main(small)
        pretrained=read(root/'factored/checkpoints/latest/state.pt')
        assert pretrained['family']=='learned_benefit_factored_v2'
        assert pretrained['learner']['objective_variant']=='factored_decision_v2'
        before=read(root/'factored/checkpoints/initial/state.pt')
        assert_nested_equal(before['table'],pretrained['table'])
        small.phase='allocation';small.resume=True;done=module.main(small)
        resumed=options(parent,root/'factored_resumed');resumed.benefit_variant='factored'
        module.main(resumed,max_updates=1);resumed.resume=True;module.main(resumed)
        resumed.phase='allocation';module.main(resumed,max_updates=1);module.main(resumed)
    a=read(root/'factored/checkpoints/latest/state.pt');b=read(root/'factored_resumed/checkpoints/latest/state.pt')
    for key in ('learner','table','latest_q'):assert_nested_equal(a[key],b[key])
    assert_nested_equal(before['state_dict'],a['state_dict'])
    assert done['family']=='learned_benefit_factored_v2'
    rows=[json.loads(s) for s in (root/'factored/metrics/loss.jsonl').read_text().splitlines()]
    from statistics import median
    summary=json.loads((root/'factored/comparison/summary.json').read_text())
    assert summary['timing']['optimization_step_p50_seconds']==median(r['step_seconds'] for r in rows if r['phase']=='allocation')


def test_fresh_factored_offline_uses_fixed_replay_and_training_only_scales(source):
    from gaussian_jscc import benefit_diagnostics as diagnostic
    parent,root=source;run=root/'replay_source'
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        entry().main(options(parent,run))
    path=run/'checkpoints/latest/state.pt';source_hash=hashlib.sha256(path.read_bytes()).hexdigest();state=read(path)
    args=diagnostic.parser().parse_args(['--run',str(run),'--out',str(root/'fresh'),
        '--fresh-factored','--updates','2','--evaluate-every','1','--cpu-threads','1'])
    with contextlib.redirect_stdout(io.StringIO()):result=diagnostic.main(args)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source_hash
    assert result['model_variant']=='factored' and result['initial_network_updates']==0
    assert result['source_baseline']['validation']['positive']['count']>0
    settings=json.loads((root/'fresh/settings.json').read_text())
    assert settings['optimizer_initialization']=='fresh Adam' and settings['renderer_calls']==0
    final=read(root/'fresh/offline_final.pt')
    assert final['model_variant']=='factored'
    for key in state['learner']['network']:
        if key.endswith(('_mean','_std','_continuous')) or key in ('scales','ready'):
            torch.testing.assert_close(final['network'][key],state['learner']['network'][key],rtol=0,atol=0)
    original=diagnostic.restore_learner(state,'cpu')
    altered=read(path)
    for row in altered['learner']['validation']['groups']:row['target'].fill_(1e6)
    altered=diagnostic.restore_learner(altered,'cpu')
    from gaussian_jscc.factored_benefit import fresh_factored
    a=fresh_factored(original,42);b=fresh_factored(altered,42)
    torch.testing.assert_close(a.network.common_scale,b.network.common_scale,rtol=0,atol=0)
    torch.testing.assert_close(a.network.increment_scale,b.network.increment_scale,rtol=0,atol=0)


def test_factored_launcher_selects_new_family_without_creating_output(source):
    import os
    from pathlib import Path
    import subprocess
    import sys
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():pytest.skip('Git Bash unavailable')
    parent,root=source;out=root/'factored_dry_script'
    script=Path(__file__).resolve().parents[1]/'scripts/test_factored_benefit_allocation.sh'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
    result=subprocess.run([str(bash),str(script),parent.as_posix(),'truck',out.as_posix(),
        '--beta','.1','--calibration-views','1','--benefit-variant','legacy','--dry-run'],
        env=env,capture_output=True,text=True,encoding='utf-8',timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    plan=json.loads(result.stdout[result.stdout.index('{'):])
    assert plan['version']=='learned_benefit_factored_v2' and not out.exists()
