import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from unittest.mock import patch

import pytest
import torch

import test_budgeted_allocation as fixture
from test_structured_allocation_integration import source


def entry():
    assert importlib.util.find_spec('gaussian_jscc.learned_benefit_allocation') is not None, 'missing learned allocation entry'
    from gaussian_jscc import learned_benefit_allocation
    return learned_benefit_allocation


def cameras(*args,**kwargs):
    result=fixture.cameras(*args,**kwargs)
    for camera in result:
        camera.world_view_transform=torch.eye(4);camera.full_proj_transform=torch.eye(4)
    return result


def options(parent,out):
    return entry().parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(out),'--beta','.1',
        '--pretrain-windows','3','--fit-windows','1','--allocation-steps','2','--benefit-steps','1',
        '--benefit-batch-size','4','--benefit-validate-every','1','--calibration-views','1','--window-views','4',
        '--control-every','1','--validate-every','1','--save-every','1','--profile-every','1',
        '--test-views','2','--trials','1','--image-views','1','--cpu-threads','1','--device','cpu','--no-lpips'])


def read(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def assert_nested_equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_nested_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested_equal(x,y)
    else:assert a==b


def test_pretrain_dry_run_and_phase_a_freeze_table_and_source(source):
    module=entry();parent,root=source;out=root/'benefit_pretrain'
    original=hashlib.sha256((parent/'checkpoints/final/codec.pt').read_bytes()).hexdigest()
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        args=options(parent,out);args.dry_run=True
        assert module.main(args)['version']==module.FAMILY and not out.exists()
        args.dry_run=False;result=module.main(args)
    assert result['status']=='pretrain_complete'
    initial=read(out/'checkpoints/initial/state.pt');latest=read(out/'checkpoints/latest/state.pt')
    assert_nested_equal(initial['table'],latest['table'])
    assert_nested_equal(initial['state_dict'],latest['state_dict'])
    assert latest['learner']['updates']>0 and bool(latest['learner']['network']['ready'])
    assert (out/'comparison/benefit_pretrain.json').exists()
    assert hashlib.sha256((parent/'checkpoints/final/codec.pt').read_bytes()).hexdigest()==original


def test_exact_resume_across_normalization_and_allocation_matches_continuous(source):
    module=entry();parent,root=source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        complete=root/'benefit_full';a=options(parent,complete);module.main(a)
        a.phase='allocation';a.resume=True;result=module.main(a)
        interrupted=root/'benefit_resumed';b=options(parent,interrupted)
        module.main(b,max_updates=1);b.resume=True;module.main(b)
        b.phase='allocation';module.main(b,max_updates=1);module.main(b)
    full=read(complete/'checkpoints/latest/state.pt');resumed=read(interrupted/'checkpoints/latest/state.pt')
    for key in ('table','blocks','learner','views','objective_reference','latest_q'):
        assert_nested_equal(full[key],resumed[key])
    assert result['status']=='complete' and result['frozen_codec_verified']
    assert result['allocation_step']==2
    rows=[json.loads(line) for line in (complete/'metrics/loss.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==[1,2,3,4,5]
    assert all(r['benefit_exact_render_calls']<=32 for r in rows)
    from statistics import median
    summary=json.loads((complete/'comparison/summary.json').read_text())
    assert summary['timing']['optimization_step_p50_seconds']==median(r['step_seconds'] for r in rows if r['phase']=='allocation')
    assert (complete/'comparison/endpoints.csv').exists()
    assert (complete/'deployment/selected_tiers.npy').exists()
    assert not list((complete/'evaluation/calibration_000004').rglob('*.png'))


def test_configuration_changes_are_rejected_and_allocation_needs_pretraining(source):
    module=entry();parent,root=source
    args=options(parent,root/'benefit_invalid');args.phase='allocation'
    with pytest.raises(ValueError,match='pretrain|resume'):module.main(args)
    args=options(parent,root/'benefit_mismatch')
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        module.main(args,max_updates=1);args.resume=True;args.mask_fraction=.1
        with pytest.raises(ValueError,match='mismatch'):module.main(args)


def test_validation_report_uses_paired_true_cost_and_counts_rescues():
    assert importlib.util.find_spec('gaussian_jscc.allocation_benefit_report') is not None, 'missing benefit reporting'
    from gaussian_jscc.allocation_benefit_report import summarize_records
    from test_allocation_benefit import record
    a=record();a['prediction']=torch.tensor([[-.1,0.]])
    a['prediction_masked']=torch.tensor([[.1,0.]]);a['first_order']=torch.tensor([[.2,0.]])
    a['target']=torch.tensor([[-.2,0.]]);a['rates']=torch.zeros(1);a['network_ready']=True
    b=copy.deepcopy(a);b['step']=2;b['target']=torch.tensor([[.2,0.]])
    result=summarize_records([a,b])
    assert result['positive']['stages']['network']['sign_errors']==1
    assert result['positive']['stages']['network']['regret_mean']==pytest.approx(.1)
    assert result['positive']['rescued_count']==1 and result['positive']['broken_count']==1
    assert 'window_paired_ci95' in result['positive']


def test_rank_correlation_of_repeated_constants_has_no_nan():
    from gaussian_jscc.allocation_benefit_report import correlation
    assert correlation([.1]*7,[1.,2.,3.,4.,5.,6.,7.]) is None


def test_launcher_preflights_both_phase_and_configuration_without_writing(source):
    import os
    from pathlib import Path
    import subprocess
    import sys
    parent,root=source;bash=Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():pytest.skip('Git Bash unavailable')
    script=Path(__file__).resolve().parents[1]/'scripts/test_learned_benefit_allocation.sh'
    assert script.exists(),'missing learned launcher'
    command=[str(bash),str(script),parent.as_posix(),'truck',(root/'benefit_dry_launcher').as_posix()]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
    completed=subprocess.run(command+['--beta','.1','--calibration-views','1','--dry-run'],
        env=env,capture_output=True,text=True,encoding='utf-8',timeout=40)
    assert completed.returncode==0,completed.stdout+completed.stderr
    plan=json.loads(completed.stdout[completed.stdout.index('{'):])
    assert plan['version']==entry().FAMILY and plan['settings']['pretrain_windows']==200
    assert not (root/'benefit_dry_launcher').exists()
    bad=subprocess.run(command+['--dry-run'],env=env,capture_output=True,text=True,encoding='utf-8',timeout=40)
    assert bad.returncode!=0 and 'beta' in bad.stderr


@pytest.mark.parametrize('field,value,match',[('finite_samples',3,'budget|4'),('replay_capacity',1,'capacity|replay')])
def test_impossible_supervision_settings_fail_before_creating_output(source,field,value,match):
    parent,root=source;args=options(parent,root/('invalid_'+field));setattr(args,field,value)
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         contextlib.redirect_stdout(io.StringIO()),pytest.raises(ValueError,match=match):entry().main(args)
    assert not __import__('pathlib').Path(args.out).exists()


def test_q0_component_diagnostics_use_the_same_conditional_mixture_as_rd():
    from gaussian_jscc.allocation_benefit_report import summarize_records
    from test_allocation_benefit import record
    r=record('keep');r['conditional']=torch.tensor([.5,.25,.25],dtype=torch.float64)
    r['target']=torch.tensor([[1.,0.],[2.,0.],[4.,0.]])
    r['prediction']=torch.tensor([[0.,0.],[2.,0.],[8.,0.]])
    r['prediction_masked']=r['prediction'].clone();r['rates'].zero_()
    got=summarize_records([r])['keep']['stages']['network']
    assert got['component_mae']==pytest.approx([.5,0.])
    assert got['component_rmse']==pytest.approx([.5,0.])
    assert got['distortion_mae']==pytest.approx(.5)


def test_resume_releases_original_replay_checkpoint_before_next_window(source):
    import weakref
    module=entry();parent,root=source;args=options(parent,root/'benefit_release')
    real_restore=module.restore_state;real_window=module.BenefitWindow;refs=[]
    def restore(*a,**kw):
        state=real_restore(*a,**kw)
        refs.append(weakref.ref(state['learner']['replay']['groups'][0]['features']['point']))
        return state
    def window(*a,**kw):
        assert refs and refs[0]() is None,'original checkpoint replay retained during training'
        return real_window(*a,**kw)
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        module.main(args,max_updates=1);args.resume=True
        with patch.object(module,'restore_state',side_effect=restore),patch.object(module,'BenefitWindow',side_effect=window):
            module.main(args,max_updates=1)


def test_pending_validation_failure_recovers_without_repeated_learning(source):
    module=entry();parent,root=source;real=module.validate_network
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        baseline=root/'benefit_pending_full';module.main(options(parent,baseline))
        out=root/'benefit_pending_resume';args=options(parent,out)
        def fail(learner,step):
            if step==1:raise RuntimeError('pending benefit evaluation interrupted')
            return real(learner,step)
        with patch.object(module,'validate_network',side_effect=fail),pytest.raises(RuntimeError,match='interrupted'):
            module.main(args)
        args.resume=True;module.main(args)
    a=read(baseline/'checkpoints/latest/state.pt');b=read(out/'checkpoints/latest/state.pt')
    for key in ('table','learner','blocks','views'):assert_nested_equal(a[key],b[key])


def test_resume_before_fitting_preserves_all_normalization_samples(source):
    module=entry();parent,root=source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        baseline=root/'benefit_before_fit_full';a=options(parent,baseline);a.fit_windows=2;module.main(a)
        out=root/'benefit_before_fit_resume';b=options(parent,out);b.fit_windows=2
        module.main(b,max_updates=1)
        state=read(out/'checkpoints/latest/state.pt')
        assert not bool(state['learner']['network']['ready'])
        assert len(state['learner']['normalization_records'])>0
        b.resume=True;module.main(b)
    a=read(baseline/'checkpoints/latest/state.pt');b=read(out/'checkpoints/latest/state.pt')
    assert_nested_equal(a['learner'],b['learner'])
