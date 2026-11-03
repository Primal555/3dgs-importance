import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from gaussian_jscc import hybrid_tier_allocation as original
from gaussian_jscc import hybrid_tier_step_controls as entry
from test_hybrid_tier_pipeline import environment,options
from test_structured_allocation_integration import source
from test_learned_benefit_allocation import read,assert_nested_equal


def args(source,out,*extra):
    return entry.parser().parse_args(['--source',str(source),'--out',str(out),'--device','cpu',*extra])


def test_snapshot_same_start_source_unchanged_and_original_cap_exact(source):
    parent,root=source;base=root/'step_base';snap=root/'step_snapshot';fork=root/'step_fork'
    with environment():
        original.main(options(parent,base),max_updates=5)
        before=(base/'allocation/checkpoints/latest/state.pt').read_bytes()
        captured=entry.main(args(base,snap,'--snapshot-only'))
        assert captured['source_step']==1
        result=entry.main(args(snap,fork,'--tier-max-log-odds','.25'))
        assert (base/'allocation/checkpoints/latest/state.pt').read_bytes()==before
        resume=options(parent,base);resume.resume=True;original.main(resume)
    a=read(base/'allocation/checkpoints/latest/state.pt');b=read(fork/'allocation/checkpoints/latest/state.pt')
    for key in ('table','blocks','views','rng','positive_provider'):assert_nested_equal(a[key],b[key])
    assert result['fork_start_step']==1 and result['step']==2 and result['continued_updates']==1
    assert (fork/'review.tar.gz').is_file()
    assert (snap/'source_state.pt').is_file()
    assert read(fork/'source_state.pt')['progress']['step']==1
    with tarfile.open(fork/'review.tar.gz') as archive:
        names=archive.getnames()
    assert 'review_summary.json' in names
    assert not any('checkpoint' in name or 'source_state' in name or 'replay' in name for name in names)


def test_wide_and_unbounded_resume_use_pinned_source_and_record_actual_updates(source):
    parent,root=source;base=root/'step_wide_base';snap=root/'step_wide_snapshot'
    with environment():
        original.main(options(parent,base),max_updates=5)
        entry.main(args(base,snap,'--snapshot-only'))
        for mode,extra in [('wide',[]),('unbounded',['--unbounded-tier-updates'])]:
            full=root/(mode+'_full');split=root/(mode+'_split')
            entry.main(args(snap,full,*extra,'--target-step','3'))
            entry.main(args(snap,split,*extra,'--target-step','3'),max_updates=1)
            # Source availability is not required by a resumed fork.
            pinned=snap/'source_state.pt';hidden=snap/'source_state.hidden'
            pinned.rename(hidden)
            try:entry.main(args(snap,split,*extra,'--target-step','3','--resume'))
            finally:hidden.rename(pinned)
            a=read(full/'allocation/checkpoints/latest/state.pt');b=read(split/'allocation/checkpoints/latest/state.pt')
            for key in ('table','blocks','views','rng','positive_provider'):assert_nested_equal(a[key],b[key])
            rows=[json.loads(x) for x in (full/'allocation/metrics/loss.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows]==[2,3]
            assert all(r['tier_update_mode']==('bounded' if mode=='wide' else 'unbounded') for r in rows)
            assert all(r['tier_update_rows']>0 and r['tier_raw_log_odds_step_mean']>=r['tier_applied_log_odds_step_mean']-1e-12 for r in rows)


def test_refuse_online_and_invalid_source_without_training(source):
    import pytest
    parent,root=source;base=root/'step_reject'
    with environment():original.main(options(parent,base,mode='online'),max_updates=5)
    with pytest.raises(ValueError,match='frozen'):
        entry.main(args(base,root/'step_rejected','--snapshot-only'))


def test_snapshot_shell_no_gpu_and_pinned_tensor_identity(source):
    import pytest
    parent,root=source;base=root/'step_shell_base';snap=root/'step_shell_snapshot'
    with environment():original.main(options(parent,base),max_updates=5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1')
    run=subprocess.run([sys.executable,'-B','-X','utf8','-m','gaussian_jscc.hybrid_tier_step_controls',
        '--source',str(base),'--out',str(snap),'--snapshot-only'],env=env,capture_output=True,
        text=True,encoding='utf8',timeout=40)
    assert run.returncode==0,run.stdout+run.stderr
    saved=read(snap/'source_state.pt');original_saved=read(base/'allocation/checkpoints/latest/state.pt')
    assert_nested_equal(saved,original_saved)
    with pytest.raises(ValueError,match='target step'):
        entry.allocation_options(saved,args(snap,root/'invalid_target','--target-step','1'),root/'invalid_target')
    with pytest.raises(FileExistsError):entry.main(args(base,snap,'--snapshot-only'))
    nested=base/'allocation/evaluation/new_fork'
    with pytest.raises(ValueError,match='independent'):
        entry.main(args(base/'allocation/checkpoints/latest/state.pt',nested,'--snapshot-only'))
    assert not nested.exists()
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if bash.exists():
        env.update(CUDA_VISIBLE_DEVICES='1',PYTHON_BIN=Path(sys.executable).as_posix())
        run=subprocess.run([str(bash),'scripts/test_hybrid_tier_step_controls.sh',str(snap),
            str(root/'shell_help'),'--help'],env=env,capture_output=True,text=True,encoding='utf8',timeout=40)
        assert run.returncode==0,run.stdout+run.stderr
        assert '--unbounded-tier-updates' in run.stdout
