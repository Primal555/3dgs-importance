import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from gaussian_jscc import hybrid_tier_allocation as original
from gaussian_jscc import fresh_hybrid_allocation as entry
from gaussian_jscc.hybrid_tier_step_controls import digest
from test_hybrid_tier_pipeline import environment,options
from test_structured_allocation_integration import source
from test_learned_benefit_allocation import read,assert_nested_equal


def args(source,out,*extra):
    return entry.parser().parse_args(['--source',str(source),'--out',str(out),'--device','cpu',*extra])


def test_fresh_state_without_old_checkpoint_and_frozen_network(source):
    from gaussian_jscc.budgeted_allocation_kernel import budgeted_mask_step
    parent,root=source;base=root/'fresh_source';out=root/'fresh_result'
    with environment():original.main(options(parent,base))
    previous=read(base/'allocation/checkpoints/latest/state.pt')
    assert previous['blocks']['counts'].sum()>0
    assert (previous['table']['keep_logits'][:,1]-previous['table']['keep_logits'][:,0]).std()>0
    before=digest(base/'benefit_network.pt')
    (base/'allocation/checkpoints').rename(base/'allocation/checkpoints.hidden')
    observed=[]
    def inspect(model,table,features,ids,geometry,task,blocks,*pos,**kw):
        if not observed:
            torch.testing.assert_close(table.keep_logits.softmax(-1)[:,1],torch.full((16,),.99,dtype=torch.float64))
            torch.testing.assert_close(table.logits.softmax(-1),torch.full((16,3),1/3,dtype=torch.float64))
            for key in ('counts','observations','finite_counts','keep_observations','weight','keep_weight'):
                assert torch.count_nonzero(getattr(blocks,key))==0,key
            observed.append(True)
        return budgeted_mask_step(model,table,features,ids,geometry,task,blocks,*pos,**kw)
    with environment(),patch('gaussian_jscc.budgeted_allocation.budgeted_mask_step',side_effect=inspect):
        result=entry.main(args(base,out,'--allocation-steps','2'))
    assert observed and result['step']==2 and result['allocation_initialization']=='fresh'
    assert result['frozen_network_verified'] is True and result['network_updates']==3
    assert digest(base/'benefit_network.pt')==before
    assert not (out/'source_state.pt').exists()
    rows=[json.loads(x) for x in (out/'allocation/metrics/loss.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==[1,2]
    assert all(r['tier_max_log_odds']==1. for r in rows)
    with tarfile.open(out/'review.tar.gz') as archive:
        names=archive.getnames()
    assert 'review_summary.json' in names
    assert not any('checkpoint' in p or 'benefit_network.pt' in p or 'source_state' in p for p in names)


@pytest.mark.parametrize('flags',[[],['--unbounded-tier-updates']])
def test_exact_resume_without_source_allocation_directory(source,flags):
    parent,root=source;base=root/'fresh_resume_source';full=root/'fresh_full';split=root/'fresh_split'
    with environment():
        original.main(options(parent,base))
        entry.main(args(base,full,*flags,'--allocation-steps','2'))
        entry.main(args(base,split,*flags,'--allocation-steps','2'),max_updates=1)
        moved=root/'hidden_source';base.rename(moved)
        try:entry.main(args(base,split,*flags,'--allocation-steps','2','--resume'))
        finally:moved.rename(base)
    a=read(full/'allocation/checkpoints/latest/state.pt');b=read(split/'allocation/checkpoints/latest/state.pt')
    for key in ('table','blocks','views','rng','positive_provider'):assert_nested_equal(a[key],b[key])
    with pytest.raises(ValueError,match='resume settings'):
        entry.main(args(base,split,*flags,'--allocation-steps','3','--resume'))


def test_defaults_directory_guard_and_shell(source):
    parent,root=source;base=root/'fresh_guard_source'
    default=args(base,root/'unused')
    assert default.allocation_steps==2000 and default.tier_max_log_odds==1.
    with environment():original.main(options(parent,base))
    nested=base/'evaluation/fresh'
    with pytest.raises(ValueError,match='independent'):entry.main(args(base,nested))
    assert not nested.exists()
    source_plan=json.loads((base/'allocation/plan.json').read_text())
    metadata_path=base/'allocation/plan.json'
    source_plan['settings']['aligned_spatial_inputs']=not source_plan['settings'].get('aligned_spatial_inputs',False)
    metadata_path.write_text(json.dumps(source_plan))
    with pytest.raises(ValueError,match='spatial input layout'):entry.source_info(base)
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if bash.exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
        run=subprocess.run([str(bash),'scripts/test_fresh_hybrid_allocation.sh',str(base),str(root/'help_out'),
            '--help'],env=env,capture_output=True,text=True,encoding='utf8',timeout=40)
        assert run.returncode==0,run.stdout+run.stderr
        assert '--unbounded-tier-updates' in run.stdout


def test_legacy_fresh_run_keeps_recorded_layout(source):
    parent,root=source;base=root/'legacy_source';out=root/'legacy_fresh'
    settings=options(parent,base);settings.aligned_spatial_inputs=False
    with environment():
        original.main(settings)
        entry.main(args(base,out,'--allocation-steps','1'))
    saved=read(out/'allocation/checkpoints/latest/state.pt')
    assert saved['plan']['settings']['aligned_spatial_inputs'] is False
