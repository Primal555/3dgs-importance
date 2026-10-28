import copy
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from gaussian_jscc.allocation_update_inspection import hypothetical_update, conditional_keep_label, restore_history
from gaussian_jscc.structured_evidence import BoundedMirrorTable
from test_calibrated_evidence import make, positive
from gaussian_jscc.allocation_update_inspection import UpdateInspector
from gaussian_jscc.budgeted_allocation_kernel import budgeted_mask_step
from gaussian_jscc.calibrated_evidence import CalibratedEvidence
from gaussian_jscc.allocation_distortion import ComponentRenderTask,ImageDistortion
from test_successive_allocation import small_successive
from test_mask_gradient import Reference,toy_mask_render
from test_structured_allocation_integration import source,options
import test_budgeted_allocation as fixture


def test_hypothetical_update_matches_real_bounded_mirror_without_mutating_table():
    table=BoundedMirrorTable(4)
    table.keep_trust=torch.tensor([.3,.4,.5,.6])
    table.tier_trust=torch.tensor([.7,.6,.5,.4])
    before=copy.deepcopy(table.state_dict())
    cost=torch.tensor([.03,-.02,.07],dtype=torch.float64)
    change=hypothetical_update(table,1,cost,1,2,.1,.4,711691)
    real=BoundedMirrorTable(4);real.load_state_dict(before)
    real.keep_trust=table.keep_trust;real.tier_trust=table.tier_trust[1:2]
    old=float(real.logits[1,1]-real.logits[1,0])
    real.update(torch.zeros(4,2),torch.tensor([1]),cost[None],.1,.4,711691)
    assert change==pytest.approx(float(real.logits[1,1]-real.logits[1,0])-old,abs=1e-14)
    for k,v in before.items(): torch.testing.assert_close(table.state_dict()[k],v,rtol=0,atol=0)


def test_conditional_keep_label_matches_on_minus_off_under_any_baseline():
    deltas=torch.tensor([[0.,0.],[2.,4.],[3.,6.],[5.,10.]],dtype=torch.float64)
    probability=torch.tensor([.2,.3,.5],dtype=torch.float64)
    assert conditional_keep_label(deltas,probability,.5,2.,.1)==pytest.approx(3.9)
    shifted=deltas-deltas[2]
    assert conditional_keep_label(shifted,probability,.5,2.,.1)==pytest.approx(3.9)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_conditional_keep_label_accepts_cuda_deltas_and_cpu_probabilities():
    deltas=torch.tensor([[0.,0.],[2.,4.],[3.,6.],[5.,10.]],device='cuda',dtype=torch.float64)
    assert conditional_keep_label(deltas,torch.tensor([.2,.3,.5]),.5,2.,.1)==pytest.approx(3.9)


def test_restore_history_undoes_virtual_observation_and_keeps_audit_coverage():
    s=make();positive(s,1,checked=True);state=s.state_dict()
    positive(s,2,views=(4,5,6,7));s.counts[0]+=1
    restore_history(s,state)
    assert int(s.counts[0])==1
    for name in ('mean','second','weight','observations','evidence_step'):
        torch.testing.assert_close(getattr(s,name),state['evidence'][name],rtol=0,atol=0)
    for k,v in state['calibrated']['tensors'].items():
        torch.testing.assert_close(getattr(s,k),v,rtol=0,atol=0)


def test_summary_keeps_sampled_q0_checks_separate_from_history():
    from gaussian_jscc.allocation_update_audit import summarize
    from gaussian_jscc.allocation_update_inspection import assessment
    rows=[dict(branch='keep_sampled',checks={'history_update':assessment(.2,.1)},
        sampled_reference={'raw_current':assessment(-.1,.1),'corrected_current':assessment(.1,.1)},
        delta_log_odds=-.05,update_direction_correct=True)]
    got=summarize(rows)
    assert got['keep_sampled']['stages']['history_update']['sign_errors']==0
    assert got['keep_sampled_reference']['stages']['raw_current']['sign_errors']==1
    assert got['keep_sampled_reference']['stages']['corrected_current']['sign_errors']==0


def test_exact_labels_do_not_change_predicted_costs_or_rng():
    outputs=[]
    for scale in (1.,10.):
        torch.manual_seed(419);torch.set_num_threads(1)
        _,geometry,features,model=small_successive();model.eval().requires_grad_(False)
        ids=[torch.arange(16).reshape(2,8)];table=BoundedMirrorTable(16,.7)
        s=CalibratedEvidence(ids,finite_samples=0);s.configure_views(1,1);s.set_view_window([0])
        inspector=UpdateInspector(6,.1,.4)
        task=ComponentRenderTask([SimpleNamespace(factor=1.,source_count=16)],Reference(),1,
            distortion=ImageDistortion('mse_ssim',.0125))
        original=task.delta_components
        # Only labels differ; the real renderer still supplies all gradients and candidates.
        def labels(scene):return original(scene)*scale
        task.delta_components=labels
        with patch('gaussian_jscc.rendering.render',side_effect=toy_mask_render):
            got=budgeted_mask_step(model,table,[features],ids,geometry,task,s,10.,'awgn',d_ref=.1,
                price=.1,rate_normalizer=1000,side_proxy=dict(step=1,xyz_per_retained=1.,fixed_uses=8.),
                timings=None,inspector=inspector)
        outputs.append((got['tier_costs'].clone(),got['keep_costs'].clone(),torch.get_rng_state()))
    for a,b in zip(outputs[0],outputs[1]):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('closed',[False,True])
@pytest.mark.parametrize('warm',[False,True])
def test_kernel_inspector_matches_q0_labels_and_never_absorbs_exact_values(closed,warm):
    torch.set_num_threads(1);torch.manual_seed(42)
    _,geometry,features,model=small_successive();model.eval().requires_grad_(False)
    ids=[torch.arange(16).reshape(2,8)]
    table=BoundedMirrorTable(16,.7)
    if closed:table.keep_logits[:,0]=100.;table.keep_logits[:,1]=-100.
    before=copy.deepcopy(table.state_dict())
    s=CalibratedEvidence(ids,finite_samples=0);s.configure_views(1,1);s.set_view_window([0])
    if warm:
        s.current_q=torch.ones(16,dtype=torch.long)
        for step in (1,2,3):
            s.accumulate_keep(torch.full((16,),step),torch.zeros(16,2),torch.full((16,3),1/3),step,.0125)
    inspector=UpdateInspector(6,.1,.4)
    task=ComponentRenderTask([SimpleNamespace(factor=1.,source_count=16)],Reference(),1,
        distortion=ImageDistortion('mse_ssim',.0125))
    with patch('gaussian_jscc.rendering.render',side_effect=toy_mask_render):
        got=budgeted_mask_step(model,table,[features],ids,geometry,task,s,10.,'awgn',d_ref=.1,
            price=.1,rate_normalizer=1000,side_proxy=dict(step=4 if warm else 1,xyz_per_retained=1.,fixed_uses=8.),
            timings=None,inspector=inspector)
    assert 3<=inspector.render_calls<=6
    assert len(s.positive_corrections.keys)==len(s.keep_corrections.keys)==0
    assert int(s.positive_exact_count.sum()+s.keep_exact_count.sum())==0
    for k,v in before.items():torch.testing.assert_close(table.state_dict()[k],v,rtol=0,atol=0)
    for row in inspector.records:
        if row['branch'].startswith('keep_'):
            assert row['branch']==('keep_mixture' if warm else 'keep_sampled')
            if warm:
                want=conditional_keep_label(torch.tensor(row['exact_components_by_tier']),
                    torch.tensor(row['conditional']),.0125,.1,row['rate_delta'])
            else:
                deltas=torch.tensor(row['exact_components_by_tier'])
                part=deltas[row['sampled_reference']['tier']]-deltas[0]
                want=float((part[0]+.0125*part[1])/.1)+row['rate_delta']
            assert row['checks']['history_update']['truth']==pytest.approx(want,rel=1e-6,abs=1e-10)
        else:
            position=(got['tier_ids']==row['source_row']).nonzero().item()
            old,new=row['old_tier']-1,row['new_tier']-1
            want=float(got['tier_costs'][position,new]-got['tier_costs'][position,old])
            assert row['checks']['history_update']['prediction']==pytest.approx(want)
    assert inspector.records


def test_server_entry_is_read_only_and_writes_only_small_audit_files(source):
    from gaussian_jscc import allocation_update_audit as audit
    from gaussian_jscc import structured_evidence_allocation as train
    parent,root=source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
        run=root/'audit_source';train.main(options(parent,run))
        checkpoint=run/'checkpoints/best_rd/state.pt'
        before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        out=root/'audit_output'
        args=audit.parser().parse_args(['--run',str(run),'--out',str(out),'--windows','2',
            '--samples','6','--device','cpu','--cpu-threads','1'])
        args.dry_run=True;audit.main(args);assert not out.exists()
        bash=shutil.which('bash') if os.name!='nt' else 'C:/softwares/Git/bin/bash.exe'
        if bash and Path(bash).is_file():
            launcher_out=root/'launcher_output'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=Path(sys.executable).as_posix(),
                DRY_RUN='1',WINDOWS='1',AUDIT_SAMPLES='6',PYTHONUTF8='1')
            launch=subprocess.run([bash,'scripts/test_allocation_update_audit.sh',str(launcher_out),str(run)],
                cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=45)
            assert launch.returncode==0,launch.stdout+launch.stderr
            assert 'allocation_update_audit_v1' in launch.stdout
            assert not launcher_out.exists()
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        invalid=copy.deepcopy(state);invalid['plan']['settings']['snr']=5.
        with patch.object(audit.torch,'load',return_value=invalid),pytest.raises(ValueError,match='10dB'):
            audit.main(args)
        args.dry_run=False;audit.main(args)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
        complete=json.loads((out/'complete.json').read_text())
        assert complete['status']=='complete' and complete['windows']==2
        assert complete['codec_unchanged'] and complete['probabilities_unchanged'] and complete['predictive_history_unchanged']
        rows=[json.loads(s) for s in (out/'windows.jsonl').read_text().splitlines()]
        assert len(rows)==2 and all(x['counterfactual_renders']<=6 for x in rows)
        record=json.loads((run/'training.json').read_text())
        for r in rows:assert set(r['views'])<=set(record['allocation_train_view_names'])
        assert not list(out.rglob('*.pt')) and not list(out.rglob('*.png'))
        with pytest.raises(FileExistsError):audit.main(args)
