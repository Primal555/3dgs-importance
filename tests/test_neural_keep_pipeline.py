import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from gaussian_jscc import neural_keep_allocation as entry
from gaussian_jscc.neural_dense_keep import FAMILY
from test_hybrid_tier_pipeline import environment,options as hybrid_options
from test_structured_allocation_integration import source
from test_learned_benefit_allocation import read,assert_nested_equal


def options(parent,out):
    args=entry.parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(out)])
    vars(args).update(vars(hybrid_options(parent,out)))
    args.keep_hidden=8;args.keep_chunk=5
    return args


def test_neural_keep_default_budget_and_family(source):
    parent,root=source
    args=entry.parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(root/'neural_dry')])
    assert args.pretrain_windows==700 and args.allocation_steps==2000 and args.network_mode=='frozen'
    assert args.keep_network_lr==1e-4 and args.keep_chunk==65536
    args.calibration_views=1;args.dry_run=True
    with environment():result=entry.main(args)
    assert result['version']==FAMILY and not (root/'neural_dry').exists()
    bash=Path('C:/softwares/Git/bin/bash.exe')
    if bash.exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
        run=subprocess.run([str(bash),'scripts/test_neural_keep_allocation.sh',parent.as_posix(),'truck',
            str(root/'shell_neural'),'--calibration-views','1','--dry-run'],env=env,
            capture_output=True,text=True,encoding='utf8',timeout=40)
        assert run.returncode==0,run.stdout+run.stderr
        printed=json.loads(run.stdout[run.stdout.index('{'):])
        assert printed['version']==FAMILY and printed['pretrain_windows']==700
        assert not (root/'shell_neural').exists()


def test_learned_keep_all_points_real_calibration_and_resume(source):
    parent,root=source;full=root/'neural_full';split=root/'neural_split'
    with environment():
        result=entry.main(options(parent,full))
        entry.main(options(parent,split),max_updates=5)
        args=options(parent,split);args.resume=True;entry.main(args)
    a=read(full/'allocation/checkpoints/latest/state.pt');b=read(split/'allocation/checkpoints/latest/state.pt')
    assert a['family']==FAMILY and result['keep_estimator']=='dense_neural_gate'
    assert result['keep_network_updates']==2 and result['frozen_network_verified'] is True
    assert_nested_equal(a['table'],b['table'])
    assert_nested_equal(a['positive_provider'],b['positive_provider'])
    assert_nested_equal(a['rng'],b['rng'])
    state=a['positive_provider']['keep_policy']
    assert state['updates']==2 and state['optimizer']['state']
    assert state['network']['output.weight'].norm()>0
    metadata=json.loads((full/'allocation/training.json').read_text())
    assert metadata['trainable_modules']==['dense_keep_policy.network']
    assert metadata['aligned_spatial_inputs'] is True
    logs=[json.loads(x) for x in (full/'allocation/metrics/loss.jsonl').read_text().splitlines()]
    assert all(r['dense_keep_rows']==16 and r['keep_estimator']=='dense_neural_gate' for r in logs)
    assert all(r['keep_network_grad_norm']>0 for r in logs)
    assert all(r['neural_keep_label_groups']==min(16,(32-r['finite_corrections'])//3) for r in logs)
    assert all(r['neural_keep_render_calls']==3*r['neural_keep_label_groups'] for r in logs)
    assert all(r['finite_corrections']+r['neural_keep_render_calls']<=32 for r in logs)
    assert all(r['finite_keep_corrections']==0 and len(r['keep_calibration_records'])==r['neural_keep_label_groups'] for r in logs)
    for r in logs:
        for label in r['keep_calibration_records']:assert abs(sum(label['conditional'])-1)<1e-10


def test_aligned_first_order_control_and_pretrain_identity(source):
    parent,root=source;neural=root/'aligned_neural';control=root/'aligned_control'
    with environment():
        entry.main(options(parent,neural))
        args=options(parent,control);args.keep_estimator='first_order';result=entry.main(args)
    a=read(neural/'pretrain/checkpoints/latest/state.pt')
    b=read(control/'pretrain/checkpoints/latest/state.pt')
    assert_nested_equal(a['learner'],b['learner'])
    state=read(control/'allocation/checkpoints/latest/state.pt')
    assert 'keep_policy' not in state['positive_provider']
    rows=[json.loads(x) for x in (control/'allocation/metrics/loss.jsonl').read_text().splitlines()]
    assert all(r['keep_estimator']=='dense_first_order' for r in rows)
    seen=state['blocks']['structured']['tensors']['keep_observations'].sum(1)
    assert torch.equal(seen,torch.full_like(seen,2))
    assert result['frozen_network_verified'] is True


def test_all_sampled_closed_still_trains_keep_and_observes_context(source):
    from gaussian_jscc.neural_keep_provider import NeuralAllocationProvider
    parent,root=source;out=root/'neural_closed'
    class ClosedProvider(NeuralAllocationProvider):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            policy=self.keep_policy
            def closed_sample(ids,snr):
                safe=ids.clamp_min(0);keep,tier=policy.table.branch_scores(safe,snr)
                noise=torch.zeros_like(keep);noise[...,0]=10.;noise[...,1]=-10.
                valid=ids>=0;policy.noise[ids[valid].cpu()]=noise[valid].cpu()
                return policy.table.sample(safe,snr,noise=torch.cat((noise,torch.zeros_like(tier)),-1))
            policy.sample=closed_sample
    with environment(),patch.object(entry,'NeuralAllocationProvider',ClosedProvider):entry.main(options(parent,out))
    rows=[json.loads(x) for x in (out/'allocation/metrics/loss.jsonl').read_text().splitlines()]
    assert all(r['sampled_tier_counts'][0]==16 and r['neural_candidates']==0 for r in rows)
    assert all(r['dense_keep_rows']==16 and r['keep_network_grad_norm']>0 for r in rows)
    assert all(r['neural_keep_label_groups']>0 and r['neural_keep_render_calls']<=32 for r in rows)
