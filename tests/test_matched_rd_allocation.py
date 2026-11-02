import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_structured_allocation_integration import source
from test_calibrated_spatial_benefit_integration import options as benefit_options
from test_spatial_benefit_integration import Window
from test_learned_benefit_allocation import cameras,read,assert_nested_equal
from test_direct_rd_allocation import FrozenWindow
import test_budgeted_allocation as fixture


def module():
    assert importlib.util.find_spec('gaussian_jscc.matched_rd_allocation'), 'matched entry missing'
    from gaussian_jscc import matched_rd_allocation
    return matched_rd_allocation


class MatchedFixture(FrozenWindow):
    cache_hits=0;cache_misses=0
    def __init__(self,model,scene,table,scheduler,reference,args):
        super().__init__(model,scene,table,scheduler,reference,args)
        self.model,self.scene,self.reference=model,scene,reference
        self.seeds=[]
    def set_condition(self,seed):self.seeds.append(seed)
    def score(self,ids,learner,seeds,cameras):
        assert seeds==[1042]
        return torch.tensor([[2.,3.,4.]]).repeat(len(ids),1)
    def exact_costs(self,ids,seeds,cameras):return torch.tensor([[2.,3.,4.]]).repeat(len(ids),1)
    def measure(self,q,seeds,cameras,meter):
        from gaussian_jscc.budgeted_allocation_evaluation import evaluate_map
        from gaussian_jscc.allocation_distortion import ImageDistortion
        args=SimpleNamespace(trials=len(seeds),seed=42,snr=10.,lpips=False,image_views=1,
            allocation_image_objective=ImageDistortion('mse_ssim',self.reference['lambda_s']))
        row=evaluate_map(self.model,self.scene,q,cameras,args)
        cost=meter.details(q)
        return dict(row,**cost,rd_score=row['source_distortion']/self.reference['d_ref']+
                    self.args.beta*cost['total_channel_uses']/self.reference['c_ref'])


def options(entry,parent,benefit,out):
    return entry.parser().parse_args(['--run',str(parent),'--benefit-run',str(benefit),'--scene','truck',
        '--out',str(out),'--device','cpu','--calibration-views','1','--allocation-steps','3',
        '--audit-every','2','--save-every','1','--accept-trials','1','--trials','1',
        '--test-views','2','--image-views','1','--no-lpips','--full-verify-every','1'])


def test_entry_resume_shared_conditions_budgets_and_output(source):
    entry=module()
    from gaussian_jscc import calibrated_spatial_benefit_allocation as benefit_entry
    parent,root=source;benefit=root/'matched_benefit'
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch.object(benefit_entry.spatial,'SpatialWindow',Window), \
         patch.object(benefit_entry.spatial,'preflight',return_value={}), \
         patch.object(entry,'MatchedWindow',MatchedFixture), \
         patch.object(entry,'batched_preflight',return_value={}),contextlib.redirect_stdout(io.StringIO()):
        benefit_entry.main(benefit_options(parent,benefit))
        full=options(entry,parent,benefit,root/'matched_full');entry.main(full)
        part=options(entry,parent,benefit,root/'matched_resume');entry.main(part,max_updates=1)
        part.resume=True;entry.main(part)
        a=read(root/'matched_full/checkpoints/latest/state.pt');b=read(root/'matched_resume/checkpoints/latest/state.pt')
        for key in ('q','bank','scheduler','frozen','progress','calibration'):
            assert_nested_equal(a[key],b[key])
        done=json.loads((root/'matched_full/complete.json').read_text())
        assert done['frozen_codec_verified'] and done['frozen_benefit_verified']
        train=json.loads((root/'matched_full/training.json').read_text())
        assert train['optimization_view_names']==train['calibration_view_names']
        assert train['condition_seeds']==[1042]
        audits=[json.loads(s) for s in (root/'matched_full/metrics/audits.jsonl').read_text().splitlines()]
        assert all(len(r['attempts'])<=12 and r['rescored']<=512 and r['exact_groups']<=64 for r in audits)
        assert all(r['after_rd']<=r['before_rd'] for r in audits)
        assert (root/'matched_full/review.tar.gz').is_file()
        dry=options(entry,parent,benefit,root/'matched_dry');dry.dry_run=True
        assert entry.main(dry)['version']==entry.FAMILY and not (root/'matched_dry').exists()
        bash=Path('C:/softwares/Git/bin/bash.exe')
        if bash.is_file():
            output=root/'matched_launcher'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
            r=subprocess.run([str(bash),'scripts/test_matched_rd_allocation.sh',parent.as_posix(),
                benefit.as_posix(),'truck',output.as_posix(),'--calibration-views','1','--dry-run'],
                env=env,capture_output=True,text=True,encoding='utf-8',timeout=45)
            assert r.returncode==0,r.stdout+r.stderr
            assert json.loads(r.stdout[r.stdout.index('{'):])['version']==entry.FAMILY
            assert not output.exists()
        part.beta=.2
        with pytest.raises(ValueError,match='mismatch'):entry.main(part)
