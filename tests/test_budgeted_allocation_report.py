import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from gaussian_jscc.budgeted_allocation_evaluation import tier_hash


class ReportTests(unittest.TestCase):
    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.budgeted_allocation_report'))
        from gaussian_jscc import budgeted_allocation_report
        return budgeted_allocation_report

    def test_report_checks_actual_feasibility_binding_and_keeps_optional_metric_missing(self):
        entry=self.entry()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder); (out/'metrics').mkdir(); (out/'deployment').mkdir()
            q=torch.tensor([1,2,0,3]); np.save(out/'deployment/selected_tiers.npy',q.numpy().astype(np.uint8))
            selected=dict(step=2,event='selected',source_psnr=24.,source_ssim=.9,source_lpips=None,
                photo_psnr=23.,photo_ssim=.88,photo_lpips=None,total_channel_uses=90,budget=100,
                budget_feasible=True,tier_counts=[1,1,1,1],codec_id='codec',tier_sha256=tier_hash(q))
            (out/'metrics/validation.jsonl').write_text(json.dumps(selected)+'\n')
            (out/'metrics/loss.jsonl').write_text('\n'.join(json.dumps(dict(step=i,step_seconds=i,
                expected_cost_proxy=80,sampled_cost_proxy=85,price=.1,budget=100,evaluated_rows=i,
                complete_coverages=0)) for i in (1,2)))
            result=dict(status='complete',budget_feasible=True,selected_total_channel_uses=90,budget=100,
                        codec_id='codec',tier_sha256=tier_hash(q))
            summary=entry.summarize(out,result=result)
            self.assertEqual(summary['units'],'total complex channel uses')
            self.assertEqual(summary['timing']['optimization_step_p50_seconds'],1.5)
            self.assertIsNone(summary['endpoints'][0]['source_lpips'])
            self.assertTrue((out/'charts/budget_cost_price.png').exists())
            selected['total_channel_uses']=110
            (out/'metrics/validation.jsonl').write_text(json.dumps(selected)+'\n')
            with self.assertRaisesRegex(ValueError,'feasib|budget|binding'): entry.summarize(out,result=result)


class LauncherTests(unittest.TestCase):
    def test_real_launcher_budget_preflight_resume_and_error_status(self):
        import test_budgeted_allocation as fixture
        fixture.BudgetedEntryTests.setUpClass()
        self.addCleanup(fixture.BudgetedEntryTests.tearDownClass)
        source=fixture.BudgetedEntryTests.source
        bash=Path('C:/softwares/Git/bin/bash.exe')
        if not bash.exists(): self.skipTest('Git Bash unavailable')
        script=Path('scripts/test_budgeted_allocation.sh')
        self.assertTrue(script.exists(),'missing budgeted allocation launcher')
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'dry'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
            base=[str(bash),str(script),source.as_posix(),'truck',out.as_posix()]
            missing=subprocess.run(base+['--dry-run'],env=env,capture_output=True,text=True,encoding='utf-8')
            self.assertNotEqual(missing.returncode,0); self.assertFalse(out.exists())
            command=base+['--budget-total-uses','100000','--calibration-views','1','--dry-run']
            run=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(run.returncode,0,run.stderr)
            plan=json.loads(run.stdout[run.stdout.index('{'):])
            self.assertEqual(plan['settings']['allocation_steps'],2000)
            self.assertEqual(plan['settings']['window_views'],4); self.assertFalse(out.exists())
            resume=subprocess.run(base+['--budget-total-uses','100000','--calibration-views','1'],
                env=dict(env,RESUME='1'),capture_output=True,text=True,encoding='utf-8')
            self.assertNotEqual(resume.returncode,0)
            self.assertIn('missing exact allocation checkpoint',resume.stderr)


if __name__=='__main__': unittest.main()
