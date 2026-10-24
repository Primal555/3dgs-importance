import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from gaussian_jscc.budgeted_allocation_evaluation import tier_hash
import test_budgeted_allocation as fixture


class FixedRDReportTests(unittest.TestCase):
    def make_report(self, out):
        (out/'metrics').mkdir(); (out/'deployment').mkdir()
        q = torch.tensor([0, 1, 2, 3])
        np.save(out/'deployment/selected_tiers.npy', q.numpy().astype(np.uint8))
        row = dict(step=2, event='selected', objective_mode='fixed_beta', beta=.2, d_ref=.02, c_ref=1000,
            source_mse=.01, source_psnr=20., source_ssim=.8, source_lpips=None,
            photo_psnr=19., photo_ssim=.7, photo_lpips=None, total_channel_uses=2000,
            rd_score=.9, normalized_distortion=.5, normalized_cost=2., tier_counts=[1,1,1,1],
            codec_id='codec', tier_sha256=tier_hash(q))
        (out/'metrics/validation.jsonl').write_text(json.dumps(row)+'\n')
        (out/'metrics/loss.jsonl').write_text(json.dumps(dict(step=1,step_seconds=2.,
            expected_cost_proxy=1500,sampled_cost_proxy=1600,loss=.8,render_loss=.01,
            objective_mode='fixed_beta',beta=.2,d_ref=.02,c_ref=1000,evaluated_rows=2,complete_coverages=0))+'\n')
        result = dict(status='complete', objective_mode='fixed_beta', beta=.2, d_ref=.02, c_ref=1000,
            rd_score=.9, selection='best_rd', selected_step=1, selection_rd_score=.8,
            selected_total_channel_uses=2000, codec_id='codec', tier_sha256=tier_hash(q))
        return row, result

    def test_rd_report_recomputes_measured_scores_without_budget_fields(self):
        from gaussian_jscc.budgeted_allocation_report import summarize
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder); row, result=self.make_report(out)
            try: summary=summarize(out,result=result)
            except KeyError as exc: self.fail('fixed RD report still requires budget fields: '+str(exc))
            self.assertEqual(summary['objective_mode'],'fixed_beta')
            self.assertEqual(summary['endpoints'][0]['rd_score'],.9)
            self.assertIsNone(summary['endpoints'][0]['source_lpips'])
            self.assertTrue((out/'charts/rate_distortion_training.png').exists())
            self.assertTrue((out/'charts/deployment_rd_score.png').exists())
            self.assertFalse((out/'charts/budget_cost_price.png').exists())
            self.assertNotIn('budget_feasible',(out/'comparison/endpoints.csv').read_text())
            for updates in (dict(total_channel_uses=1000),dict(rd_score=.1),dict(beta=.3)):
                broken=dict(row,**updates)
                (out/'metrics/validation.jsonl').write_text(json.dumps(broken)+'\n')
                with self.subTest(updates=updates),self.assertRaisesRegex(ValueError,'RD|score|binding|objective'):
                    summarize(out,result=result)


class FixedRDCompleteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.BudgetedEntryTests.setUpClass()
        cls.source=fixture.BudgetedEntryTests.source;cls.root=fixture.BudgetedEntryTests.root

    @classmethod
    def tearDownClass(cls): fixture.BudgetedEntryTests.tearDownClass()

    def options(self, out, beta=.2):
        from test_fixed_beta_allocation import FixedBetaEntryTests
        args=FixedBetaEntryTests('test_zero_beta_dry_run_has_distinct_mode_and_never_writes')
        args.source=self.source
        return args.options(out,beta)

    def test_complete_exports_calibration_best_and_test_metrics_without_budget_gate(self):
        from gaussian_jscc import budgeted_allocation as entry
        out=self.root/'rd_complete'
        with patch('gaussian_jscc.rendering.load_cameras',side_effect=fixture.cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
            try: result=entry.main(self.options(out))
            except KeyError as exc: self.fail('RD completion uses a budget-only field: '+str(exc))
        self.assertEqual(result['status'],'complete');self.assertEqual(result['selection'],'best_rd')
        self.assertNotIn('budget_feasible',result);self.assertNotIn('budget',result)
        rows=[json.loads(s) for s in (out/'metrics/validation.jsonl').read_text().splitlines()]
        calibration=[r for r in rows if r['event'].startswith('calibration_')]
        selected=min(calibration,key=lambda r:(r['source_mse']/result['d_ref']+
            .2*r['total_channel_uses']/result['c_ref'],r['source_mse'],r['total_channel_uses']))
        self.assertEqual(result['selected_step'],selected['step'])
        self.assertAlmostEqual(result['selection_rd_score'],selected['rd_score'])
        endpoint=next(r for r in rows if r['event']=='selected')
        self.assertAlmostEqual(result['rd_score'],endpoint['source_mse']/result['d_ref']+
                               .2*endpoint['total_channel_uses']/result['c_ref'])
        self.assertTrue((out/'complete.json').exists())
        state=torch.load(out/'checkpoints/latest/state.pt',weights_only=True)
        self.assertIsNone(state['controller'])

    def test_interrupted_cost_refresh_resumes_with_same_beta_and_unique_events(self):
        from gaussian_jscc import budgeted_allocation as entry
        out=self.root/'rd_pending'; args=self.options(out,0.)
        real=entry.evaluate_map
        def fail(*a,**kw):
            result=real(*a,**kw)
            if kw.get('label')=='calibration_000001':raise RuntimeError('calibration interrupted')
            return result
        with patch('gaussian_jscc.rendering.load_cameras',side_effect=fixture.cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=fixture.render),contextlib.redirect_stdout(io.StringIO()):
            with patch.object(entry,'evaluate_map',side_effect=fail),self.assertRaisesRegex(RuntimeError,'interrupted'):
                entry.main(args)
            state=torch.load(out/'checkpoints/latest/state.pt',weights_only=True)
            self.assertEqual(state['progress']['pending'],['calibration'])
            self.assertIsNone(state['controller']);args.resume=True
            try: result=entry.main(args)
            except KeyError as exc:self.fail('RD resume/report requires budget fields: '+str(exc))
        self.assertEqual(result['beta'],0.)
        controls=[json.loads(s) for s in (out/'metrics/control.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in controls],[1,2])
        self.assertTrue(all(r['beta']==0. and r['event']=='cost_refresh' for r in controls))

    def test_real_launcher_requires_beta_preserves_errors_and_dry_run_does_not_write(self):
        bash=Path('C:/softwares/Git/bin/bash.exe')
        if not bash.exists():self.skipTest('Git Bash unavailable')
        script=Path('scripts/test_rate_distortion_allocation.sh')
        self.assertTrue(script.exists(),'fixed-beta allocation launcher missing')
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'dry'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
            base=[str(bash),str(script),self.source.as_posix(),'truck',out.as_posix()]
            for options in (['--dry-run'],['--beta','.2','--budget-total-uses','100','--dry-run']):
                run=subprocess.run(base+options,env=env,capture_output=True,text=True,encoding='utf-8')
                self.assertNotEqual(run.returncode,0);self.assertFalse(out.exists())
            run=subprocess.run(base+['--beta','0','--calibration-views','1','--dry-run'],
                               env=env,capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(run.returncode,0,run.stderr)
            plan=json.loads(run.stdout[run.stdout.index('{'):])
            self.assertEqual(plan['settings']['beta'],0.)
            self.assertEqual(plan['settings']['allocation_steps'],2000)
            self.assertEqual(plan['version'],'fixed_beta_allocation_v1');self.assertFalse(out.exists())
            resumed=subprocess.run(base+['--beta','.2','--calibration-views','1'],
                env=dict(env,RESUME='1'),capture_output=True,text=True,encoding='utf-8')
            self.assertNotEqual(resumed.returncode,0)
            self.assertIn('missing exact allocation checkpoint',resumed.stderr)


if __name__=='__main__':unittest.main()
