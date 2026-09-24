import json
from pathlib import Path
import tempfile
import unittest
from scripts.analyze_training_phases import analyze


class PhaseAnalysisTests(unittest.TestCase):
    def test_phase_local_windows_boundary_dedup_and_metric_definitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'training.json').write_text(json.dumps({'position_bits':12}),encoding='utf-8')
            losses=[{'step':i,'phase':'bootstrap' if i<=4 else 'render','step_seconds':.1 if i<=4 else 2.,'lr':1e-4}
                    for i in range(1,9)]
            records=[]
            for step,phase in [(0,'initial'),(2,'bootstrap'),(4,'bootstrap'),(4,'render'),(6,'render'),(8,'render')]:
                records.append({'step':step,'phase':phase,'layouts':[
                    {'layout':tier,'source_psnr':10+step+j,'photo_psnr':8+step+j,'source_mse':1/(step+1)}
                    for j,tier in enumerate(('1','2','3','mixed'))]})
            local=[{'step':i,'layouts':[{'layout':'1','loss':.1/(i+1)}]} for i in (0,2,4)]
            for name,rows in [('loss',losses),('validation',records),('bootstrap_validation',local)]:
                (root/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            report,points,windows=analyze(root,window=2)
            self.assertEqual([p['updates'] for p in report['phases']],[4,4])
            self.assertAlmostEqual(report['phases'][0]['recorded_training_seconds'],.4)
            self.assertEqual(report['phases'][1]['recorded_training_seconds'],8.)
            render=[p for p in points if p['phase']=='render']
            self.assertEqual([p['phase_step'] for p in render],[0,2,4])
            self.assertEqual(render[0]['source_psnr'],15.5)  # Mean of dB values, not PSNR(mean MSE).
            self.assertIsNone(render[0]['local_validation_loss'])
            self.assertEqual(report['phases'][1]['best_by_source_mse']['step'],8)
            self.assertEqual(len(windows),4)
            self.assertIsNone(windows[2]['mean_psnr_gain_vs_previous_window'])
            self.assertEqual(windows[3]['mean_psnr_gain_vs_previous_window'],2.)


if __name__=='__main__':
    unittest.main()
