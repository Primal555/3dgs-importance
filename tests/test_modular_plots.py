import json
import tempfile
import unittest
from pathlib import Path


class ModularPlotTests(unittest.TestCase):
    def test_phase_separated_curves_and_snr_gaps(self):
        from gaussian_jscc.modular_plots import plot_run
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'metrics').mkdir()
            paths=plot_run(root); self.assertEqual(paths,[])
            rows=[{'step':i+1,'phase_step':1,'phase':p,'loss':.1,'step_seconds':1,
                   'gradient_groups':{},'updates':{}} for i,p in enumerate(('semantic_bootstrap','semantic_render','channel_bootstrap','channel_render'))]
            (root/'metrics/loss.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            vals=[{'step':1,'path':'clean','layout':'clean','actual_snr':None,'source_psnr':25,'channel_trained':False},
                  {'step':1,'path':'finite_noiseless','layout':'3','actual_snr':10,'source_psnr':24,'channel_trained':False},
                  {'step':1,'path':'awgn','layout':'3','actual_snr':10,'source_psnr':23,'channel_trained':False}]
            (root/'metrics/validation.jsonl').write_text('\n'.join(json.dumps(r) for r in vals))
            paths=plot_run(root)
            self.assertTrue(all(p.exists() and p.stat().st_size>1000 for p in paths))
            self.assertIn('objectives_by_phase.png',[p.name for p in paths])
            self.assertNotIn('psnr_snr.png',[p.name for p in paths])
            for r in vals: r['channel_trained']=True
            (root/'metrics/test.jsonl').write_text('\n'.join(json.dumps(r) for r in vals))
            self.assertIn('psnr_snr.png',[p.name for p in plot_run(root)])

if __name__=='__main__': unittest.main()
