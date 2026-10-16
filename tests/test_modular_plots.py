import json
import tempfile
import unittest
from pathlib import Path


class ModularPlotTests(unittest.TestCase):
    def test_missing_metrics_are_not_fabricated_and_latest_measured_boundary_wins(self):
        from gaussian_jscc.modular_plots import quality_series
        base=dict(path='awgn',layout='3',actual_snr=10)
        rows=[dict(base,step=0),dict(base,step=500,source_lpips=.3),
              dict(base,step=500,source_lpips=.2),dict(base,step=1000,source_lpips=None),
              dict(base,step=1500,source_lpips=float('nan')),
              dict(base,step=2000,source_lpips=.1,actual_snr=20)]
        selected=quality_series(rows,'awgn','3','source_lpips')
        self.assertEqual([(r['step'],r['source_lpips']) for r in selected],[(500,.2)])

    def test_quality_plots_include_ssim_lpips_and_skip_missing_legacy_values(self):
        from gaussian_jscc.modular_plots import plot_run
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'metrics').mkdir()
            rows=[]
            for step in (0,500,1000):
                for path,layout in [('clean','clean'),('awgn','1'),('awgn','2'),('awgn','3'),('awgn','mixed')]:
                    r=dict(step=step,path=path,layout=layout,actual_snr=None if path=='clean' else 10,
                           source_psnr=25.,photo_psnr=24.,source_ssim=.9,photo_ssim=.85)
                    if step>0: r.update(source_lpips=.12,photo_lpips=.16)
                    rows.append(r)
            (root/'metrics/validation.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (root/'metrics/test.jsonl').write_text('\n'.join(json.dumps(r) for r in rows if r['step']==1000))
            names={p.name for p in plot_run(root)}
            self.assertTrue({'validation_ssim.png','validation_lpips.png','ssim_snr.png','lpips_snr.png'}<=names)

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
            for r in vals: r.update(latent_mse=.1,latent_nmse=.2)
            rows[-1].update(render_loss=.04,weighted_latent_loss=.06)
            (root/'metrics/loss.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (root/'metrics/validation.jsonl').write_text('\n'.join(json.dumps(r) for r in vals))
            (root/'metrics/test.jsonl').write_text('\n'.join(json.dumps(r) for r in vals))
            names=[p.name for p in plot_run(root)]
            self.assertIn('psnr_snr.png',names)
            self.assertIn('latent_reconstruction.png',names)

if __name__=='__main__': unittest.main()
