import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_modular_experiment import source_fixture
import test_modular_experiment as experiment_fixture
from test_render_first import synthetic_render
import test_render_first as render_fixture


class ModularSNRControlTests(unittest.TestCase):
    def test_lpips_control_option_and_comparison_preserve_both_references(self):
        from gaussian_jscc.modular_snr_controls import parser,comparisons
        args=parser().parse_args(['--run','trained','--out','new','--lpips'])
        self.assertTrue(args.lpips)
        r=dict(path='awgn',actual_snr=10.,layout='3',condition_policy='matched',views=4,
               source_psnr=27.,photo_psnr=24.,source_ssim=.9,photo_ssim=.85,
               source_lpips=.1,photo_lpips=.2,latent_mse=.001,latent_nmse=.03,
               tx_condition_snr=10.,rx_condition_snr=10.)
        records=[dict(r,layout=tier) for tier in ('1','2','3','mixed')]
        measured=next(x for x in comparisons(records) if x['layout']=='3')['matched']
        self.assertEqual(measured['source_lpips'],.1)
        self.assertEqual(measured['photo_lpips'],.2)
        self.assertEqual(measured['photo_ssim'],.85)

    def test_existing_checkpoint_four_conditions_without_training(self):
        from gaussian_jscc.modular_experiment import main as fit
        from gaussian_jscc.modular_snr_controls import main,parser
        with tempfile.TemporaryDirectory() as temp,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()):
            root=Path(temp); source=source_fixture(root); run=root/'trained'
            opts=experiment_fixture.ModularExperimentTests().args(source,run); opts.train_snrs=[10.]
            fit(opts)
            checkpoint=run/'checkpoints/final/codec.pt'
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            # All scene identity/split metadata needed for diagnosis is in the
            # completed modular run. The original metadata directory can move.
            source.rename(root/'source_moved')
            out=root/'controls'
            args=parser().parse_args(['--run',str(run),'--out',str(out),'--device','cpu','--snrs','10','20','--trials','1',
                                     '--manifest',str(root/'manifest.json')])
            summary=main(args)
            self.assertEqual(summary['status'],'complete')
            self.assertEqual(summary['optimizer_updates'],0)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),digest)
            records=[json.loads(s) for s in (out/'metrics/test.jsonl').read_text().splitlines()]
            high=[r for r in records if r['path']=='awgn' and r['actual_snr']==20 and r['layout']=='3']
            actual={(r['condition_policy'],r['tx_condition_snr'],r['rx_condition_snr']) for r in high}
            self.assertEqual(actual,{('matched',20.,20.),('fixed_both',10.,10.),('fixed_tx',10.,20.),('fixed_rx',20.,10.)})
            at_reference=[r for r in records if r['path']=='awgn' and r['actual_snr']==10 and r['layout']=='3']
            self.assertEqual(len({(r['source_psnr'],r['latent_mse']) for r in at_reference}),1)
            self.assertTrue((out/'charts/snr_controls.png').exists())
            self.assertEqual(len(summary['comparison']),8)
            with self.assertRaises(FileExistsError): main(args)
            # Exported inference checkpoints carry top-level training support.
            # It must take precedence over the source run's train_snrs=[10].
            from gaussian_jscc.transport import load_checkpoint,save_checkpoint
            exported=root/'exported.pt'; model=load_checkpoint(checkpoint,'cpu')
            model.set_training_snrs([5.]); save_checkpoint(exported,model,8)
            args.out=str(root/'export-controls'); args.checkpoint=str(exported); args.policies=['fixed_both']; args.snrs=[20.]
            exported_summary=main(args)
            self.assertEqual(exported_summary['training_snrs'],[5.])
            self.assertEqual(exported_summary['reference_snr'],5.)


if __name__=='__main__': unittest.main()
