import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_modular_experiment import source_fixture
from test_render_first import synthetic_render
from test_perceptual_metrics import backend
import test_render_first as render_fixture


def cameras():
    result=render_fixture.RenderFirstTests().cameras()
    for i,camera in enumerate(result):
        camera.camera_center=torch.tensor([4.,2.+i,3.])
    return result


class PayloadExperimentTests(unittest.TestCase):
    def options(self,source,out):
        from gaussian_jscc.xyz_payload_experiment import parser
        return parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu',
            '--semantic-bootstrap-steps','4','--semantic-render-steps','1',
            '--channel-bootstrap-steps','2','--channel-render-steps','1',
            '--validate-every','4','--save-every','4','--validation-trials','1',
            '--context-centers','4','2','--context-neighbors','3','4','--channel-hidden','24','--no-plots'])

    def test_all_phases_report_actual_coordinates_rates_and_frozen_encoder(self):
        from gaussian_jscc.xyz_payload_experiment import main
        from gaussian_jscc.xyz_payload_checkpoint import load_checkpoint
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.perceptual_metrics._load_lpips_model',side_effect=backend), \
             patch('gaussian_jscc.rendering.load_cameras',return_value=cameras()):
            root=Path(t); source=source_fixture(root); out=root/'run'
            options=self.options(source,out); options.lpips=True; main(options)
            done=json.loads((out/'complete.json').read_text()); self.assertEqual(sum(done['completed'].values()),8)
            rows=[json.loads(s) for s in (out/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),8)
            self.assertEqual(rows[3]['objective'],'directional_color_mse_unclipped_v1')
            self.assertEqual(rows[-1]['updates']['semantic_encoder']['update_norm'],0)
            self.assertGreater(rows[-1]['scene_gradient_norms']['xyz'],0)
            rec=[json.loads(s) for s in (out/'metrics/test.jsonl').read_text().splitlines()]
            self.assertEqual(len(rec),9)
            self.assertLess(next(r for r in rec if r['path']=='clean')['world_distance_rmse'],1e-5)
            comm=[r for r in rec if r['path']!='clean']
            self.assertTrue(all(r['xyz_bytes']==0 and r['world_distance_rmse']>0 for r in comm))
            self.assertTrue(all('source_ssim' in r and 'photo_ssim' in r for r in rec))
            self.assertTrue(all('source_lpips' in r and 'photo_lpips' in r for r in rec))
            a=load_checkpoint(out/'checkpoints/end_semantic_render/codec.pt')
            b=load_checkpoint(out/'checkpoints/final/codec.pt')
            for x,y in zip(a.semantic_encoder.parameters(),b.semantic_encoder.parameters()):
                torch.testing.assert_close(x,y,atol=0,rtol=0)
            self.assertFalse((out/'packet/xyz.bin').exists())
            self.assertTrue((out/'packet/received.npy').exists())
            with self.assertRaises(FileExistsError): main(self.options(source,out))

    def test_resume_midphase_and_at_boundary_match_uninterrupted(self):
        from gaussian_jscc.xyz_payload_experiment import main
        from gaussian_jscc.xyz_payload_checkpoint import load_checkpoint
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras',return_value=cameras()):
            root=Path(t); source=source_fixture(root); main(self.options(source,root/'full'))
            for stop in (3,5,7):
                out=root/f'stop{stop}'; main(self.options(source,out),max_updates=stop)
                options=self.options(source,out); options.resume=str(out/'checkpoints/latest/codec.pt')
                options.xyz_latent_weight=.5
                with self.assertRaisesRegex(ValueError,'settings|objective'): main(options)
                options.xyz_latent_weight=1.; main(options)
                expected=load_checkpoint(root/'full/checkpoints/final/codec.pt')
                actual=load_checkpoint(out/'checkpoints/final/codec.pt')
                for name,value in expected.state_dict().items():
                    torch.testing.assert_close(value,actual.state_dict()[name],atol=0,rtol=0)
                rows=[json.loads(s) for s in (out/'metrics/loss.jsonl').read_text().splitlines()]
                self.assertEqual([r['step'] for r in rows],list(range(1,9)))

    def test_metadata_only_random_start_and_explicit_semantic_fork(self):
        from gaussian_jscc.xyz_payload_experiment import main
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras',return_value=cameras()):
            root=Path(t); source=source_fixture(root); out=root/'semantic'
            options=self.options(source,out); options.channel_bootstrap_steps=0; options.channel_render_steps=0
            main(options)
            record=json.loads((out/'training.json').read_text()); self.assertTrue(record['random_initialization'])
            options=self.options(source,root/'fork'); options.init_semantic=str(out/'checkpoints/end_semantic_render/codec.pt')
            main(options)
            record=json.loads((root/'fork/training.json').read_text())
            self.assertEqual(record['phase_budgets']['semantic_bootstrap'],0)
            self.assertEqual(record['phase_budgets']['semantic_render'],0)

    def test_settings_rejected_before_loading_source_and_shell_requires_gpu(self):
        import os
        import shutil
        import subprocess
        from gaussian_jscc.xyz_payload_experiment import main,parser
        options=parser().parse_args(['--run','missing','--out','missing-new-output'])
        self.assertEqual((options.semantic_bootstrap_steps,options.semantic_render_steps,
                          options.channel_bootstrap_steps,options.channel_render_steps),(4000,1000,4000,2000))
        self.assertFalse(options.joint_semantic_encoder)
        options.channel_bootstrap_steps=-1
        with self.assertRaisesRegex(ValueError,'budgets'): main(options)
        shell=shutil.which('bash') if os.name!='nt' else 'C:/softwares/Git/bin/bash.exe'
        if not shell or not Path(shell).exists(): self.skipTest('bash unavailable')
        script=Path(__file__).resolve().parents[1]/'scripts/test_xyz_payload_truck.sh'
        result=subprocess.run([shell,'-n',str(script)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)
        environment=dict(os.environ); environment.pop('CUDA_VISIBLE_DEVICES',None)
        result=subprocess.run([shell,str(script),'source','out'],env=environment,capture_output=True,text=True,encoding='utf-8')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Set CUDA_VISIBLE_DEVICES',result.stderr)


if __name__=='__main__': unittest.main()
