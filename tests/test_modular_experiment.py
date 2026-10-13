import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_gaussian_jscc import fixture
from test_render_first import synthetic_render
import test_render_first as render_fixture
from gaussian_jscc.data import write_ply
from gaussian_jscc.allocation import scene_fingerprint


def source_fixture(root):
    raw,_=fixture(16,1); write_ply(root/'truck.ply',raw,1)
    (root/'source_images').mkdir()
    manifest=root/'manifest.json'; manifest.write_text(json.dumps({'scenes':[{'name':'truck','role':'train','ply':'truck.ply','source':'source_images','prior':'none'}]}))
    source=root/'source'; (source/'scenes/truck').mkdir(parents=True)
    (source/'training.json').write_text(json.dumps({'manifest':str(manifest),'selected_scenes':[{'name':'truck','role':'train'}],
        'codec_config':{'hidden':16,'depth':2,'grid_dim':4,'levels':[2],'decoder_window':4,'attention_heads':4,'block_size':8},
        'blocks_per_batch':2,'resolution':2,'images':'images','views_per_step':1,'validation_views':1}))
    (source/'scenes/truck/scene.json').write_text(json.dumps({'fingerprint':scene_fingerprint(raw),'source_gaussians':16,
            'train_view_names':['1.3'],'validation_view_names':['0.4']}))
    # Deliberately invalid old checkpoint: random experiments must never read it.
    (source/'codec.pt').write_bytes(b'not a checkpoint')
    return source


class ModularExperimentTests(unittest.TestCase):
    def args(self,source,out):
        from gaussian_jscc.modular_experiment import parser
        return parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu','--validate-every','2','--save-every','2',
            '--semantic-bootstrap-steps','2','--semantic-render-steps','2','--channel-bootstrap-steps','2','--channel-render-steps','2',
            '--train-snrs','0','10','20'])

    def test_four_phase_cpu_experiment_and_log_integrity(self):
        from gaussian_jscc.modular_experiment import main
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()):
            root=Path(t); source=source_fixture(root); out=root/'run'
            main(self.args(source,out))
            done=json.loads((out/'complete.json').read_text()); self.assertEqual(sum(done['completed'].values()),8)
            rows=[json.loads(l) for l in (out/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),8); self.assertEqual(len({r['step'] for r in rows}),8)
            for r in rows:
                self.assertEqual(len(r['updates']),4); self.assertIsNone(r['peak_memory_allocated_bytes'])
                self.assertEqual(r['snr'],r['tx_condition_snr'])
                if r['phase']=='channel_bootstrap':
                    self.assertEqual(r['objective'],'latent_mse')
                    self.assertAlmostEqual(r['loss'],r['latent_mse'],places=6)
                if r['phase']=='channel_render':
                    self.assertAlmostEqual(r['loss'],r['render_loss']+r['weighted_latent_loss'],places=6)
                    self.assertGreater(r['weighted_latent_loss'],0)
            rec=json.loads((out/'training.json').read_text()); self.assertTrue(rec['random_initialization'])
            self.assertEqual(rec['training_objective']['version'],'latent_reconstruction_v2')
            self.assertTrue(rec['reference_scene_identity_and_splits_verified'])
            self.assertEqual(len(list((out/'images/final').glob('*.png'))),5)  # One heldout view in fixture.
            self.assertTrue((out/'checkpoints/end_semantic_render/codec.pt').exists())
            self.assertTrue((out/'metrics/phase_summary.json').exists())
            with self.assertRaises(FileExistsError): main(self.args(source,out))

    def test_exact_midphase_resume_and_zero_stages(self):
        from gaussian_jscc.modular_experiment import main
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()):
            root=Path(t); source=source_fixture(root); a=root/'a'; b=root/'b'
            main(self.args(source,a)); main(self.args(source,b),max_updates=5)
            self.assertFalse((b/'complete.json').exists())
            opts=self.args(source,b); opts.resume=str(b/'checkpoints/latest/codec.pt')
            opts.channel_latent_weight=0.5
            with self.assertRaisesRegex(ValueError,'objective'): main(opts)
            opts.channel_latent_weight=1.
            record_path=b/'training.json'; record=json.loads(record_path.read_text())
            legacy=dict(record); legacy.pop('training_objective')
            record_path.write_text(json.dumps(legacy))
            with self.assertRaisesRegex(ValueError,'objective'): main(opts)
            record_path.write_text(json.dumps(record))
            main(opts)
            ma=load_checkpoint(a/'checkpoints/final/codec.pt','cpu'); mb=load_checkpoint(b/'checkpoints/final/codec.pt','cpu')
            for x,y in zip(ma.parameters(),mb.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
            ra=[json.loads(l) for l in (a/'metrics/loss.jsonl').read_text().splitlines()]
            rb=[json.loads(l) for l in (b/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([(r['layout'],r['snr']) for r in ra],[(r['layout'],r['snr']) for r in rb])
            opts=self.args(source,root/'zero'); opts.semantic_bootstrap_steps=0; opts.channel_bootstrap_steps=0; opts.channel_render_steps=0
            main(opts)
            self.assertFalse(json.loads((root/'zero/complete.json').read_text())['channel_trained'])

    def test_cli_rejects_bad_paths_and_missing_gpu(self):
        from gaussian_jscc.modular_experiment import main
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError): main(self.args(Path(t)/'missing',Path(t)/'out'))
            for weight in (0,-1,float('nan'),float('inf')):
                opts=self.args(Path(t)/'missing',Path(t)/'out'); opts.channel_latent_weight=weight
                with self.assertRaisesRegex(ValueError,'latent weight'): main(opts)
        bash=Path('C:/softwares/Git/bin/bash.exe')
        if bash.exists():
            env=dict(os.environ); env.pop('CUDA_VISIBLE_DEVICES',None)
            proc=subprocess.run([str(bash),'scripts/test_modular_codec_truck.sh','missing','unused'],env=env,text=True,capture_output=True)
            self.assertNotEqual(proc.returncode,0); self.assertIn('CUDA_VISIBLE_DEVICES',proc.stderr)

    def test_phase_boundary_resume_and_semantic_fork(self):
        from gaussian_jscc.modular_experiment import main
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()):
            root=Path(t); source=source_fixture(root); out=root/'boundary'
            main(self.args(source,out),max_updates=4)
            opts=self.args(source,out); opts.resume=str(out/'checkpoints/latest/codec.pt'); main(opts)
            rows=[json.loads(l) for l in (out/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows],list(range(1,9)))
            opts=self.args(source,root/'fork'); opts.init_semantic=str(out/'checkpoints/end_semantic_render/codec.pt')
            main(opts)
            rows=[json.loads(l) for l in (root/'fork/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual({r['phase'] for r in rows},{'channel_bootstrap','channel_render'})
            self.assertEqual(len(rows),4)

if __name__=='__main__': unittest.main()
