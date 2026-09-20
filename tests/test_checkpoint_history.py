"""Real codec and fixed noisy decoding; CPU renderer stand-in for offline history."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch

from gaussian_jscc.checkpoint_history import build_parser, checkpoint_paths, evaluate_history, file_hash
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import save_checkpoint
from test_learned_joint import setup
from test_render_first import synthetic_render


class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_missing_checkpoints_are_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'codec_500.pt').touch()
            with self.assertRaisesRegex(FileNotFoundError,'codec_1000.pt'):
                checkpoint_paths(tmp,500,1000,500)
            with self.assertRaises(ValueError):
                checkpoint_paths(tmp,500,750,500)

    def test_fixed_noise_images_metrics_and_readonly_weights(self):
        torch.manual_seed(42)
        raw,_,_,model=setup(16)
        cameras=[SimpleNamespace(factor=v,original_image=torch.full((3,8,8),.5),image_name=str(i))
                 for i,v in enumerate((.4,.7,1.3))]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); train=root/'training';train.mkdir()
            ply=root/'input.ply';write_ply(ply,raw,0)
            for step in (500,1000):
                save_checkpoint(train/f'codec_{step}.pt',model,step,{'source_gaussians':len(raw)})
            hashes={p.name:file_hash(p) for p in train.iterdir()}
            args=build_parser().parse_args(['--training',str(train),'--ply',str(ply),'--source','mock',
                                          '--out',str(root/'eval'),'--stop','1000','--views','2',
                                          '--blocks-per-batch','2','--trials','2'])
            def render(*values):
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(values[0].requires_grad)
                return synthetic_render(*values)
            with patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=cameras) as load, \
                 patch('gaussian_jscc.rendering.render',side_effect=render), \
                 patch('torch.optim.Adam',side_effect=AssertionError('no optimization')):
                results=evaluate_history(args)
            self.assertEqual(load.call_count,1)
            self.assertEqual(results[0]['layouts'],results[1]['layouts'])
            self.assertEqual(hashes,{p.name:file_hash(p) for p in train.iterdir()})
            for step in ('000500','001000'):
                for tier in ('1','2','3','mixed'):
                    self.assertTrue((root/'eval/validation_images'/step/f'{tier}_view00.png').exists())
            for name in ('metrics.csv','results.json','quality_vs_step.png','quality_vs_step.svg','evaluation_config.json'):
                self.assertTrue((root/'eval'/name).exists())

    def test_cpu_render_request_rejected(self):
        args=build_parser().parse_args(['--training','missing','--ply','missing','--source','missing',
                                      '--out','missing','--device','cpu'])
        with self.assertRaisesRegex(ValueError,'CUDA'):
            evaluate_history(args)


if __name__=='__main__':
    unittest.main()
