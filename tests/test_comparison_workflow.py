"""Real CPU LDPC/file pipeline; replace only the unavailable CUDA renderer/preflight."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from comparison_baselines import run
from comparison_baselines.evaluate import evaluate_plys
from gaussian_jscc.data import write_ply


@unittest.skipUnless(importlib.util.find_spec('sionna'), 'requires the independent Sionna dependency')
class RealDigitalWorkflowTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def fixture(self,root,extra=()):
        source=root/'photos';source.mkdir()
        raw=torch.zeros(2,14);raw[:,7]=1.;raw[:,11:14]=.2
        ply=root/'source.ply';write_ply(ply,raw,0)
        args=run.parser().parse_args(['--source',str(source),'--ply',str(ply),'--out',str(root/'out'),
            '--methods','maskgaussian','--snrs','-20','30','30.000001','--trials','2',
            '--device','cpu','--link-device','cpu','--no-lpips','--views','2','--image-views','0',*extra])
        return args,run.build_plan(args)

    def executor(self):
        counts={'transmit':0,'evaluate':0}
        cameras=[SimpleNamespace(image_name=str(i),original_image=torch.full((3,16,16),.1+.1*i),
            world_view_transform=torch.eye(4),full_proj_transform=torch.eye(4),camera_center=torch.zeros(3),
            FoVx=1.,FoVy=1.) for i in range(2)]
        def renderer(raw,camera,degree,white_background=False):
            return raw[0,11].expand(3,16,16)
        def execute(python,module,arguments,log):
            arguments=list(map(str,arguments))
            if module.endswith('.preflight'):return
            if module.endswith('.digital_link'):
                from comparison_baselines.digital_link import main
                if arguments[0]=='transmit':counts['transmit']+=1
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    code=main(arguments)
                if code:raise RuntimeError('real LDPC preflight failed')
            elif module.endswith('.evaluate'):
                counts['evaluate']+=1
                def arg(key):return arguments[arguments.index('--'+key)+1]
                with patch('gaussian_jscc.rendering.load_cameras',return_value=cameras):
                    evaluate_plys(arg('source'),arg('source-ply'),arg('received-ply'),arg('out'),
                        resolution=int(arg('resolution')),images=arg('images'),views=int(arg('views')),
                        image_views=int(arg('image-views')),lpips=False,device='cpu',renderer=renderer)
            else:raise AssertionError(module)
        return execute,counts

    def test_real_fec_pipeline_failure_success_cache_costs_and_compact_review(self):
        with tempfile.TemporaryDirectory() as directory:
            args,plan=self.fixture(Path(directory));executor,counts=self.executor()
            with patch.object(run,'_execute',side_effect=executor),contextlib.redirect_stdout(io.StringIO()):
                run.execute(args,plan)
                summary=json.loads((Path(args.out)/'summary.json').read_text(encoding='utf8'))
                self.assertEqual(len(summary['trials']),6)
                self.assertEqual([r['success_rate'] for r in summary['results']],[0.,1.,1.])
                self.assertIsNone(summary['results'][0]['metrics'])
                self.assertEqual(counts['transmit'],6)
                self.assertEqual(counts['evaluate'],1)
                self.assertTrue(all(r['total_channel_uses']%1536==0 for r in summary['trials']))
                self.assertTrue((Path(args.out)/'results/maskgaussian/snr_30.000001/trial_000/result.json').is_file())
                args.resume=True
                run.execute(args,plan)
                self.assertEqual(counts['transmit'],6)
            with tarfile.open(Path(args.out)/'review.tar.gz') as archive:
                names=archive.getnames()
                self.assertIn('comparison.csv',names)
                self.assertFalse(any(name.endswith(('.ply','.zlib','.zip')) for name in names))

    def test_resume_after_metrics_and_intermediate_write_before_trial_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            args,plan=self.fixture(Path(directory),['--keep-intermediates','--snrs','30','--trials','1'])
            executor,counts=self.executor();original=run._json
            def crash_once(path,record):
                if Path(path).name=='result.json':raise RuntimeError('simulated trial commit interruption')
                return original(path,record)
            with patch.object(run,'_execute',side_effect=executor),patch.object(run,'_json',side_effect=crash_once),\
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'simulated trial'):
                    run.execute(args,plan)
            self.assertEqual(counts['evaluate'],1)
            args.resume=True
            with patch.object(run,'_execute',side_effect=executor),contextlib.redirect_stdout(io.StringIO()):
                run.execute(args,plan)
            self.assertEqual(counts['evaluate'],1)
            self.assertTrue((Path(args.out)/'complete.json').is_file())
            self.assertTrue((Path(args.out)/'received/maskgaussian/snr_30.0/trial_000/received.ply').is_file())


if __name__=='__main__':unittest.main()
