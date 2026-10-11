"""Full experiment-3 plumbing; CPU synthetic images aren't real quality evidence."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.data import write_ply
from gaussian_jscc import multiscene
from gaussian_jscc.multiscene_experiments import evaluate, parser as evaluation_parser
from test_gaussian_jscc import fixture
from test_render_first import synthetic_render
import test_render_first as render_fixture


class ResidualControlTests(unittest.TestCase):
    def test_separate_truck_pair_runs_fair_phases_compact_outputs_and_diagnostics(self):
        try:
            controls = importlib.import_module('gaussian_jscc.residual_controls')
        except ModuleNotFoundError:
            self.fail('missing separate residual-prefix experiment entry')
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw,_ = fixture(12,degree=1)
            write_ply(root/'truck.ply',raw,1)
            (root/'images').mkdir()
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'scenes':[
                {'name':'truck','role':'train','ply':'truck.ply','source':'images'},
                {'name':'train','role':'train','ply':'not_loaded.ply','source':'not_loaded'}]}))
            source = root/'source'
            arguments = multiscene.parser().parse_args(['--manifest',str(manifest),'--out',str(source),
                '--train-scenes','truck','--device','cpu','--bootstrap-steps','1','--render-steps','1',
                '--allocation-steps','0','--joint-steps','0','--hidden','16','--depth','1','--block-size','8',
                '--decoder-window','4','--blocks-per-batch','2','--validation-views','1','--validation-trials','1',
                '--views-per-step','1','--output-profile','compact'])
            args = controls.parser().parse_args(['--run',str(source),'--out',str(root/'controls'),
                '--device','cpu','--bootstrap-steps','4','--render-steps','4','--validate-every','4',
                '--test-views','1','--trials','1','--image-views','1'])
            def dispatch(command, check):
                module = command[command.index('-m')+1]
                options = command[command.index('-m')+2:]
                if module=='gaussian_jscc.multiscene':
                    multiscene.train(multiscene.parser().parse_args(options))
                elif module=='gaussian_jscc.multiscene_experiments':
                    evaluate(evaluation_parser().parse_args(options))
                else:
                    self.fail(module)
            cameras = render_fixture.RenderFirstTests().cameras()
            with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=cameras), \
                 patch('gaussian_jscc.residual_controls.subprocess.run',side_effect=dispatch):
                multiscene.train(arguments)
                controls.main(args)
            out = root/'controls'
            complete = json.loads((out/'complete.json').read_text())
            self.assertTrue(complete['paired_initializer_verified'])
            self.assertFalse(complete['allocator_trained'])
            rows = json.loads((out/'comparison.json').read_text())
            self.assertEqual(len(rows),6)
            self.assertEqual({r['scene'] for r in rows},{'truck'})
            for mode in ('direct','residual'):
                run = out/mode
                training = json.loads((run/'training.json').read_text())
                self.assertEqual(training['decoder_mode'],mode)
                self.assertEqual(training['drop'],0)
                losses = [json.loads(line) for line in (run/'loss.jsonl').read_text().splitlines()]
                self.assertEqual(len(losses),8)
                self.assertEqual({r['layout'] for r in losses},{'1','2','3','mixed'})
                self.assertTrue(all(r['retained_gaussians']==len(raw) for r in losses if r['phase']=='render'))
                images = list((run/'scenes/truck/validation_images').iterdir())
                self.assertEqual({p.name for p in images},{'000000','000008'})
                self.assertTrue((run/'test/prefix_gains.jsonl').is_file())
            diagnostics = json.loads((out/'residual/stage_diagnostics.json').read_text())
            self.assertEqual(diagnostics['source_gaussians'],len(raw))
            self.assertGreater(diagnostics['corrections'][0]['normalized_rms'],0)
            self.assertGreater(diagnostics['corrections'][1]['normalized_rms'],0)
            self.assertTrue((out/'charts/truck_residual_final_test.png').is_file())
            self.assertTrue((out/'charts/truck_residual_prefix_gains.png').is_file())
            # The shell entry refuses unset RUN/GPU, and dry-run doesn't start training.
            bash = Path('C:/softwares/Git/bin/bash.exe')
            if os.name=='nt' and bash.exists():
                env = dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=sys.executable.replace('\\','/'))
                launch = [str(bash),'scripts/test_residual_prefix_truck.sh',str(source),str(root/'dry'), '--dry-run']
                ready = subprocess.run(launch,env=env,capture_output=True,text=True)
                self.assertEqual(ready.returncode,0,ready.stderr)
                self.assertIn('decoder_residual:',ready.stdout)
                self.assertFalse((root/'dry').exists())


if __name__=='__main__':
    unittest.main()
