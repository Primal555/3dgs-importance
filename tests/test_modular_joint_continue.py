"""Continuation must retain Adam/RNG and not rerun representation/channel bootstrap."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_modular_experiment import source_fixture
import test_modular_experiment as experiment_fixture
from test_render_first import synthetic_render
import test_render_first as render_fixture


class JointContinuationTests(unittest.TestCase):
    def test_fork_preserves_optimizer_then_matches_uninterrupted_joint_training(self):
        from gaussian_jscc.modular_experiment import main as train
        from gaussian_jscc.modular_joint_continue import parser, main
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),\
                patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()):
            root=Path(t);source=source_fixture(root);parent=root/'parent';out=root/'continued'
            opts=experiment_fixture.ModularExperimentTests().args(source,parent)
            opts.semantic_bootstrap_steps=1;opts.semantic_render_steps=0
            opts.channel_bootstrap_steps=1;opts.channel_render_steps=2
            opts.image_objective='mse_ssim';opts.semantic_decoder_lr=1e-5;opts.cpu_threads=1
            train(opts)
            checkpoint=parent/'checkpoints/final/codec.pt'
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            original=torch.load(checkpoint,map_location='cpu',weights_only=True)
            whole=copy.deepcopy(opts);whole.out=str(root/'whole');whole.channel_render_steps=4;train(whole)
            args=parser().parse_args(['--run',str(parent),'--out',str(out),'--additional-steps','2',
                '--device','cpu','--cpu-threads','1','--validate-every','1','--save-every','1'])
            main(args,max_updates=1)
            initial=torch.load(out/'checkpoints/initial/codec.pt',weights_only=True)
            for name,value in original['state_dict'].items():
                torch.testing.assert_close(value,initial['state_dict'][name],rtol=0,atol=0)
            before=original['training']['optimizer'];after=initial['training']['optimizer']
            self.assertEqual(before['param_groups'],after['param_groups'])
            for key,state in before['state'].items():
                for field,value in state.items():torch.testing.assert_close(value,after['state'][key][field],rtol=0,atol=0)
            self.assertEqual(initial['training']['progress']['phase_step'],2)
            self.assertFalse(initial['training']['progress']['phase_complete'])
            self.assertEqual(initial['training']['progress']['phase_budgets']['channel_render'],4)
            with self.assertRaises(FileExistsError):main(args)
            args.resume=True;args.dry_run=True
            plan=main(args)
            self.assertEqual(plan['target_step'],6)
            self.assertFalse((out/'checkpoints/final/codec.pt').exists())
            args.dry_run=False;main(args)
            result=torch.load(out/'checkpoints/final/codec.pt',weights_only=True)
            expected=torch.load(root/'whole/checkpoints/final/codec.pt',weights_only=True)
            for name,value in expected['state_dict'].items():
                torch.testing.assert_close(value,result['state_dict'][name],rtol=0,atol=0,msg=name)
            rows=[json.loads(x) for x in (out/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([(r['step'],r['phase_step'],r['layout']) for r in rows],[(5,3,'3'),(6,4,'mixed')])
            self.assertTrue(all(r['phase']=='channel_render' for r in rows))
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),digest)
            record=json.loads((out/'training.json').read_text())
            self.assertTrue(record['continuation']['optimizer_preserved'])
            self.assertEqual(record['continuation']['additional_steps'],2)

    def test_invalid_budget_and_missing_source_fail_before_output_creation(self):
        from gaussian_jscc.modular_joint_continue import parser, main
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);args=parser().parse_args(['--run',str(root/'missing'),'--out',str(root/'out'),'--device','cpu'])
            with self.assertRaises(FileNotFoundError):main(args)
            args.additional_steps=0
            with self.assertRaisesRegex(ValueError,'positive'):main(args)
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
