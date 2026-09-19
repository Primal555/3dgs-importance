"""Phase-local validation LR, logged rates and early-stop interaction."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.optimization import ValidationLRSchedule
import test_render_first as render_fixture
from test_render_first import synthetic_render
from test_learned_joint import setup


class ValidationLRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def optimizer(self):
        return torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.))],lr=1e-4)

    def test_three_bad_checks_halve_lr_floor_and_improvement_reset(self):
        optimizer=self.optimizer()
        scheduler=ValidationLRSchedule(optimizer,min_lr=2.5e-5)
        self.assertFalse(scheduler.observe(1.)['reduced'])
        self.assertFalse(scheduler.observe(1.)['reduced'])
        self.assertFalse(scheduler.observe(.9)['reduced'])
        self.assertFalse(scheduler.observe(.9)['reduced'])
        self.assertFalse(scheduler.observe(.9)['reduced'])
        self.assertEqual(scheduler.observe(.9)['lr_after'],[5e-5])
        for _ in range(3):
            event=scheduler.observe(.9)
        self.assertEqual(event['lr_after'],[2.5e-5])
        for _ in range(6):
            self.assertFalse(scheduler.observe(.9)['reduced'])

    def test_constant_and_nonfinite_guard(self):
        scheduler=ValidationLRSchedule(self.optimizer(),mode='constant')
        for score in (1.,2.,3.,4.,5.):
            self.assertEqual(scheduler.observe(score)['lr_after'],[1e-4])
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            scheduler.observe(float('nan'))

    def test_render_cli_logs_decay_and_gives_reduced_lr_time(self):
        from gaussian_jscc.cli import main
        from gaussian_jscc.data import write_ply
        raw,_,_,_=setup(16)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--source','mock','--device','cuda','--render-steps','12','--validation-views','1',
                  '--validate-every','1','--hidden','16','--depth','1','--grid-dim','4','--levels','2',
                  '--block-size','8','--decoder-window','4','--lr-patience','2','--patience','2',
                  '--min-lr','0.000025']
            with patch('sys.argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()*2), \
                 patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                 patch('gaussian_jscc.learned_train.validate_render',return_value={'score':1.}), \
                 patch('gaussian_jscc.plots.safe_plot'):
                main()
            rows=[json.loads(s) for s in (root/'run'/'loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['lr'] for r in rows],[1e-4,1e-4,5e-5,5e-5,2.5e-5,2.5e-5])
            self.assertTrue(all(r['render_backward']=='direct' for r in rows))
            events=[json.loads(s) for s in (root/'run'/'lr_schedule.jsonl').read_text().splitlines()]
            self.assertEqual([e['step'] for e in events if e['reduced']],[2,4])
            self.assertTrue(events[0]['baseline'])


if __name__=='__main__':
    unittest.main()
