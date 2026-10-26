import contextlib
import io
import json
import unittest
from unittest.mock import patch
import torch
import test_budgeted_allocation as fixture


class EvidenceEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.BudgetedEntryTests.setUpClass()
        cls.root, cls.source = fixture.BudgetedEntryTests.root, fixture.BudgetedEntryTests.source

    @classmethod
    def tearDownClass(cls): fixture.BudgetedEntryTests.tearDownClass()

    def options(self, out):
        from gaussian_jscc.evidence_allocation import parser
        return parser().parse_args(['--run',str(self.source),'--scene','truck','--out',str(out),'--beta','.1',
            '--allocation-steps','2','--window-views','4','--calibration-views','1','--control-every','1',
            '--validate-every','1','--save-every','1','--profile-every','0','--test-views','2',
            '--trials','1','--image-views','1','--device','cpu','--cpu-threads','1','--no-lpips'])

    def test_exact_resume_includes_memory_finite_coverage_and_frozen_codec(self):
        from gaussian_jscc.evidence_allocation import main
        with patch('gaussian_jscc.rendering.load_cameras',side_effect=fixture.cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=fixture.render), contextlib.redirect_stdout(io.StringIO()):
            out = self.root/'evidence_full'; main(self.options(out))
            args = self.options(self.root/'evidence_resume'); main(args,max_updates=1)
            args.resume=True; main(args)
        full=torch.load(out/'checkpoints/latest/state.pt',weights_only=True)
        resumed=torch.load(self.root/'evidence_resume/checkpoints/latest/state.pt',weights_only=True)
        self.assertEqual(full['family'],'evidence_allocation_v1')
        for key,value in full['table'].items(): torch.testing.assert_close(value,resumed['table'][key],rtol=0,atol=0)
        for key in ('mean','weight','observations','finite_counts','context'):
            torch.testing.assert_close(full['blocks']['evidence'][key],resumed['blocks']['evidence'][key],rtol=0,atol=0)
        self.assertEqual(full['objective_reference'],resumed['objective_reference'])
        self.assertTrue((out/'comparison/evidence.json').exists())
        self.assertTrue((out/'comparison/uniform_controls.csv').exists())
        rows=[json.loads(x) for x in (out/'metrics/loss.jsonl').read_text().splitlines()]
        self.assertTrue(all(x['finite_corrections']>0 for x in rows))
        self.assertTrue(all('candidate_evaluations_per_second' in x for x in rows))
        self.assertTrue(all(x['codec_updated'] is False for x in rows))
        # An incompatible memory setting must not mutate exact checkpoints.
        from gaussian_jscc.evidence_allocation import main
        args.evidence_memory=False
        with patch('gaussian_jscc.rendering.load_cameras',side_effect=fixture.cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=fixture.render), self.assertRaisesRegex(ValueError,'mismatch'):
            main(args)

    def test_independent_switches_and_dry_run_reject_non_rd_mode(self):
        from gaussian_jscc.evidence_allocation import main
        args=self.options(self.root/'evidence_dry'); args.dry_run=True
        args.finite_feedback=False; args.evidence_memory=False; args.candidate_policy='round_robin'
        with contextlib.redirect_stdout(io.StringIO()): plan=main(args)
        self.assertFalse(plan['writes_output']); self.assertFalse((self.root/'evidence_dry').exists())
        self.assertFalse(plan['settings']['finite_feedback'])
        args.beta=None; args.budget_total_uses=10000
        with self.assertRaisesRegex(ValueError,'beta'): main(args)


if __name__=='__main__': unittest.main()
