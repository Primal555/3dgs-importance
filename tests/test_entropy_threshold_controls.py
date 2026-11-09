import contextlib
import io
import json
import tarfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import test_budgeted_allocation as fixtures
from test_budgeted_allocation import cameras, render
from gaussian_jscc import entropy_stage_one as stage
from gaussian_jscc import entropy_threshold_controls as entry


class ThresholdEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.BudgetedEntryTests.setUpClass()
        cls.root = fixtures.BudgetedEntryTests.root
        cls.codec = fixtures.BudgetedEntryTests.source
        cls.source = cls.root/'entropy_threshold_source'
        cfg = torch.load(cls.codec/'checkpoints/final/codec.pt',weights_only=True)['config']
        dim = cfg['latent_dim']
        args = stage.parser().parse_args(['--run',str(cls.codec),'--out',str(cls.source),
            '--scene','truck','--groups',str(dim),'--hidden','12','--epochs','1',
            '--batch-size','8','--eta-count','3','--calibration-views','1','--test-views','2',
            '--trials','1','--image-views','0','--device','cpu','--cpu-threads','1','--no-lpips'])
        with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            stage.main(args)

    @classmethod
    def tearDownClass(cls): fixtures.BudgetedEntryTests.tearDownClass()

    def options(self,out):
        return entry.parser().parse_args(['--source',str(self.source),'--out',str(out),
            '--quantiles','0','.25','.5','.75','1','--image-views','0','--device','cpu','--cpu-threads','1'])

    def test_frozen_shared_measurements_and_exact_resume(self):
        out = self.root/'entropy_threshold_output'; args = self.options(out)
        score_path = self.source/'cache/information_scores.pt'; before = score_path.read_bytes()
        source_files = sorted(str(p.relative_to(self.source)) for p in self.source.rglob('*'))
        with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            partial = entry.main(args,max_candidates=4)
            self.assertEqual(partial['status'],'calibration_incomplete')
            self.assertFalse((out/'selection.json').exists())
            args.resume = True; result = entry.main(args)
            # A completed resume must consume the saved measurements without rendering.
            with patch('gaussian_jscc.entropy_threshold_controls.evaluate_map',side_effect=AssertionError('rerender')):
                again = entry.main(args)
        self.assertEqual(result,again)
        self.assertEqual(before,score_path.read_bytes())
        self.assertEqual(source_files,sorted(str(p.relative_to(self.source)) for p in self.source.rglob('*')))
        self.assertTrue(result['frozen_codec_verified']); self.assertEqual(result['optimizer_updates'],0)
        summary = json.loads((out/'comparison/summary.json').read_text())
        self.assertEqual(set(summary['selections']),{'beta_0p1','beta_0p5','q2_budget'})
        budget = summary['uniform']['uniform_q2']['cost']['total_channel_uses']
        self.assertLessEqual(summary['selections']['q2_budget']['cost']['total_channel_uses'],budget)
        oldref = json.loads((self.source/'objective_reference.json').read_text())
        for key in ('d_ref','c_ref','lambda_s'): self.assertEqual(summary['reference'][key],oldref[key])
        selection = json.loads((out/'selection.json').read_text())
        rows = selection['candidates']; self.assertEqual(len(rows),len({r['tier_sha256'] for r in rows}))
        self.assertGreater(sum(r['measurement_origin']=='source_cache' for r in rows),2)
        for row in summary['selections'].values():
            self.assertEqual(row['metrics']['tier_counts'][0],0)
            c = row['cost']
            self.assertEqual(c['total_channel_uses'],sum(c[k] for k in
                ('payload_complex_symbols','coordinate_channel_uses','metadata_channel_uses','scale_channel_uses')))
        with tarfile.open(out/'review.tar.gz') as archive:
            self.assertIn('comparison/summary.json',archive.getnames())
            self.assertFalse(any(n.endswith('.pt') for n in archive.getnames()))
        args.betas=[.2]
        with self.assertRaisesRegex(ValueError,'settings|identity'): entry.main(args)

    def test_reject_changed_scores_before_writing_output(self):
        path=self.source/'cache/information_scores.pt'; original=path.read_bytes()
        try:
            scores=torch.load(path,weights_only=True); scores[0]+=1; torch.save(scores,path)
            with self.assertRaisesRegex(ValueError,'score'):
                entry.main(self.options(self.root/'bad_entropy_scores'))
            self.assertFalse((self.root/'bad_entropy_scores').exists())
        finally: path.write_bytes(original)


if __name__=='__main__':unittest.main()
