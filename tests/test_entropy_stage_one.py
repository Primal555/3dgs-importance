import contextlib
import io
import json
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc import entropy_stage_one as entry
from gaussian_jscc.entropy_evaluation import calibrate_maps
import test_budgeted_allocation as fixtures
from test_budgeted_allocation import cameras, render


class EntropySelectionTests(unittest.TestCase):
    def test_partial_final_block_cache_signature_preserves_batch_partition(self):
        groups=[torch.arange(512).reshape(2,256),torch.arange(512,529).reshape(1,17)]
        identity=entry.block_signature(groups)
        self.assertEqual(identity,entry.block_signature([g.clone() for g in groups]))
        regrouped=[groups[0][:1],groups[0][1:],groups[1]]
        self.assertNotEqual(identity,entry.block_signature(regrouped))
        groups[-1][0,0]+=1
        self.assertNotEqual(identity,entry.block_signature(groups))

    def test_calibration_deduplicates_maps_and_uses_actual_rd(self):
        with tempfile.TemporaryDirectory() as directory:
            args = entry.parser().parse_args(['--run','unused','--out',directory,'--scene','truck',
                '--etas','.1','.11','1','10','--no-lpips','--calibration-views','1'])
            model = SimpleNamespace(cfg=SimpleNamespace(rates=(0,8,24,48)))
            scene = SimpleNamespace(raw=torch.zeros(3,1),cameras=[SimpleNamespace(image_name='cal')])
            calls=[]
            class Meter:
                def details(self,q): return dict(total_channel_uses=int(torch.tensor([0,8,24,48])[q].sum()))
            def evaluate(model,scene,q,views,args,**kwargs):
                self.assertEqual([v.image_name for v in views],['cal'])
                calls.append(q.tolist())
                mse={1:.4,2:.102,3:.1}[int(q[0])]
                return dict(source_objective_mse=mse,source_structure_error=.2,
                    source_distortion=mse+args.allocation_image_objective.lambda_s*.2,
                    source_psnr=20.,photo_psnr=19.)
            with patch('gaussian_jscc.entropy_evaluation.evaluate_map',side_effect=evaluate):
                selected,record,reference = calibrate_maps(model,scene,torch.tensor([20.,20.,20.]),args,
                    Meter(),Path(directory))
            self.assertEqual(selected.tolist(),[2,2,2])
            self.assertEqual(record['selected_eta'],1.)
            self.assertEqual(len(calls),3)  # q3 reference, q1, q2; duplicate eta maps reused.
            self.assertEqual(len(record['candidates']),4)
            self.assertEqual(reference['c_ref'],144)


class EntropyEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.BudgetedEntryTests.setUpClass()
        cls.source=fixtures.BudgetedEntryTests.source;cls.root=fixtures.BudgetedEntryTests.root

    @classmethod
    def tearDownClass(cls): fixtures.BudgetedEntryTests.tearDownClass()

    def options(self,out):
        saved=torch.load(self.source/'checkpoints/final/codec.pt',weights_only=True)
        dim=saved['config']['latent_dim']
        return entry.parser().parse_args(['--run',str(self.source),'--out',str(out),'--scene','truck',
            '--epochs','2','--batch-size','5','--groups',str(dim//2),str(dim-dim//2),
            '--hidden','12','--calibration-views','1','--test-views','2','--trials','1',
            '--image-views','0','--eta-count','3','--device','cpu','--cpu-threads','1','--no-lpips'])

    def test_real_codec_path_frozen_accounted_resumable_and_small_review(self):
        source_path=self.source/'checkpoints/final/codec.pt';before=source_path.read_bytes()
        out=self.root/'entropy';args=self.options(out)
        with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
             patch('gaussian_jscc.rendering.render',side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            pending=entry.main(args,max_epochs=1)
            self.assertEqual(pending['status'],'density_incomplete')
            self.assertFalse((out/'complete.json').exists())
            args.resume=True;result=entry.main(args)
            resumed=entry.main(args)
            self.assertEqual(result,resumed)
        self.assertEqual(before,source_path.read_bytes())
        self.assertTrue(result['frozen_codec_verified'])
        summary=json.loads((out/'comparison/summary.json').read_text())
        self.assertEqual(set(summary['test']),{'selected','uniform_q1','uniform_q2','uniform_q3'})
        for row in summary['test'].values():
            self.assertEqual(row['metrics']['tier_counts'][0],0)
            cost=row['cost']
            self.assertEqual(cost['total_channel_uses'],sum(cost[k] for k in
                ('payload_complex_symbols','coordinate_channel_uses','metadata_channel_uses','scale_channel_uses')))
        with tarfile.open(out/'review.tar.gz') as archive:
            names=archive.getnames()
            self.assertIn('comparison/summary.json',names)
            self.assertFalse(any(n.startswith(('cache/','checkpoints/')) for n in names))
        changed=self.options(out);changed.resume=True;changed.bin_width=.5
        with self.assertRaisesRegex(ValueError,'settings|identity'):
            entry.main(changed)


if __name__=='__main__':unittest.main()
