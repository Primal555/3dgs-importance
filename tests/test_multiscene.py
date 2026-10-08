"""Real codec/Adam/packet integration; synthetic renderer is NOT quality evidence."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.data import write_ply, read_ply, prepare
from gaussian_jscc.multiscene import manifest, shared_statistics, parser, train
from gaussian_jscc.multiscene_experiments import packet_cost, shuffled_tiers, evaluate, parser as eval_parser
from gaussian_jscc.transport import load_checkpoint, model_id, transmit
from gaussian_jscc.route2 import load_mask
from test_dense_prefix import dense
from test_render_first import synthetic_render
import test_render_first as render_fixture


class MultiSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_equal_scene_statistics_not_point_weighted(self):
        a=torch.zeros(2,5)
        b=torch.zeros(20,5)
        b[:,3:]=10
        mean,std=shared_statistics([a,b])
        torch.testing.assert_close(mean,torch.full((2,),5.))
        torch.testing.assert_close(std,torch.full((2,),5.))

    def test_positive_shuffle_preserves_deletion_locations(self):
        q=torch.tensor([0,1,2,3,0,3,1,2]*8)
        for positive in (False,True):
            result=shuffled_tiers(q,9,positive)
            torch.testing.assert_close(torch.bincount(q),torch.bincount(result))
            self.assertFalse(torch.equal(q,result))
            if positive:
                torch.testing.assert_close(q==0,result==0)

    def test_packet_cost_matches_real_transport(self):
        raw,_,_,model=dense(16)
        q=torch.arange(16)%len(model.cfg.rates)
        ordered,g,oq=prepare(raw,model.cfg.morton_bits,q)
        cost=packet_cost(model,g,ordered,oq,10,'none',2.)
        with tempfile.TemporaryDirectory() as temp:
            packet=Path(temp)/'packet'
            stats=transmit(model,raw,q,10,'none',42,packet,code_rate=1.,modulation_bits=2)
            self.assertEqual(cost['total_channel_uses'],stats['total_channel_uses'])
            self.assertEqual(cost['metadata_bytes'],stats['metadata_bytes'])
            self.assertEqual(cost['xyz_bytes'],(packet/'xyz.bin').stat().st_size)

    def test_manifest_rejects_duplicates_and_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'manifest.json'
            scene={'name':'a','role':'train','ply':'a.ply','source':'a'}
            for scenes in ([scene,scene],[dict(scene,role='test')]):
                path.write_text(json.dumps({'scenes':scenes}))
                with self.assertRaises(ValueError):
                    manifest(path)

    def test_shared_training_adaptation_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            raw,_,_,_=dense(16)
            specifications=[]
            originals=[]
            for name,role,shift in [('fixture_a','train',0.),('fixture_b','train',.5),('fixture_holdout','heldout',10.)]:
                points=raw.clone()
                points[:,3:]+=shift
                write_ply(root/f'{name}.ply',points,0)
                originals.append(read_ply(root/f'{name}.ply')[0])
                specifications.append({'name':name,'role':role,'ply':f'{name}.ply','source':name})
            path=root/'manifest.json'
            path.write_text(json.dumps({'scenes':specifications}))
            run=root/'run'
            args=parser().parse_args(['--manifest',str(path),'--out',str(run),'--device','cpu',
                '--bootstrap-steps','4','--render-steps','1','--allocation-steps','1','--joint-steps','1',
                '--hidden','16','--depth','1','--block-size','8','--decoder-window','4',
                '--blocks-per-batch','2','--views-per-step','1','--validation-views','1',
                '--validation-trials','1','--validate-every','4','--save-every','4'])
            cameras=render_fixture.RenderFirstTests().cameras()
            with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                 patch('gaussian_jscc.mask_checks.check_masked_renderer',return_value={'synthetic_test':True}), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=cameras) as loader:
                train(args)
                self.assertTrue(all('fixture_holdout' not in str(c.args[0]) for c in loader.call_args_list))
                a=load_checkpoint(run/'checkpoints/end_render/codec.pt','cpu')
                b=load_checkpoint(run/'checkpoints/end_allocation/codec.pt','cpu')
                self.assertEqual(model_id(a),model_id(b))
                final=load_checkpoint(run/'checkpoints/final/codec.pt','cpu')
                self.assertNotEqual(model_id(final),model_id(b))
                expected_mean,expected_std=shared_statistics(originals[:2])
                torch.testing.assert_close(final.attr_mean,expected_mean)
                torch.testing.assert_close(final.attr_std,expected_std)
                for spec,points in zip(specifications[:2],originals[:2]):
                    load_mask(run/f'checkpoints/final/{spec["name"]}.pt',points,final,'cpu')
                rows=[json.loads(line) for line in (run/'loss.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),14)
                for name in ('fixture_a','fixture_b'):
                    self.assertEqual([r['layout'] for r in rows if r['scene']==name and r['phase']=='bootstrap'],['1','2','3','mixed'])
                adapt=root/'adapt'
                adaptation=parser().parse_args(['--adapt','--manifest',str(path),'--out',str(adapt),'--device','cpu',
                    '--checkpoint',str(run/'checkpoints/final/codec.pt'),'--bootstrap-steps','0','--render-steps','0',
                    '--joint-steps','0','--allocation-steps','1','--validation-views','1','--views-per-step','1',
                    '--validation-trials','1','--blocks-per-batch','2'])
                train(adaptation)
                adapted=load_checkpoint(adapt/'checkpoints/final/codec.pt','cpu')
                self.assertEqual(model_id(final),model_id(adapted))
                load_mask(adapt/'checkpoints/final/fixture_holdout.pt',originals[2],adapted,'cpu')
                control=root/'beta_control'
                control_args=parser().parse_args(['--allocation-only','--manifest',str(path),'--out',str(control),
                    '--device','cpu','--checkpoint',str(run/'checkpoints/end_render/codec.pt'),
                    '--train-scenes','fixture_a','--bootstrap-steps','0','--render-steps','0','--joint-steps','0',
                    '--allocation-steps','1','--beta','0','--validation-views','1','--views-per-step','1',
                    '--validation-trials','1','--blocks-per-batch','2'])
                train(control_args)
                self.assertEqual(model_id(a),model_id(load_checkpoint(control/'checkpoints/final/codec.pt','cpu')))
                self.assertFalse((control/'scenes/fixture_b').exists())
                evaluation=eval_parser().parse_args(['--manifest',str(path),'--checkpoint',str(adapt/'checkpoints/final/codec.pt'),
                    '--role','heldout','--device','cpu','--out',str(root/'evaluation'),'--trials','1',
                    '--shuffle-seeds','101','--test-views','1','--snrs','0','10'])
                evaluate(evaluation)
                results=[json.loads(line) for line in (root/'evaluation/results.jsonl').read_text().splitlines()]
                self.assertEqual(len(results),12)
                learned=next(r for r in results if r['layout']=='mask')
                for r in results:
                    self.assertGreater(r['total_channel_uses'],r['payload_complex_symbols'])
                    self.assertEqual(r['codec_id'],model_id(final))
                    if r['layout'].startswith('shuffle'):
                        self.assertEqual(r['tier_counts'],learned['tier_counts'])
                self.assertTrue(list((root/'evaluation/charts').glob('*.png')))
                self.assertTrue((root/'evaluation/charts/fixture_holdout_snr_sweep.png').exists())
                self.assertTrue(any(r['digital_efficiency_exceeds_same_snr_capacity'] for r in results))
                self.assertTrue(list((run/'charts').glob('*.pdf')))
                artifact=os.environ.get('MULTISCENE_TEST_ARTIFACTS')
                if artifact:
                    shutil.copytree(root,artifact,dirs_exist_ok=False)


if __name__=='__main__':
    unittest.main()
