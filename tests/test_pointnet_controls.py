import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_modular_experiment import source_fixture
from test_render_first import synthetic_render
import test_render_first as render_fixture


class PointControlsTests(unittest.TestCase):
    def test_plan_matches_training_and_only_changes_context(self):
        from gaussian_jscc.pointnet_context_controls import parser,build_plan
        opts=parser().parse_args(['--run','source','--out','new','--device','cpu'])
        jobs=build_plan(opts)['jobs']
        self.assertEqual([j['group'] for j in jobs],['baseline','encoder','decoder','both'])
        from gaussian_jscc.modular_experiment import parser as train_parser
        parsed=[vars(train_parser().parse_args(j['command'][4:])) for j in jobs]
        for d in parsed: d.pop('context_variant'); d.pop('out')
        self.assertTrue(all(d==parsed[0] for d in parsed))
        opts.groups=['both','both']
        with self.assertRaises(ValueError): build_plan(opts)

    def test_aggregate_uses_actual_prefix_gains_and_requires_matched_scenes(self):
        from gaussian_jscc.pointnet_context_controls import summarize
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for name,psnrs in [('baseline',[20.,22.,23.]),('both',[21.,24.,24.5])]:
                p=root/name; (p/'metrics').mkdir(parents=True)
                (p/'training.json').write_text(json.dumps({'scene':{'fingerprint':'same','validation_view_names':['v']},
                    'phase_budgets':{'semantic_render':1},'module_lrs':{'semantic_encoder':.0001},'train_snrs':[10],
                    'seed':42,'parameters':{'semantic_encoder':100},'codec_config':{'context_variant':name}}))
                (p/'complete.json').write_text('{"status":"complete"}')
                (p/'metrics/test.jsonl').write_text('\n'.join(json.dumps({'path':'awgn','layout':str(i+1),'actual_snr':10.,
                    'source_psnr':v,'photo_psnr':v-1,'source_ssim':.8,'photo_ssim':.7,
                    'symbols_per_gaussian':[4,12,24][i],'latent_nmse':.1,'condition_policy':'trained_range'})
                    for i,v in enumerate(psnrs)))
            result=summarize(root,['baseline','both'])
            both=next(r for r in result['enhancement_gains'] if r['group']=='both')
            self.assertEqual((both['q2_minus_q1_db'],both['q3_minus_q2_db']),(3.,.5))
            self.assertTrue((root/'comparison/quality_by_tier.png').exists())
            changed=root/'both/training.json'; record=json.loads(changed.read_text()); record['scene']['fingerprint']='different'
            changed.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'matched'): summarize(root,['baseline','both'])

    def test_spatial_four_phases_resume_and_from_modular_source(self):
        from gaussian_jscc.modular_experiment import parser,main
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()):
            root=Path(t); source=source_fixture(root)
            common=['--run',str(source),'--device','cpu','--context-variant','both','--context-centers','4','2',
                    '--context-neighbors','2','4','--validate-every','2','--save-every','2']
            for phase in ('semantic-bootstrap','semantic-render','channel-bootstrap','channel-render'):
                common += ['--'+phase+'-steps','1']
            a=parser().parse_args(common+['--out',str(root/'a')]); main(a)
            b=parser().parse_args(common+['--out',str(root/'b')]); main(b,max_updates=3)
            b.resume=str(root/'b/checkpoints/latest/codec.pt'); main(b)
            m1=load_checkpoint(root/'a/checkpoints/final/codec.pt','cpu')
            m2=load_checkpoint(root/'b/checkpoints/final/codec.pt','cpu')
            for x,y in zip(m1.parameters(),m2.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
            # A modular run is also valid metadata input; its codec is not loaded.
            c=parser().parse_args(common+['--out',str(root/'c')]); c.run=str(root/'a')
            main(c,max_updates=1)
            self.assertTrue(json.loads((root/'c/training.json').read_text())['random_initialization'])


if __name__=='__main__': unittest.main()
