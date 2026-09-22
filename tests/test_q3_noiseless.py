"""Fixed full payload, identity channel, unchanged bootstrap and q3-only history."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from gaussian_jscc.cli import main
from gaussian_jscc.codec import channel, prefix_mask
from gaussian_jscc.data import write_ply
from gaussian_jscc.learned_training import hard_layout
from gaussian_jscc.transport import load_checkpoint
from gaussian_jscc.checkpoint_history import build_parser, evaluate_history
from test_point_attention import setup
from test_render_first import synthetic_render


class Q3NoiselessTests(unittest.TestCase):
    xyz_decoder='additive'
    decoder_attention='window'
    decoder_memory='none'
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_payload_identity_and_padding(self):
        q=hard_layout(torch.tensor([[0,1,2,-1]]),3,0.)
        self.assertEqual(q.tolist(),[[3,3,3,0]])
        mask=prefix_mask(q.flatten(),(0,8,16,32))
        self.assertTrue(mask[:3].all());self.assertFalse(mask[3].any())
        x=torch.randn(3,64,requires_grad=True)
        rng=torch.random.get_rng_state().clone()
        y=channel(x,10,'none')
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        y.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))

    def test_training_and_history_only_use_q3(self):
        raw,_,_,_=setup(17)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ply=root/'input.ply';write_ply(ply,raw,0)
            argv=['gaussian_jscc','train-learned','--architecture','learned_split_logcov',
                  '--context-mode','multiscale_self','--encoder-attention','geometric_point',
                  '--xyz-decoder',self.xyz_decoder,
                  '--decoder-attention',self.decoder_attention,
                  '--decoder-memory',self.decoder_memory,
                  '--bootstrap-tier','3','--channel','none','--ply',str(ply),'--out',str(root/'run'),
                  '--device','cpu','--bootstrap-objective','spatial-response','--bootstrap-steps','5',
                  '--render-steps','0','--joint-steps','0','--hidden','16','--depth','1',
                  '--block-size','8','--decoder-window','4','--blocks-per-batch','2',
                  '--validation-blocks','1','--validation-trials','1','--validate-every','1','--save-every','5']
            def layout(ids,tier,drop):
                self.assertEqual(tier,3);self.assertEqual(drop,0)
                return hard_layout(ids,tier,drop)
            with patch('sys.argv',argv),patch('gaussian_jscc.learned_train.hard_layout',side_effect=layout),patch('gaussian_jscc.plots.safe_plot'):
                main()
            model=load_checkpoint(root/'run/codec.pt','cpu')
            self.assertEqual(model.cfg.rates,(0,8,16,32))
            self.assertEqual(model.cfg.encoder_attention,'geometric_point')
            self.assertEqual(model.cfg.xyz_decoder,self.xyz_decoder)
            self.assertEqual(model.cfg.decoder_attention,self.decoder_attention)
            self.assertEqual(model.cfg.decoder_memory,self.decoder_memory)
            if self.decoder_attention=='transformer_trunk':
                record=json.loads((root/'run/training.json').read_text())
                self.assertEqual(record['decoder_xyz_taps'],[1,2,4])
                self.assertEqual(record['codec_config']['decoder_depth'],4)
                self.assertIn('no receiver Context gate',record['context_gate_initialization'])
            rows=[json.loads(s) for s in (root/'run/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),5)
            for row in rows:
                self.assertEqual(row['layout'],'3')
                self.assertEqual(row['tier_counts'][:3],[0,0,0])
                self.assertEqual(row['symbols_per_source_gaussian'],32)
                self.assertEqual(row['objective'],'spatial_logcov_v1')
            val=[json.loads(s) for s in (root/'run/bootstrap_validation.jsonl').read_text().splitlines()]
            self.assertTrue(all([v['layout'] for v in r['layouts']]==['3'] and r['channel']=='none' for r in val))
            self.assertTrue(all(v['layouts'][0]['block_xyz_common_mse']>=0 and
                                v['layouts'][0]['block_xyz_relative_mse']>=0 for v in val))
            args=build_parser().parse_args(['--training',str(root/'run'),'--ply',str(ply),'--source','mock',
                                          '--start','5','--stop','5','--every','5','--tier','3',
                                          '--channel','none','--trials','1','--views','1'])
            camera=SimpleNamespace(factor=.4,original_image=torch.full((3,8,8),.5),image_name='view')
            with patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=[camera]), \
                 patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
                result=evaluate_history(args)[0]
            self.assertEqual([e['layout'] for e in result['layouts']],['3'])
            self.assertEqual(result['score'],result['layouts'][0]['source_mse'])
            self.assertEqual(result['layouts'][0]['symbols_per_source_gaussian'],32)
            panels=list((root/'run/render_history/validation_images/000005').glob('*.png'))
            self.assertEqual([p.name for p in panels],['3_view00.png'])
            bad=argv+['--drop','0.1']
            with patch('sys.argv',bad),self.assertRaisesRegex(ValueError,'drop=0'):
                main()
            if self.decoder_attention!='window':
                bad=argv+['--init',str(root/'run/codec.pt'),'--decoder-attention','window','--out',str(root/'mismatch')]
                with patch('sys.argv',bad),self.assertRaisesRegex(ValueError,'decoder attention mismatch'):
                    main()
            if self.decoder_memory!='none':
                bad=argv+['--init',str(root/'run/codec.pt'),'--decoder-memory','none','--out',str(root/'memory_mismatch')]
                with patch('sys.argv',bad),self.assertRaisesRegex(ValueError,'decoder memory mismatch'):
                    main()


if __name__=='__main__':
    unittest.main()
