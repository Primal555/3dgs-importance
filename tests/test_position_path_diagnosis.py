import unittest
import torch
from diagnose_position_path import decompose, aggregate


class PositionPathTests(unittest.TestCase):
    xyz_decoder='additive'
    decoder_attention='window'
    def test_translation_and_exact_energy_decomposition(self):
        torch.manual_seed(42)
        target=torch.randn(32,3,dtype=torch.float64)
        result=decompose(target+torch.tensor([2.,3.,4.]),target,torch.ones(32))
        self.assertAlmostEqual(result['common_fraction'],1.)
        self.assertLess(result['centered_rmse'],1e-12)
        self.assertAlmostEqual(result['sse'],result['common_sse']+result['relative_sse'])

    def test_affine_oracle_not_confused_with_centering(self):
        target=torch.randn(32,3,dtype=torch.float64)
        result=decompose(target*2+3,target,torch.ones(32))
        self.assertGreater(result['centered_rmse'],.1)
        self.assertLess(result['affine_oracle_rmse'],1e-10)
        self.assertAlmostEqual(result['centered_spread_ratio'],2.)

    def test_collapse_and_point_weighted_aggregation(self):
        target=torch.randn(32,3,dtype=torch.float64)
        result=decompose(torch.zeros_like(target),target,torch.ones(32))
        self.assertGreater(result['affine_oracle_rmse'],0)
        self.assertEqual(result['centered_spread_ratio'],0)
        both=aggregate([result,result])
        self.assertEqual(both['points'],64)
        self.assertAlmostEqual(both['rmse'],result['xyz_rmse'])

    def test_readonly_real_codec_report(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from diagnose_position_path import run
        from test_point_attention import setup
        from gaussian_jscc.codec import GaussianCodec, CodecConfig
        from gaussian_jscc.data import write_ply
        from gaussian_jscc.transport import save_checkpoint
        from gaussian_jscc.checkpoint_history import file_hash
        torch.set_num_threads(1)
        raw,_,_,old=setup(17)
        config=old.cfg.to_dict();config['rates']=[0,8,16,32];config['block_size']=8
        config['xyz_decoder']=self.xyz_decoder
        config['decoder_attention']=self.decoder_attention
        model=GaussianCodec(CodecConfig.from_dict(config))
        model.attr_mean.copy_(old.attr_mean);model.attr_std.copy_(old.attr_std)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_ply(root/'input.ply',raw,0)
            save_checkpoint(root/'codec.pt',model,10,{'source_gaussians':17,'bootstrap_validation_blocks':[0]})
            digest=file_hash(root/'codec.pt')
            args=SimpleNamespace(checkpoint=str(root/'codec.pt'),ply=str(root/'input.ply'),out=str(root/'report'),
                                 device='cpu',max_blocks=0,blocks_per_batch=2,cpu_threads=1,snr=10.)
            result=run(args)
            if self.decoder_attention=='transformer_trunk':
                self.assertFalse(any(key.startswith('self_only') for key in result['summary']))
                self.assertIn('not applicable',result['self_only_warning'])
            self.assertEqual(result['summary']['full/all']['points'],17)
            self.assertEqual(result['summary']['full/heldout']['points'],8)
            self.assertEqual(result['checkpoint_sha256'],digest)
            self.assertEqual(file_hash(root/'codec.pt'),digest)
            self.assertTrue((root/'report/blocks.csv').is_file())
            with self.assertRaises(FileExistsError):
                run(args)


if __name__=='__main__':
    unittest.main()
