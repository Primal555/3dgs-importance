import unittest
from unittest.mock import patch
import torch
import test_transformer_trunk as trunk_tests
import test_multiscale_codec as contracts
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.optimization import parameter_group


def setup(n=17):
    raw,g,f,old=trunk_tests.setup(n)
    cfg=old.cfg.to_dict();cfg['decoder_localization']='token_translation'
    model=GaussianCodec(CodecConfig.from_dict(cfg))
    model.attr_mean.copy_(old.attr_mean);model.attr_std.copy_(old.attr_std)
    return raw,g,f,model


class LocalizationCLI(trunk_tests.TransformerTrunkCLI):
    decoder_localization='token_translation'


class LocalizationDiagnosis(trunk_tests.TransformerTrunkDiagnosis):
    decoder_localization='token_translation'


class LocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_exact_equality_and_config(self):
        _,_,f,old=trunk_tests.setup()
        cfg=old.cfg.to_dict()
        torch.manual_seed(42);a=GaussianCodec(CodecConfig.from_dict(cfg))
        cfg['decoder_localization']='token_translation'
        torch.manual_seed(42);b=GaussianCodec(CodecConfig.from_dict(cfg))
        for name,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[name],atol=0,rtol=0)
        q=torch.arange(len(f))%4
        pa,pb=a(f,f[:,:3],q,10,'none'),b(f,f[:,:3],q,10,'none')
        torch.testing.assert_close(pa,pb,atol=0,rtol=0)
        pa.square().sum().backward();pb.square().sum().backward()
        bp=dict(b.named_parameters())
        for name,p in a.named_parameters():
            torch.testing.assert_close(p.grad,bp[name].grad,atol=0,rtol=0)
        loc=b.learned.dec_trunk.localization
        self.assertGreater(float(loc.translation.weight.grad.norm()),0)
        self.assertEqual(float(loc.query.grad.norm()),0)
        self.assertNotIn('decoder_localization',a.cfg.to_dict())
        self.assertEqual(b.cfg.to_dict()['decoder_localization'],'token_translation')
        with self.assertRaises(ValueError):
            CodecConfig(decoder_localization='token_translation')

    def test_translation_preserves_relative_positions_and_attributes(self):
        _,_,_,m=setup()
        q=torch.full((2,17),3,dtype=torch.long);q[0,3]=0;q[1]=0
        z=torch.randn(2,17,2*m.cfg.rates[-1])
        old=m.learned.decode(z,q,10)
        loc=m.learned.dec_trunk.localization
        with torch.no_grad():
            loc.translation.weight.normal_(0,.01)
            loc.translation.bias.copy_(torch.tensor([.1,-.2,.3]))
        new=m.learned.decode(z,q,10)
        delta=(new-old)[0,q[0]>0,:3]
        torch.testing.assert_close(delta,delta[:1].expand_as(delta),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(new[...,3:],old[...,3:],atol=0,rtol=0)
        self.assertEqual(float(new[q==0].detach().abs().sum()),0)
        perm=torch.randperm(17)
        torch.testing.assert_close(m.learned.decode(z[:,perm],q[:,perm],10),new[:,perm],atol=2e-6,rtol=2e-5)
        z.requires_grad_();m.learned.decode(z,q,10)[0,0,:3].sum().backward()
        for layer in loc.attention:
            self.assertGreater(float(layer.in_proj_weight.grad.norm()),0)
        self.assertEqual(float(z.grad[q==0].abs().sum()),0)
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertIn('block_localization',{parameter_group(n) for n,_ in m.named_parameters()})

    def test_contracts_and_actual_fit(self):
        for name in ('test_transport_power_drop_and_replay_contracts','test_short_fit_and_all_backward_modes'):
            with self.subTest(name=name),patch('test_multiscale_codec.setup',setup):
                getattr(contracts.MultiScaleTests(name),name)()

    def test_larger_runner_end_to_end(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.compare_localization_large import build_parser,run
        from gaussian_jscc.data import write_ply
        from gaussian_jscc.transport import load_checkpoint
        raw,_,_,_=trunk_tests.old_setup(2048)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_ply(root/'input.ply',raw,0)
            args=build_parser().parse_args(['--ply',str(root/'input.ply'),'--out',str(root/'run'),
                                          '--blocks','8','--steps','2','--every','1','--save-every','2',
                                          '--seeds','42','--threads','1','--blocks-per-batch','2'])
            result=run(args)
            self.assertEqual(len(result['rows']),2)
            rows=[json.loads(s) for s in (root/'run/token_translation_seed42/validation.jsonl').read_text().splitlines()]
            baseline=json.loads((root/'run/none_seed42/validation.jsonl').read_text().splitlines()[0])
            self.assertEqual(rows[0]['heldout']['mse'],baseline['heldout']['mse'])
            self.assertAlmostEqual(rows[-1]['heldout']['relative_mse'],rows[-1]['heldout']['without_translation_relative_mse'],places=6)
            m=load_checkpoint(root/'run/token_translation_seed42/codec_2.pt','cpu')
            self.assertEqual(m.cfg.decoder_localization,'token_translation')
            self.assertTrue((root/'run/common_mse.png').is_file())


if __name__=='__main__':
    unittest.main()
