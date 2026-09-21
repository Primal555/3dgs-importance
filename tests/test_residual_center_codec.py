"""Baseline-preserving XYZ translation residual and transport contracts."""
import unittest
from unittest.mock import patch
import torch
import test_multiscale_codec as contracts
import test_q3_noiseless as q3_contracts
import test_position_path_diagnosis as diagnosis_contracts
from test_point_attention import setup as old_setup
from gaussian_jscc.codec import CodecConfig,GaussianCodec


def setup(n=17):
    raw,g,f,old=old_setup(n)
    cfg=old.cfg.to_dict();cfg['xyz_decoder']='residual_center'
    model=GaussianCodec(CodecConfig.from_dict(cfg))
    model.attr_mean.copy_(old.attr_mean);model.attr_std.copy_(old.attr_std)
    return raw,g,f,model


class ResidualCenterCLI(q3_contracts.Q3NoiselessTests):
    xyz_decoder='residual_center'


class ResidualCenterDiagnosis(diagnosis_contracts.PositionPathTests):
    xyz_decoder='residual_center'


class SymbolSkipCLI(q3_contracts.Q3NoiselessTests):
    xyz_decoder='symbol_skip'


class SymbolSkipDiagnosis(diagnosis_contracts.PositionPathTests):
    xyz_decoder='symbol_skip'


class ResidualCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_baseline_at_initialization_and_shared_gradients(self):
        _,_,f,old=old_setup(17)
        cfg=old.cfg.to_dict()
        torch.manual_seed(42);baseline=GaussianCodec(CodecConfig.from_dict(cfg))
        cfg['xyz_decoder']='residual_center'
        torch.manual_seed(42);new=GaussianCodec(CodecConfig.from_dict(cfg))
        q=torch.arange(17)%4
        a=baseline(f,f[:,:3],q,10,'none');b=new(f,f[:,:3],q,10,'none')
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().mean().backward();b.square().mean().backward()
        shared=dict(baseline.named_parameters())
        for name,p in new.named_parameters():
            if name in shared and p.grad is not None:
                torch.testing.assert_close(p.grad,shared[name].grad,rtol=0,atol=0)
        self.assertGreater(float(new.learned.dec_xyz_center[-1][-1].weight.grad.norm()),0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None))

    def test_center_translation_preserves_relative_geometry(self):
        _,_,f,m=setup(17)
        q=torch.full((17,),3,dtype=torch.long);q[1]=0
        a=m(f,f[:,:3],q,10,'none')
        shift=torch.tensor([.1,-.2,.3])
        with torch.no_grad():
            m.learned.dec_xyz_center[-1][-1].bias.copy_(shift)
        b=m(f,f[:,:3],q,10,'none')
        torch.testing.assert_close(b[q>0,:3]-a[q>0,:3],shift.expand(16,3))
        torch.testing.assert_close(b[:,3:],a[:,3:],rtol=0,atol=0)
        self.assertEqual(float(b[q==0].detach().abs().sum()),0.)

    def test_local_comparison_uses_matched_inputs_and_initial_predictions(self):
        import tempfile
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from gaussian_jscc.data import write_ply
        from scripts.compare_xyz_decoders_local import run
        raw,_,_,_=old_setup(1024)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_ply(root/'input.ply',raw,0)
            args=SimpleNamespace(ply=str(root/'input.ply'),out=str(root/'run'),steps=1,
                                 every=1,threads=1,seeds=[42],modes=['additive','residual_center'],blocks=[0,1,2,3])
            result=run(args)
            self.assertEqual(result['additive_seed42']['initial'],result['residual_center_seed42']['initial'])
            self.assertTrue((root/'run/protocol.json').is_file())
            self.assertTrue((root/'run/residual_center_seed42/codec.pt').is_file())
            rows=(root/'run/residual_center_seed42/loss.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(rows[0])['step'],1)

    def test_existing_contracts(self):
        for name in ('test_transport_power_drop_and_replay_contracts',
                     'test_mixed_tiers_batching_and_all_heads_receive_gradients',
                     'test_short_fit_and_all_backward_modes'):
            with self.subTest(name=name),patch('test_multiscale_codec.setup',setup):
                getattr(contracts.MultiScaleTests(name),name)()

    def test_symbol_skip_starts_as_baseline_and_receives_gradients(self):
        _,g,f,old=old_setup(17)
        cfg=old.cfg.to_dict()
        torch.manual_seed(42);baseline=GaussianCodec(CodecConfig.from_dict(cfg))
        cfg['xyz_decoder']='symbol_skip'
        torch.manual_seed(42);new=GaussianCodec(CodecConfig.from_dict(cfg))
        q=torch.arange(17)%4
        a=baseline(f,f[:,:3],q,10,'none');b=new(f,f[:,:3],q,10,'none')
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().mean().backward();b.square().mean().backward()
        self.assertGreater(float(new.learned.dec_xyz_symbols.weight.grad.norm()),0.)
        shared=dict(baseline.named_parameters())
        for name,p in new.named_parameters():
            if name in shared and p.grad is not None:
                torch.testing.assert_close(p.grad,shared[name].grad,rtol=0,atol=0)

    def test_symbol_skip_transport_fit_and_replay(self):
        def symbol_setup(n=17):
            raw,g,f,old=old_setup(n)
            cfg=old.cfg.to_dict();cfg['xyz_decoder']='symbol_skip'
            m=GaussianCodec(CodecConfig.from_dict(cfg))
            m.attr_mean.copy_(old.attr_mean);m.attr_std.copy_(old.attr_std)
            return raw,g,f,m
        with patch('test_multiscale_codec.setup',symbol_setup):
            for name in ('test_transport_power_drop_and_replay_contracts','test_short_fit_and_all_backward_modes'):
                getattr(contracts.MultiScaleTests(name),name)()

    def test_screen_rejects_one_bad_seed_and_incomplete_run(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.compare_xyz_decoders_local import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol=dict(ply='same.ply',steps=300,every=50,blocks=[0,1,2,3],seeds=[42,43],
                          config={},modes=['additive','block_center'])
            (root/'protocol.json').write_text(json.dumps(protocol))
            for seed in protocol['seeds']:
                for mode in protocol['modes']:
                    folder=root/f'{mode}_seed{seed}';folder.mkdir()
                    mse=2. if mode=='additive' else (1. if seed==42 else 3.)
                    metric=dict(mse=mse,common_mse=mse*.5,relative_mse=mse*.5)
                    rows=[dict(step=s,fit=metric,heldout=metric) for s in (200,250,300)]
                    (folder/'validation.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            result=summarize([root],root/'report')
            self.assertFalse(result['decisions']['block_center']['local_screen_pass'])
            self.assertEqual(result['decisions']['block_center']['seeds_passing'],1)
            (root/'block_center_seed43/validation.jsonl').write_text(json.dumps(dict(step=250,fit=metric,heldout=metric)))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                summarize([root],root/'report2')


if __name__=='__main__':
    unittest.main()
