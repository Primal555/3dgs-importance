"""Experimental attribute residuals: real codec, channel, transport and autograd."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import prepare, to_features, to_raw
from gaussian_jscc.transport import load_checkpoint, model_id, receive, save_checkpoint, transmit
from test_gaussian_jscc import fixture


class ResidualPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pair(self, degree=1):
        self.assertTrue('decoder_mode' in CodecConfig.__dataclass_fields__, 'missing opt-in residual decoder')
        raw, _ = fixture(8, degree=degree)
        raw, geometry, _ = prepare(raw, 16)
        cfg = dict(sh_degree=degree, hidden=16, depth=1, grid_dim=4, levels=(2,),
                   block_size=8, decoder_window=4, rates=(0,4,12,24), prefix_mode='progressive',
                   position_delivery='quantized', position_bits=16, position_compression='delta_zlib')
        torch.manual_seed(67)
        direct = GaussianCodec(CodecConfig(**cfg))
        torch.manual_seed(67)
        residual = GaussianCodec(CodecConfig(**cfg, decoder_mode='residual'))
        for model in (direct, residual):
            model.attr_mean.copy_(raw[:,3:].mean(0))
            model.attr_std.copy_(raw[:,3:].std(0,unbiased=False).clamp_min(.01))
        features, _ = to_features(raw,geometry,direct)
        return raw, geometry, features, direct, residual

    def open_corrections(self, model):
        # Zero-init corrections could hide future-symbol leakage; exercise learned heads.
        with torch.no_grad():
            for stage in model.learned.refinements:
                for head in stage.heads.values():
                    head.weight.normal_(std=.03)
                    head.bias.normal_(std=.02)

    def test_opt_in_preserves_legacy_config_shared_weights_and_encoder_codeword(self):
        _, _, f, direct, residual = self.pair()
        self.assertNotIn('decoder_mode', direct.cfg.to_dict())
        self.assertEqual(CodecConfig.from_dict(residual.cfg.to_dict()).decoder_mode,'residual')
        for name, value in direct.state_dict().items():
            torch.testing.assert_close(value,residual.state_dict()[name],atol=0,rtol=0)
        active = torch.ones(len(f),dtype=torch.bool)
        torch.testing.assert_close(direct.encode_full(f,f[:,:3],active,10),
                                   residual.encode_full(f,f[:,:3],active,10),atol=0,rtol=0)
        # Extra enhancement initialization must not shift subsequent shared RNG draws.
        torch.manual_seed(91)
        GaussianCodec(direct.cfg)
        after_direct = torch.rand(10)
        torch.manual_seed(91)
        GaussianCodec(residual.cfg)
        torch.testing.assert_close(torch.rand(10),after_direct,atol=0,rtol=0)
        for values in ({'prefix_mode':'adaptive'}, {'position_delivery':'learned'}, {'rates':(0,4,12,24,32)}):
            cfg = dict(residual.cfg.to_dict(), **values)
            if values.get('position_delivery')=='learned':
                cfg['position_compression']='none'
            with self.assertRaises(ValueError):
                CodecConfig(**cfg)

    def test_zero_initialized_enhancements_start_at_base_and_do_not_touch_xyz(self):
        _, _, f, direct, model = self.pair()
        received = torch.randn(1,len(f),48)
        q = torch.tensor([[0,1,2,3,1,2,3,0]])
        stages, deltas = model.learned.decode_stages(received,q,10)
        self.assertEqual(len(stages),3)
        self.assertEqual(len(deltas),2)
        self.assertTrue(all(torch.count_nonzero(d)==0 for d in deltas))
        for stage in stages[1:]:
            torch.testing.assert_close(stage,stages[0],atol=0,rtol=0)
        expected = direct.learned.decode(received,(q>0).long(),10)
        torch.testing.assert_close(stages[0],expected,atol=0,rtol=0)
        self.open_corrections(model)
        stages, deltas = model.learned.decode_stages(received,q,10)
        torch.testing.assert_close(stages[1][...,3:],stages[0][...,3:]+deltas[0])
        torch.testing.assert_close(stages[2][...,3:],stages[1][...,3:]+deltas[1])
        self.assertTrue(all(torch.count_nonzero(stage[q==0])==0 for stage in stages))
        for stage in stages[1:]:
            torch.testing.assert_close(stage[...,:3],stages[0][...,:3],atol=0,rtol=0)

    def test_lower_stages_cannot_read_future_symbols_or_neighbor_upgrades(self):
        _, _, f, _, model = self.pair()
        self.open_corrections(model)
        received = torch.randn(1,len(f),48)
        q = torch.tensor([[1,2,3,1,2,3,2,1]])
        original, _ = model.learned.decode_stages(received,q,10)
        altered = received.clone(); altered[...,8:] += 100
        changed, _ = model.learned.decode_stages(altered,q,10)
        torch.testing.assert_close(original[0],changed[0],atol=0,rtol=0)
        altered = received.clone(); altered[...,24:] -= 100
        changed, _ = model.learned.decode_stages(altered,q,10)
        torch.testing.assert_close(original[1],changed[1],atol=0,rtol=0)
        lower, _ = model.learned.decode_stages(received,q.clamp_max(2),10)
        torch.testing.assert_close(original[1],lower[1],atol=0,rtol=0)
        base = model.learned.decode(received,torch.ones_like(q),10)
        deployed = model.learned.decode(received,q,10)
        torch.testing.assert_close(base[q==1],deployed[q==1],atol=0,rtol=0)
        # Upper layers still have an observable effect, so the invariance isn't a dead decoder.
        self.assertGreater(float((original[2]-changed[2]).detach().abs().sum()),0.)

    def test_gradients_reach_corrections_and_only_transmitted_encoder_rows(self):
        _, geometry, f, _, model = self.pair()
        for tier in (1,2,3):
            model.zero_grad(set_to_none=True)
            q = torch.full((1,len(f)),tier)
            pred, _, _ = model.forward_tier_batches(f[None],f[None,:,:3],F.one_hot(q,4).float(),10,'none')
            (pred[...,3:]-f[None,:,3:]).square().mean().backward()
            grad = model.learned.symbol_head.weight.grad
            cutoff = 2*model.cfg.rates[tier]
            self.assertGreater(float(grad[:8].norm()),0.)
            self.assertTrue((grad[cutoff:]==0).all())
            for index, stage in enumerate(model.learned.refinements, start=2):
                gradients = [h.weight.grad for h in stage.heads.values()]
                if index<=tier:
                    self.assertGreater(sum(float(g.norm()) for g in gradients if g is not None),0.)
                else:
                    self.assertTrue(all(g is None for g in gradients))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            pred = model(f,f[:,:3],torch.full((len(f),),3),10,'none')
            (pred[:,3:]-f[:,3:]).square().mean().backward()
            optimizer.step()
        self.assertGreater(float(model.learned.refinements[0].previous.weight.grad.norm()),0.)
        self.assertGreater(float(model.learned.symbol_head.weight.grad[24:].norm()),0.)
        raw = to_raw(pred, geometry, model)
        self.assertTrue(torch.isfinite(raw).all())
        torch.testing.assert_close(raw[:,7:11].norm(dim=-1),torch.ones(len(raw)))

    def test_mixed_q0_and_degree_zero_roundtrip_checkpoint_and_actual_packet(self):
        raw, geometry, f, _, model = self.pair(degree=0)
        self.open_corrections(model)
        q = torch.tensor([0,1,2,3,0,1,2,3])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_checkpoint(root/'codec.pt',model,17)
            loaded = load_checkpoint(root/'codec.pt','cpu')
            self.assertEqual(model_id(model),model_id(loaded))
            stats = transmit(model,raw,q,10,'none',42,root/'packet')
            self.assertEqual(stats['payload_complex_symbols'],80)
            restored = receive(loaded,root/'packet')
            ordered,g,oq = prepare(raw,16,q)
            features,_ = to_features(ordered,g,model)
            expected = to_raw(model(features,features[:,:3],oq,10,'none')[oq>0],g,model)
            torch.testing.assert_close(restored,expected)
            inactive = model.learned.decode(torch.randn(1,len(f),48),torch.zeros(1,len(f),dtype=torch.long),10)
            self.assertEqual(int(torch.count_nonzero(inactive)),0)

    def test_allocator_and_legacy_checkpoint_conversion_are_explicitly_rejected(self):
        _, _, f, _, model = self.pair()
        q = torch.full((1,len(f)),3)
        with self.assertRaisesRegex(ValueError,'residual'):
            model.forward_st_prefix_batches(f[None],f[None,:,:3],q,F.one_hot(q,4).float(),10)
        from gaussian_jscc.multiscene import parser, train
        args = parser().parse_args(['--out','unused','--decoder-mode','residual'])
        with self.assertRaisesRegex(ValueError,'residual'):
            train(args)

    def test_real_replay_and_checkpoint_backward_match_with_mixed_residual_stages(self):
        from unittest.mock import patch
        from gaussian_jscc.training import full_scene_step
        from gaussian_jscc.render_objective import MultiViewRenderTask
        from test_render_first import synthetic_render, Reference
        import test_render_first as renders
        _, geometry, f, _, model = self.pair()
        self.open_corrections(model)
        q = torch.tensor([[0,1,2,3,1,2,3,3]])
        observations = []
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            for mode in ('replay','checkpoint'):
                trial = copy.deepcopy(model)
                torch.manual_seed(77)
                task = MultiViewRenderTask(renders.RenderFirstTests().cameras(),Reference(),1)
                loss,_ = full_scene_step(trial,[(f[None],q)],geometry,10,'awgn',task,attr_weight=0,mode=mode)
                observations.append((loss,{n:p.grad for n,p in trial.named_parameters()}))
        torch.testing.assert_close(observations[0][0],observations[1][0])
        for name, gradient in observations[0][1].items():
            if gradient is not None:
                torch.testing.assert_close(gradient,observations[1][1][name],atol=2e-6,rtol=2e-4)


if __name__=='__main__':
    unittest.main()
