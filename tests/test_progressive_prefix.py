"""Progressive code invariants and CPU protocol/autograd integration.

Synthetic rendering tests are not evidence of real-scene fidelity gains.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from test_learned_joint import setup
from test_render_first import synthetic_render, Reference
import test_render_first as render_fixture
from gaussian_jscc.codec import CodecConfig, GaussianCodec, pack, prefix_mask
from gaussian_jscc.data import prepare, to_features, to_raw, write_ply
from gaussian_jscc.render_objective import MultiViewRenderTask
from gaussian_jscc.render_validation import validate_render
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.transport import save_checkpoint, load_checkpoint, model_id, transmit, receive


def progressive(n=16):
    raw, geometry, features, old = setup(n)
    cfg = CodecConfig(**dict(old.cfg.to_dict(), prefix_mode='progressive',
                            position_delivery='quantized', position_bits=16,
                            position_compression='delta_zlib', rates=(0,8,16,32)))
    model = GaussianCodec(cfg)
    model.attr_mean.copy_(old.attr_mean)
    model.attr_std.copy_(old.attr_std)
    return raw, geometry, features, model


class ProgressiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_encode_once_matches_all_prefixes_and_mixed_neighbor_tiers(self):
        _, _, f, model = progressive(8)
        retained = torch.tensor([True,False,True,True,True,True,True,True])
        full = model.encode_full(f,f[:,:3],retained,10)
        for q in (retained.long(),retained.long()*2,retained.long()*3,
                  torch.tensor([3,0,1,2,3,1,2,1])):
            actual = model.encode(f,f[:,:3],q,10)
            torch.testing.assert_close(actual,pack(full,q,model.cfg.rates),rtol=0,atol=0)
            self.assertEqual(actual.shape,(int(torch.tensor(model.cfg.rates)[q].sum()),2))
        # Positive tier embedding is receiver-only, including context gates.
        with torch.no_grad():
            model.learned.tier.weight.add_(100)
        torch.testing.assert_close(full,model.encode_full(f,f[:,:3],retained,10),rtol=0,atol=0)
        self.assertGreater(float((full-model.encode_full(f,f[:,:3],retained,0)).detach().abs().max()),0)

    def test_independent_layer_power_and_suffix_changes_do_not_rescale_base(self):
        _,_,f,model = progressive(8)
        active = torch.ones(8,dtype=torch.bool)
        full = model.encode_full(f,f[:,:3],active,10)
        for a,b in zip(model.cfg.rates[:-1],model.cfg.rates[1:]):
            energy = full[:,2*a:2*b].square().sum(-1)/(b-a)
            self.assertTrue((energy <= 1+1e-6).all())
            self.assertTrue((energy > 0).all())
        with torch.no_grad():
            model.learned.symbol_head.weight[16:].mul_(50)
            model.learned.symbol_head.bias[16:].add_(25)
        changed = model.encode_full(f,f[:,:3],active,10)
        torch.testing.assert_close(full[:,:16],changed[:,:16],rtol=0,atol=0)
        self.assertFalse(torch.equal(full[:,16:],changed[:,16:]))

    def test_dropped_points_and_padding_do_not_leak(self):
        _,_,f,model = progressive(8)
        q = torch.tensor([1,0,2,3,1,0,2,3])
        original = model(f,f[:,:3],q,10,'none')
        altered = f.clone()
        altered[q==0] = 100
        torch.testing.assert_close(original,model(altered,altered[:,:3],q,10,'none'))
        zero = torch.zeros_like(q)
        self.assertEqual(model.encode(f,f[:,:3],zero,10).shape,(0,2))
        self.assertTrue(torch.equal(model(f,f[:,:3],zero,10,'none'),torch.zeros_like(f)))

    def test_unsent_symbols_and_gradients_are_zero(self):
        _,_,f,model = progressive(8)
        for tier in (1,2,3):
            model.zero_grad(set_to_none=True)
            q = torch.full((8,),tier)
            cutoff = 2*model.cfg.rates[tier]
            z = model.learned.encode(f[None],f[None,:,:3],q[None],10)
            self.assertTrue((z[...,cutoff:] == 0).all())
            clean = model.learned.decode(z,q[None],10)
            corrupted = z.detach().clone()
            corrupted[...,cutoff:] = 1000
            torch.testing.assert_close(clean,model.learned.decode(corrupted,q[None],10))
            clean[...,3:].square().mean().backward()
            grad = model.learned.symbol_head.weight.grad
            self.assertTrue((grad[cutoff:] == 0).all())
            self.assertGreater(float(grad[:cutoff].norm()),0)
            self.assertTrue(torch.isfinite(grad).all())

    def test_paired_noise_is_fixed_per_slot_and_packed_matches_batched_forward(self):
        _,_,f,model = progressive(8)
        qs = [torch.full((1,8),q) for q in (1,2,3)]
        qs.append(torch.tensor([[3,0,1,2,1,2,3,0]]))
        received = []
        original = model.learned.decode
        def capture(z,q,snr):
            received.append(z.detach().clone())
            return original(z,q,snr)
        with patch.object(model.learned,'decode',side_effect=capture):
            for q in qs:
                torch.manual_seed(123)
                model.forward_tier_batches(f[None],f[None,:,:3],F.one_hot(q,4).float(),10,'awgn',paired_noise=True)
        for tier in (1,2):
            cutoff = 2*model.cfg.rates[tier]
            torch.testing.assert_close(received[tier-1][...,:cutoff],received[2][...,:cutoff],rtol=0,atol=0)
        # Even a different retained set has the same noise in shared slots.
        noises = []
        for q,z in zip(qs,received):
            noises.append(z-model.learned.encode(f[None],f[None,:,:3],q,10))
        mask = prefix_mask(qs[-1].flatten(),model.cfg.rates).reshape_as(received[-1])
        torch.testing.assert_close(noises[-1][mask],noises[2][mask],rtol=1e-5,atol=2e-7)
        torch.manual_seed(23)
        packed = model(f,f[:,:3],qs[-1][0],10,'awgn')
        torch.manual_seed(23)
        batched = model.forward_tier_batches(f[None],f[None,:,:3],F.one_hot(qs[-1],4).float(),10,'awgn')[0][0]
        torch.testing.assert_close(packed,batched)

    def test_checkpoint_packet_roundtrip_and_legacy_identity(self):
        raw,_,_,model = progressive(17)
        q = torch.arange(len(raw))%4
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_checkpoint(root/'codec.pt',model,0)
            loaded = load_checkpoint(root/'codec.pt','cpu')
            self.assertEqual(loaded.cfg.prefix_mode,'progressive')
            self.assertEqual(model_id(loaded),model_id(model))
            stats = transmit(model,raw,q,10,'none',42,root/'packet')
            actual = receive(loaded,root/'packet')
            ordered,g,oq = prepare(raw,16,q)
            expected = []
            for start in range(0,len(raw),model.cfg.block_size):
                f,_ = to_features(ordered[start:start+model.cfg.block_size],g,model)
                qb = oq[start:start+len(f)]
                pred = model(f,f[:,:3],qb,10,'none')
                expected.append(to_raw(pred[qb>0],g,model))
            torch.testing.assert_close(actual,torch.cat(expected))
            self.assertEqual(stats['payload_complex_symbols'],int(torch.tensor(model.cfg.rates)[q].sum()))
        config = model.cfg.to_dict()
        config.pop('prefix_mode')
        legacy = CodecConfig.from_dict(config)
        self.assertEqual(legacy.prefix_mode,'adaptive')
        self.assertNotIn('prefix_mode',legacy.to_dict())
        other = GaussianCodec(legacy)
        other.load_state_dict(model.state_dict())
        self.assertNotEqual(model_id(other),model_id(model))
        with self.assertRaisesRegex(ValueError,'progressive'):
            other.encode_full(torch.zeros(8,14),torch.zeros(8,3),torch.ones(8,dtype=torch.bool),10)

    def test_replay_matches_checkpoint_with_render_only_gradients(self):
        _,g,f,base = progressive(16)
        batches = [(f[:8][None],torch.tensor([[1,0,2,3,1,2,3,1]])),
                   (f[8:][None],torch.tensor([[3,2,1,3,2,1,2,0]]))]
        results = []
        cameras = render_fixture.RenderFirstTests().cameras()
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            for mode in ('replay','checkpoint'):
                model = copy.deepcopy(base)
                torch.manual_seed(123)
                task = MultiViewRenderTask(cameras,Reference(),0)
                loss,stats = full_scene_step(model,batches,g,10,'awgn',task,attr_weight=0,mode=mode)
                self.assertEqual(stats['aux_loss'],0)
                results.append((loss,{n:p.grad for n,p in model.named_parameters()}))
        torch.testing.assert_close(results[0][0],results[1][0])
        for name,grad in results[0][1].items():
            if grad is not None:
                torch.testing.assert_close(grad,results[1][1][name],rtol=2e-4,atol=2e-6,msg=name)
                self.assertTrue(torch.isfinite(grad).all())

    def test_validation_logs_paired_prefix_gains_and_preserves_rng(self):
        raw,g,f,model = progressive(16)
        groups,ids = [f.reshape(2,8,-1)],[torch.arange(16).reshape(2,8)]
        with tempfile.TemporaryDirectory() as temp,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            state = torch.get_rng_state().clone()
            a = validate_render(model,groups,ids,raw,g,render_fixture.RenderFirstTests().cameras(),Reference(),10,'awgn',2,42,temp,0,'initial')
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            b = validate_render(model,groups,ids,raw,g,render_fixture.RenderFirstTests().cameras(),Reference(),10,'awgn',2,42,temp,1,'render')
            self.assertEqual(a['layouts'],b['layouts'])
            self.assertTrue(a['paired_prefix_noise'])
            self.assertEqual(len(a['prefix_gains']),2)
            gain = a['prefix_gains'][0]
            self.assertAlmostEqual(gain['source_psnr_gain_db'],a['layouts'][1]['source_psnr']-a['layouts'][0]['source_psnr'])
            self.assertEqual(len(list((Path(temp)/'validation_images'/'000000').glob('*.png'))),8)

    def test_cli_four_layout_render_smoke_and_initializer_mismatch(self):
        from gaussian_jscc.cli import main
        raw,_,_,_ = progressive(16)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_ply(root/'input.ply',raw,0)
            argv = ['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                    '--source','mock','--device','cuda','--render-steps','4','--validation-views','1',
                    '--validation-trials','1','--validate-every','4','--save-every','4',
                    '--hidden','16','--depth','1','--grid-dim','4','--levels','2','--block-size','8',
                    '--decoder-window','4','--prefix-mode','progressive','--position-delivery','quantized',
                    '--position-bits','16','--position-compression','delta_zlib','--render-lr','0.0001']
            with patch('sys.argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()*2), \
                 patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
                main()
            rows = [json.loads(line) for line in (root/'run'/'loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['layout'] for r in rows],['1','2','3','mixed'])
            self.assertTrue(all(r['aux_loss']==0 and r['grad_norm']>0 for r in rows))
            for suffix in ('png','svg','csv'):
                self.assertTrue((root/'run'/'charts'/f'prefix_gains.{suffix}').exists())
            loaded = load_checkpoint(root/'run'/'codec.pt','cpu')
            self.assertEqual(loaded.cfg.prefix_mode,'progressive')
            argv[argv.index(str(root/'run'))] = str(root/'mismatch')
            argv[argv.index('progressive')] = 'adaptive'
            argv.extend(['--init',str(root/'run'/'codec.pt')])
            with patch('sys.argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')):
                with self.assertRaisesRegex(ValueError,'initializer prefix_mode'):
                    main()


if __name__ == '__main__':
    unittest.main()
