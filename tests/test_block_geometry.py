"""CPU wire, optimization, migration and fixed-SNR contracts for block geometry."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from gaussian_jscc.block_geometry import BlockGeometry, position_objective
from gaussian_jscc.codec import CodecConfig, GaussianCodec, unpack, channel
from gaussian_jscc.data import prepare, to_features, write_ply
from gaussian_jscc.losses import add_arguments, configure_training
from gaussian_jscc.optimization import all_tier_geometry_step, clip_codec_gradients
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.transport import save_checkpoint, load_checkpoint, model_id, transmit, receive
from test_gaussian_jscc import fixture


class BlockGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def scene(self, n=17):
        raw, _ = fixture(n, degree=0)
        cfg = CodecConfig(sh_degree=0, hidden=16, grid_dim=4, depth=1, levels=(2, 3),
                          block_size=8, position_head='block_relative_v4', loss_profile='position_v3')
        model = GaussianCodec(cfg)
        raw, geometry, _ = prepare(raw, 16)
        f, _ = to_features(raw, geometry, model)
        return raw, model, geometry, f

    def test_full_range_noiseless_identity_all_tiers_and_mixed(self):
        _, model, _, f = self.scene()
        f[:8, :3] = torch.tensor([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)])
        for q in [torch.full((len(f),), t) for t in (1,2,3)] + [torch.arange(len(f)) % 3 + 1]:
            symbols = model.encode(f, f[:,:3], q, 10.)
            pred = model.decode(symbols, q, 10.)
            torch.testing.assert_close(pred[:,:3], f[:,:3], atol=2e-6, rtol=2e-6)
            self.assertEqual(len(symbols), sum(model.cfg.rates[int(t)] for t in q))
            self.assertAlmostEqual(float(symbols.square().sum(-1).mean().detach()), 1., places=5)
            dense = unpack(symbols, q, model.cfg.rates)[..., model.geometry_slots]
            torch.testing.assert_close(dense.square().sum(-1), model.block_geometry.rates[q].float())

    def test_power_and_active_masks_after_learning(self):
        _, model, _, f = self.scene()
        torch.nn.init.normal_(model.block_geometry.encoder[-1].weight, std=2.)
        q = torch.arange(len(f)) % 4
        z = model.block_geometry.encode(f[:,:3], q, 10.)
        torch.testing.assert_close(z.square().sum(-1), model.block_geometry.rates[q].float())
        self.assertEqual(float(z[q == 0].detach().abs().max()), 0.)
        empty = model.block_geometry.encode(f[:,:3], q*0, 10.)
        self.assertTrue(torch.isfinite(empty).all())
        self.assertEqual(float(empty.detach().abs().max()), 0.)

    def test_padded_batch_matches_independent_wire(self):
        _, model, _, f = self.scene(8)
        features = torch.stack((f, f))
        features[1, :5, :3] = features[1, :5, :3] * .01 + .9
        q = torch.tensor([[1,2,3,1,2,3,1,2], [3,1,2,1,3,0,0,0]])
        pred, _, _ = model.forward_tier_batches(features, features[...,:3], F.one_hot(q,4).float(), 10., 'none')
        for i in range(2):
            keep = q[i] > 0
            expected = model(features[i,keep], features[i,keep,:3], q[i,keep], 10., 'none')
            torch.testing.assert_close(pred[i,keep], expected, atol=3e-6, rtol=3e-5)
        choices = F.one_hot(q,4).float().requires_grad_()
        with self.assertRaisesRegex(ValueError, 'joint mask'):
            model.forward_tier_batches(features, features[...,:3], choices, 10.)

    def test_receiver_needs_no_source_or_encoder_cache(self):
        raw, model, _, _ = self.scene()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(root/'codec.pt', model, 0)
            fresh = load_checkpoint(root/'codec.pt', 'cpu')
            self.assertEqual(model_id(model), model_id(fresh))
            transmit(model, raw, torch.arange(len(raw))%3+1, 10., 'none', 3, root/'packet')
            recovered = receive(fresh, root/'packet')
            # This fixture is already in sender Morton order.
            torch.testing.assert_close(recovered[:,:3], raw[:,:3], atol=3e-6, rtol=2e-5)

    def test_geometry_freeze_and_local_gradient(self):
        _, model, geometry, f = self.scene()
        for n,p in model.named_parameters():
            p.requires_grad_(n.startswith('block_geometry.'))
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss, _ = all_tier_geometry_step(model, f, geometry, 10., 'awgn')
        norm, stats = clip_codec_gradients(model, 1., 'branch')
        self.assertTrue(torch.isfinite(norm))
        self.assertEqual(set(stats['gradient_groups']), {'geometry_encoder','geometry_decoder'})
        optimizer.step()
        self.assertTrue(any(not torch.equal(t,model.state_dict()[n]) for n,t in before.items() if n.startswith('block_geometry.')))
        for n,t in before.items():
            if not n.startswith('block_geometry.'):
                torch.testing.assert_close(t,model.state_dict()[n],rtol=0,atol=0)
        target = torch.full((8,3), .5)
        pred = (target + .1).requires_grad_()
        loss, _ = position_objective(pred,target)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_explicit_migration_preserves_old_weights(self):
        _, model, _, _ = self.scene()
        cfg = copy.deepcopy(model.cfg); cfg.position_head = 'sigmoid'
        old = GaussianCodec(cfg)
        before = copy.deepcopy(old.state_dict())
        parser = argparse.ArgumentParser(); add_arguments(parser)
        argv = ['--position-head','block_relative_v4','--loss-profile','position_v3']
        with self.assertRaisesRegex(ValueError,'upgrade-position-head'):
            configure_training(old, parser.parse_args(argv))
        configure_training(old, parser.parse_args(argv + ['--upgrade-position-head']))
        for n,t in before.items():
            torch.testing.assert_close(t,old.state_dict()[n],rtol=0,atol=0)

    def test_replay_matches_checkpoint(self):
        _, model, geometry, f = self.scene(8)
        results = []
        for mode in ('replay','checkpoint'):
            model.zero_grad(set_to_none=True); torch.manual_seed(42)
            loss,_ = full_scene_step(model, [(f[None], torch.full((1,len(f)),2))], geometry,10.,'awgn',
                                     lambda raw:raw.square().mean(), attr_weight=1.,mode=mode)
            results.append((loss,{n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0],results[1][0])
        for n in results[0][1]:
            torch.testing.assert_close(results[0][1][n],results[1][1][n],atol=2e-5,rtol=2e-4)

    def test_fixed_snr_learning_reduces_independent_noise_objective(self):
        torch.manual_seed(123)
        model = BlockGeometry((0,4,8,16),32)
        xyz = torch.rand(4,64,3)*.2 + torch.rand(4,1,3)*.7
        q = torch.ones((4,64),dtype=torch.long)
        def predict():
            z = model.encode(xyz,q,10.)
            return model.decode(channel(z.reshape(-1,2),10.,'awgn').reshape_as(z),q,10.)
        def metric():
            with torch.random.fork_rng(), torch.no_grad():
                torch.manual_seed(456)
                return float(position_objective(predict(),xyz)[0])
        before = metric()
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        for _ in range(150):
            optimizer.zero_grad(set_to_none=True)
            loss,_ = position_objective(predict(),xyz)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        self.assertLess(metric(),before*.95)

    def test_cpu_cli_fixed_snr_best_checkpoint_and_early_stop(self):
        from gaussian_jscc.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw, model, _, _ = self.scene(8)
            write_ply(root/'input.ply',raw,0); save_checkpoint(root/'init.pt',model,0)
            argv = ['codec','train','--ply',str(root/'input.ply'),'--init',str(root/'init.pt'),
                    '--out',str(root/'run'),'--device','cpu','--training-data-device','cpu',
                    '--position-head','block_relative_v4','--loss-profile','position_v3',
                    '--geometry-only','--fixed-snr','10','--steps','5','--tier-training','all',
                    '--attribute-drop','0','--position-eval-every','1','--position-eval-blocks','1',
                    '--position-patience','1','--position-min-delta','1000000','--profile-every','1']
            with patch.object(sys,'argv',argv), patch('gaussian_jscc.plots.safe_plot'):
                main()
            logs = [json.loads(s) for s in (root/'run/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(logs),1)
            self.assertEqual(logs[0]['snr'],10.)
            self.assertEqual(logs[0]['phase'],'geometry')
            saved = torch.load(root/'run/codec.pt',weights_only=True)
            self.assertEqual(saved['step'],1)
            best = torch.load(root/'run/codec_best.pt',weights_only=True)
            self.assertEqual(best['step'],0)
            rows = json.loads((root/'run/position_evaluation.json').read_text())
            self.assertEqual({r['snr'] for r in rows},{10.})


if __name__ == '__main__':
    unittest.main()
