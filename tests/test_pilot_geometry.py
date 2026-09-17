"""v5 wire contracts: no true gain/coordinate side channel, mixed tiers and q0."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import torch
from torch.nn import functional as F

from gaussian_jscc.block_geometry import PilotBlockGeometry
from gaussian_jscc.codec import channel
from gaussian_jscc.losses import add_arguments, configure_training
from gaussian_jscc.transport import load_checkpoint, save_checkpoint, transmit, receive, model_id
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.data import write_ply
from gaussian_jscc.optimization import all_tier_geometry_step
import test_block_geometry as block_tests


class PilotGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def scene(self, n=17):
        raw, model, geometry, features=block_tests.BlockGeometryTests().scene(n)
        model.cfg.geometry_group_size=4
        model.enable_block_geometry('block_pilot_v5')
        return raw,model,geometry,features

    def test_noiseless_mixed_tiers_q0_power_and_compaction(self):
        _,model,_,f=self.scene()
        q=torch.arange(len(f))%4
        g=model.block_geometry
        for learned in (False,True):
            if learned:
                torch.nn.init.normal_(g.encoder[-1].weight,std=1.)
            z=g.encode(f[:,:3],q,10.)
            self.assertAlmostEqual(float(z.square().sum().detach()/g.rates[q].sum()),1.,places=5)
            self.assertEqual(float(z[q==0].detach().abs().max()),0.)
            keep=q>0
            torch.testing.assert_close(z[keep],g.encode(f[keep,:3],q[keep],10.))
            y=g.decode(z,q,10.)
            torch.testing.assert_close(y[keep],g.decode(z[keep],q[keep],10.))
            if not learned:
                torch.testing.assert_close(y[keep],f[keep,:3],atol=2e-6,rtol=2e-6)
        z=g.encode(f[:,:3],q*0,10.)
        self.assertEqual(float(z.abs().max()),0.)
        self.assertTrue(torch.isfinite(g.decode(z,q*0,10.)).all())

    def test_noiseless_full_range_and_group_tail(self):
        g=PilotBlockGeometry((0,4,8,16),16,256)
        for n in (1,2,255,256,257,513):
            xyz=torch.rand(n,3);xyz[0]=0.
            if n>1: xyz[-1]=1.
            for tier in (1,2,3):
                q=torch.full((n,),tier)
                torch.testing.assert_close(g.decode(g.encode(xyz,q,10.),q,10.),xyz,atol=3e-6,rtol=3e-6)

    def test_padded_batch_and_independent_receiver(self):
        raw,model,_,f=self.scene(17)
        q=torch.arange(len(f))%4
        features=torch.stack([f,f])
        tiers=torch.stack([q,q.flip(0)])
        result,_,_=model.forward_tier_batches(features,features[...,:3],F.one_hot(tiers,4).float(),10.,'none')
        for b in range(2):
            keep=tiers[b]>0
            expected=model(features[b,keep],features[b,keep,:3],tiers[b,keep],10.,'none')
            torch.testing.assert_close(result[b,keep],expected,atol=5e-6,rtol=5e-5)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save_checkpoint(root/'codec.pt',model,0)
            fresh=load_checkpoint(root/'codec.pt','cpu')
            self.assertEqual(model_id(model),model_id(fresh))
            for kind in ('none','awgn'):
                stats=transmit(model,raw,q,10.,kind,42,root/kind)
                decoded=receive(fresh,root/kind)
                torch.testing.assert_close(decoded,receive(model,root/kind))
                self.assertEqual(stats['geometry_pilot_real_slots_per_retained_gaussian'],1)
                self.assertFalse(stats['block_reference_in_metadata'])
                self.assertEqual(stats['geometry_complex_symbols'],int(model.block_geometry.rates[q].sum()))
                if kind=='none':
                    torch.testing.assert_close(decoded[:,:3],raw[q>0,:3],atol=4e-6,rtol=3e-5)

    def test_migration_resets_only_geometry_and_old_hash_is_stable(self):
        raw,model,_,_=block_tests.BlockGeometryTests().scene()
        before=copy.deepcopy(model.state_dict()); old_hash=model_id(model)
        model.cfg.geometry_group_size=13
        self.assertEqual(old_hash,model_id(model)) # inactive option not in old wire hash
        parser=argparse.ArgumentParser(); add_arguments(parser)
        argv=['--position-head','block_pilot_v5','--geometry-group-size','4','--loss-profile','position_v3']
        with self.assertRaisesRegex(ValueError,'upgrade-position-head'):
            configure_training(model,parser.parse_args(argv))
        configure_training(model,parser.parse_args(argv+['--upgrade-position-head']))
        self.assertEqual(model.block_geometry.group_size,4)
        for name,value in before.items():
            if not name.startswith('block_geometry.'):
                torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        self.assertNotEqual(old_hash,model_id(model))

    def test_short_packet_adverse_pilots_remain_finite(self):
        g=PilotBlockGeometry((0,4,8,16),16,4)
        for n in (1,3,9):
            x=torch.rand(n,3);q=torch.ones(n,dtype=torch.long)
            z=g.encode(x,q,10.);z[:,7]=-1000.
            pred=g.decode(z,q,10.)
            pred.square().mean().backward()
            self.assertTrue(torch.isfinite(pred).all())
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in g.parameters() if p.grad is not None))

    def test_replay_matches_checkpoint(self):
        _,model,geometry,f=self.scene(8);results=[]
        for mode in ('replay','checkpoint'):
            model.zero_grad(set_to_none=True);torch.manual_seed(42)
            loss,_=full_scene_step(model,[(f[None],torch.full((1,len(f)),2))],geometry,10.,'awgn',
                                  lambda raw:raw.square().mean(),attr_weight=1.,mode=mode)
            results.append((loss,{n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0],results[1][0])
        for n in results[0][1]:
            torch.testing.assert_close(results[0][1][n],results[1][1][n],atol=3e-5,rtol=3e-4)

    def test_geometry_clean_objective_and_frozen_other_weights(self):
        _,model,geometry,f=self.scene()
        for name,p in model.named_parameters():
            p.requires_grad_(name.startswith('block_geometry.'))
        before=copy.deepcopy(model.state_dict())
        loss,stats=all_tier_geometry_step(model,f,geometry,10.,'awgn',clean_weight=1.)
        self.assertAlmostEqual(float(loss),stats['noisy_position_loss']+stats['clean_position_loss'],places=5)
        torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-4).step()
        for name,value in before.items():
            if not name.startswith('block_geometry.'):
                torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)

    def test_v5_cli_default_clean_loss_and_checkpoint(self):
        from gaussian_jscc.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,model,_,_=self.scene(8)
            write_ply(root/'input.ply',raw,0);save_checkpoint(root/'init.pt',model,0)
            argv=['codec','train','--ply',str(root/'input.ply'),'--init',str(root/'init.pt'),
                  '--out',str(root/'run'),'--device','cpu','--training-data-device','cpu',
                  '--position-head','block_pilot_v5','--loss-profile','position_v3',
                  '--geometry-only','--fixed-snr','10','--steps','2','--tier-training','all',
                  '--attribute-drop','0','--position-eval-every','1','--position-eval-blocks','1']
            with patch.object(sys,'argv',argv),patch('gaussian_jscc.plots.safe_plot'):
                main()
            info=json.loads((root/'run/training.json').read_text())
            self.assertEqual(info['geometry_clean_weight'],1.)
            logs=[json.loads(s) for s in (root/'run/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(logs),2)
            self.assertIn('clean_position_loss',logs[0])
            trained=load_checkpoint(root/'run/codec.pt','cpu')
            self.assertEqual(trained.cfg.position_head,'block_pilot_v5')
            self.assertEqual(trained.block_geometry.group_size,4)


if __name__=='__main__':
    unittest.main()
