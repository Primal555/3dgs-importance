"""Position migration, bounded geometry gradients, clipping and training contracts."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gaussian_jscc.data import prepare, to_features, write_ply
from gaussian_jscc.losses import (add_arguments, configure_training, initialize_position_head,
                                  position_v3_terms, reconstruction_loss)
from gaussian_jscc.optimization import all_tier_attribute_step, clip_codec_gradients, preserved_rng
from gaussian_jscc.transport import load_checkpoint, model_id, save_checkpoint, transmit, receive
from gaussian_jscc.training import full_scene_step
from test_gaussian_jscc import fixture


class PositionV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def options(self, upgrade=True):
        parser=argparse.ArgumentParser()
        add_arguments(parser)
        return parser.parse_args(['--loss-profile','position_v3']+(['--upgrade-position-head'] if upgrade else []))

    def scene(self):
        raw,model=fixture(n=8,degree=0)
        raw,geometry,_=prepare(raw,16)
        configure_training(model,self.options())
        initialize_position_head(model,raw,geometry)
        f,_=to_features(raw,geometry,model)
        return raw,model,geometry,f

    def test_explicit_head_migration_preserves_other_weights_and_loaded_head(self):
        raw,model=fixture(n=8,degree=0)
        before=copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError,'upgrade-position-head'):
            configure_training(model,self.options(False))
        configure_training(model,self.options())
        for name,tensor in before.items():
            if not name.startswith('position_seed.'):
                torch.testing.assert_close(tensor,model.state_dict()[name],rtol=0,atol=0)
        raw,geometry,_=prepare(raw,16)
        initialize_position_head(model,raw,geometry)
        self.assertFalse(model.position_head_needs_initialization)
        self.assertFalse(model.detach_attribute_context_xyz)
        torch.testing.assert_close(.5+.25*model.position_seed.bias,geometry.normalize(raw[:,:3]).median(0).values)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'codec.pt'
            save_checkpoint(path,model,0)
            loaded=load_checkpoint(path,'cpu')
            initialize_position_head(loaded,raw+10,geometry)
            self.assertEqual(model_id(model),model_id(loaded))
            self.assertEqual(loaded.cfg.position_head,'normalized_affine_v3')
            self.assertEqual(loaded.cfg.geometry_weight,10.)

    def test_bounded_nonvanishing_geometry_gradient_even_outside_bbox(self):
        _,model,geometry,target=self.scene()
        identity=reconstruction_loss(target,target,geometry,model)
        self.assertLess(float(identity),1e-10)
        norms=[]
        for offset in (.1,1.,100.):
            pred=target.clone()
            pred[:,:3]+=offset
            pred.requires_grad_()
            base,tail=position_v3_terms(pred,target,model.cfg)
            loss=model.cfg.geometry_weight*(base+model.cfg.position_tail_weight*tail).mean()
            loss.backward()
            norms.append(pred.grad[:,:3].norm())
            self.assertTrue((pred.grad[:,:3]>0).all())
            bound=model.cfg.geometry_weight*(1+model.cfg.position_tail_weight)/(3*len(pred))
            self.assertLessEqual(float(pred.grad.abs().max()),bound+1e-6)
        torch.testing.assert_close(norms[0],norms[-1])

    def test_normalized_head_geometry_weight_gradient_bound(self):
        _,model,_,_=self.scene()
        norms=[]
        h=torch.randn(8,model.cfg.hidden)
        for magnitude in (1.,10000.):
            model.zero_grad(set_to_none=True)
            xyz=model.predict_position(h*magnitude)
            target=torch.full_like(xyz,-100.)
            base,tail=position_v3_terms(xyz,target,model.cfg)
            (10*(base+2*tail).mean()).backward()
            norm=model.position_seed.weight.grad.norm()
            bound=10*3*.25*math.sqrt(model.cfg.hidden/3)
            self.assertLessEqual(float(norm),bound+1e-5)
            norms.append(norm)
        torch.testing.assert_close(norms[0],norms[1],rtol=1e-4,atol=1e-5)

    def test_branch_clipping_does_not_shrink_unrelated_small_gradient(self):
        _,model,_,_=self.scene()
        values=[]
        for mode in ('global','branch'):
            model.zero_grad(set_to_none=True)
            model.position_seed.bias.grad=torch.full_like(model.position_seed.bias,1000.)
            attribute=next(model.heads.parameters())
            attribute.grad=torch.full_like(attribute,.01)
            norm,stats=clip_codec_gradients(model,1.,mode)
            self.assertGreater(float(norm),1000.)
            values.append(attribute.grad.clone())
            self.assertLessEqual(stats['gradient_groups']['geometry_decoder']['after'],1.000001)
            self.assertLessEqual(stats['post_clip_total_norm'],math.sqrt(2)+1e-6)
        torch.testing.assert_close(values[1],torch.full_like(values[1],.01))
        self.assertLess(float(values[0].norm()),float(values[1].norm())/1000)
        model.position_seed.bias.grad.fill_(float('inf'))
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            clip_codec_gradients(model,1.,'branch')

    def test_all_tiers_matches_mean_gradient_and_does_not_update_weights(self):
        _,model,geometry,f=self.scene()
        before=copy.deepcopy(model.state_dict())
        torch.manual_seed(77)
        loss,metrics=all_tier_attribute_step(model,f,geometry,10.,'awgn')
        actual={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        self.assertEqual(set(metrics['per_tier']),{'1','2','3'})
        for n,t in before.items():
            torch.testing.assert_close(t,model.state_dict()[n],rtol=0,atol=0)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(77)
        losses=[]
        for tier in (1,2,3):
            pred=model(f,f[:,:3],torch.full((len(f),),tier),10.,'awgn')
            losses.append(reconstruction_loss(pred,f,geometry,model))
        expected=sum(losses)/3
        expected.backward()
        torch.testing.assert_close(loss,expected)
        for n,p in model.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(actual[n],p.grad,atol=1e-5,rtol=1e-4)

    def test_replay_matches_checkpoint_with_new_head_and_loss(self):
        _,model,geometry,f=self.scene()
        batches=[(f[None],torch.full((1,len(f)),2))]
        gradients=[]; losses=[]
        for mode in ('checkpoint','replay'):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(19)
            loss,_=full_scene_step(model,batches,geometry,10.,'awgn',lambda raw:raw.square().mean(),
                                   attr_weight=1.,mode=mode)
            losses.append(loss)
            gradients.append({n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
        torch.testing.assert_close(*losses)
        for n in gradients[0]:
            torch.testing.assert_close(gradients[0][n],gradients[1][n],rtol=2e-4,atol=2e-5)

    def test_preserved_rng(self):
        def seed():
            random.seed(19); np.random.seed(19); torch.manual_seed(19)
        def draw():
            return random.random(),np.random.rand(),float(torch.rand(()))
        seed(); expected=draw(); seed()
        with preserved_rng(torch.device('cpu')):
            draw(); draw()
        self.assertEqual(expected,draw())

    def test_cpu_cli_checkpoint_fixed_evaluation_and_charts(self):
        from gaussian_jscc.cli import main
        from gaussian_jscc.plots import plot_training
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw,model=fixture(n=8,degree=0)
            write_ply(root/'source.ply',raw,0)
            save_checkpoint(root/'init.pt',model,0)
            initial=(root/'init.pt').read_bytes()
            argv=['codec','train','--ply',str(root/'source.ply'),'--init',str(root/'init.pt'),
                  '--out',str(root/'run'),'--device','cpu','--training-data-device','cpu',
                  '--steps','2','--save-every','1','--profile-every','1',
                  '--loss-profile','position_v3','--upgrade-position-head',
                  '--tier-training','all','--attribute-drop','0','--clip-mode','branch',
                  '--position-eval-every','1','--position-eval-blocks','1','--position-eval-snrs','10']
            with patch.object(sys,'argv',argv),patch('gaussian_jscc.plots.safe_plot'):
                main()
            self.assertEqual(initial,(root/'init.pt').read_bytes())
            rows=[json.loads(r) for r in (root/'run/loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            for r in rows:
                self.assertEqual(r['loss_profile'],'position_v3')
                self.assertEqual(r['clip_mode'],'branch')
                self.assertIn('updates',r)
                self.assertAlmostEqual(r['loss'],sum(r[k+'_contribution'] for k in ('geometry','shape','scale','opacity','dc','sh')),places=4)
            evaluations=json.loads((root/'run/position_evaluation.json').read_text())
            self.assertEqual(len(evaluations),18)
            self.assertEqual({r['step'] for r in evaluations},{0,1,2})
            self.assertIn('unclipped_position_rmse',evaluations[0])
            trained=load_checkpoint(root/'run/codec.pt','cpu')
            self.assertEqual(trained.cfg.position_head,'normalized_affine_v3')
            transmit(trained,raw,torch.full((len(raw),),2),10.,'none',42,root/'packet')
            recovered=receive(trained,root/'packet')
            self.assertTrue(torch.isfinite(recovered).all())
            result=plot_training(root/'run')
            self.assertTrue((root/'run/charts/training_gradient_groups.png').exists())
            self.assertTrue((root/'run/charts/training_fixed_positions.png').exists())
            self.assertTrue(result['charts'])


if __name__=='__main__':
    unittest.main()
