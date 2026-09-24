"""CPU checks: real codec/channel/autograd, synthetic differentiable rendering.

These verify implementation, not real CUDA rasterizer fidelity or convergence.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from test_learned_joint import setup
from gaussian_jscc.render_objective import image_distortion, MultiViewRenderTask, split_cameras
from gaussian_jscc.render_validation import validate_render
from gaussian_jscc.training import full_scene_step


def synthetic_render(scene,camera,*args):
    # Nonlinear XYZ/attribute coupling with different camera observations.
    xyz = scene[:,:3].mean(0) if len(scene) else scene.new_zeros(3)
    attr = scene[:,3:].mean() if len(scene) else scene.new_zeros(())
    return (xyz*camera.factor+attr*.02).sigmoid()[:,None,None].expand(3,8,8)


class Reference:
    def get(self,camera,device):
        return torch.full((3,8,8),.6,device=device)


class RenderFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def cameras(self):
        return [SimpleNamespace(factor=v,original_image=torch.full((3,8,8),.5),image_name=str(v)) for v in (.4,1.3)]

    def test_mse_no_clamping_and_target_detached(self):
        image=torch.tensor([2.],requires_grad=True)
        target=torch.tensor([.5],requires_grad=True)
        image_distortion(image,target).backward()
        self.assertEqual(float(image.grad),3.)
        self.assertIsNone(target.grad)

    def test_view_split_disjoint_spaced(self):
        train,val,ti,vi=split_cameras(list(range(21)),4,6)
        self.assertEqual(vi,[0,7,13,20])
        self.assertEqual(len(train),6)
        self.assertFalse(set(ti)&set(vi))
        self.assertEqual(val,vi)

    def test_streamed_multiview_matches_checkpoint_no_auxiliary(self):
        _,geometry,f,model=setup(16)
        batches=[(f[:8][None],torch.tensor([[1,2,3,1,2,3,0,1]])),
                 (f[8:][None],torch.tensor([[3,2,1,3,2,1,2,3]]))]
        other=copy.deepcopy(model)
        outputs=[]
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.training.reconstruction_loss',side_effect=AssertionError('auxiliary must not execute')):
            for mode,m in [('replay',model),('checkpoint',other)]:
                torch.manual_seed(125)
                task=MultiViewRenderTask(self.cameras(),Reference(),0)
                loss,stats=full_scene_step(m,batches,geometry,10,'awgn',task,attr_weight=0,mode=mode)
                outputs.append(loss)
                self.assertEqual(stats['aux_loss'],0)
                self.assertEqual(len(task.stats['view_mse']),2)
        torch.testing.assert_close(outputs[0],outputs[1])
        for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4,msg=name)

    def test_render_equivalence_not_parameter_copy(self):
        raw,_,_,_=setup(16)
        raw.requires_grad_()
        permuted=raw.flip(0)
        camera=self.cameras()[0]
        # Gaussian permutation does not change the rendered task; a per-row
        # parameter-copy loss would penalize it.
        self.assertGreater(float((raw-permuted).square().mean().detach()),0)
        self.assertLess(float(image_distortion(synthetic_render(raw,camera),synthetic_render(permuted,camera)).detach()),1e-10)

    def test_fixed_validation_preserves_rng_and_layouts(self):
        raw,g,f,model=setup(16)
        groups=[f.reshape(2,8,-1)]
        ids=[torch.arange(16).reshape(2,8)]
        with tempfile.TemporaryDirectory() as temp,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            root=Path(temp)
            state=torch.get_rng_state().clone()
            first=validate_render(model,groups,ids,raw,g,self.cameras(),Reference(),10,'awgn',2,42,root,0,'initial')
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            second=validate_render(model,groups,ids,raw,g,self.cameras(),Reference(),10,'awgn',2,42,root,1,'render')
            self.assertEqual(first['layouts'],second['layouts'])
            self.assertEqual([e['layout'] for e in first['layouts']],['1','2','3','mixed'])
            self.assertEqual(len(list((root/'validation_images'/'000000').glob('*.png'))),8)
            self.assertTrue(model.training)

    def test_zero_auxiliary_replay_records_geometry_gradient(self):
        _,g,f,model=setup(8)
        task=MultiViewRenderTask(self.cameras(),Reference(),0)
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            _,stats=full_scene_step(model,[(f[None],torch.ones(1,8,dtype=torch.long))],g,10,'none',task,attr_weight=0)
        self.assertGreater(stats['scene_gradient_norms']['xyz'],0)
        self.assertGreater(stats['scene_gradient_norms']['attributes'],0)

    def test_nonfinite_image_objective_stops(self):
        _,g,f,model=setup(8)
        task=MultiViewRenderTask(self.cameras(),Reference(),0)
        with patch('gaussian_jscc.rendering.render',return_value=torch.full((3,8,8),float('nan'))):
            with self.assertRaisesRegex(RuntimeError,'nonfinite image'):
                full_scene_step(model,[(f[None],torch.ones(1,8,dtype=torch.long))],g,10,'none',task,attr_weight=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_initializer_skips_bootstrap_and_selects_render_checkpoint(self):
        import sys
        from gaussian_jscc.cli import main
        from gaussian_jscc.data import write_ply
        from gaussian_jscc.transport import save_checkpoint,load_checkpoint
        raw,_,_,model=setup(16)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'input.ply',raw,0)
            save_checkpoint(root/'init.pt',model,2000,{'objective':'old_parameter_objective'})
            argv=['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--init',str(root/'init.pt'),'--source','mock','--device','cuda','--render-steps','2',
                  '--validation-views','1','--validation-trials','1','--validate-every','1']
            with patch.object(sys,'argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()+self.cameras()), \
                 patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),patch('gaussian_jscc.plots.safe_plot'):
                main()
            rows=[json.loads(line) for line in (root/'run'/'loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['render','render'])
            self.assertTrue((root/'run'/'codec_best_render.pt').exists())
            self.assertFalse((root/'run'/'route2.pt').exists())
            loaded=load_checkpoint(root/'run'/'codec.pt','cpu')
            self.assertEqual(loaded.cfg.rates,model.cfg.rates)

    def test_random_render_training_never_loads_checkpoint(self):
        from gaussian_jscc.cli import main
        from gaussian_jscc.data import write_ply
        raw,_,_,_=setup(16)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--source','mock','--device','cuda','--render-steps','2','--validation-views','1',
                  '--validation-trials','1','--validate-every','1','--hidden','16','--depth','1',
                  '--grid-dim','4','--levels','2','--block-size','8','--decoder-window','4']
            with patch('sys.argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=self.cameras()+self.cameras()), \
                 patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),patch('gaussian_jscc.plots.safe_plot'), \
                 patch('gaussian_jscc.learned_train.load_checkpoint',side_effect=AssertionError('random must not load')):
                main()
            record=json.loads((root/'run'/'training.json').read_text())
            self.assertEqual(record['initialization'],{'mode':'random','checkpoint':None,'seed':42,
                                                      'bootstrap_steps':0,'feature_statistics':'computed from input PLY'})
            rows=[json.loads(line) for line in (root/'run'/'loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['render','render'])
            self.assertTrue(all(r['aux_loss']==0 for r in rows))


if __name__=='__main__':
    unittest.main()
