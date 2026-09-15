"""CPU contracts for the paired gradient experiment; no CUDA claims."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.utils.rnn import pad_sequence

from gaussian_jscc.data import prepare, to_features, write_ply
from gaussian_jscc.gradient_diagnostics import GradientProbe, main, evaluate_render, make_plots
from gaussian_jscc.losses import reconstruction_loss
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.transport import save_checkpoint, load_checkpoint, model_id
from test_gaussian_jscc import fixture


class GradientDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_detach_preserves_forward_and_direct_geometry_gradient(self):
        raw, model = fixture(n=17,degree=0)
        raw, geom, _ = prepare(raw,16)
        f,_ = to_features(raw,geom,model)
        q=torch.full((len(f),),3,dtype=torch.long)
        outputs, xyz_grads, attr_grads=[],[],[]
        identity=model_id(model)
        for detach in (False,True):
            model.detach_attribute_context_xyz=detach
            captured=[]
            hook=model.position_seed.register_forward_hook(lambda m,i,o: captured.append(o))
            pred=model(f,f[:,:3],q,10.,'none')
            outputs.append(pred.detach())
            xyz_grads.append(torch.autograd.grad(pred[:,:3].square().mean(),captured[0],retain_graph=True)[0])
            grad=torch.autograd.grad(pred[:,3:].square().mean(),captured[0],allow_unused=True)[0]
            attr_grads.append(torch.zeros_like(captured[0]) if grad is None else grad)
            hook.remove()
            self.assertEqual(model_id(model),identity)
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
        torch.testing.assert_close(xyz_grads[0],xyz_grads[1],rtol=0,atol=0)
        self.assertGreater(attr_grads[0].norm().item(),0)
        self.assertEqual(attr_grads[1].norm().item(),0)

    def test_observer_is_non_mutating_and_replay_components_sum(self):
        raw,model=fixture(n=19,degree=0)
        raw,geom,_=prepare(raw,16)
        f,_=to_features(raw,geom,model)
        blocks=list(f.split(8)); qs=[torch.arange(len(b))%3+1 for b in blocks]
        batches=[(pad_sequence(blocks[:2],batch_first=True),pad_sequence(qs[:2],batch_first=True)),
                 (blocks[2][None],qs[2][None])]
        for detached in (False,True):
            model.detach_attribute_context_xyz=detached
            outputs=[]
            for enabled in (False,True):
                torch.manual_seed(101); model.zero_grad(set_to_none=True)
                probe=GradientProbe(model) if enabled else None
                loss,_=full_scene_step(model,batches,geom,10.,'awgn',lambda x:x.square().mean(),
                                       attr_weight=1.,gradient_observer=probe)
                outputs.append((loss,[None if p.grad is None else p.grad.clone() for p in model.parameters()],torch.rand(3)))
                if probe:
                    report=probe.report()
                    self.assertLess(report['component_sum_relative_error'],1e-5)
                    self.assertGreater(report['render']['groups']['geometry_decoder'],0)
                    if detached:
                        self.assertLess(report['attributes']['groups']['geometry_decoder'],1e-7)
            torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=0,atol=0)
            torch.testing.assert_close(outputs[0][2],outputs[1][2],rtol=0,atol=0)
            for a,b in zip(outputs[0][1],outputs[1][1]):
                self.assertEqual(a is None,b is None)
                if a is not None:
                    torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)

    def test_cpu_paired_cli_outputs_and_original_checkpoint(self):
        raw,model=fixture(n=19,degree=0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ply=root/'source.ply'; ckpt=root/'codec.pt'; out=root/'experiment'
            write_ply(ply,raw,0); save_checkpoint(ckpt,model,10)
            original=ckpt.read_bytes()
            main(['--ply',str(ply),'--checkpoint',str(ckpt),'--out',str(out),
                  '--device','cpu','--training-data-device','cpu','--steps','2','--render-steps','0',
                  '--test-views','0','--eval-blocks','1','--eval-snrs','10','--eval-every','1',
                  '--probe-every','1','--no-plots'])
            self.assertEqual(original,ckpt.read_bytes())
            summary=json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['paired_schedule_verified'])
            for variant in ('attached','detached'):
                logs=[json.loads(x) for x in (out/variant/'loss.jsonl').read_text().splitlines()]
                self.assertEqual(len(logs),2)
                self.assertIn('updates',logs[0])
                self.assertLess(logs[0]['component_gradients']['component_sum_relative_error'],1e-5)
                saved=load_checkpoint(out/variant/'codec.pt',torch.device('cpu'))
                self.assertFalse(saved.detach_attribute_context_xyz)
                self.assertEqual(saved.cfg.to_dict(),model.cfg.to_dict())
            self.assertTrue((out/'fixed_evaluation.csv').exists())
            try:
                import matplotlib
            except ImportError:
                return  # Core test still runs without optional plotting support.
            logs={v:[json.loads(s) for s in (out/v/'loss.jsonl').read_text().splitlines()]
                  for v in ('attached','detached')}
            ev={v:json.loads((out/v/'evaluation.json').read_text()) for v in logs}
            make_plots(out,logs,ev)
            self.assertTrue((out/'diagnostics.png').exists())
            self.assertTrue((out/'gradient_components.png').exists())

    def test_full_scene_fixed_render_uses_all_rows_and_writes_panels(self):
        raw,model=fixture(n=19,degree=0)
        raw,geom,_=prepare(raw,16)
        f,_=to_features(raw,geom,model)
        args=SimpleNamespace(blocks_per_batch=2,seed=42,render_eval_tier=2,
                             render_eval_snr=10.,white_background=False)
        camera=SimpleNamespace(image_name='fixed',original_image=torch.zeros(3,8,8))
        sizes=[]
        def fake_render(scene,*args):
            sizes.append(len(scene))
            return scene[:,:3].mean(0).sigmoid()[:,None,None].expand(3,8,8)
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.rendering.render',side_effect=fake_render):
            out=Path(tmp)/'test'
            result=evaluate_render(model,list(f.split(8)),raw,geom,[camera],0,args,out)
            self.assertIn('received_vs_reference_psnr',result)
            self.assertTrue((out/'00000.png').exists())
            self.assertTrue((out/'metrics.json').exists())
            self.assertEqual(set(sizes),{19})


if __name__=='__main__':
    unittest.main()
