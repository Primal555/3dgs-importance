"""Task-aware channel loss: real decoder/response gradients and exact resume."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class ChannelResponseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.channel_response_training'),
                             'task-aware channel training entry is missing')
        from gaussian_jscc import channel_response_training
        return channel_response_training

    def fixture(self):
        from test_modular_training import setup
        from gaussian_jscc.modular_training import configure_phase
        m, g, f, q = setup()
        m.set_scene_geometry(g)
        configure_phase(m, 'channel_bootstrap', {})
        c = self.entry()
        teacher = c.frozen_teacher(m)
        with torch.no_grad(): y = m.semantic_encoder(f, f[..., :3], q > 0)
        return c, m, teacher, g, f, q, y

    def test_response_keeps_teacher_fixed_but_backpropagates_into_received_latent(self):
        c, m, teacher, g, f, q, y = self.fixture()
        from gaussian_jscc.position_delivery import delivered_positions
        xyz = delivered_positions(f[..., :3], q, m.cfg)
        predicted = y.clone().requires_grad_()
        target = y.clone().requires_grad_()
        directions = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        loss, _ = c.response_loss(m, teacher, target, predicted, q, xyz, g, directions)
        self.assertLess(float(loss.detach()), 1e-11)
        shifted = (y + torch.linspace(-.2, .2, y.shape[-1])).requires_grad_()
        loss, _ = c.response_loss(m, teacher, target, shifted, q, xyz, g, directions)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertGreater(float(shifted.grad.norm()), 0)
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        before = float(loss.detach())
        with torch.no_grad():
            for p in m.semantic_decoder.parameters(): p.add_(1)
        after = c.response_loss(m, teacher, y, shifted.detach(), q, xyz, g, directions)[0]
        self.assertAlmostEqual(before, float(after), places=8)

    def test_gradient_calibration_has_hand_computed_ratio_and_rejects_degenerate_loss(self):
        c = self.entry()
        v = torch.tensor([1., 2.], requires_grad=True)
        record = c.gradient_calibration(v.square().mean(), 4*v.square().mean(), v)
        self.assertAlmostEqual(record['response_weight'], .25, places=7)
        self.assertIsNone(v.grad)
        v = torch.ones(2, requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            c.gradient_calibration(v.square().mean(), (v*0).sum(), v)

    def test_response_channel_step_updates_both_channel_ends_not_semantic_modules(self):
        c, m, teacher, g, f, q, y = self.fixture()
        from gaussian_jscc.position_delivery import delivered_positions
        xyz = delivered_positions(f[..., :3], q, m.cfg)
        recovered = c.transmit(m, y, q, xyz, 17)
        loss, stats = c.channel_loss(m, teacher, y, recovered, q, xyz, g,
                                     c.directions(3, 21, y.device), .5)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), stats['latent_mse']+.5*stats['response_mse'], places=7)
        for name in ('channel_encoder', 'channel_decoder'):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(m, name).parameters()
                                   if p.grad is not None), 0)
        for name in ('semantic_encoder', 'semantic_decoder'):
            self.assertTrue(all(p.grad is None for p in getattr(m, name).parameters()))

    def test_padded_points_never_change_the_loss(self):
        c,m,teacher,g,f,q,y=self.fixture()
        from gaussian_jscc.position_delivery import delivered_positions
        q[:, -1]=0
        xyz=delivered_positions(f[...,:3],q,m.cfg)
        altered=y+.1
        a=c.channel_loss(m,teacher,y,altered,q,xyz,g,c.directions(2,17,y.device),.5)[0]
        y=y.clone(); altered=altered.clone()
        y[:, -1]=100; altered[:, -1]=-100
        b=c.channel_loss(m,teacher,y,altered,q,xyz,g,c.directions(2,17,y.device),.5)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_complete_and_exact_resume_freeze_both_semantic_modules(self):
        c = self.entry()
        import test_modular_experiment as fixture
        import test_render_first as renders
        from gaussian_jscc.modular_experiment import main as train_teacher
        from gaussian_jscc.channel_normalization_controls import parser as power_parser, run_arm
        def cameras(*args, **kwargs):
            result = renders.RenderFirstTests().cameras()
            if args[-1]=='test':
                for i, camera in enumerate(result): camera.image_name='test_'+str(i)
            return result
        with tempfile.TemporaryDirectory() as t, \
             patch('gaussian_jscc.rendering.render', side_effect=renders.synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
            root=Path(t); source=fixture.source_fixture(root)
            args=fixture.ModularExperimentTests().args(source, root/'teacher')
            args.channel_variant='progressive'; args.channel_hidden=16; args.rates=[8,24,48]
            args.context_variant='both'; args.context_centers=[4,2]; args.context_neighbors=[4,4]
            args.channel_bootstrap_steps=args.channel_render_steps=0
            train_teacher(args)
            power=power_parser().parse_args(['--run',str(root/'teacher'),'--out',str(root/'power'),
                '--steps','3','--device','cpu','--save-every','1','--validate-every','3',
                '--diagnostic-blocks','1','--response-blocks','1','--trials','2','--render-trials','1','--no-lpips'])
            run_arm(power,'block_awgn10')
            parent=root/'power/block_awgn10'; endpoint=parent/'checkpoints/end_channel_bootstrap/codec.pt'
            endpoint.parent.mkdir(); shutil.copyfile(parent/'checkpoints/final/codec.pt',endpoint)
            original=endpoint.read_bytes()
            args=c.parser().parse_args(['--run',str(parent),'--out',str(root/'full'),
                '--steps','4','--calibration-batches','2','--response-directions','2',
                '--device','cpu','--validate-every','2','--save-every','1','--test-views','2','--render-trials','1','--no-lpips'])
            result=c.main(args)
            self.assertEqual(result['status'],'complete')
            a=torch.load(root/'full/checkpoints/final/codec.pt',weights_only=True)
            rows=[json.loads(s) for s in (root/'full/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['channel_response']*4)
            self.assertEqual([r['layout'] for r in rows],['1','2','3','mixed'])
            for row in rows:
                self.assertTrue(row['response_mse']>=0)
                self.assertAlmostEqual(row['loss'],row.get('render_loss',0)+row['latent_mse']+
                                       row['response_weight']*row['response_mse'], places=5)
            self.assertEqual(len(set(r['response_weight'] for r in rows)),1)
            end=torch.load(root/'full/checkpoints/end_channel_bootstrap/codec.pt',weights_only=True)
            teacher=torch.load(endpoint,weights_only=True)
            for name,value in teacher['state_dict'].items():
                if name.startswith(('semantic_encoder.','semantic_decoder.')):
                    torch.testing.assert_close(value,end['state_dict'][name],rtol=0,atol=0)
                    torch.testing.assert_close(value,a['state_dict'][name],rtol=0,atol=0)
            args.out=str(root/'resumed')
            c.main(args,max_updates=2)
            args.resume=True; c.main(args)
            b=torch.load(root/'resumed/checkpoints/final/codec.pt',weights_only=True)
            for name,value in a['state_dict'].items():
                torch.testing.assert_close(value,b['state_dict'][name],rtol=0,atol=0)
            self.assertEqual(a['training']['progress']['response_weight'],b['training']['progress']['response_weight'])
            for key, state in a['training']['optimizer']['state'].items():
                for field,value in state.items():
                    torch.testing.assert_close(value,b['training']['optimizer']['state'][key][field],rtol=0,atol=0)
            args.out=str(root/'pending'); args.resume=False
            evaluator=c.evaluate_saved
            def interrupted_evaluation(*values,**options):
                if options['label']=='response_000002': raise RuntimeError('evaluation interrupted')
                return evaluator(*values,**options)
            with patch.object(c,'evaluate_saved',side_effect=interrupted_evaluation):
                with self.assertRaisesRegex(RuntimeError,'evaluation interrupted'): c.main(args)
            args.resume=True; c.main(args)
            repaired=torch.load(root/'pending/checkpoints/final/codec.pt',weights_only=True)
            for name,value in a['state_dict'].items():
                torch.testing.assert_close(value,repaired['state_dict'][name],rtol=0,atol=0)
            self.assertEqual(endpoint.read_bytes(),original)
            self.assertTrue((root/'full/checkpoints/end_channel_bootstrap/codec.pt').exists())
            self.assertFalse((root/'full/evaluation/response_000002/images').exists())
            self.assertTrue((root/'full/evaluation/final/images/final').exists())


if __name__=='__main__': unittest.main()
