import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gaussian_jscc.axis_position import fit_axis_floor, teacher_axis_position
from gaussian_jscc.covariance import pack_symmetric
from gaussian_jscc.data import Geometry, write_ply
from gaussian_jscc.representation_train import train, check_args
from gaussian_jscc.representation_validation import objective
from gaussian_jscc.spatial_response import spatial_response_loss
from gaussian_jscc.transport import load_checkpoint
from test_representation_codec import setup, args_for


def analytic(scales=(.1, .2, .4), delta=(.1, 0., 0.), rotation=None, span=(1., 1., 1.)):
    dtype = torch.float64
    axes = torch.eye(3, dtype=dtype) if rotation is None else rotation.to(dtype)
    source = torch.zeros(1, 13, dtype=dtype)
    source[:, 4:10] = pack_symmetric(axes @ torch.diag(2*torch.tensor(scales, dtype=dtype).log()) @ axes.T)
    geometry = Geometry([0, 0, 0], span, 16)
    pred = source.clone()
    pred[:, :3] = torch.as_tensor(delta, dtype=dtype)/geometry.span.double()
    model = SimpleNamespace(cfg=SimpleNamespace(architecture='learned_split_logcov'),
                            attr_std=torch.ones(10, dtype=dtype), attr_mean=torch.zeros(10, dtype=dtype))
    return pred.requires_grad_(), source.requires_grad_(), geometry, model


class TeacherAxisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_analytical_anisotropic_loss(self):
        p, t, g, m = analytic()
        loss, stats = teacher_axis_position(p, t, g, m, .001)
        self.assertAlmostEqual(float(loss.detach()), math.sqrt(2)-1, places=10)
        loss.backward()
        self.assertAlmostEqual(float(p.grad[0, 0]), 10/math.sqrt(2), places=9)
        self.assertIsNone(t.grad)
        self.assertEqual(float(p.grad[:, 3:].abs().sum()), 0)
        self.assertAlmostEqual(stats['axis_native_radius_p50'], 1.)
        p2, t2, g2, m2 = analytic(delta=(0., 0., .1))
        looser, _ = teacher_axis_position(p2, t2, g2, m2, .001)
        self.assertLess(float(looser.detach()), float(loss.detach()))

    def test_rotation_and_axis_sign_invariance(self):
        angle = .67
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
                                 [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float64)
        delta = torch.tensor([.12, -.25, .3], dtype=torch.float64)
        expected, _ = teacher_axis_position(*analytic(delta=delta), .001)
        for r in (rotation, rotation @ torch.diag(torch.tensor([-1., 1., -1.], dtype=torch.float64))):
            actual, _ = teacher_axis_position(*analytic(delta=rotation @ delta, rotation=r), .001)
            torch.testing.assert_close(actual, expected)

    def test_zero_and_repeated_eigenvalues(self):
        for delta in ((0., 0., 0.), (.1, -.2, .3)):
            p, t, g, m = analytic(scales=(.2, .2, .2), delta=delta)
            loss, _ = teacher_axis_position(p, t, g, m, .001)
            loss.backward()
            self.assertTrue(torch.isfinite(p.grad).all())
            if not any(delta):
                self.assertEqual(float(loss.detach()), 0)
                self.assertEqual(float(p.grad.abs().sum()), 0)

    def test_floor_limits_xyz_not_parameter_gradient(self):
        p, t, g, m = analytic(scales=(1e-8, 1e-5, .1), delta=(10., 2., 1.))
        loss, stats = teacher_axis_position(p, t, g, m, .01)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertLessEqual(float(p.grad[:, :3].norm()), 100.+1e-8)
        self.assertAlmostEqual(stats['axis_clamped_fraction'], 2/3)
        self.assertGreater(stats['axis_native_radius_p50'], stats['axis_effective_radius_p50'])

    def test_world_units_and_bbox_do_not_reweight_loss(self):
        expected, _ = teacher_axis_position(*analytic(delta=(.1, .2, .3)), .01)
        changed_bbox, _ = teacher_axis_position(*analytic(delta=(.1, .2, .3), span=(100, 200, 300)), .01)
        torch.testing.assert_close(changed_bbox, expected)
        scaled, _ = teacher_axis_position(*analytic(scales=(1, 2, 4), delta=(1, 2, 3), span=(10, 10, 10)), .1)
        torch.testing.assert_close(scaled, expected)

    def test_prediction_shape_cannot_relax_position_objective(self):
        p, t, g, m = analytic()
        loss, _ = teacher_axis_position(p, t, g, m, .01)
        modified = p.detach().clone()
        modified[:, 3:] += 30
        changed, _ = teacher_axis_position(modified, t, g, m, .01)
        torch.testing.assert_close(loss, changed)

    def test_fixed_floor_fit_and_validation(self):
        raw = torch.zeros(2, 14)
        raw[:, 4:7] = torch.tensor([[.001, .002, .003], [.004, .005, .006]]).log()
        info = fit_axis_floor(raw, 50)
        self.assertAlmostEqual(info['world'], .0035, places=8)
        self.assertEqual(info['clamped_axis_fraction'], .5)
        self.assertEqual(info['affected_point_fraction'], .5)
        explicit = fit_axis_floor(raw, 1, .0045)
        self.assertEqual(explicit['world'], .0045)
        self.assertEqual(explicit['method'], 'explicit_world')
        for kw in ({'percentile': -1}, {'percentile': float('nan')}, {'override': 0}, {'override': float('inf')}):
            with self.assertRaises(ValueError):
                fit_axis_floor(raw, **kw)

    def test_replaces_old_position_preserves_attribute_gradient(self):
        _, g, f, m = setup()
        p = (f+torch.randn_like(f)*.05).requires_grad_()
        args = args_for('unused', 'unused')
        directions = torch.randn(4, 3)
        old, _, old_terms = spatial_response_loss(p, f, g, m, directions=directions, return_components=True)
        default, _ = objective(p, f, g, m, args, directions)
        torch.testing.assert_close(default, old, atol=0, rtol=0)
        args.position_objective = 'teacher-axis'
        args.axis_floor_world = .01
        new, stats, terms = objective(p, f, g, m, args, directions, return_components=True)
        position, _ = teacher_axis_position(p, f, g, m, .01)
        torch.testing.assert_close(new, (position/3+old_terms['shape']+old_terms['appearance']).to(new))
        old_grad, = torch.autograd.grad(old, p, retain_graph=True)
        new_grad, = torch.autograd.grad(new, p, retain_graph=True)
        position_grad, = torch.autograd.grad(position/3, p)
        torch.testing.assert_close(new_grad[:, 3:], old_grad[:, 3:], atol=0, rtol=0)
        torch.testing.assert_close(new_grad[:, :3], position_grad[:, :3])
        self.assertIn('legacy_scene_position_diagnostic', stats)
        self.assertNotIn('spatial_fine_contribution', stats)
        self.assertEqual(set(terms), {'position', 'shape', 'appearance'})

    def test_args_and_old_resume_defaults(self):
        args = args_for('unused', 'unused')
        args.position_objective = 'teacher-axis'
        with self.assertRaisesRegex(ValueError, 'representation-only'):
            check_args(args)
        args.adapter_steps = args.joint_steps = 0
        check_args(args)
        old = copy.deepcopy(args)
        for key in ('position_objective', 'axis_floor_world', 'axis_floor_percentile'):
            delattr(old, key)
        check_args(old)
        self.assertEqual(old.position_objective, 'scene-scale')

    def test_trainer_diagnostics_frozen_channel_and_exact_resume(self):
        raw, _, _, _ = setup(40)
        options = ['--adapter-steps', '0', '--joint-steps', '0', '--position-objective', 'teacher-axis',
                   '--representation-steps', '3']
        with tempfile.TemporaryDirectory() as tmp, patch('gaussian_jscc.representation_plots.plot_run'):
            root = Path(tmp)
            write_ply(root/'source.ply', raw, 0)
            full, stopped = root/'full', root/'stopped'
            train(args_for(root/'source.ply', full, options))
            actual_step = torch.optim.Adam.step
            calls = [0]
            def interrupted(opt, *a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('test interruption')
                return actual_step(opt, *a, **kw)
            with patch.object(torch.optim.Adam, 'step', interrupted):
                with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                    train(args_for(root/'source.ply', stopped, options))
            resume = args_for(root/'source.ply', stopped, options)
            resume.resume = str(stopped/'training_state.pt')
            train(resume)
            a, b = load_checkpoint(full/'codec.pt', 'cpu'), load_checkpoint(stopped/'codec.pt', 'cpu')
            initial = load_checkpoint(full/'codec_0.pt', 'cpu')
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
                if name.startswith(('learned.channel_encoder.', 'learned.channel_decoder.')):
                    torch.testing.assert_close(value, initial.state_dict()[name], atol=0, rtol=0)
            log = json.loads((full/'loss.jsonl').read_text().splitlines()[0])
            self.assertEqual(set(log['objective_module_grad_norms']), {'position', 'shape', 'appearance'})
            self.assertEqual(log['clip_factor'], 1.)
            self.assertGreater(log['objective_module_grad_norms']['position']['representation_decoder'], 0)
            state = torch.load(stopped/'training_state.pt', weights_only=True)
            config = json.loads((full/'training.json').read_text())
            self.assertEqual(state['axis_floor'], config['axis_floor'])
            self.assertEqual(state['axis_floor']['method'], 'source_axis_percentile')


if __name__ == '__main__':
    unittest.main()
