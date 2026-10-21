import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


class JointStabilityAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def audit(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.joint_stability_audit'))
        from gaussian_jscc import joint_stability_audit
        return joint_stability_audit

    def test_large_derived_seed_keeps_torch_seed_and_maps_only_numpy_to_uint32(self):
        from gaussian_jscc.cli import seed_all
        try:
            seed_all(4295077352)
        except ValueError as exc:
            self.fail(f'joint step8590 must not abort: {exc}')
        self.assertEqual(torch.initial_seed(), 4295077352)
        np.testing.assert_array_equal(np.random.rand(3), np.random.RandomState(110056).rand(3))
        seed_all(42)
        np.testing.assert_array_equal(np.random.rand(3), np.random.RandomState(42).rand(3))

    def test_attribute_profile_keeps_source_ids_and_separates_scale_and_rotation(self):
        audit = self.audit()
        raw = torch.zeros(3, 14); raw[:, 7] = 1
        gradient = torch.zeros_like(raw); gradient[1, 4] = 12; gradient[2, 7] = 5
        result = audit.gradient_profile(raw, gradient, torch.tensor([10, 22, 35]), top_k=2)
        self.assertEqual(result['groups']['scale']['norm'], 12.)
        self.assertEqual(result['groups']['rotation']['norm'], 5.)
        self.assertEqual(result['groups']['attributes']['norm'], 13.)
        self.assertEqual([r['source_id'] for r in result['top_points']], [22, 35])
        self.assertEqual(result['top_points'][0]['log_scale'], [0., 0., 0.])

    def test_instrumentation_preserves_real_joint_gradient_and_adam_update(self):
        audit = self.audit()
        from test_modular_training import setup
        from test_render_first import Reference, RenderFirstTests, synthetic_render
        from gaussian_jscc.modular_training import configure_phase, bootstrap_step
        from gaussian_jscc.power_followup import joint_optimizer, joint_update
        model, geometry, features, tiers = setup()
        old = configure_phase(model, 'channel_bootstrap', {}, freeze_semantic=True)
        bootstrap_step(model, features, tiers, geometry, 10., 'awgn', 'channel_bootstrap'); old.step()
        reference = copy.deepcopy(model)
        other_old = configure_phase(reference, 'channel_bootstrap', {}, freeze_semantic=True)
        other_old.load_state_dict(copy.deepcopy(old.state_dict()))
        optimizer = joint_optimizer(model, old, 1e-4, 1e-5)
        other = joint_optimizer(reference, other_old, 1e-4, 1e-5)
        scene = SimpleNamespace(groups=[features[:1], features[1:]],
            group_ids=[torch.arange(8)[None], torch.arange(8, 16)[None]],
            geometry=geometry, cameras=RenderFirstTests().cameras(), views_per_step=2,
            reference=Reference(), degree=1, white_background=False)
        args = SimpleNamespace(seed=42, top_k=3)
        with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
            reference.train(); reference.semantic_encoder.eval(); other.zero_grad(set_to_none=True)
            loss, stats = joint_update(reference, scene, 8001, 1, args)
            before = {n: p.grad.clone() for n, p in reference.named_parameters() if p.grad is not None}
            other.step()
            record = audit.audit_joint_step(model, optimizer, scene, 8001, 1, args, instrumented=True)
        self.assertAlmostEqual(record['before']['loss'], float(loss), places=6)
        self.assertLess(record['decomposition']['gradient_sum_relative_error'], 1e-4)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, dict(reference.named_parameters())[name], atol=2e-7, rtol=2e-5, msg=name)
            if name in before:
                torch.testing.assert_close(value.grad, before[name], atol=1e-6, rtol=2e-4, msg=name)
        self.assertEqual(record['training_view_indices'], stats['training_view_indices'])
        self.assertTrue(np.isfinite(record['after']['loss']))
        self.assertEqual(set(record['decomposition']['parameter_norms']), {'mse', 'ssim', 'latent'})
        self.assertGreater(record['updates']['channel_decoder']['update_norm'], 0)
        self.assertEqual(len(record['raw_render_gradient']['top_points']), 3)

    def test_pairing_report_failure_does_not_abort_independent_probe_jobs(self):
        from gaussian_jscc import power_stability_controls as suite
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root/'out'
            args = suite.parser().parse_args(['--power-run', str(root/'power'), '--followup', str(root/'source'),
                '--out', str(out), '--resume-suite'])
            checkpoint = out/'block_continuation/block/checkpoints/end_channel_bootstrap/codec.pt'
            checkpoint.parent.mkdir(parents=True); checkpoint.touch()
            plan = dict(block_options=[], point_checkpoint=str(root/'point.pt'), point_run=str(root/'point'))
            (out/'plan.json').write_text(json.dumps(plan), encoding='utf-8')
            with patch.object(suite, 'build_plan', return_value=plan), \
                    patch.object(suite.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                    patch.object(suite, 'paired_endpoints', side_effect=ValueError('unpaired endpoint')), \
                    patch.object(suite, 'summarize_probes', return_value=[]):
                try:
                    result = suite.main(args)
                except ValueError as exc:
                    self.fail(f'comparison error must not abort probe scheduling: {exc}')
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(len(list(out.glob('*plain.log'))), 2)
            self.assertEqual(len(list(out.glob('*instrumented.log'))), 2)


if __name__ == '__main__':
    unittest.main()
