import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from test_successive_joint import source_run, cameras as old_cameras
from test_mask_gradient import toy_mask_render


def cameras(*args, **kwargs):
    result = old_cameras(*args, **kwargs)
    if args[-1] != 'test':
        for i in range(4):
            result.append(SimpleNamespace(factor=1.5+i*.1, original_image=torch.full((3, 8, 8), .5),
                image_name='extra'+str(i), camera_center=torch.tensor([5., 6., 7.])))
    for camera in result: camera.source_count = 16
    return result


def render(raw, camera, degree, white_background=False, existence=None, colors=None):
    if colors is not None:
        value = colors.mean(0) if len(colors) else raw.new_zeros(3)
        return value[:, None, None].expand_as(camera.original_image)
    image = toy_mask_render(raw, camera, degree, white_background, existence)
    return F.interpolate(image[None], size=(8, 8), mode='nearest')[0]


class BudgetedEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            parent = source_run(cls.root)
            path = parent / 'checkpoints/final/codec.pt'
            saved = torch.load(path, weights_only=True)
            saved['training']['scene_record']['train_view_names'] += ['extra'+str(i) for i in range(4)]
            torch.save(saved, path)
            meta_path = parent / 'training.json'; meta = json.loads(meta_path.read_text())
            meta['scene'] = saved['training']['scene_record']; meta_path.write_text(json.dumps(meta))
            from gaussian_jscc.successive_joint import main, parser
            cls.source = cls.root / 'jointA'
            main(parser().parse_args(['--run', str(parent), '--out', str(cls.source), '--device', 'cpu',
                                      '--steps', '1', '--test-views', '2', '--render-trials', '1', '--no-lpips']))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def entry(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.budgeted_allocation'))
        from gaussian_jscc import budgeted_allocation
        return budgeted_allocation

    def options(self, out, budget=100000):
        return self.entry().parser().parse_args(['--run', str(self.source), '--scene', 'truck', '--out', str(out),
            '--budget-total-uses', str(budget), '--allocation-steps', '2', '--calibration-views', '1',
            '--window-views', '4', '--control-every', '1', '--validate-every', '1', '--save-every', '1',
            '--audit-every', '0', '--profile-every', '0', '--test-views', '2', '--trials', '1',
            '--image-views', '1', '--cpu-threads', '1', '--device', 'cpu', '--no-lpips'])

    def test_budget_required_and_preflight_never_renders_or_writes(self):
        entry = self.entry(); base = ['--run', str(self.source), '--scene', 'truck', '--out', str(self.root/'dry')]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): entry.parser().parse_args(base)
            with self.assertRaises(SystemExit):
                entry.parser().parse_args(base+['--budget-total-uses', '100', '--budget-ratio-to-q3', '.5'])
        args = self.options(self.root/'dry'); args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()), patch('gaussian_jscc.rendering.render', side_effect=AssertionError('preflight rendered')):
            plan = entry.main(args)
        self.assertFalse(Path(args.out).exists()); self.assertFalse(plan['writes_output'])
        self.assertEqual(plan['settings']['window_views'], 4)

    def test_actual_cost_matches_packet_accounting_and_caches_xyz(self):
        self.entry()
        from gaussian_jscc.budgeted_allocation_evaluation import ExactDeploymentCost
        from gaussian_jscc.modular_allocation import measured_cost
        from gaussian_jscc.transport import model_id
        from test_successive_allocation import small_successive
        raw, geometry, _, model = small_successive()
        order = torch.tensor([5, 2, 7, 0, 8, 1, 4, 3, 6, 9, 10, 11, 15, 14, 13, 12])
        scene = SimpleNamespace(raw=raw[order], geometry=geometry, order=order,
            group_ids=[order.reshape(2, 8)], net_bits_per_use=2.)
        meter = ExactDeploymentCost(model, scene, model_id(model), 10., 2.)
        q = torch.arange(16) % 4
        expected = measured_cost(model, scene, q[order], 10., model_id(model))
        result = meter.details(q)
        for key in ('xyz_bytes', 'metadata_bytes', 'scale_stream_bytes', 'payload_complex_symbols', 'total_channel_uses'):
            self.assertEqual(result[key], expected[key], key)
        other = torch.where(q>0, torch.full_like(q, 3), q)
        self.assertTrue(meter.details(other)['xyz_cache_hit'])
        empty = meter.details(torch.zeros_like(q)); self.assertEqual(empty['payload_complex_symbols'], 0)
        self.assertGreater(empty['total_channel_uses'], 0)

    def test_frozen_end_to_end_and_resume_match_table_price_coverage_and_maps(self):
        entry = self.entry()
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            full_out = self.root/'full'; result = entry.main(self.options(full_out))
            self.assertEqual(result['status'], 'complete'); self.assertTrue(result['budget_feasible'])
            final = torch.load(full_out/'checkpoints/latest/state.pt', weights_only=True)
            source = torch.load(self.source/'checkpoints/final/codec.pt', weights_only=True)
            for k, v in source['state_dict'].items(): torch.testing.assert_close(final['state_dict'][k], v, rtol=0, atol=0)
            record = json.loads((full_out/'training.json').read_text())
            self.assertFalse(set(record['allocation_train_view_names']) & set(record['calibration_view_names']))
            self.assertEqual(len(record['allocation_train_view_names']), 4)
            out = self.root/'resumed'; args = self.options(out)
            entry.main(args, max_updates=1); args.resume = True; entry.main(args)
            resumed = torch.load(out/'checkpoints/latest/state.pt', weights_only=True)
            for k, v in final['table'].items(): torch.testing.assert_close(v, resumed['table'][k], rtol=0, atol=0)
            self.assertEqual(final['controller'], resumed['controller'])
            torch.testing.assert_close(final['blocks']['counts'], resumed['blocks']['counts'], rtol=0, atol=0)
            torch.testing.assert_close(final['latest_q'], resumed['latest_q'], rtol=0, atol=0)
            self.assertEqual(final['budget_reference'], resumed['budget_reference'])
            rows = [json.loads(s) for s in (out/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows], [1, 2])
            self.assertTrue(all(r['views_per_step']==4 and r['codec_updated'] is False for r in rows))

    def test_infeasible_budget_is_reported_without_fake_feasible_export(self):
        entry = self.entry(); out = self.root/'unmet'
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            result = entry.main(self.options(out, 1))
        self.assertEqual(result['status'], 'budget_unmet'); self.assertFalse(result['budget_feasible'])
        self.assertFalse((out/'checkpoints/best_feasible/state.pt').exists())
        self.assertTrue((out/'checkpoints/lowest_violation/state.pt').exists())

    def test_shared_source_allows_trained_scene_and_rejects_heldout_or_incomplete(self):
        entry = self.entry()
        from test_successive_multiscene import source_and_manifest
        from gaussian_jscc import successive_multiscene
        root = self.root/'shared_fixture'; root.mkdir()
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            parent, manifest = source_and_manifest(root)
            source = root/'shared'
            successive_multiscene.main(successive_multiscene.parser().parse_args([
                '--run',str(parent),'--manifest',str(manifest),'--out',str(source),'--steps-per-scene','1',
                '--validation-views','1','--test-views','2','--render-trials','1','--cpu-threads','1','--device','cpu','--no-lpips']))
            args = self.options(root/'dry'); args.run=str(source); args.scene='train'; args.dry_run=True
            self.assertFalse(entry.main(args)['writes_output'])
            args.scene='drjohnson'
            with self.assertRaisesRegex(ValueError,'train|heldout'): entry.main(args)
            args.scene='train'; complete=source/'complete.json'; before=complete.read_bytes()
            complete.write_text(json.dumps(dict(status='interrupted')))
            try:
                with self.assertRaisesRegex(ValueError,'completed'): entry.main(args)
            finally: complete.write_bytes(before)
        self.assertFalse(Path(args.out).exists())

    def test_ratio_reference_is_floored_and_missing_side_rate_is_rejected(self):
        entry = self.entry(); out = self.root/'ratio'
        args=self.options(out); args.budget_total_uses=None; args.budget_ratio_to_q3=.873
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            entry.main(args,max_updates=1)
        state=torch.load(out/'checkpoints/latest/state.pt',weights_only=True)
        reference=state['budget_reference']
        self.assertEqual(reference['budget'],int(.873*reference['q3_cost']['total_channel_uses']))
        metadata=self.source/'training.json'; old=metadata.read_bytes()
        record=json.loads(old); record.pop('net_bits_per_use')
        metadata.write_text(json.dumps(record))
        try:
            args.dry_run=True
            with self.assertRaisesRegex(ValueError,'bits/use'): entry.main(args)
        finally: metadata.write_bytes(old)

    def test_pending_calibration_resumes_without_double_control_or_duplicate_rows(self):
        entry = self.entry(); out=self.root/'pending'; args=self.options(out)
        real=entry.evaluate_map
        def fail(*a,**kw):
            result=real(*a,**kw)
            if kw.get('label')=='calibration_000001': raise RuntimeError('simulated calibration interruption')
            return result
        with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
             patch('gaussian_jscc.rendering.render', side_effect=render), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(entry,'evaluate_map',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'calibration interruption'): entry.main(args)
            before=torch.load(out/'checkpoints/latest/state.pt',weights_only=True)
            self.assertEqual(before['progress']['pending'],['calibration'])
            self.assertEqual(before['controller']['observations'],1)
            args.resume=True; entry.main(args)
            final=torch.load(out/'checkpoints/latest/state.pt',weights_only=True)
        self.assertEqual(final['controller']['observations'],2)
        controls=[json.loads(s) for s in (out/'metrics/control.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in controls],[1,2])
        validations=[json.loads(s) for s in (out/'metrics/validation.jsonl').read_text().splitlines()]
        self.assertEqual(len({(r['step'],r['event']) for r in validations}),len(validations))


if __name__ == '__main__': unittest.main()
