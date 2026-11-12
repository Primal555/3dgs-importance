import contextlib
import io
import json
import tarfile
from unittest.mock import patch
import pytest
import torch
from gaussian_jscc.student_keep_feedback import sampling_probabilities, corrected_signal
from gaussian_jscc.student_keep_controls import losses_for, heldout_update_summary


def test_residual_expectation_matches_dense_exact_shared_gradient():
    # Enumerate every possible one-draw outcome: no Monte Carlo tolerance.
    theta = torch.tensor([.2, -.4], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[1., 2.], [-3., 1.], [.5, .7]], dtype=torch.float64)
    probability = (x@theta).sigmoid()
    approx = torch.tensor([.01, -.02, .05], dtype=torch.float64)
    truth = torch.tensor([-.04, .03, .006], dtype=torch.float64)
    sampling = torch.tensor([.2, .3, .5], dtype=torch.float64)
    expected = 0.
    for i in range(3):
        signal = corrected_signal(approx, truth[i:i+1], torch.tensor([i]), torch.ones(1), sampling[i:i+1], 1, 'weighted_residual')
        expected = expected+sampling[i]*(probability*signal).sum()
    gradient, = torch.autograd.grad(expected, theta, retain_graph=True)
    true_gradient, = torch.autograd.grad((probability*truth).sum(), theta)
    torch.testing.assert_close(gradient, true_gradient, atol=1e-15, rtol=1e-14)


def test_duplicate_draws_and_component_units_are_preserved():
    base = torch.tensor([[1., 5.], [2., 6.], [3., 7.]], dtype=torch.float64)
    truth = torch.tensor([[10., 50.], [30., 70.]], dtype=torch.float64)
    ids = torch.tensor([0, 2]); counts = torch.tensor([3, 1]); s = torch.tensor([.25, .5], dtype=torch.float64)
    result = corrected_signal(base, truth, ids, counts, s, 4, 'weighted_residual')
    torch.testing.assert_close(result[0], base[0]+3*(truth[0]-base[0]))
    torch.testing.assert_close(result[2], base[2]+.5*(truth[1]-base[2]))
    torch.testing.assert_close(result[1], base[1])
    replaced = corrected_signal(base, truth, ids, counts, s, 4, 'replacement')
    torch.testing.assert_close(replaced[ids], truth)


def test_sampling_has_full_support_for_retained_and_dropped_points():
    p = torch.tensor([0., .25, .5, .75, 1.])
    influence = torch.tensor([0., 1., 4., 1., 0.])
    s = sampling_probabilities(p, influence)
    assert (s >= .5/5).all() and s[2] == s.max()
    assert float(s.sum()) == pytest.approx(1.)
    torch.testing.assert_close(sampling_probabilities(p, influence*0), torch.full((5,), .2, dtype=torch.float64))


def test_holdout_uses_keep_minus_drop_sign_for_deletion_and_restoration():
    before = (torch.tensor([[0., 1.], [1., 0.]]), torch.zeros(2, 3))
    after = (before[0].flip(1), before[1])
    labels = [dict(source_row=i, exact_keep_cost=v, approximate_keep_cost=-v, population=10)
              for i,v in enumerate([2., -3.])]
    result = heldout_update_summary(labels, before, after)
    assert result['hard_keep_update_exact']['population_sum_estimate'] == -25.
    assert result['sign_error_fraction'] == 1.


def test_two_arms_resume_match_and_report_only_original_test_views(source):
    from gaussian_jscc import student_policy_experiment as student
    from gaussian_jscc import structured_evidence_allocation as teacher_entry
    from gaussian_jscc import student_keep_controls as controls
    from gaussian_jscc.student_policy_feedback import policy_surrogate
    from test_student_policy_experiment import args_for, capture_view
    from test_structured_allocation_integration import options
    import test_budgeted_allocation as fixture
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch('gaussian_jscc.student_policy_features.capture_view', side_effect=capture_view), \
         contextlib.redirect_stdout(io.StringIO()):
        teacher = root/'teacher'; teacher_entry.main(options(parent, teacher))
        source_run = root/'student'
        student.main(args_for(parent, teacher, source_run, '--distill-target', 'soft_deployment',
            '--teacher-weight-schedule', 'constant', '--cost-protocol', 'digital_side_file_v1'))
        original = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in source_run.rglob('*') if p.is_file()}
        def cli(out):
            return controls.parser().parse_args(['--source', str(source_run), '--out', str(out), '--checkpoint', 'latest',
                '--device', 'cpu', '--steps', '2', '--keep-draws', '3', '--holdout-draws', '2',
                '--evaluate-every', '1', '--diagnostic-every', '1'])
        # Check that separating loss components preserves the actual summed surrogate.
        original_losses = controls.losses_for
        from gaussian_jscc.student_keep_feedback import KeepResidualFeedback
        from gaussian_jscc.budgeted_allocation_kernel import raw_batches, set_rng
        from gaussian_jscc.optimization import preserved_rng
        original_measure = KeepResidualFeedback._measure
        def check_measure(self, model, features, id_batches, cache, geometry, task, scene_tensor,
                          points, keep_tiers, execution, snr, kind, d_ref):
            before_scene = scene_tensor.detach().clone()
            labels = original_measure(self, model, features, id_batches, cache, geometry, task, scene_tensor,
                                      points, keep_tiers, execution, snr, kind, d_ref)
            # Independent full original-batch decode verifies row offsets, padded/nonsequential
            # point IDs, paired noise and semantic re-encoding for keep-set changes.
            with torch.no_grad():
                for point in points.tolist():
                    batch, row, col, _ = self.lookup[point].tolist()
                    q, _, _, _, rng = cache[batch]
                    f = features[batch].to(scene_tensor.device); alt = q.clone()
                    alt[row, col] = labels[point]['new']
                    with preserved_rng(f.device):
                        set_rng(rng, f.device)
                        pred = model.forward_tier_batches(f, f[..., :3], torch.nn.functional.one_hot(alt, 4).to(f),
                            snr, kind, paired_noise=True)[0]
                    raw = raw_batches(pred, geometry, model); valid = id_batches[batch].to(f.device) >= 0
                    offset = sum(int((x >= 0).sum()) for x in id_batches[:batch]); count = int(valid.sum())
                    changed = before_scene.clone()
                    changed[offset:offset+count] = torch.cat((raw[valid], (alt[valid]>0).to(raw)[:, None]), -1)
                    expected = task.delta_components(changed)/d_ref*labels[point]['sign']
                    torch.testing.assert_close(labels[point]['parts'], expected.cpu(), atol=1e-8, rtol=2e-4)
            torch.testing.assert_close(scene_tensor, before_scene, atol=0, rtol=0)
            return labels
        def check_losses(heads, keep, ids, costs, inverse, sampler):
            pieces = original_losses(heads, keep, ids, costs, inverse, sampler)
            combined = policy_surrogate(*heads, keep, ids, costs, inverse)[0]
            torch.testing.assert_close(sum(pieces.values()), combined)
            assert set(sampler.last_correction['ids'].tolist()).isdisjoint(r['source_row'] for r in sampler.holdout)
            return pieces
        with patch.object(controls, 'losses_for', side_effect=check_losses), \
             patch.object(KeepResidualFeedback, '_measure', check_measure):
            full = root/'full'; assert controls.main(cli(full))['status'] == 'complete'
            resumed = root/'resumed'; args = cli(resumed)
            assert controls.main(args, max_updates=1)['status'] == 'interrupted'
            args.resume = True; assert controls.main(args)['status'] == 'complete'
        assert original == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in source_run.rglob('*') if p.is_file()}
        source_plan = json.loads((source_run/'plan.json').read_text())
        for mode in controls.ARMS:
            a = torch.load(full/mode/'checkpoints/latest/student.pt', weights_only=True)
            b = torch.load(resumed/mode/'checkpoints/latest/student.pt', weights_only=True)
            for key in a['student']: torch.testing.assert_close(a['student'][key], b['student'][key], atol=0, rtol=0)
            torch.testing.assert_close(a['rng']['torch'], b['rng']['torch'])
            rows = [json.loads(l) for l in (full/mode/'metrics/validation.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows] == [0, 1, 2]
            for r in rows:
                assert {v['view'] for v in r['per_view_trial']} == set(source_plan['test_view_names'])
                assert r['total_channel_uses'] == r['comparison_accounting']['total_channel_uses']
                ref = source_plan['reference']
                assert r['rd_score'] == pytest.approx(r['source_distortion']/ref['d_ref']+ref['beta']*r['total_channel_uses']/ref['c_ref'])
            losses = [json.loads(l) for l in (full/mode/'metrics/loss.jsonl').read_text().splitlines()]
            assert all(r['correction']['keep_batch_execution_verified'] for r in losses)
            assert all(0 <= r['correction']['finite_positive_corrections'] <= 3 for r in losses)
            assert all(r['teacher_weight'] == .1 for r in losses)
            assert all(not (set(r['camera_names']) & set(source_plan['test_view_names'])) for r in losses)
        first = [json.loads((full/m/'checks/000001.json').read_text())['actions'] for m in controls.ARMS]
        for a,b in zip(*first):
            assert a['source_row'] == b['source_row'] and a['new_tier'] == b['new_tier']
            assert a['measured_delta_rd'] == b['measured_delta_rd']
        with tarfile.open(full/'review.tar.gz') as archive:
            names = archive.getnames()
            assert 'review_summary.json' in names
            assert not any('/checkpoints/' in n or n.endswith('.pt') for n in names)


from test_structured_allocation_integration import source
