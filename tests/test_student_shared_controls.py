import contextlib
import copy
import io
import itertools
import json
import tarfile
from unittest.mock import patch
import pytest
import torch
from gaussian_jscc.student_shared_feedback import stratified_draws, combine_keep_weights, SharedResidualFeedback
from gaussian_jscc.student_output_step import output_controlled_step


def test_shared_two_stage_residual_expectation_matches_full_population_gradient():
    # Enumerate all outer candidate subsets, inner groups and global keep draws.
    # Includes a dropped point, unequal within-candidate probabilities and shared parameters.
    torch.manual_seed(24)
    parameter = torch.randn(2, dtype=torch.float64, requires_grad=True)
    x = torch.randn(4, 2, dtype=torch.float64)
    keep = (x@parameter).sigmoid()
    tier = (x[:, :, None]*parameter[None, :, None]*torch.tensor([1., -2., 3.])).sum(1).softmax(-1)
    retained = torch.tensor([False, True, True, True])
    a_keep = torch.tensor([.2, .3, -.2, .4], dtype=torch.float64)
    t_keep = torch.tensor([-.1, .2, -.7, .6], dtype=torch.float64)
    a_tier = torch.randn(4, 3, dtype=torch.float64)
    t_tier = torch.randn(4, 3, dtype=torch.float64)
    expected = parameter.sum()*0
    for candidates in itertools.combinations((1, 2, 3), 2):
        candidate_ids = torch.tensor(candidates)
        for j, probability in enumerate((.3, .7)):
            point = candidate_ids[j:j+1]
            for global_id in range(4):
                ids, weights = combine_keep_weights(torch.tensor([global_id]), torch.tensor([4.], dtype=torch.float64),
                    point, torch.tensor([1.5/probability], dtype=torch.float64), retained, .6)
                signal = a_keep.clone(); signal[ids] += weights*(t_keep[ids]-a_keep[ids])
                positive = a_tier[candidate_ids].clone()
                positive[j] += (t_tier[point[0]]-positive[j])/probability
                loss = (keep*signal).sum()+1.5*(tier[candidate_ids]*positive).sum()
                expected = expected+loss*probability/12
    truth = (keep*t_keep).sum()+(tier[retained]*t_tier[retained]).sum()
    got, = torch.autograd.grad(expected, parameter, retain_graph=True)
    wanted, = torch.autograd.grad(truth, parameter)
    torch.testing.assert_close(got, wanted, atol=1e-14, rtol=1e-13)


def test_duplicate_keep_draws_and_zero_priority_strata_preserve_population_sum():
    strata = torch.tensor([0, 0, 1, 2, 2, 2])
    values = torch.tensor([2., 2., -3., 5., 5., 5.], dtype=torch.float64)
    ids, weights, info = stratified_draws(strata, torch.tensor([0., 0., 100., 0., 0., 0.]), 7,
        torch.Generator().manual_seed(11))
    assert set(strata[ids].tolist()) == {0, 1, 2}
    assert sum(info['draws']) == 7
    torch.testing.assert_close((values[ids]*weights).sum(), values.sum())
    retained = torch.tensor([False, True, True])
    ids, weights = combine_keep_weights(torch.tensor([0, 0, 1]), torch.tensor([2., 2., 4.]),
        torch.tensor([1, 1, 2]), torch.tensor([5., 5., 7.]), retained, .8)
    assert ids.tolist() == [0, 1, 2]
    torch.testing.assert_close(weights, torch.tensor([4., 5.2, 1.4], dtype=torch.float64))


@pytest.mark.parametrize('diagnostic', [False, True])
def test_output_control_scales_one_adam_displacement_and_preserves_moments(diagnostic):
    torch.manual_seed(3); torch.set_num_threads(1)
    model = torch.nn.Linear(3, 5).double()
    original = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.4)
    reference_optimizer = torch.optim.Adam(original.parameters(), lr=.4)
    inputs = torch.randn(11, 3, dtype=torch.float64)
    def heads():
        with torch.no_grad():
            v = model(inputs)
            return v[:, :2].clone(), v[:, 2:].clone()
    before = heads(); start = [p.detach().clone() for p in model.parameters()]
    loss = model(inputs).square().sum()
    result = output_controlled_step(model, optimizer, dict(numerical=loss), 1.,
        before=before, heads=heads, diagnostic=diagnostic)
    reference_optimizer.zero_grad(); original(inputs).square().sum().backward()
    torch.nn.utils.clip_grad_norm_(original.parameters(), 1.); reference_optimizer.step()
    control = result['output_step']; scale = control['applied_scale']
    assert 0 < scale < 1 and control['backtrack_forwards'] > 0
    for name in ('keep', 'tier'): assert control['accepted'][name]['mean'] <= 2e-5
    for p, full, initial in zip(model.parameters(), original.parameters(), start):
        torch.testing.assert_close(p, initial+scale*(full-initial))
    for actual, reference in zip(optimizer.state.values(), reference_optimizer.state.values()):
        for name in actual: torch.testing.assert_close(actual[name], reference[name], atol=0, rtol=0)
        assert actual['step'] == 1
    if diagnostic:
        assert result['directional_product_scope']
        assert result['parameter_step_norm'] < .4


def test_output_control_restores_parameters_and_adam_after_failed_forward():
    model = torch.nn.Linear(2, 5); inputs = torch.ones(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    optimizer.zero_grad(); model(inputs).sum().backward(); optimizer.step()
    saved_model = copy.deepcopy(model.state_dict()); saved_opt = copy.deepcopy(optimizer.state_dict())
    value = model(inputs)
    with pytest.raises(RuntimeError, match='output failure'):
        output_controlled_step(model, optimizer, dict(loss=value.square().sum()), 1.,
            before=(value[:, :2].detach(), value[:, 2:].detach()), diagnostic=False,
            heads=lambda: (_ for _ in ()).throw(RuntimeError('output failure')))
    for name, value in model.state_dict().items(): torch.testing.assert_close(value, saved_model[name], atol=0, rtol=0)
    for key, state in optimizer.state_dict()['state'].items():
        for name, value in state.items(): torch.testing.assert_close(value, saved_opt['state'][key][name], atol=0, rtol=0)


@pytest.mark.parametrize('retain', [False, True])
def test_shared_teacher_continuation_and_resume_match_independent_decodes(source, retain):
    from gaussian_jscc import student_policy_experiment as student
    from gaussian_jscc import structured_evidence_allocation as teacher_entry
    from gaussian_jscc import student_shared_controls as controls
    from gaussian_jscc.student_policy_feedback import policy_surrogate
    from gaussian_jscc.budgeted_allocation_kernel import raw_batches, set_rng
    from gaussian_jscc.optimization import preserved_rng
    from test_student_policy_experiment import args_for, capture_view
    from test_structured_allocation_integration import options
    import test_budgeted_allocation as fixture
    parent, root = source
    measured_cases = set()
    original_measure = SharedResidualFeedback._measure_states
    def check_measure(self, model, features, id_batches, cache, geometry, task, scene_tensor, requests,
                      execution, snr, kind, d_ref):
        before_scene = scene_tensor.clone(); before_rng = torch.get_rng_state().clone()
        measured, stats = original_measure(self, model, features, id_batches, cache, geometry, task,
            scene_tensor, requests, execution, snr, kind, d_ref)
        assert stats['alternative_renders']+stats['baseline_states_reused'] == len(set(requests))
        with torch.no_grad():
            for point, k in sorted(set(requests)):
                batch, row, col, _ = self.lookup[point].tolist(); q, _, _, _, rng = cache[batch]
                f = features[batch]; alt = q.clone(); alt[row, col] = k
                measured_cases.add('same' if k == q[row, col] else 'positive' if k*q[row, col] > 0
                                   else 'restore' if q[row, col] == 0 else 'delete')
                with preserved_rng(f.device):
                    set_rng(rng, f.device)
                    pred = model.forward_tier_batches(f, f[..., :3], torch.nn.functional.one_hot(alt, 4).to(f),
                        snr, kind, paired_noise=True)[0]
                raw = raw_batches(pred, geometry, model); valid = id_batches[batch] >= 0
                offset = sum(int((x >= 0).sum()) for x in id_batches[:batch]); count = int(valid.sum())
                changed = before_scene.clone()
                changed[offset:offset+count] = torch.cat((raw[valid], (alt[valid]>0).to(raw)[:, None]), -1)
                expected = task.delta_components(changed)/d_ref
                torch.testing.assert_close(measured[point, k], expected.cpu(), atol=1e-8, rtol=2e-4)
        torch.testing.assert_close(scene_tensor, before_scene, atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
        return measured, stats
    from gaussian_jscc.student_shared_feedback import shared_losses
    def check_losses(heads, keep, ids, costs, inverse, sampler):
        pieces = shared_losses(heads, keep, ids, costs, inverse, sampler)
        torch.testing.assert_close(sum(pieces.values()), policy_surrogate(*heads, keep, ids, costs, inverse)[0])
        assert set(sampler.last_correction['ids'].tolist()).isdisjoint(r['source_row'] for r in sampler.holdout)
        return pieces
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch('gaussian_jscc.student_policy_features.capture_view', side_effect=capture_view), \
         contextlib.redirect_stdout(io.StringIO()):
        teacher = root/'teacher'; teacher_entry.main(options(parent, teacher))
        source_run = root/'student'
        student.main(args_for(parent, teacher, source_run, '--distill-target', 'soft_deployment',
            '--teacher-weight-schedule', 'constant', '--cost-protocol', 'digital_side_file_v1'))
        if retain:
            path = source_run/'checkpoints/latest/student.pt'
            state = torch.load(path, weights_only=True)
            state['student']['keep_head.3.bias'] += torch.tensor([-5., 5.])
            torch.save(state, path)
        original_files = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in source_run.rglob('*') if p.is_file()}
        def cli(out):
            return controls.parser().parse_args(['--source', str(source_run), '--out', str(out), '--checkpoint', 'latest',
                '--device', 'cpu', '--steps', '2', '--full-groups', '2', '--keep-draws', '18',
                '--holdout-draws', '2', '--evaluate-every', '1', '--diagnostic-every', '1'])
        with patch.object(SharedResidualFeedback, '_measure_states', check_measure), \
             patch.object(controls, 'shared_losses', side_effect=check_losses):
            full = root/'full'; assert controls.main(cli(full))['status'] == 'complete'
            resumed = root/'resumed'; args = cli(resumed)
            assert controls.main(args, max_updates=1)['status'] == 'interrupted'
            args.resume = True; assert controls.main(args)['status'] == 'complete'
        assert original_files == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in source_run.rglob('*') if p.is_file()}
        mode = 'shared_residual'
        a = torch.load(full/mode/'checkpoints/latest/student.pt', weights_only=True)
        b = torch.load(resumed/mode/'checkpoints/latest/student.pt', weights_only=True)
        for key in a['student']: torch.testing.assert_close(a['student'][key], b['student'][key], atol=0, rtol=0)
        torch.testing.assert_close(a['rng']['torch'], b['rng']['torch'], atol=0, rtol=0)
        for name in ('joint_rng', 'residual_rms', 'residual_seen'):
            torch.testing.assert_close(a['sampler']['shared_residual'][name], b['sampler']['shared_residual'][name], atol=0, rtol=0)
        plan = json.loads((source_run/'plan.json').read_text())
        rows = [json.loads(l) for l in (full/mode/'metrics/validation.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == [0, 1, 2]
        for r in rows:
            assert {v['view'] for v in r['per_view_trial']} == set(plan['test_view_names'])
            assert r['total_channel_uses'] == r['comparison_accounting']['total_channel_uses']
            ref = plan['reference']
            assert r['rd_score'] == pytest.approx(r['source_distortion']/ref['d_ref']+ref['beta']*r['total_channel_uses']/ref['c_ref'])
        losses = [json.loads(l) for l in (full/mode/'metrics/loss.jsonl').read_text().splitlines()]
        assert all(r['teacher_weight'] == .1 for r in losses)
        assert all(r['shared_feedback']['positive_checked_actions'] <= 4 for r in losses)
        assert all(r['output_step']['accepted'][head]['mean'] <= 2e-5 for r in losses for head in ('keep', 'tier'))
        assert all(not set(r['camera_names']) & set(plan['test_view_names']) for r in losses)
        with tarfile.open(full/'review.tar.gz') as archive:
            assert 'review_summary.json' in archive.getnames()
            assert not any(p.endswith('.pt') for p in archive.getnames())
    assert ({'same', 'positive', 'delete'} if retain else {'restore'}) <= measured_cases


from test_structured_allocation_integration import source
