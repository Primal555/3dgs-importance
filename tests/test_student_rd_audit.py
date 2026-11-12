import contextlib
import io
import json
from unittest.mock import patch
import torch
import pytest
from gaussian_jscc.student_rd_audit_tools import split_action_maps, transition_stats, take_step, paired_decode, direction_summary
from gaussian_jscc.student_policy_model import hard_tiers
from gaussian_jscc.modular_checkpoint import capture_rng


def test_maps_separate_shared_spread_and_keep_tier_interaction():
    before_keep = torch.tensor([[0., 1.], [0., 1.], [1., 0.], [0., 1.]])
    before_tier = torch.tensor([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.], [3., 0., 0.]])
    after_keep = before_keep.clone(); after_keep[2] = torch.tensor([0., 1.]); after_keep[3] = torch.tensor([1., 0.])
    after_tier = before_tier.roll(1, 1)
    maps = split_action_maps((before_keep, before_tier), (after_keep, after_tier), torch.tensor([0]))
    assert maps['baseline'].tolist() == [1, 2, 0, 1]
    assert maps['full'].tolist() == [2, 3, 1, 0]
    assert maps['keep_only'].tolist() == [1, 2, 3, 0]
    assert maps['positive_only'].tolist() == [2, 3, 0, 2]
    assert maps['positive_evaluated'].tolist() == [2, 2, 0, 1]
    assert maps['positive_unseen'].tolist() == [1, 3, 0, 2]
    stats = transition_stats((before_keep, before_tier), (after_keep, after_tier), torch.tensor([0]))
    assert stats['positive_unseen_changes'] == 1 and stats['changed_points'] == 4
    assert stats['positive_only_map_unseen_changes'] == 2
    assert sum(map(sum, stats['transition_counts'])) == 4


def test_gradient_inspection_preserves_adam_step_and_measures_conflict():
    model = torch.nn.Linear(2, 2); other = torch.nn.Linear(2, 2)
    other.load_state_dict(model.state_dict())
    a = torch.optim.AdamW(model.parameters(), lr=.01); b = torch.optim.AdamW(other.parameters(), lr=.01)
    x = torch.tensor([[.4, -.3]])
    out = model(x); keep = out.sum(); positive = -out.sum()*2; teacher = out.square().sum()
    report = take_step(model, a, dict(keep=keep, positive=positive, teacher=.1*teacher), .5)
    b.zero_grad(); o = other(x); (o.sum()-2*o.sum()+.1*o.square().sum()).backward()
    torch.nn.utils.clip_grad_norm_(other.parameters(), .5); b.step()
    for p, q in zip(model.parameters(), other.parameters()): torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert report['gradients']['all']['cosine']['keep:positive'] == pytest.approx(-1.)
    assert 'teacher' in report['gradient_dot_parameter_step']


def test_fixed_surrogate_uses_double_precision_for_both_sides():
    from gaussian_jscc.student_rd_step_audit import diagnostic_surrogate
    heads = (torch.randn(4, 2), torch.randn(4, 3))
    keep = torch.randn(4, 2); ids = torch.tensor([0, 2]); costs = torch.randn(2, 3); inv = torch.tensor([2., 3.])
    first = diagnostic_surrogate(heads, keep, ids, costs, inv)
    same = diagnostic_surrogate(tuple(x.double() for x in heads), keep.double(), ids, costs.double(), inv.double())
    assert first == same


def test_direction_maps_hold_keep_mask_fixed_and_summary_counts_costly_errors():
    keep = torch.tensor([[0., 1.], [0., 1.], [1., 0.], [0., 1.]])
    old = torch.nn.functional.one_hot(torch.tensor([2, 0, 2, 1]), 3).float()
    new = torch.nn.functional.one_hot(torch.tensor([0, 2, 0, 2]), 3).float()
    maps = split_action_maps((keep, old), (keep.flip(1), new), torch.tensor([0, 1]), directions=True)
    assert maps['positive_down_only'].tolist() == [1, 1, 0, 2]
    assert maps['positive_up_only'].tolist() == [3, 3, 0, 3]
    records = [dict(old_tier=3, new_tier=1, feedback_kind='positive', selection_group='boundary',
                    first_order_delta_rd=p, measured_delta_rd=t) for p,t in [(-2., 3.), (1., -2.)]]
    summary = direction_summary(records)
    down = summary['down']['all']
    assert down['false_beneficial_cost_sum'] == 3.
    assert down['missed_beneficial_cost_sum'] == 2.
    assert down['regret_mean'] == 2.5 and down['sign_error_fraction'] == 1.
    assert down['prediction_bias_mean'] == -1.
    assert summary['up']['all']['count'] == 0


@pytest.mark.parametrize('soft,digital', [(False, False), (True, False), (True, True)])
def test_audit_runs_from_pinned_checkpoint_without_changing_source(source, soft, digital):
    from gaussian_jscc import student_policy_experiment as student
    from gaussian_jscc import structured_evidence_allocation as teacher_entry
    from gaussian_jscc import student_rd_step_audit as audit
    from test_student_policy_experiment import args_for, capture_view
    from test_structured_allocation_integration import options
    import test_budgeted_allocation as fixture
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch('gaussian_jscc.student_policy_features.capture_view', side_effect=capture_view), \
         contextlib.redirect_stdout(io.StringIO()):
        teacher = root/'teacher'; teacher_entry.main(options(parent, teacher))
        run = root/'student'
        extra = ['--distill-target', 'soft_deployment', '--teacher-weight-schedule', 'constant'] if soft else []
        if digital: extra += ['--cost-protocol', 'digital_side_file_v1', '--calibration-views', '2', '--window-views', '2']
        args = args_for(parent, teacher, run, *extra)
        student.main(args, max_updates=3)
        original = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
        out = root/'audit'
        diagnostic_options = ['--diagnostic-views', '3', '--direction-maps'] if digital else []
        result = audit.main(audit.parser().parse_args(['--source', str(run), '--out', str(out),
            '--device', 'cpu', '--probes', '2', *diagnostic_options]))
        assert result['status'] == 'complete' and result['probes'] == 2
        assert original == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
        report = json.loads((out/'review_summary.json').read_text())
        rows = report['probes']
        if digital:
            assert report['plan']['cost_protocol'] == 'digital_side_file_v1'
            assert len(report['plan']['original_calibration_view_names']) == 2
            diagnostic = report['plan']['diagnostic_views']
            assert len(diagnostic['camera_names']) == 3
            assert diagnostic['overlap_with_student_training']
        for row in rows:
            assert row['baseline_gradient_render_error'] < 1e-9
            assert {'full', 'small_step', 'keep_only', 'positive_only', 'positive_evaluated', 'positive_unseen'} <= set(row['maps'])
            assert ('anchor_restored' in row['maps']) == (not soft)
            assert row['teacher_weight'] == (.1 if soft else 0.)
            groups = {'training', 'calibration', 'expanded_diagnostic'} if digital else {'training', 'calibration'}
            assert all(set(m['evaluation']) == groups for m in row['maps'].values())
            if digital:
                assert {'positive_down_only', 'positive_up_only'} <= set(row['maps'])
                for m in row['maps'].values():
                    assert m['cost']['total_channel_uses'] == m['cost']['comparison_accounting']['total_channel_uses']
                    for value in m['evaluation'].values():
                        ref = report['plan']['reference']
                        assert value['rd'] == pytest.approx(value['distortion']/ref['d_ref']+
                            ref['beta']*m['cost']['total_channel_uses']/ref['c_ref'])
            assert row['source_weights_reloaded'] and row['source_optimizer_reloaded']
        assert rows[0]['source_model_hash'] == rows[1]['source_model_hash']
        assert (out/'review.tar.gz').is_file()
        # The first probe must reproduce the next ordinary training update,
        # including saved optimizer moments, sampling schedule and AWGN state.
        args.resume = True; student.main(args, max_updates=1)
        from gaussian_jscc.student_policy_model import StudentConfig, StudentPolicy
        from gaussian_jscc.transport import model_id
        state = torch.load(run/'checkpoints/latest/student.pt', weights_only=True)
        net = StudentPolicy(StudentConfig(**state['config'])); net.load_state_dict(state['student'])
        assert model_id(net) == rows[0]['full_model_hash']


from test_structured_allocation_integration import source
