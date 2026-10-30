import copy

import pytest
import torch


def test_weighted_decision_report_uses_original_sampling_weights():
    from gaussian_jscc.physical_benefit_audit_metrics import decision_report
    # First group should drop but predicts keep; second correctly keeps q1.
    target = torch.tensor([[[1., 0.]]*3, [[-1., 0.]]*3], dtype=torch.float64)
    pred = torch.zeros_like(target); pred[..., 0] = -1.
    report = decision_report(pred, target, torch.zeros(2, 3), torch.tensor([1., 3.]))
    assert report['conditional_keep']['regret_mean'] == pytest.approx(.25)
    assert report['four_state_regret'] == pytest.approx(.25)
    assert report['conditional_keep']['false_beneficial_fraction'] == pytest.approx(.25)
    assert report['effective_sample_size'] == pytest.approx(1.6)
    assert report['common_error_rmse'] == pytest.approx(1.)


def test_common_oracle_changes_keep_decisions_but_preserves_positive_ranking():
    from gaussian_jscc.physical_benefit_audit_metrics import error_oracles
    target = torch.tensor([[[-3., 0.], [-2., 0.], [-1., 0.]]], dtype=torch.float64)
    pred = target + torch.tensor([4., 0.])
    result = error_oracles(pred, target, torch.zeros(1, 3), torch.ones(1))
    assert result['original']['four_state_regret'] == 3.
    assert result['correct_common']['four_state_regret'] == 0.
    assert result['correct_common']['three_tier_regret'] == result['original']['three_tier_regret']
    assert result['correct_centered']['four_state_regret'] == 3.


def test_invalid_weights_are_rejected():
    from gaussian_jscc.physical_benefit_audit_metrics import decision_report
    x = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError):
        decision_report(x, x, torch.zeros(1, 3), torch.tensor([0.]))


def test_physical_features_can_identically_hide_opposite_render_benefits():
    from gaussian_jscc.physical_benefit_audit_metrics import feature_collision_probe
    result = feature_collision_probe()
    assert result['features_identical']
    assert result['feature_max_abs'] == 0.
    assert result['rd_a'] < 0 < result['rd_b']
    assert result['full_mse_a'] < 0 < result['full_mse_b']
    assert result['teacher_full_mse_max_abs'] < 1e-7


def test_actual_successive_codec_preserves_noise_and_batched_interventions():
    from gaussian_jscc.physical_benefit_audit_metrics import codec_pairing_probe
    from test_successive_allocation import small_successive
    torch.set_num_threads(1)
    _, geometry, features, model = small_successive()
    model.eval().requires_grad_(False)
    state = copy.deepcopy(model.state_dict())
    result = codec_pairing_probe(model, features, seed=31)
    assert result['baseline_noise_max_abs'] < 1e-6
    assert result['candidate_batch_max_abs'] < 2e-5
    assert result['rate_delta_max_abs'] == 0.
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])


def test_fixed_fit_counts_visits_and_keeps_source_learner_unchanged():
    from gaussian_jscc.physical_benefit_audit import fixed_fit
    from gaussian_jscc.physical_benefit import PhysicalLearner, FEATURE_DIM
    from test_physical_benefit import record
    torch.set_num_threads(1)
    groups = [record(i, .01*(i+1), FEATURE_DIM) for i in range(4)]
    class Replay:
        def training_records(self): return iter(groups)
        def sample(self, count): return groups[:count]
    learner = PhysicalLearner(Replay(), 'cpu', hidden=8)
    learner.fit_scales()
    learner.train_updates(1, 2)
    before = copy.deepcopy(learner.state_dict())
    fork, history, visits = fixed_fit(learner, groups, updates=4, batch_size=2,
        evaluate_every=2, seed=7, observer=lambda model: {'updates': model.updates})
    assert visits == [2, 2, 2, 2]
    assert [row['additional_updates'] for row in history] == [0, 2, 4]
    assert fork.updates == learner.updates + 4
    for key, value in learner.network.state_dict().items():
        assert torch.equal(value, before['network'][key])
    for parameter, values in before['optimizer']['state'].items():
        for key, value in values.items():
            actual = learner.optimizer.state_dict()['state'][parameter][key]
            assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual


def test_complete_audit_relocates_replay_and_does_not_modify_source(tmp_path):
    from gaussian_jscc.physical_benefit_audit import main, parser, file_hash
    from gaussian_jscc.physical_benefit import PhysicalLearner, FEATURE_DIM
    from gaussian_jscc.pixel_benefit_replay import PixelReplay
    from test_physical_benefit import record
    from test_successive_allocation import small_successive
    torch.set_num_threads(1); torch.manual_seed(8)
    run, out = tmp_path/'source', tmp_path/'audit'
    replay = PixelReplay(run/'replay', max_groups=16, validation_groups=4)
    for i in range(12):
        replay.add(record(i, .001*(i+1), FEATURE_DIM))
    for i in range(4):
        replay.add(record(100+i, .002*(i+1), FEATURE_DIM), validation=True)
    learner = PhysicalLearner(replay, 'cpu', hidden=8)
    learner.fit_scales()
    _, geometry, _, codec = small_successive()
    # The legacy test fixture mutates a progressive config after construction.
    # Persist a valid successive configuration for the actual checkpoint loader.
    codec.cfg.channel_denoiser_mode = 'none'
    replay_state = replay.state_dict()
    replay_state['root'] = '/server/original/run/replay'
    state = dict(family='physical_benefit_allocation_v1',
        progress=dict(phase='pretrain', pretrain_complete=True),
        replay=replay_state, learner=learner.state_dict(), state_dict=codec.state_dict(),
        plan=dict(settings=dict(hidden=8, dropout=.05, decision_weight=1., tie_fraction=.02,
            weight_decay=1e-4, batch_groups=4), codec_config=codec.cfg.to_dict()),
        scene_record=dict(geometry=dict(lower=geometry.lower, span=geometry.span, bits=geometry.bits)))
    checkpoint = run/'checkpoints/latest/state.pt'
    checkpoint.parent.mkdir(parents=True)
    torch.save(state, checkpoint)
    before = file_hash(checkpoint)
    args = parser().parse_args(['--run', str(run), '--out', str(out), '--fit-updates', '2',
        '--fit-groups', '4', '--batch-groups', '2', '--evaluate-every', '1', '--cpu-threads', '1'])
    result = main(args)
    assert result['status'] == 'complete'
    assert result['source_checkpoint_unchanged']
    assert file_hash(checkpoint) == before
    assert result['fixed_fit_visit_min'] == result['fixed_fit_visit_max'] == 1
    assert (out/'before.json').is_file()
    assert (out/'mechanism_checks.json').is_file()
    with pytest.raises(ValueError, match='new audit directory'):
        main(args)
