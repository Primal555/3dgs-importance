import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from gaussian_jscc import structured_evidence_allocation as entry
from gaussian_jscc.allocation_distortion import ComponentRenderTask, ImageDistortion
from gaussian_jscc.structured_evidence import StructuredEvidence, BoundedMirrorTable
from gaussian_jscc.budgeted_allocation_kernel import budgeted_mask_step
from test_successive_allocation import small_successive
from test_mask_gradient import toy_mask_render, Reference
import test_budgeted_allocation as fixture


def run_kernel(finite, closed=False, mode='mse_ssim'):
    torch.set_num_threads(1); torch.manual_seed(213)
    _, geometry, features, model = small_successive(); model.eval().requires_grad_(False)
    ids = [torch.arange(16).reshape(2, 8)]
    table = BoundedMirrorTable(16, .6)
    if closed: table.keep_logits[:, 0] = 100.; table.keep_logits[:, 1] = -100.
    scheduler = StructuredEvidence(ids, rounds=2, finite_samples=finite, memory=False)
    obj = ImageDistortion(mode, .0125 if mode != 'mse' else 0.)
    task = ComponentRenderTask([SimpleNamespace(factor=1., source_count=16)], Reference(), 1, distortion=obj)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    with patch('gaussian_jscc.rendering.render', side_effect=toy_mask_render):
        result = budgeted_mask_step(model, table, [features], ids, geometry, task, scheduler,
            10., 'awgn', d_ref=.1, price=.1, rate_normalizer=1000,
            side_proxy=dict(xyz_per_retained=1., fixed_uses=8., step=1), timings=None)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    return result, scheduler, torch.get_rng_state()


def test_exact_component_corrections_enter_primary_history_once_and_preserve_rng():
    base, _, rng = run_kernel(0)
    got, scheduler, rng1 = run_kernel(8)
    assert torch.equal(rng, rng1)
    assert got['source_distortion'] == pytest.approx(got['source_mse']+.0125*got['source_structure_error'])
    assert got['view_mse'] == base['view_mse']
    assert int(scheduler.observations.sum()) == len(got['tier_ids'])
    for row in got['audit']:
        assert row['applied_to_training']
        if row['feedback_kind'] != 'positive': continue
        j = (got['tier_ids'] == row['source_row']).nonzero().item()
        new, old = row['new_tier']-1, row['old_tier']-1
        expected = (row['measured_delta_mse']+.0125*row['measured_delta_structure'])/.1+row['rate_delta_cost']
        assert float(got['tier_costs'][j, new]-got['tier_costs'][j, old]) == pytest.approx(expected, abs=1e-10)
        assert scheduler.positive_exact_count[row['source_row']].sum() == 1
    assert got['finite_corrections'] > 2
    assert {'boundary', 'uncertainty', 'coverage'} <= {a['selection_group'] for a in got['audit']}


def test_all_closed_points_have_keep_observations_not_only_exact_subset():
    got, scheduler, _ = run_kernel(8, closed=True)
    assert got['q0_feedback_rows'] == 16
    assert scheduler.keep_observations.sum() == 16
    assert got['tier_ids'].numel() == 0
    assert got['finite_keep_corrections'] == 8
    assert got['keep_costs'].norm() > 0


@pytest.fixture
def source():
    fixture.BudgetedEntryTests.setUpClass()
    try: yield fixture.BudgetedEntryTests.source, fixture.BudgetedEntryTests.root
    finally: fixture.BudgetedEntryTests.tearDownClass()


def options(source, out):
    return entry.parser().parse_args(['--run', str(source), '--scene', 'truck', '--out', str(out),
        '--beta', '.1', '--allocation-steps', '2', '--calibration-views', '1', '--window-views', '4',
        '--control-every', '1', '--validate-every', '1', '--save-every', '1', '--profile-every', '0',
        '--test-views', '2', '--trials', '1', '--image-views', '1', '--cpu-threads', '1',
        '--device', 'cpu', '--no-lpips', '--finite-samples', '4'])


def test_full_entry_resume_calibration_selection_and_legacy_rejection(source):
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), contextlib.redirect_stdout(io.StringIO()):
        out = root/'structured'; entry.main(options(parent, out))
        full = torch.load(out/'checkpoints/latest/state.pt', weights_only=True)
        dest = root/'resumed'; args = options(parent, dest)
        entry.main(args, max_updates=1); args.resume = True; entry.main(args)
        resumed = torch.load(dest/'checkpoints/latest/state.pt', weights_only=True)
        for k,v in full['table'].items(): torch.testing.assert_close(v, resumed['table'][k], rtol=0, atol=0)
        for k,v in full['blocks']['structured']['tensors'].items():
            torch.testing.assert_close(v, resumed['blocks']['structured']['tensors'][k], rtol=0, atol=0)
        assert full['objective_reference'] == resumed['objective_reference']
        ref = full['objective_reference']; assert ref['structure_reference_fraction'] == pytest.approx(.2)
        assert full['family'] == entry.FAMILY
        assert full['progress']['frozen_codec_verified']
        history = json.loads((out/'comparison/structured_evidence.json').read_text())
        assert history['selected_step'] == full['progress']['best']['step']
        assert history['primary_observation_rule'] == 'one resolved observation per action per step'
        assert (out/'charts/structured_evidence.png').exists()
        rows = [json.loads(s) for s in (out/'metrics/validation.jsonl').read_text().splitlines()]
        for row in rows:
            assert row['source_distortion'] == pytest.approx(row['source_mse']+ref['lambda_s']*row['source_structure_error'], abs=1e-8)
            assert row['rd_score'] == pytest.approx(row['source_distortion']/ref['d_ref']+.1*row['total_channel_uses']/ref['c_ref'])
        calibration = [r for r in rows if r['event'].startswith('calibration_')]
        assert full['progress']['best']['rd_score'] == pytest.approx(min(r['rd_score'] for r in calibration))
        assert not list((out/'evaluation/calibration_000001').rglob('*.png'))
        args.lambda_s=.01
        with pytest.raises(ValueError, match='mismatch'): entry.main(args)
        args.lambda_s=None; args.max_log_odds=.5
        with pytest.raises(ValueError, match='mismatch'): entry.main(args)
        from gaussian_jscc.budgeted_allocation_state import restore_state
        from gaussian_jscc import evidence_allocation
        legacy = dict(full['plan'], version=evidence_allocation.FAMILY)
        with pytest.raises(ValueError, match='mismatch'):
            restore_state(out/'checkpoints/latest/state.pt', legacy, full['scene_record'])


def test_mse_only_kernel_uses_zero_structure_without_ssim_gradients():
    got, scheduler, _ = run_kernel(4, mode='mse')
    assert got['source_structure_error'] == 0
    assert got['source_distortion'] == got['source_mse']
    assert torch.count_nonzero(scheduler.mean[..., 1]) == 0


def test_real_launcher_validates_beta_and_preflights_without_output(source):
    import os
    from pathlib import Path
    import subprocess
    import sys
    parent, root = source
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists(): pytest.skip('Git Bash unavailable')
    out = root/'launcher_dry'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONUTF8='1')
    command = [str(bash), 'scripts/test_structured_evidence_allocation.sh', parent.as_posix(), 'truck', out.as_posix()]
    run = subprocess.run(command+['--beta', '.1', '--calibration-views', '1', '--dry-run'],
        env=env, capture_output=True, text=True, encoding='utf-8')
    assert run.returncode == 0, run.stderr
    plan = json.loads(run.stdout[run.stdout.index('{'):])
    assert plan['version'] == entry.FAMILY and not out.exists()
    assert plan['settings']['finite_samples'] == 32
    assert plan['settings']['distortion'] == 'mse_ssim'
    bad = subprocess.run(command+['--dry-run'], env=env, capture_output=True, text=True, encoding='utf-8')
    assert bad.returncode != 0 and 'beta' in bad.stderr
