import contextlib
import copy
import io
import json
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc import fast_first_order_allocation as teacher
from gaussian_jscc.allocation_reference import anchored_reference, load_reference_anchor
from gaussian_jscc.budgeted_allocation_inputs import load_inputs
from test_structured_allocation_integration import source
from test_fast_first_order_entry import args_for
import test_budgeted_allocation as fixture


def configured(parent, out, anchor=None):
    args = args_for(parent, out)
    args.cost_protocol = 'digital_side_file_v1'
    args.tier_max_log_odds = 2.
    args.early_stop = False
    args.allocation_steps = 2
    args.window_views = 2
    if anchor is not None:
        args.calibration_views = 2
        args.objective_reference_run = str(anchor)
    return args


def test_coverage_preserves_objective_and_resume(source):
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         contextlib.redirect_stdout(io.StringIO()):
        original = root/'original_reference'
        teacher.main(configured(parent, original))
        source_training = json.loads((original/'training.json').read_text())
        expected = source_training['objective_reference']
        out = root/'expanded_calibration'
        done = teacher.main(configured(parent, out, original))
        assert done['status'] == 'complete'
        training = json.loads((out/'training.json').read_text())
        assert len(training['calibration_view_names']) == 2
        assert len(training['allocation_train_view_names']) == 3
        assert not set(training['calibration_view_names']) & set(training['allocation_train_view_names'])
        assert not set(training['test_view_names']) & set(training['calibration_view_names'])
        ref = training['objective_reference']
        for key in ('d_ref', 'lambda_s', 'c_ref', 'mse_ref', 'structure_ref', 'calibration_views'):
            assert ref[key] == expected[key]
        assert ref['normalization_source']['view_names'] == source_training['calibration_view_names']
        assert not (out/'evaluation/q3_reference').exists()
        rows = [json.loads(s) for s in (out/'metrics/validation.jsonl').read_text().splitlines()]
        for row in rows:
            assert row['d_ref'] == expected['d_ref'] and row['c_ref'] == expected['c_ref']
            assert row['lambda_s'] == expected['lambda_s']
            if row['event'].startswith('calibration'):
                assert row['views'] == 2
            assert row['rd_score'] == pytest.approx(row['source_distortion']/expected['d_ref'] +
                row['beta']*row['total_channel_uses']/expected['c_ref'])
        resumed_out = root/'expanded_resume'
        args = configured(parent, resumed_out, original)
        assert teacher.main(args, max_updates=1)['status'] == 'interrupted'
        args.resume = True
        assert teacher.main(args)['status'] == 'complete'
        full = torch.load(out/'checkpoints/latest/state.pt', weights_only=True)
        resumed = torch.load(resumed_out/'checkpoints/latest/state.pt', weights_only=True)
        assert full['objective_reference'] == resumed['objective_reference']
        for key, value in full['table'].items():
            torch.testing.assert_close(value, resumed['table'][key], atol=0, rtol=0)
        assert torch.equal(full['rng']['torch'], resumed['rng']['torch'])
        assert torch.equal(full['latest_q'], resumed['latest_q'])

        # Changed source files must not silently modify a resumed objective.
        training_path = original/'training.json'
        before = training_path.read_bytes()
        training_path.write_bytes(before+b'\n')
        with pytest.raises(ValueError, match='resume source/settings/scene mismatch'):
            teacher.main(args)
        training_path.write_bytes(before)

        args = configured(parent, root/'unused', original)
        _, cfg, _, old, _, provenance = load_inputs(args)
        anchor = load_reference_anchor(args, old, cfg.to_dict(), provenance)
        altered_cost = copy.deepcopy(anchor['reference']['q3_cost'])
        altered_cost['metadata_bytes'] += 1
        with pytest.raises(ValueError, match='accounting changed'):
            anchored_reference(anchor, altered_cost)
        altered_scene = dict(old, fingerprint='changed')
        with pytest.raises(ValueError, match='scene changed'):
            load_reference_anchor(args, altered_scene, cfg.to_dict(), provenance)
        args.beta *= 2
        with pytest.raises(ValueError, match='setting changed: beta'):
            load_reference_anchor(args, old, cfg.to_dict(), provenance)


def test_reference_cli_is_opt_in():
    base = ['--run', 'codec', '--scene', 'truck', '--out', 'new', '--beta', '.5']
    args = teacher.parser().parse_args(base)
    assert not hasattr(args, 'objective_reference_run')
    args = teacher.parser().parse_args(base+['--objective-reference-run', 'teacher'])
    assert args.objective_reference_run == 'teacher'
