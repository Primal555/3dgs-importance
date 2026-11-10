import contextlib
import io
import json
import tarfile
import pytest
from unittest.mock import patch
import torch
from gaussian_jscc import fast_first_order_allocation as entry
from gaussian_jscc.first_order_stopping import FirstOrderStopping
from test_structured_allocation_integration import source, options
import test_budgeted_allocation as fixture


def args_for(parent, out):
    args = options(parent, out)
    extra = dict(candidate_batch=8, early_stop=True, stop_min_steps=0, stop_patience=1,
        stop_relative_improvement=.9, stop_map_tolerance=1., stop_cost_tolerance=.99,
        fast_execution_version='batched_first_order_v1')
    for k, v in extra.items(): setattr(args, k, v)
    args.allocation_steps = 8
    return args


@pytest.mark.parametrize('tier_cap', [None, 1.])
def test_early_finish_resume_and_reports(source, tier_cap):
    parent, root = source
    def configured(out):
        args = args_for(parent, out)
        if tier_cap is not None: args.tier_max_log_odds = tier_cap
        return args
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), contextlib.redirect_stdout(io.StringIO()):
        out = root/'fast'; result = entry.main(configured(out))
        assert result['step'] == 2 and result['stop_reason'] == 'stable_calibration_plateau'
        full = torch.load(out/'checkpoints/latest/state.pt', weights_only=True)
        dest = root/'fast_resume'; args = configured(dest)
        assert entry.main(args, max_updates=1)['status'] == 'interrupted'
        args.resume = True; resumed_result = entry.main(args)
        assert resumed_result['step'] == 2
        resumed = torch.load(dest/'checkpoints/latest/state.pt', weights_only=True)
        for k, value in full['table'].items():
            torch.testing.assert_close(value, resumed['table'][k], rtol=0, atol=0)
        assert full['progress']['stopping']['last_event'] == resumed['progress']['stopping']['last_event']
        assert torch.equal(full['rng']['torch'], resumed['rng']['torch'])
        rows = [json.loads(s) for s in (dest/'metrics/stopping.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == [0, 1, 2]
        assert [r['action'] for r in rows] == ['improved', 'reduce_lr', 'stop']
        with tarfile.open(dest/'review.tar.gz') as archive:
            names = archive.getnames()
        assert 'execution_check.json' in names and 'review_summary.json' in names
        assert 'deployment/selected_tiers.npy' in names
        assert not any(n.startswith('checkpoints/') for n in names)
        assert resumed_result['frozen_codec_verified']
        if tier_cap is not None:
            assert resumed['plan']['settings']['tier_max_log_odds'] == tier_cap
            updates = [json.loads(s) for s in (dest/'metrics/loss.jsonl').read_text().splitlines()]
            assert all(r['tier_max_log_odds'] == tier_cap and r['max_log_odds'] == .25 for r in updates)
