import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from gaussian_jscc import neural_capacity_history as entry
from gaussian_jscc.full_history_spatial_replay import FullHistorySpatialReplay
from gaussian_jscc.spatial_benefit import SpatialBenefitNetwork
from test_hybrid_tier_pipeline import environment
from test_neural_correction_pipeline import options as old_options
from test_structured_allocation_integration import source
from test_learned_benefit_allocation import read, assert_nested_equal


def options(parent, out):
    args = entry.parser().parse_args(['--run', str(parent), '--scene', 'truck', '--out', str(out)])
    # Small CPU protocol exercises both phases and forces capacity overflow.
    vars(args).update(vars(old_options(parent, out)))
    args.replay_capacity = 4
    return args


def test_defaults_only_change_capacity_and_replay_policy():
    args = entry.parser().parse_args(['--run', 'codec', '--scene', 'truck', '--out', 'new'])
    assert args.hidden == 256 and args.network_mode == 'online'
    assert args.pretrain_windows == 700 and args.allocation_steps == 2000
    assert args.tier_max_log_odds == 1. and args.max_log_odds == .25
    assert args.full_history_policy == FullHistorySpatialReplay.policy
    assert sum(p.numel() for p in SpatialBenefitNetwork(256).parameters()) == 206082


@pytest.mark.parametrize('stop', [2, 5])
def test_both_phases_keep_history_and_resume_exactly(source, stop):
    from test_spatial_benefit_integration import Window
    parent, root = source
    full, split = root/'wide_full', root/'wide_split'
    with environment(), patch.object(entry, 'pretrain_window', Window):
        result = entry.main(options(parent, full))
        entry.main(options(parent, split), max_updates=stop)
        args = options(parent, split)
        args.resume = True
        entry.main(args)
    a = read(full/'allocation/checkpoints/latest/state.pt')
    b = read(split/'allocation/checkpoints/latest/state.pt')
    pre = read(full/'pretrain/checkpoints/latest/state.pt')
    assert len(pre['replay']['training']) == 4
    assert len(a['replay']['training']) > len(pre['replay']['training'])
    assert a['replay']['full_history_policy'] == FullHistorySpatialReplay.policy
    assert pre['replay']['full_history_policy'] == FullHistorySpatialReplay.policy
    for key in ('table', 'views', 'rng', 'positive_provider'):
        assert_nested_equal(a[key], b[key])
    assert [a['replay']['usage'][e['id']] for e in a['replay']['training']] == [
        b['replay']['usage'][e['id']] for e in b['replay']['training']]
    assert result['frozen_codec_verified'] and result['network_updates'] == 5
    assert result['benefit_replay_policy'] == FullHistorySpatialReplay.policy
    for state in (pre, a):
        for item in state['replay']['training']:
            assert (Path(state['replay']['root'])/item['path']).is_file()
    gaps = [read(Path(a['replay']['root'])/e['path']) for e in a['replay']['training']]
    assert any(g.get('supervision') == 'action_gap' for g in gaps)
    records = a['blocks']['calibrated']['positive']['records']
    assert records
    assert all(v['record_id'] in a['replay']['priorities'] for v in records.values())
    assert (full/'review.tar.gz').is_file()


def test_script_dry_run_records_new_protocol(source):
    import json
    parent, root = source
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():
        import pytest
        pytest.skip('Git Bash unavailable')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHON_BIN=Path(sys.executable).as_posix(),
               PYTHONUTF8='1')
    run = subprocess.run([str(bash), 'scripts/test_neural_capacity_history.sh',
        parent.as_posix(), 'truck', (root/'script_new').as_posix(),
        '--calibration-views', '1', '--dry-run'], env=env, capture_output=True,
        text=True, encoding='utf8', timeout=40)
    assert run.returncode == 0, run.stdout+run.stderr
    plan = json.loads(run.stdout[run.stdout.index('{'):])
    assert plan['settings']['hidden'] == 256
    assert plan['settings']['full_history_policy'] == FullHistorySpatialReplay.policy
    assert plan['automatic_transition'] and plan['allocation_steps'] == 2000
    assert not (root/'script_new').exists()
