import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from gaussian_jscc import fast_first_order_allocation as fast
from gaussian_jscc import teacher_constant_continue as entry
from test_fast_first_order_entry import args_for
from test_structured_allocation_integration import source
import test_budgeted_allocation as fixture
from test_successive_multisnr import digital_runner


def assert_same(a, b):
    if isinstance(a, torch.Tensor): torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_same(x, y)
    else: assert a == b


def teacher_args(parent, out, steps):
    args = args_for(parent, out)
    args.early_stop = False; args.allocation_steps = steps
    args.tier_max_log_odds = 2.; args.cost_protocol = 'digital_side_file_v1'
    args.keep_lr, args.tier_lr = .1, .4
    return args


def options(source, out):
    return entry.parser().parse_args(['--source', str(source), '--out', str(out), '--extra-steps', '2',
        '--device', 'cpu', '--digital-python', sys.executable, '--digital-device', 'cpu'])


def test_continue_preserves_allocation_trajectory_resume_and_delivery(source):
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch('comparison_baselines.run._execute', side_effect=digital_runner):
        old, full, continued, resumed = [root/n for n in ('old', 'full', 'continued', 'resumed')]
        fast.main(teacher_args(parent, old, 2)); fast.main(teacher_args(parent, full, 4))
        before = entry.load_checkpoint(old/'checkpoints/latest/state.pt')
        result = entry.main(options(old, continued))
        assert result['step'] == 4 and result['additional_updates'] == 2
        assert result['actual_side_delivery_completed']
        assert entry.main(options(old, resumed), max_updates=1)['status'] == 'interrupted'
        args = options(old, resumed); args.resume = True
        entry.main(args)
        expected = entry.load_checkpoint(full/'checkpoints/latest/state.pt')
        for path in (continued, resumed):
            saved = entry.load_checkpoint(path/'checkpoints/latest/state.pt')
            for name, value in expected['table'].items():
                torch.testing.assert_close(value, saved['table'][name], rtol=0, atol=0)
            assert torch.equal(expected['rng']['torch'], saved['rng']['torch'])
            assert torch.equal(expected['latest_q'], saved['latest_q'])
            assert_same(expected['views'], saved['views'])
            assert_same(expected['blocks'], saved['blocks'])
            assert_same(expected['objective_reference'], saved['objective_reference'])
            for key in ('d_ref', 'c_ref', 'lambda_s'):
                assert saved['objective_reference'][key] == before['objective_reference'][key]
            rows = entry.read(path/'actual_delivery/summary.json')['trials']
            assert len(rows) == 3 and all(r['status'] == 'success' for r in rows)
            assert (path/'review.tar.gz').exists()
        after = entry.load_checkpoint(old/'checkpoints/latest/state.pt')
        assert after['progress']['step'] == 2 and after['plan'] == before['plan']
        args = options(old, continued); args.resume = True
        with patch.object(entry, 'verify_delivery', wraps=entry.verify_delivery), \
             patch('comparison_baselines.run._execute', side_effect=AssertionError('duplicate delivery')):
            entry.main(args)


def test_reject_scheduled_or_changed_configuration(source):
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render):
        old = root/'old'; fast.main(teacher_args(parent, old, 2))
        state = entry.load_checkpoint(old/'checkpoints/latest/state.pt')
        complete = entry.read(old/'complete.json')
        broken = copy.deepcopy(state); broken['plan']['settings']['early_stop'] = True
        with pytest.raises(ValueError, match='constant-LR'):
            entry.validate_source(broken['plan'], broken, complete)
        args = options(old, root/'new')
        entry.main(args, max_updates=1)
        args.resume = True; args.extra_steps = 3
        with pytest.raises(ValueError, match='settings mismatch'):
            entry.main(args)


def test_launcher_dry_run_keeps_original_plan(source):
    parent, root = source
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists(): pytest.skip('Git Bash unavailable')
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render):
        old = root/'old'; fast.main(teacher_args(parent, old, 2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', PYTHON_BIN=Path(sys.executable).as_posix(),
               DIGITAL_PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8='1', RESUME='0')
    script = Path('scripts/continue_teacher_constant_lr.sh').resolve()
    run = subprocess.run([str(bash), script.as_posix(), old.as_posix(), (root/'dry').as_posix(), '--dry-run'],
        env=env, capture_output=True, text=True, encoding='utf8', timeout=45)
    assert run.returncode == 0, run.stdout+run.stderr
    plan = json.loads(run.stdout[run.stdout.index('{'):])
    assert plan['source_step'] == 2 and plan['target_step'] == 502
    assert plan['original_plan'] == entry.read(old/'plan.json')
    assert not (root/'dry').exists()
