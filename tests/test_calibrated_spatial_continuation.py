"""Continuation must preserve the experiment and extend only its stopping point."""
import contextlib
import io
import json
from unittest.mock import patch

import pytest
import torch

from test_structured_allocation_integration import source
from test_learned_benefit_allocation import cameras, read, assert_nested_equal
from test_calibrated_spatial_benefit_integration import options
from test_spatial_benefit_integration import Window
import test_budgeted_allocation as fixture
from gaussian_jscc import calibrated_spatial_benefit_allocation as module


@contextlib.contextmanager
def small_experiment():
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch.object(module.spatial, 'SpatialWindow', Window), \
         patch.object(module.spatial, 'preflight', return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):
        yield


def test_completed_legacy_run_extends_and_restores_target_on_interruption(source):
    parent, root = source
    out = root/'extended'
    args = options(parent, out)
    with small_experiment():
        module.main(args)
        path = out/'checkpoints/latest/state.pt'
        before = read(path)
        # The server's existing 200-window checkpoint predates this progress field.
        legacy = read(path)
        legacy['progress'].pop('pretrain_target', None)
        torch.save(legacy, path)
        args.resume = True
        args.pretrain_until = 5
        result = module.main(args, max_updates=1)
        midway = read(path)
        assert result['step'] == 4
        assert midway['progress']['pretrain_target'] == 5
        assert not midway['progress']['pretrain_complete']
        assert midway['learner']['updates'] == before['learner']['updates'] + 1
        assert_nested_equal(before['learner']['scales'], midway['learner']['scales'])
        for key in ('state_dict', 'table', 'objective_reference', 'plan'):
            assert_nested_equal(before[key], midway[key])
        assert before['replay']['validation'] == midway['replay']['validation']
        assert not (out/'pretrain_complete.json').exists()
        archived = json.loads((out/'pretrain_complete_000003.json').read_text())
        assert archived['step'] == 3
        args.pretrain_until = None
        result = module.main(args)
        assert result['step'] == 5 and result['pretrain_target'] == 5
        assert result['network_updates'] == 4
        final = read(path)
        assert final['progress']['pretrain_complete']
        assert final['plan']['settings']['pretrain_windows'] == 3
        assert 'pretrain_until' not in final['plan']['settings']
        rows = [json.loads(x) for x in (out/'metrics/benefit_validation.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == [1, 2, 3, 4, 5]
        module.main(args)
        assert_nested_equal(final['learner'], read(path)['learner'])


def test_extension_matches_continuous_training(source):
    parent, root = source
    with small_experiment():
        full = options(parent, root/'continuous'); full.pretrain_windows = 5
        module.main(full)
        part = options(parent, root/'continued')
        module.main(part)
        part.resume = True; part.pretrain_until = 5
        module.main(part)
    a = read(root/'continuous/checkpoints/latest/state.pt')
    b = read(root/'continued/checkpoints/latest/state.pt')
    for key in ('learner', 'table', 'state_dict', 'blocks', 'views', 'latest_q', 'rng'):
        assert_nested_equal(a[key], b[key])
    assert sorted(a['replay']['usage'].values()) == sorted(b['replay']['usage'].values())


@pytest.mark.parametrize('resume,phase,target', [
    (False, 'pretrain', 5), (True, 'allocation', 5), (True, 'pretrain', 2),
])
def test_extension_rejects_fresh_wrong_phase_and_shortened_target(source, resume, phase, target):
    parent, root = source
    args = options(parent, root/'invalid')
    with small_experiment():
        module.main(args)
        args.resume = resume; args.phase = phase; args.pretrain_until = target
        with pytest.raises(ValueError, match='pretrain'):
            module.main(args)


def test_saved_extension_cannot_shrink_or_change_original_configuration(source):
    parent, root = source
    args = options(parent, root/'identity')
    with small_experiment():
        module.main(args)
        args.resume = True; args.pretrain_until = 6
        module.main(args, max_updates=1)
        args.pretrain_until = 5
        with pytest.raises(ValueError, match='pretrain'):
            module.main(args)
        args.pretrain_until = 6; args.benefit_lr /= 10
        with pytest.raises(ValueError, match='mismatch'):
            module.main(args)


def test_parser_exposes_explicit_absolute_target():
    args = module.parser().parse_args([
        '--run', 'source', '--scene', 'truck', '--out', 'same_out', '--beta', '.1',
        '--resume', '--phase', 'pretrain', '--pretrain-until', '700',
    ])
    assert args.pretrain_until == 700 and args.pretrain_windows == 200


def test_interrupted_marker_archive_resumes_with_saved_target(source):
    from gaussian_jscc import pixel_benefit_allocation as base
    parent, root = source
    out = root/'archive_recovery'
    args = options(parent, out)
    with small_experiment():
        module.main(args)
        args.resume = True; args.pretrain_until = 5
        with patch.object(base, '_archive_pretrain_completion', side_effect=OSError('archive interrupted')):
            with pytest.raises(OSError, match='archive interrupted'):
                module.main(args)
        state = read(out/'checkpoints/latest/state.pt')
        assert state['progress']['pretrain_target'] == 5
        assert state['progress']['pretrain_windows_done'] == 3
        assert not state['progress']['pretrain_complete']
        args.pretrain_until = None; args.phase = 'allocation'
        with pytest.raises(ValueError, match='finish pretraining'):
            module.main(args)
        args.phase = 'pretrain'
        assert module.main(args)['step'] == 5
        assert json.loads((out/'pretrain_complete_000003.json').read_text())['step'] == 3
