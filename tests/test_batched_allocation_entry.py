import contextlib
import io
import json
from unittest.mock import patch

import pytest

from test_structured_allocation_integration import source
from test_learned_benefit_allocation import cameras, read, assert_nested_equal
from test_calibrated_spatial_benefit_integration import options
from test_spatial_benefit_integration import Window
import test_budgeted_allocation as fixture
from gaussian_jscc import calibrated_spatial_benefit_allocation as entry


class FastWindow(Window):
    def __init__(self, *args):
        super().__init__(*args)
        self.batch_size = args[-1].allocation_batch

    @property
    def execution_settings(self):
        return dict(mode='batched_spatial_v1', allocation_batch=self.batch_size)


def test_original_allocation_checkpoint_resumes_with_larger_execution_batch(source):
    parent, root = source
    args = options(parent, root/'fast_resume')
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch.object(entry.spatial, 'SpatialWindow', Window), \
         patch.object(entry.spatial, 'preflight', return_value={}), \
         patch.object(entry, 'BatchedSpatialWindow', FastWindow), \
         patch.object(entry, 'verify_batched_execution', return_value={'batched_projection_max_abs': 0.}), \
         contextlib.redirect_stdout(io.StringIO()):
        entry.main(args)
        args.resume = True; args.phase = 'allocation'
        entry.main(args, max_updates=1)
        path = root/'fast_resume/checkpoints/latest/state.pt'
        before = read(path)
        args.allocation_batch = 64
        entry.main(args)
        after = read(path)
        assert after['progress']['allocation_step'] == 2
        assert after['learner']['updates'] == before['learner']['updates']+1
        assert_nested_equal(before['state_dict'], after['state_dict'])
        assert after['plan'] == before['plan']
        assert 'allocation_batch' not in after['plan']['settings']
        assert after['execution']['allocation_batch'] == 64
        rows = [json.loads(x) for x in (root/'fast_resume/metrics/loss.jsonl').read_text().splitlines()]
        assert rows[-1]['execution']['allocation_batch'] == 64
        args.allocation_batch = 128
        entry.main(args)
        assert read(path)['learner']['updates'] == after['learner']['updates']
        args.beta = .5
        with pytest.raises(ValueError, match='mismatch'):
            entry.main(args)


def test_runtime_batch_parser_and_preflight():
    from gaussian_jscc.batched_spatial_window import verify_batched_execution
    args = entry.parser().parse_args(['--run', 'source', '--scene', 'truck', '--out', 'same',
        '--beta', '.1', '--resume', '--phase', 'allocation', '--allocation-batch', '64'])
    assert args.allocation_batch == 64
    report = verify_batched_execution('cpu')
    assert report['batched_projection_max_abs'] < 2e-6
    assert report['batched_features_max_abs'] < 2e-5


@pytest.mark.parametrize('phase,batch', [('pretrain', 64), ('allocation', 0), ('allocation', -1)])
def test_invalid_runtime_batch_is_rejected_before_source_loading(phase, batch):
    args = entry.parser().parse_args(['--run', 'missing', '--scene', 'truck', '--out', 'same',
        '--beta', '.1', '--phase', phase, '--resume'])
    args.allocation_batch = batch
    with pytest.raises(ValueError, match='allocation'):
        entry.main(args)
