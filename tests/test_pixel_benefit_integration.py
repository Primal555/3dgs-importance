import contextlib
import io
from unittest.mock import patch

import pytest
import torch

from test_structured_allocation_integration import source
from test_learned_benefit_allocation import cameras, read, assert_nested_equal
from test_pixel_benefit_learning import view
import test_budgeted_allocation as fixture


def options(parent, out):
    from gaussian_jscc.pixel_benefit_allocation import parser
    return parser().parse_args(['--run', str(parent), '--scene', 'truck', '--out', str(out), '--beta', '.1',
        '--pretrain-windows', '3', '--fit-windows', '1', '--allocation-steps', '2', '--net-updates', '1',
        '--batch-groups', '1', '--calibration-views', '1', '--window-views', '4', '--control-every', '1',
        '--validate-every', '1', '--save-every', '1', '--test-views', '2', '--trials', '1',
        '--image-views', '1', '--cpu-threads', '1', '--device', 'cpu', '--renderer-backend', 'cpu',
        '--benefit-validate-every', '1', '--no-lpips'])


class Window:
    def __init__(self, model, scene, table, scheduler, reference, args):
        self.table, self.difficulty = table, torch.zeros(1)
    def state_dict(self): return {'difficulty': self.difficulty}
    def load_state_dict(self, state): self.difficulty = state['difficulty']
    def collect(self, step, cameras, learner, replay, pretrain):
        a = view(1, .1)
        replay.add(dict(source=1, step=step, views=[a], importance_weight=1.))
        if not replay.validation_frozen:
            replay.add(dict(source=2, step=step, views=[view(1, .2)]), validation=True)
        n = len(self.table.logits)
        return dict(keep_costs=torch.zeros(n, 2), tier_ids=torch.tensor([0]),
                    tier_costs=torch.tensor([[.1, .2, .3]]), q=torch.ones(n, dtype=torch.long),
                    metrics=dict(source_mse=.1, source_distortion=.1))


def test_dry_run_has_own_family_and_no_output(source):
    from gaussian_jscc import pixel_benefit_allocation as module
    parent, root = source
    args = options(parent, root/'dry'); args.dry_run = True
    with contextlib.redirect_stdout(io.StringIO()): plan = module.main(args)
    assert plan['version'] == 'pixel_benefit_allocation_v1'
    assert not (root/'dry').exists()


def test_pretrain_resume_and_last_allocation_state(source):
    from gaussian_jscc import pixel_benefit_allocation as module
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch.object(module, 'PixelWindow', Window), patch.object(module, 'preflight', return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):
        args = options(parent, root/'run')
        module.main(args, max_updates=1)
        a = read(root/'run/checkpoints/latest/state.pt')
        assert a['progress']['pretrain_windows_done'] == 1 and a['replay']['validation_frozen']
        args.resume = True
        module.main(args)
        latest = read(root/'run/checkpoints/latest/state.pt')
        initial = read(root/'run/checkpoints/initial/state.pt')
        assert_nested_equal(initial['table'], latest['table'])
        assert latest['learner']['updates'] == 2
        args.phase = 'allocation'
        result = module.main(args)
        final = read(root/'run/checkpoints/final/state.pt')
        assert result['status'] == 'complete' and final['progress']['allocation_step'] == 2
        assert final['learner']['updates'] == 4
        assert (root/'run/deployment/latest_probabilities.npy').exists()
        assert (root/'run/evaluation/calibration_000004').exists()
        import json
        row = json.loads((root/'run/metrics/loss.jsonl').read_text().splitlines()[-1])
        assert {'beta', 'd_ref', 'c_ref', 'lambda_s', 'sampled_tier_counts', 'expected_cost_proxy',
                'peak_allocated_mib', 'network_updates'} <= row.keys()


def test_changed_plan_rejected(source):
    from gaussian_jscc import pixel_benefit_allocation as module
    parent, root = source
    args = options(parent, root/'changed')
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch.object(module, 'PixelWindow', Window), patch.object(module, 'preflight', return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):
        module.main(args, max_updates=1)
        args.resume = True; args.label_groups += 1
        with pytest.raises(ValueError, match='mismatch'): module.main(args)


def test_pending_validation_resume_does_not_repeat_training(source):
    from gaussian_jscc import pixel_benefit_allocation as module
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch.object(module, 'PixelWindow', Window), patch.object(module, 'preflight', return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):
        args = options(parent, root/'pending')
        def fail(learner):
            raise RuntimeError('interrupted validation')
        with patch.object(module.PixelLearner, 'validate', fail), pytest.raises(RuntimeError, match='interrupted'):
            module.main(args)
        before = read(root/'pending/checkpoints/latest/state.pt')
        assert before['progress']['pending'] == ['validation']
        args.resume = True
        module.main(args, max_updates=1)
        after = read(root/'pending/checkpoints/latest/state.pt')
        assert after['learner']['updates'] == before['learner']['updates'] + 1


def test_launcher_dry_run_creates_no_output(source):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():
        pytest.skip('Git Bash unavailable')
    parent, root = source
    out = root/'launcher'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONUTF8='1')
    script = Path(__file__).resolve().parents[1]/'scripts/test_pixel_benefit_allocation.sh'
    completed = subprocess.run([str(bash), str(script), parent.as_posix(), 'truck', out.as_posix(),
                                '--beta', '.1', '--calibration-views', '1', '--dry-run'],
                               env=env, capture_output=True, text=True, encoding='utf-8', timeout=40)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout[completed.stdout.index('{'):])['version'] == 'pixel_benefit_allocation_v1'
    assert not out.exists()


def test_preflight_cannot_consume_training_rng(source):
    from gaussian_jscc import pixel_benefit_allocation as module
    parent, root = source
    def consumes_rng(*args):
        torch.rand(100)
        return {}
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch.object(module, 'PixelWindow', Window), contextlib.redirect_stdout(io.StringIO()):
        baseline = options(parent, root/'plain')
        with patch.object(module, 'preflight', return_value={}): module.main(baseline)
        trial = options(parent, root/'consumed')
        with patch.object(module, 'preflight', side_effect=consumes_rng): module.main(trial)
    a = read(root/'plain/checkpoints/latest/state.pt')
    b = read(root/'consumed/checkpoints/latest/state.pt')
    assert_nested_equal(a['learner'], b['learner'])


def test_sparse_checkpoint_resume_matches_uninterrupted_training(source):
    from gaussian_jscc import pixel_benefit_allocation as module
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch.object(module, 'PixelWindow', Window), patch.object(module, 'preflight', return_value={}), \
         contextlib.redirect_stdout(io.StringIO()):
        full = options(parent, root/'full'); full.benefit_validate_every = 100; full.save_every = 100
        module.main(full)
        partial = options(parent, root/'partial'); partial.benefit_validate_every = 100; partial.save_every = 100
        module.main(partial, max_updates=1)
        partial.resume = True
        module.main(partial)
    a = read(root/'full/checkpoints/latest/state.pt')
    b = read(root/'partial/checkpoints/latest/state.pt')
    for key in ('learner', 'table', 'blocks', 'views', 'latest_q'):
        assert_nested_equal(a[key], b[key])


def test_checkpoint_manifest_survives_interrupted_replace(tmp_path):
    from gaussian_jscc import pixel_benefit_allocation as module
    from gaussian_jscc.pixel_benefit_replay import PixelReplay
    replay = PixelReplay(tmp_path/'replay', max_groups=1)
    replay.add(dict(source=1, step=0, views=[]))
    path = tmp_path/'checkpoints/latest/state.pt'
    old = replay.state_dict()
    module.save_replay_checkpoint(path, {'replay': old}, replay)
    replay.add(dict(source=2, step=1, views=[]))
    real = module._atomic_manifest
    calls = []
    def interrupt(target, data):
        calls.append(data)
        if len(calls) == 2:
            raise RuntimeError('interrupted after checkpoint replace')
        return real(target, data)
    with patch.object(module, '_atomic_manifest', side_effect=interrupt), pytest.raises(RuntimeError):
        module.save_replay_checkpoint(path, {'replay': replay.state_dict()}, replay)
    restored = PixelReplay(tmp_path/'replay', max_groups=1)
    restored.load_state_dict(read(path)['replay'])
    assert [r['source'] for r in restored.training_records()] == [2]
    assert (tmp_path/'replay'/old['training'][0]['path']).exists()
    assert module.save_replay_checkpoint(path, {'replay': replay.state_dict()}, replay) == 1
    assert not (tmp_path/'replay'/old['training'][0]['path']).exists()


def test_missing_manifest_blocks_replay_collection(tmp_path):
    from gaussian_jscc import pixel_benefit_allocation as module
    from gaussian_jscc.pixel_benefit_replay import PixelReplay
    replay = PixelReplay(tmp_path/'replay', max_groups=1)
    replay.add(dict(source=1, step=0, views=[]))
    foreign = tmp_path/'checkpoints/other/state.pt'
    foreign.parent.mkdir(parents=True)
    torch.save({'replay': replay.state_dict()}, foreign)
    with pytest.raises(FileNotFoundError, match='manifest'):
        module.save_replay_checkpoint(tmp_path/'checkpoints/latest/state.pt', {'replay': replay.state_dict()}, replay)


def test_checkpoint_interrupted_before_replace_preserves_old_state(tmp_path):
    from gaussian_jscc import pixel_benefit_allocation as module
    from gaussian_jscc.pixel_benefit_replay import PixelReplay
    replay = PixelReplay(tmp_path/'replay', max_groups=1)
    replay.add(dict(source=1, step=0, views=[]))
    path = tmp_path/'checkpoints/latest/state.pt'
    module.save_replay_checkpoint(path, {'replay': replay.state_dict()}, replay)
    replay.add(dict(source=2, step=1, views=[]))
    with patch.object(module.torch, 'save', side_effect=RuntimeError('interrupted')), pytest.raises(RuntimeError):
        module.save_replay_checkpoint(path, {'replay': replay.state_dict()}, replay)
    restored = PixelReplay(tmp_path/'replay', max_groups=1)
    restored.load_state_dict(read(path)['replay'])
    assert [r['source'] for r in restored.training_records()] == [1]


def test_gc_respects_other_checkpoint_manifest(tmp_path):
    from gaussian_jscc import pixel_benefit_allocation as module
    from gaussian_jscc.pixel_benefit_replay import PixelReplay
    replay = PixelReplay(tmp_path/'replay', max_groups=1)
    replay.add(dict(source=1, step=0, views=[]))
    older = replay.state_dict()
    module.save_replay_checkpoint(tmp_path/'checkpoints/initial/state.pt', {'replay': older}, replay)
    replay.add(dict(source=2, step=1, views=[]))
    module.save_replay_checkpoint(tmp_path/'checkpoints/latest/state.pt', {'replay': replay.state_dict()}, replay)
    restored = PixelReplay(tmp_path/'replay', max_groups=1)
    restored.load_state_dict(read(tmp_path/'checkpoints/initial/state.pt')['replay'])
    assert [r['source'] for r in restored.training_records()] == [1]
