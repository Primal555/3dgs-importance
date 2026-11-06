import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from gaussian_jscc import benefit_stage_one as entry
from gaussian_jscc.full_history_spatial_replay import FullHistorySpatialReplay
from gaussian_jscc.neural_correction_learning import CorrectionLearner
from test_spatial_benefit import collision_inputs


def source(tmp_path):
    root = tmp_path/'source'
    replay = FullHistorySpatialReplay(root/'allocation/neural_replay', max_groups=2,
                                     validation_groups=2, cache_mib=1)
    inputs = collision_inputs()[0][0]
    def record(i, validation=False):
        return dict(source=i, step=i+1, validation=validation, importance_weight=1.,
            rates=torch.tensor([.001, .002, .004], dtype=torch.float64),
            views=[dict(inputs=copy.deepcopy(inputs),
                        values=torch.tensor([[-.01, -.002], [-.02, -.004], [-.03, -.005]], dtype=torch.float64))])
    for i in range(6):
        replay.add(record(i))
    replay.add(record(100, True), True)
    replay.add(record(101, True), True)
    replay.freeze_validation()
    learner = CorrectionLearner(replay, 'cpu', hidden=16, dropout=.05)
    learner.fit_scales()
    gap = record(7)
    gap.update(supervision='action_gap', old_tier=1, new_tier=2,
               delta_values=torch.tensor([-.01, -.002]), rate_delta=.001)
    replay.add(gap)
    settings = dict(hidden=16, benefit_lr=1e-4, dropout=.05, weight_decay=1e-4,
        decision_weight=.25, tie_fraction=.02, common_weight=1., pair_weight=1.,
        training_probe_groups=4, batch_groups=2, replay_cache_mib=1.)
    state = dict(plan=dict(settings=settings), progress=dict(step=7),
        positive_provider=dict(learner=learner.state_dict()), replay=replay.state_dict())
    path = root/'allocation/checkpoints/latest/state.pt'
    path.parent.mkdir(parents=True)
    torch.save(state, path)
    return root, path


def test_zero_render_offline_pair_source_untouched_and_readable_metrics(tmp_path, capsys, monkeypatch):
    root, checkpoint = source(tmp_path)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    import gaussian_jscc.rendering
    monkeypatch.setattr(gaussian_jscc.rendering, 'render', lambda *a, **k: pytest.fail('offline renderer called'))
    out = tmp_path/'offline'
    args = entry.parser().parse_args(['--run', str(root), '--out', str(out), '--device', 'cpu',
        '--updates', '3', '--evaluate-every', '2', '--train-groups', '100', '--probe-groups', '6',
        '--cpu-threads', '1'])
    result = entry.main(args)
    assert result['status'] == 'complete' and result['renderer_calls'] == 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    assert result['probability_table_updated'] is False
    rows = [json.loads(s) for s in (out/'metrics/history.jsonl').read_text(encoding='utf8').splitlines()]
    assert [r['additional_updates'] for r in rows] == [0, 2, 3]
    assert rows[0]['original']['fixed_validation'] == rows[0]['repaired']['fixed_validation']
    assert rows[-1]['original']['fixed_validation']['positive_window']['comparisons'] == 6
    assert rows[-1]['repaired']['recent_training']['action_gap_groups'] > 0
    assert (out/'original/final.pt').is_file() and (out/'repaired/final.pt').is_file()
    assert (out/'review.tar.gz').is_file()
    assert result['validation_groups'] == 2 and result['training_groups'] == 7
    text = capsys.readouterr().out
    assert '不更新概率表' in text and '固定留出集' in text and '近期训练回放' in text
    assert '三档选优代价' in text
    with pytest.raises(FileExistsError):
        entry.main(args)


def test_evaluation_only_and_source_overlap_are_handled(tmp_path):
    root, _ = source(tmp_path)
    args = entry.parser().parse_args(['--run', str(root), '--out', str(tmp_path/'only'),
        '--updates', '0', '--device', 'cpu', '--train-groups', '4', '--cpu-threads', '1'])
    result = entry.main(args)
    assert result['additional_updates'] == 0
    args.out = str(root/'bad')
    with pytest.raises(ValueError, match='separate'):
        entry.main(args)


def test_paired_dropout_matches_for_mixed_full_and_action_records(tmp_path):
    from gaussian_jscc.benefit_stage_one_data import snapshot, read_groups
    from gaussian_jscc.benefit_stage_one_losses import fit_positive_scales
    root, _ = source(tmp_path)
    saved = snapshot(root)
    records = list(read_groups(saved['replay'], saved['replay'].training))
    mixed = [records[-1], records[0], records[1]]
    scales = fit_positive_scales(records)
    a = entry.make_learner(saved, 'cpu')
    b = entry.make_learner(saved, 'cpu', True, scales)
    captured = [[], []]
    hooks = [net.network.register_forward_hook(lambda module, inp, out, i=i:
             captured[i].append(out.detach().clone())) for i, net in enumerate((a, b))]
    for learner in (a, b):
        learner.network.train()
        torch.manual_seed(19)
        learner._batch_losses(mixed)
    for h in hooks:
        h.remove()
    assert len(captured[0]) == len(captured[1]) == 2
    for x, y in zip(*captured):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_snapshot_survives_live_checkpoint_replacement(tmp_path):
    from gaussian_jscc.benefit_stage_one_data import snapshot
    root, path = source(tmp_path)
    saved = snapshot(root)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    payload['progress']['step'] = 99
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)
    assert saved['source_step'] == 7
    assert snapshot(root)['source_step'] == 99


def test_shell_cpu_entry_runs_without_cuda_or_scene_inputs(tmp_path):
    import os
    import subprocess
    import sys
    root, _ = source(tmp_path)
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():
        pytest.skip('Git Bash unavailable')
    out = tmp_path/'shell'
    env = dict(os.environ, DEVICE='cpu', PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONUTF8='1')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    run = subprocess.run([str(bash), 'scripts/test_benefit_stage_one.sh', root.as_posix(),
        out.as_posix(), '--updates', '0', '--cpu-threads', '1'],
        env=env, capture_output=True, text=True, encoding='utf8', timeout=40)
    assert run.returncode == 0, run.stdout+run.stderr
    assert json.loads((out/'summary.json').read_text(encoding='utf8'))['renderer_calls'] == 0
