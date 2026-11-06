import hashlib
import json

import pytest
import torch

from gaussian_jscc import benefit_rd_choice as entry


def comparison(tmp_path, batch_groups=None):
    from test_benefit_stage_one import source
    from gaussian_jscc import benefit_stage_one as old
    root, checkpoint = source(tmp_path)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    torch.save(dict(settings=state['plan']['settings']), root/'benefit_network.pt')
    run = tmp_path/'comparison'
    options = ['--run', str(root), '--out', str(run), '--updates', '3',
        '--evaluate-every', '1', '--train-groups', '100', '--cpu-threads', '1']
    if batch_groups is not None:
        options += ['--batch-groups', str(batch_groups)]
    old.main(old.parser().parse_args(options))
    return root, run


def test_fair_single_arm_uniform_coverage_and_source_readonly(tmp_path, monkeypatch, capsys):
    root, run = comparison(tmp_path)
    files = [p for r in (root, run) for p in r.rglob('*') if p.is_file()]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    import gaussian_jscc.rendering
    monkeypatch.setattr(gaussian_jscc.rendering, 'render', lambda *a, **k: pytest.fail('render called'))
    out = tmp_path/'rd'
    args = entry.parser().parse_args(['--run', str(run), '--out', str(out), '--evaluate-every', '1', '--cpu-threads', '1'])
    result = entry.main(args)
    assert result['status'] == 'complete'
    assert result['additional_updates'] == 3
    assert result['source_initial_verified'] and result['reference_final_verified']
    assert result['renderer_calls'] == 0 and result['trained_arms'] == 1
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    assert (out/'review.tar.gz').is_file() and (out/'final.pt').is_file()
    history = [json.loads(line) for line in (out/'metrics/history.jsonl').read_text().splitlines()]
    assert [r['additional_updates'] for r in history] == [0, 1, 2, 3]
    assert all('ordinary' in r['validation'] and 'large_gap' in r['validation'] for r in history)
    settings = json.loads((out/'settings.json').read_text())
    assert settings['sampling'] == 'uniform shuffled cycles; original source-importance weights'
    assert result['training_coverage']['draws'] == 6
    assert result['training_coverage']['never_visited'] == 1
    text = capsys.readouterr().out
    assert '普通样本' in text and '高差距样本' in text
    with pytest.raises(FileExistsError):
        entry.main(args)


def test_missing_configuration_and_tampered_reference_fail_before_training(tmp_path):
    root, run = comparison(tmp_path)
    initial_path = run/'initial.pt'
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    initial['settings'].pop('evaluator_configuration', None)
    initial['settings'].pop('batch_groups', None)
    torch.save(initial, initial_path)
    (root/'benefit_network.pt').unlink()
    args = entry.parser().parse_args(['--run', str(run), '--out', str(tmp_path/'rd'), '--cpu-threads', '1', '--batch-groups', '2'])
    with pytest.raises(ValueError, match='evaluator settings'):
        entry.main(args)
    # Existing complete immutable metadata can be relocated explicitly.
    config = torch.load(root/'allocation/checkpoints/latest/state.pt', map_location='cpu', weights_only=True)['plan']['settings']
    torch.save(dict(settings=config), root/'benefit_network.pt')
    path = run/'summary.json'
    summary = json.loads(path.read_text())
    summary['after']['original']['fixed_validation']['three_tier']['regret_mean'] += .1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='reference final'):
        entry.main(args)
    assert not (tmp_path/'rd').exists()


def test_actual_batch_override_is_saved_and_conflicting_size_rejected(tmp_path):
    root, run = comparison(tmp_path)
    args = entry.parser().parse_args(['--run', str(run), '--out', str(tmp_path/'bad'),
        '--cpu-threads', '1', '--batch-groups', '3'])
    with pytest.raises(ValueError, match='batch size'):
        entry.main(args)
    path = run/'initial.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    saved['settings'].pop('batch_groups')
    torch.save(saved, path)
    args.batch_groups = None
    with pytest.raises(ValueError, match='--batch-groups'):
        entry.main(args)


def test_recorded_offline_override_controls_training_draws_and_preserves_normalizers(tmp_path):
    root, run = comparison(tmp_path, batch_groups=3)
    out = tmp_path/'rd'
    args = entry.parser().parse_args(['--run', str(run), '--out', str(out), '--cpu-threads', '1'])
    result = entry.main(args)
    assert result['training_coverage']['draws'] == 9
    initial = torch.load(run/'initial.pt', map_location='cpu', weights_only=True)['learner']['network']
    final = torch.load(out/'final.pt', map_location='cpu', weights_only=True)['learner']['network']
    for key in ('input_mean', 'input_scale', 'context_mean', 'context_scale', 'state_mean', 'state_scale', 'output_scale'):
        torch.testing.assert_close(initial[key], final[key], atol=0, rtol=0)


def test_shell_entry_runs_one_new_arm(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root, run = comparison(tmp_path)
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():
        pytest.skip('Git Bash unavailable')
    out = tmp_path/'shell'
    env = dict(os.environ, DEVICE='cpu', PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONUTF8='1')
    result = subprocess.run([str(bash), 'scripts/test_benefit_rd_choice.sh', run.as_posix(),
        out.as_posix(), '--cpu-threads', '1'], env=env, capture_output=True, text=True,
        encoding='utf8', timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads((out/'summary.json').read_text())['trained_arms'] == 1
