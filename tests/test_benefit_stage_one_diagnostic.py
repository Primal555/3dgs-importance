import hashlib
import json

import pytest
import torch

from gaussian_jscc import benefit_stage_one_diagnostic as diagnostic


def test_paired_choice_harm_and_component_decomposition():
    group = dict(source=7, step=19, rates=torch.tensor([0., .1, .2]),
        views=[dict(values=torch.tensor([[0., 0.], [.2, .1], [.4, .2]]))])
    original = torch.tensor([[[0., 0.], [.2, .1], [.4, .2]]])
    repaired = torch.tensor([[[1., 0.], [.2, .1], [-1., 0.]]])
    row = diagnostic.compare_record(dict(id='sample'), group, original, repaired)
    assert row['category'] == 'new_wrong_choice'
    assert row['original']['choice'] == 1 and row['repaired']['choice'] == 3
    assert row['regret_change'] == pytest.approx(.8)
    assert row['true_switch_components'] == pytest.approx([.4, .2, .2])
    summary = diagnostic.summarize_records([row], 10)
    assert summary['paired']['harmful_switches'] == 1
    assert summary['paired']['harm_total'] == pytest.approx(.8)
    assert summary['paired']['mean_regret_change'] == pytest.approx(.8)
    assert summary['paired']['top10_harm_fraction'] == 1.


def test_different_tied_best_choices_are_both_correct():
    g = dict(source=1, step=2, rates=torch.zeros(3), views=[dict(values=torch.zeros(3, 2))])
    a = torch.tensor([[[0., 0.], [1., 0.], [2., 0.]]])
    b = torch.tensor([[[2., 0.], [0., 0.], [1., 0.]]])
    row = diagnostic.compare_record(dict(id='tie'), g, a, b)
    assert row['category'] == 'both_correct'
    assert row['regret_change'] == 0
    assert diagnostic.summarize_records([row], 3)['paired']['harmful_switches'] == 0
    with pytest.raises(ValueError, match='finite'):
        diagnostic.compare_record(dict(id='bad'), g, a*float('nan'), b)


def test_completed_offline_fixture_readonly_reconciles_and_packages(tmp_path, monkeypatch):
    from test_benefit_stage_one import source
    from gaussian_jscc import benefit_stage_one as offline
    root, _ = source(tmp_path)
    run = tmp_path/'offline'
    offline.main(offline.parser().parse_args(['--run', str(root), '--out', str(run),
        '--updates', '1', '--cpu-threads', '1']))
    files = [p for r in (run, root) for p in r.rglob('*') if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    def forbidden(*a, **k):
        pytest.fail('diagnostic attempted optimizer update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    out = tmp_path/'diagnostic'
    args = diagnostic.parser().parse_args(['--run', str(run), '--out', str(out), '--cpu-threads', '1'])
    result = diagnostic.main(args)
    assert result['status'] == 'complete' and result['records'] == 2
    assert result['renderer_calls'] == result['optimizer_updates'] == 0
    assert result['reconciliation']['matched'] is True
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items())
    assert (out/'review.tar.gz').is_file()
    rows = [json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and all(r['source'] in (100, 101) for r in rows)
    with pytest.raises(FileExistsError):
        diagnostic.main(args)
    args.out = str(run/'bad')
    with pytest.raises(ValueError, match='separate'):
        diagnostic.main(args)
    manifest = json.loads((run/'data_manifest.json').read_text())
    manifest['replay_root'] = str(tmp_path/'missing_replay')
    (run/'data_manifest.json').write_text(json.dumps(manifest))
    args.out = str(tmp_path/'missing')
    with pytest.raises(FileNotFoundError, match='replay'):
        diagnostic.main(args)
    assert not (tmp_path/'missing_replay').exists()


def test_changed_recorded_metrics_are_reported_as_mismatch(tmp_path):
    from test_benefit_stage_one import source
    from gaussian_jscc import benefit_stage_one as offline
    root, _ = source(tmp_path)
    run = tmp_path/'offline'
    offline.main(offline.parser().parse_args(['--run', str(root), '--out', str(run),
        '--updates', '0', '--cpu-threads', '1']))
    path = run/'summary.json'
    summary = json.loads(path.read_text())
    summary['after']['repaired']['fixed_validation']['value_error']['window_rd_rmse'] += .1
    path.write_text(json.dumps(summary))
    out = tmp_path/'diagnostic'
    args = diagnostic.parser().parse_args(['--run', str(run), '--out', str(out), '--cpu-threads', '1'])
    with pytest.raises(RuntimeError, match='reconciliation mismatch'):
        diagnostic.main(args)
    report = json.loads((out/'summary.json').read_text())
    assert report['status'] == 'reconciliation_mismatch'
    assert report['reconciliation']['matched'] is False


def test_shell_entry_uses_fixed_comparison(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from test_benefit_stage_one import source
    from gaussian_jscc import benefit_stage_one as offline
    root, _ = source(tmp_path)
    run = tmp_path/'offline'
    offline.main(offline.parser().parse_args(['--run', str(root), '--out', str(run),
        '--updates', '0', '--cpu-threads', '1']))
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():
        pytest.skip('Git Bash unavailable')
    out = tmp_path/'shell_check'
    env = dict(os.environ, DEVICE='cpu', PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONUTF8='1')
    completed = subprocess.run([str(bash), 'scripts/test_benefit_stage_one_diagnostic.sh',
        run.as_posix(), out.as_posix(), '--cpu-threads', '1'], env=env,
        capture_output=True, text=True, encoding='utf8', timeout=30)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    assert json.loads((out/'summary.json').read_text())['status'] == 'complete'
