import json
from pathlib import Path

import pytest
import torch

from gaussian_jscc import benefit_three_task_controls as controls


def test_completed_budget_extension_matches_one_continuous_run(tmp_path, monkeypatch):
    from test_benefit_isolated_positive_controls import prepared
    source = prepared(tmp_path)
    def args(out, samples, resume=False):
        argv = ['--source', str(source), '--out', str(out), '--objective', 'view_gap',
                '--arm', 'batch64', '--samples', str(samples), '--evaluate-every-samples', '64', '--cpu-threads', '1']
        if resume: argv.append('--resume')
        return controls.parser().parse_args(argv)
    out = tmp_path/'extended'
    controls.main(args(out, 64))
    first = json.loads((out/'split_batch64/metrics/history.jsonl').read_text().splitlines()[-1])
    rename = Path.rename
    def interrupt_archive(self, target):
        if self == out/'review.tar.gz': raise OSError('archive interrupted')
        return rename(self, target)
    with monkeypatch.context() as m:
        m.setattr(Path, 'rename', interrupt_archive)
        with pytest.raises(OSError, match='archive interrupted'):
            controls.main(args(out, 128, True))
    assert json.loads((out/'split_batch64/settings.json').read_text())['sample_budget'] == 64
    assert (out/'review.tar.gz').is_file()
    controls.main(args(out, 128, True))
    controls.main(args(tmp_path/'continuous', 128))
    left = torch.load(out/'split_batch64/latest.pt', weights_only=True)
    right = torch.load(tmp_path/'continuous/split_batch64/latest.pt', weights_only=True)
    for k, v in left['learner']['network'].items():
        torch.testing.assert_close(v, right['learner']['network'][k], atol=0, rtol=0)
    torch.testing.assert_close(left['balancer']['logits'], right['balancer']['logits'], atol=0, rtol=0)
    assert left['sampler']['visits'] == right['sampler']['visits']
    assert left['gradient_totals'] == right['gradient_totals']
    assert json.loads((out/'split_batch64/complete.json').read_text())['sample_visits'] == 128
    history = [json.loads(s) for s in (out/'split_batch64/metrics/history.jsonl').read_text().splitlines()]
    assert [r['sample_visits'] for r in history] == [0, 64, 128]
    assert history[1] == first
    assert json.loads((out/'plan.json').read_text())['sample_budget'] == 128
    assert json.loads((out/'split_batch64/settings.json').read_text())['optimizer_updates'] == 2
    assert (out/'review_64.tar.gz').is_file()
    with pytest.raises(ValueError, match='settings mismatch'):
        controls.main(args(out, 64, True))
    altered = args(out, 192, True); altered.microbatch = 4
    plan_before = (out/'plan.json').read_bytes()
    with pytest.raises(ValueError, match='settings mismatch'):
        controls.main(altered)
    assert (out/'plan.json').read_bytes() == plan_before
