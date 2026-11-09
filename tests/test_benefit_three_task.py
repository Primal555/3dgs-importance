import copy
import hashlib
import json

import pytest
import torch

from gaussian_jscc import benefit_three_task as model


def test_tasks_preserve_numeric_loss_and_positive_rank_ignores_common_shift():
    from gaussian_jscc.benefit_footprint_rank_gradnorm import split_losses
    torch.manual_seed(1)
    p = torch.randn(5, 3, 2, dtype=torch.double, requires_grad=True)
    t = torch.randn_like(p); rates = torch.rand(5, 3, dtype=torch.double)
    new = model.three_losses(p, t, rates, .7, .2, .8, 1.3)
    old = split_losses(p, t, rates, .7, .2, .8, 1.3)
    torch.testing.assert_close(new['common']+new['relative'], old['magnitude'])
    shifted = model.three_losses(p+7., t, rates, .7, .2, .8, 1.3)
    torch.testing.assert_close(new['ranking'], shifted['ranking'])
    torch.testing.assert_close(new['relative'], shifted['relative'])
    assert not torch.allclose(new['common'], shifted['common'])
    assert model.three_losses(t, t, rates, .7, .2, .8, 1.3)['ranking'].abs().max() < 1e-12
    g = torch.autograd.grad(new['ranking'].sum(), p)[0]
    torch.testing.assert_close(g.sum(1), torch.zeros(5, 2, dtype=torch.double), atol=1e-12, rtol=0)


def test_accumulation_matches_full_batch_gradients_and_gradnorm():
    from gaussian_jscc.benefit_rank_gradnorm import GradNormBalancer
    torch.manual_seed(2)
    x = torch.randn(11, 2, dtype=torch.double)
    net = torch.nn.Linear(2, 3, dtype=torch.double)
    parameters = list(net.parameters())
    full = net(x).square().mean(0)
    expected = [torch.autograd.grad(v, parameters, retain_graph=True) for v in full]
    accumulator = model.TaskAccumulator(parameters, shared_indices=[0, 1])
    for start in range(0, 11, 4):
        per_record = net(x[start:start+4]).square()
        accumulator.add(per_record, torch.ones(len(per_record), dtype=torch.double)/11,
                        torch.arange(start, start+len(per_record)) % 2, diagnostics=True)
    torch.testing.assert_close(accumulator.losses, full.detach())
    for actual, want in zip(accumulator.gradients, expected):
        for a, b in zip(actual, want): torch.testing.assert_close(a, b)
    a = GradNormBalancer(list(model.TASKS), [1., 1., 1.], device='cpu')
    b = GradNormBalancer(list(model.TASKS), [1., 1., 1.], device='cpu')
    a.measure(full, parameters)
    b.measure_gradients(accumulator.losses, accumulator.shared_norms())
    a.update_weights(); b.update_weights()
    torch.testing.assert_close(a.logits, b.logits)
    weights = torch.tensor([.5, 1., 1.5], dtype=torch.double)
    accumulator.assign(weights)
    for j, p in enumerate(parameters):
        torch.testing.assert_close(p.grad, sum(weights[i]*expected[i][j] for i in range(3)))
    report = accumulator.stratum_report(weights)
    assert report['ordinary']['records'] == 6 and report['large_gap']['records'] == 5


def test_stratum_weighting_preserves_population_mean_and_resume():
    labels = [0]*90+[1]*10
    sampler = model.make_sampler(labels, 64, 14, stratified=True)
    chosen = sampler.next()
    correction = model.sampling_weights(labels, sampler, chosen)
    values = torch.tensor([1. if labels[i] == 0 else 11. for i in chosen], dtype=torch.double)
    assert float((values*correction).mean()) == pytest.approx(2.)
    state = sampler.state_dict(); next_batch = sampler.next()
    restored = model.make_sampler(labels, 64, 14, stratified=True); restored.load_state_dict(state)
    assert restored.next() == next_batch
    empty_tail = model.make_sampler([0]*5, 64, 3, stratified=True)
    assert model.sampling_weights([0]*5, empty_tail, empty_tail.next()).eq(1).all()


def test_runner_equal_draw_budgets_initialization_and_readonly_source(tmp_path):
    from test_benefit_isolated_positive_controls import prepared
    from gaussian_jscc import benefit_three_task_controls as controls
    source = prepared(tmp_path)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    out = tmp_path/'suite'
    args = controls.parser().parse_args(['--source', str(source), '--out', str(out),
        '--samples', '64', '--evaluate-every-samples', '64', '--cpu-threads', '1'])
    result = controls.main(args)
    assert result['status'] == 'complete'
    a = json.loads((out/'split_batch8/summary.json').read_text())
    b = json.loads((out/'split_batch64/summary.json').read_text())
    assert a['sample_visits'] == b['sample_visits'] == 64
    assert a['optimizer_updates'] == 8 and b['optimizer_updates'] == 1
    assert a['initial_branch_digests'] == b['initial_branch_digests']
    assert a['normalizers_unchanged'] and b['normalizers_unchanged']
    assert a['branch_changed']['common_head'] and a['branch_changed']['relative_head']
    settings = json.loads((out/'split_batch64/settings.json').read_text())
    assert settings['tasks'] == list(model.TASKS)
    assert settings['rank_pairs'] == [[1, 2], [1, 3], [2, 3]]
    rows = [json.loads(s) for s in (out/'split_batch64/metrics/optimization.jsonl').read_text().splitlines()]
    assert rows[0]['sample_visits'] == 64 and rows[0]['update'] == 1
    assert sum(rows[0]['strata'][s]['records'] for s in ('ordinary', 'large_gap')) == 64
    assert (out/'review.tar.gz').is_file()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())


def test_accumulated_runner_exact_resume_and_settings_guard(tmp_path, monkeypatch):
    from test_benefit_isolated_positive_controls import prepared
    from gaussian_jscc import benefit_three_task_controls as controls
    source = prepared(tmp_path)
    def args(out):
        return controls.parser().parse_args(['--source', str(source), '--out', str(out), '--arm', 'batch64',
            '--samples', '128', '--evaluate-every-samples', '64', '--cpu-threads', '1'])
    interrupted = tmp_path/'interrupted'; a = args(interrupted)
    step = torch.optim.AdamW.step; calls = 0
    def stop(self, *pos, **kw):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError('interrupted')
        return step(self, *pos, **kw)
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW, 'step', stop)
        with pytest.raises(RuntimeError, match='interrupted'): controls.main(a)
    for name in ('history', 'optimization'):
        with (interrupted/f'split_batch64/metrics/{name}.jsonl').open('a') as f:
            f.write('{"sample_visits":128,"unfinished":')
    a.resume = True; controls.main(a)
    controls.main(args(tmp_path/'straight'))
    left = torch.load(interrupted/'split_batch64/latest.pt', weights_only=True)
    right = torch.load(tmp_path/'straight/split_batch64/latest.pt', weights_only=True)
    for k, v in left['learner']['network'].items(): torch.testing.assert_close(v, right['learner']['network'][k], rtol=0, atol=0)
    assert torch.equal(left['balancer']['logits'], right['balancer']['logits'])
    assert left['sampler']['visits'] == right['sampler']['visits']
    (interrupted/'review.tar.gz').unlink()
    assert controls.main(a)['status'] == 'complete'
    a.microbatch = 4
    with pytest.raises(ValueError, match='settings mismatch'): controls.main(a)


def test_resume_trimming_rejects_interior_corruption(tmp_path):
    from gaussian_jscc.benefit_three_task_controls import trim_records
    path = tmp_path/'metrics.jsonl'
    text = '{"sample_visits":64}\nBROKEN\n{"sample_visits":128}\n'
    path.write_text(text)
    with pytest.raises(json.JSONDecodeError): trim_records(path, 64)
    assert path.read_text() == text
