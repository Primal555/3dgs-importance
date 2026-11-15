import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc import successive_tier_refinement as entry
from gaussian_jscc import successive_q3_refinement as base
from gaussian_jscc.multisnr_training import cached_render_step
from gaussian_jscc.render_objective import MultiViewRenderTask
from test_successive_q3_refinement import load_model_scene, assert_same_checkpoint
from test_successive_multisnr import shared, digital_runner
from test_render_first import synthetic_render
from test_successive_joint import cameras


def options(source, out):
    return entry.parser().parse_args(['--run', str(source), '--out', str(out), '--device', 'cpu',
        '--digital-python', sys.executable, '--digital-device', 'cpu', '--scenes', 'truck', 'train',
        '--heldout-scenes', 'drjohnson', '--steps-per-scene', '2', '--save-every', '1',
        '--validate-every', '1', '--trials', '1', '--cpu-threads', '1', '--no-lpips'])


@pytest.mark.parametrize('tier', [1, 2, 3])
def test_each_stage_gets_render_gradients_only_on_active_pair(shared, tmp_path, tier):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
        args, model, scene, _ = load_model_scene(shared, tmp_path)
        opt = entry.configure_optimizer(model, tier, args.lr)
        before = base.frozen_digest(model.state_dict(), entry.prefixes(tier))
        targets = [b['y'] for b in base.cache_teacher(model, scene)]
        task = MultiViewRenderTask(scene.cameras[:1], scene.reference, scene.degree)
        cached_render_step(model, base.CachedQ3Batches(scene, targets, tier), scene.geometry, 10., task, 0.)
        for prefix in entry.prefixes(tier):
            assert sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                       if n.startswith(prefix) and p.grad is not None) > 0
        assert all(n.startswith(entry.prefixes(tier)) for n, p in model.named_parameters() if p.grad is not None)
        opt.step()
        assert base.frozen_digest(model.state_dict(), entry.prefixes(tier)) == before


@pytest.mark.parametrize('interrupt', [1, 4, 5])
def test_sequential_exact_resume_stage_freezing_and_actual_cost(shared, tmp_path, interrupt):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('comparison_baselines.run._execute', side_effect=digital_runner):
        full, resumed = tmp_path/'full', tmp_path/'resumed'
        result = entry.main(options(shared, full))
        assert result['step'] == 12 and result['completed_stages'] == [1, 2, 3]
        assert result['lower_tiers_unchanged'] and result['heldout_optimizer_updates'] == 0
        assert entry.main(options(shared, resumed), max_updates=interrupt)['status'] == 'interrupted'
        args = options(shared, resumed); args.resume = True
        entry.main(args)
        a = torch.load(full/'checkpoints/final/codec.pt', weights_only=True)
        b = torch.load(resumed/'checkpoints/final/codec.pt', weights_only=True)
        assert_same_checkpoint(a, b)
        checkpoints = [torch.load(full/f'checkpoints/{name}/codec.pt', weights_only=True)
                       for name in ('initial', 'after_q1', 'after_q2', 'final')]
        for tier in (1, 2, 3):
            first, last = checkpoints[tier-1]['state_dict'], checkpoints[tier]['state_dict']
            assert base.frozen_digest(first, entry.prefixes(tier)) == base.frozen_digest(last, entry.prefixes(tier))
            for prefix in entry.prefixes(tier):
                assert any(not torch.equal(v, last[n]) for n, v in first.items() if n.startswith(prefix))
            states = checkpoints[tier]['training']['optimizer']['state'].values()
            assert {int(s['step']) for s in states} == {4}  # Adam restarted at each tier.
        report = entry.read(full/'review_summary.json')
        assert report['lower_tier_checks'] and all(c['passed'] for c in report['lower_tier_checks'])
        actual = entry.read(full/'actual_delivery/truck/summary.json')['trials']
        assert {r['method'] for r in actual} == {'1', '2', '3', 'mixed'}
        for r in actual:
            row = next(x for x in report['rows'] if x['endpoint'] == 'final' and x['scene'] == 'truck'
                       and x['path'] == 'awgn' and x['layout'] == r['method'])
            assert r['total_channel_uses'] == row['total_channel_uses']
            assert r['metrics']['source_psnr'] == pytest.approx(row['source_psnr'], abs=1e-5)
        records = [json.loads(s) for s in (resumed/'metrics/loss.jsonl').read_text().splitlines()]
        assert [r['step'] for r in records] == list(range(1, 13))
        assert (full/'review.tar.gz').exists()


def test_launcher_dry_run_resolves_stage_budget(shared, tmp_path):
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists(): pytest.skip('Git Bash unavailable')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHON_BIN=Path(sys.executable).as_posix(),
               DIGITAL_PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8='1', RESUME='0')
    script = Path('scripts/test_successive_tier_refinement.sh').resolve()
    run = subprocess.run([str(bash), script.as_posix(), shared.as_posix(), (tmp_path/'dry').as_posix(),
        '--heldout-scenes', 'drjohnson', '--dry-run'], env=env, capture_output=True, text=True, encoding='utf8', timeout=45)
    assert run.returncode == 0, run.stdout+run.stderr
    plan = json.loads(run.stdout[run.stdout.index('{'):])
    assert plan['total_updates'] == 3*3*250
    assert plan['stage_updates'] == dict(q1=750, q2=750, q3=750)
    assert plan['actual_delivery_scope']['layouts'] == ['1', '2', '3', 'mixed']
    assert not (tmp_path/'dry').exists()
