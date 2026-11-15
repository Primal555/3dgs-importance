import json
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc import successive_tier_refinement as entry
from gaussian_jscc import successive_q3_refinement as base
from gaussian_jscc.multisnr_training import cached_render_step
from gaussian_jscc.render_objective import MultiViewRenderTask
from gaussian_jscc.refinement_views import RefinementViewCycle, refinement_split, make_refinement_scene
from test_successive_q3_refinement import load_model_scene, assert_same_checkpoint
from test_successive_multisnr import shared, digital_runner
from test_render_first import synthetic_render, Reference
from test_successive_joint import cameras


def options(source, out):
    return entry.parser().parse_args(['--run', str(source), '--out', str(out), '--device', 'cpu',
        '--digital-python', sys.executable, '--digital-device', 'cpu', '--scenes', 'truck', 'train',
        '--heldout-scenes', 'drjohnson', '--steps-per-scene', '2', '--save-every', '1',
        '--validate-every', '1', '--trials', '1', '--cpu-threads', '1', '--no-lpips',
        '--views-per-step', '1', '--validation-views', '1'])


def test_refinement_partitions_and_real_scene_camera_identity(shared, tmp_path):
    record = dict(train_view_names=[f'train{i}' for i in range(215)],
                  validation_view_names=[f'val{i}' for i in range(4)],
                  test_view_names=[f'test{i}' for i in range(32)])
    before = copy.deepcopy(record)
    split = refinement_split(record, 16, 32)
    assert record == before and len(split['train_view_names']) == 203
    assert len(split['validation_view_names']) == 16
    assert set(record['validation_view_names']) <= set(split['validation_view_names'])
    assert set(split['train_view_names']).isdisjoint(split['validation_view_names'])
    assert set(split['train_view_names']+split['validation_view_names']).isdisjoint(split['test_view_names'])
    with pytest.raises(ValueError, match='enough distinct'):
        refinement_split(record, 200, 32)
    # Validate the real constructor before changing the original camera split.
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras) as loader:
        args, model, source_scene, _ = load_model_scene(shared, tmp_path)
        original = copy.deepcopy(source_scene.record)
        args.views_per_step = 1; args.validation_views = 1
        loader.reset_mock()
        raw, degree = entry.read_ply(source_scene.spec['ply'])
        scene = make_refinement_scene(source_scene.spec, raw, degree, model, args, tmp_path,
                                      original, refinement_split(original, 1, 1))
        assert loader.call_count == 2  # One load for train/validation and one for test.
        assert scene.record == original
        corrupted = copy.deepcopy(original); corrupted['camera_fingerprints']['train'] = 'changed'
        with pytest.raises(ValueError, match='camera split mismatch'):
            make_refinement_scene(source_scene.spec, raw, degree, model, args, tmp_path,
                                  corrupted, refinement_split(original, 1, 1))


def test_32_view_cycle_covers_scene_and_resumes_without_changing_channel_rng():
    names = [str(i) for i in range(203)]
    rng = torch.get_rng_state().clone()
    cycle = RefinementViewCycle(names, 42, 'truck', 1)
    seen = set()
    for _ in range(7):
        indices = cycle.next(32)
        assert len(indices) == len(set(indices)) == 32
        seen.update(indices)
    assert len(seen) == 203
    resumed = RefinementViewCycle(names, 42, 'truck', 1, json.loads(json.dumps(cycle.state_dict())))
    for _ in range(143):
        assert cycle.next(32) == resumed.next(32)
    assert cycle.state_dict() == resumed.state_dict()
    assert cycle.coverage()['total_view_uses'] == 4800
    assert cycle.coverage()['mean_visits'] == pytest.approx(4800/203)
    assert torch.equal(torch.get_rng_state(), rng)
    sequences = [RefinementViewCycle(names, 42, scene, tier).next(32)
                 for scene, tier in [('truck', 1), ('truck', 2), ('train', 1)]]
    assert len({tuple(x) for x in sequences}) == 3


def test_source_settings_cannot_silently_override_requested_views(shared, tmp_path):
    saved, metadata, cfg, identity = base.load_source(shared)
    metadata = copy.deepcopy(metadata)
    metadata['settings'].update(views_per_step=2, validation_views=4)
    for record in metadata['scene_records'].values():
        record.update(train_view_names=[f'train{i}' for i in range(215)],
                      validation_view_names=[f'val{i}' for i in range(4)],
                      test_view_names=[f'test{i}' for i in range(32)])
    args = options(shared, tmp_path/'unused')
    args.views_per_step = 32; args.validation_views = 16
    with patch.object(base, 'load_source', return_value=(saved, metadata, cfg, identity)):
        plan = entry.inputs(args)[-1]
    assert args.views_per_step == plan['settings']['views_per_step'] == 32
    assert args.validation_views == plan['settings']['validation_views'] == 16
    split = plan['camera_splits']['truck']
    assert len(split['train_view_names']) == 203 and len(split['validation_view_names']) == 16


def test_32_view_streamed_gradient_is_the_mean_not_the_sum():
    views = [SimpleNamespace(factor=.2+i*.07) for i in range(32)]
    raw = torch.linspace(-1, 1, 80).reshape(5, 16).requires_grad_()
    task = MultiViewRenderTask(views, Reference(), 1)
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
        expected = task(raw)
        expected_grad, = torch.autograd.grad(expected, raw)
        loss = task.backward_scene(raw)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(raw.grad, expected_grad, atol=1e-8, rtol=1e-6)


def test_monitor_uses_all_four_trials_and_rejects_changed_cameras():
    args = SimpleNamespace(seed=42, trials=4)
    scene = SimpleNamespace(val_cameras=[SimpleNamespace(image_name=str(i)) for i in range(16)])
    rows = [dict(trial=i, seed=1042+i, scope='validation', path='awgn', snr_db=10., layout='2',
                 views=[dict(view=str(v)) for v in range(16)], source_psnr=20.+i,
                 photo_psnr=15.+i, source_mse=.01*(i+1),
                 diagnostics=dict(latent_mse=.1*(i+1), latent_nmse=.2*(i+1))) for i in range(4)]
    result = entry.monitor_metrics(rows, args, scene, 2)
    assert result['source_psnr'] == 21.5 and result['photo_psnr'] == 16.5
    assert result['source_psnr_std_trials'] == pytest.approx(1.25**.5)
    assert result['latent_mse'] == pytest.approx(.25)
    assert result['trials'] == 4 and result['validation_views'] == 16
    rows[-1]['views'][0]['view'] = 'training_view'
    with pytest.raises(RuntimeError, match='validation trials'):
        entry.monitor_metrics(rows, args, scene, 2)


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
        assert a['training']['progress']['view_schedulers'] == b['training']['progress']['view_schedulers']
        assert a['training']['progress']['view_coverage'] == b['training']['progress']['view_coverage']
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
        assert all(r['view_coverage']['total_view_uses'] == r['scene_step'] for r in records)
        assert (full/'review.tar.gz').exists()


def test_launcher_dry_run_resolves_stage_budget(shared, tmp_path):
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists(): pytest.skip('Git Bash unavailable')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHON_BIN=Path(sys.executable).as_posix(),
               DIGITAL_PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8='1', RESUME='0')
    script = Path('scripts/test_successive_tier_refinement.sh').resolve()
    run = subprocess.run([str(bash), script.as_posix(), shared.as_posix(), (tmp_path/'dry').as_posix(),
        '--heldout-scenes', 'drjohnson', '--views-per-step', '1', '--validation-views', '1',
        '--dry-run'], env=env, capture_output=True, text=True, encoding='utf8', timeout=45)
    assert run.returncode == 0, run.stdout+run.stderr
    plan = json.loads(run.stdout[run.stdout.index('{'):])
    assert plan['total_updates'] == 3*3*150
    assert plan['stage_updates'] == dict(q1=450, q2=450, q3=450)
    assert plan['settings']['views_per_step'] == plan['settings']['validation_views'] == 1
    assert plan['validation_protocol']['trials'] == 4
    defaults = entry.parser().parse_args(['--run', str(shared), '--out', str(tmp_path/'default'),
                                         '--digital-python', sys.executable])
    assert defaults.views_per_step == 32 and defaults.validation_views == 16 and defaults.steps_per_scene == 150
    assert plan['actual_delivery_scope']['layouts'] == ['1', '2', '3', 'mixed']
    assert not (tmp_path/'dry').exists()
