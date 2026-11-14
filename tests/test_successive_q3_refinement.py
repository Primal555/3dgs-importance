"""Last-stage gradients, exact resume, fixed q2 and final digital accounting."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc import successive_q3_refinement as entry
from gaussian_jscc.model_factory import build_codec
from gaussian_jscc.modular_training import render_step
from gaussian_jscc.multisnr_training import cached_render_step
from gaussian_jscc.render_objective import MultiViewRenderTask
from test_render_first import synthetic_render
from test_successive_joint import cameras
from test_successive_multisnr import shared, digital_runner


def options(source, out):
    return entry.parser().parse_args(['--run', str(source), '--out', str(out), '--device', 'cpu',
        '--digital-python', sys.executable, '--digital-device', 'cpu', '--scenes', 'truck', 'train',
        '--heldout-scenes', 'drjohnson', '--steps-per-scene', '2', '--save-every', '1',
        '--validate-every', '1', '--trials', '1', '--cpu-threads', '1', '--no-lpips'])


def load_model_scene(shared, tmp_path):
    args = options(shared, tmp_path/'unused')
    saved, metadata, cfg, specs, _, plan = entry.inputs(args)
    model = build_codec(cfg); model.load_state_dict(saved['state_dict']); model.set_training_snrs([10.])
    raw, degree = entry.read_ply(specs[0]['ply'])
    scene = entry.make_scene(specs[0], raw, degree, model, args, tmp_path,
                             old=metadata['scene_records']['truck'])
    return args, model, scene, plan


def test_q3_render_gradients_cross_frozen_decoder_and_cache_matches(shared, tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
        args, model, scene, plan = load_model_scene(shared, tmp_path)
        optimizer = entry.configure_optimizer(model, args.lr)
        targets = [b['y'] for b in entry.cache_teacher(model, scene)]
        frozen = entry.frozen_digest(model.state_dict())
        qs = [(ids >= 0).long()*3 for ids in scene.group_ids]
        task = MultiViewRenderTask(scene.cameras[:1], scene.reference, scene.degree)
        original = model.forward_channel_batches

        def paired(*a, **kw):
            kw['paired_noise'] = True
            return original(*a, **kw)

        model.train(); model.semantic_encoder.eval(); model.semantic_decoder.eval()
        torch.manual_seed(73)
        with patch.object(model, 'forward_channel_batches', side_effect=paired):
            loss, _ = render_step(model, list(zip(scene.groups, qs)), scene.geometry, 10., 'awgn',
                                  task, 'channel_render', channel_latent_weight=1.)
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        assert gradients and all(n.startswith(entry.TRAINABLE) for n in gradients)
        optimizer.zero_grad(set_to_none=True); torch.manual_seed(73)
        with patch.object(model.semantic_encoder, 'forward', side_effect=AssertionError('cache missed')):
            cached, _ = cached_render_step(model, entry.CachedQ3Batches(scene, targets), scene.geometry, 10., task, 1.)
        torch.testing.assert_close(loss, cached, atol=1e-7, rtol=1e-6)
        for n, p in model.named_parameters():
            if n in gradients:
                torch.testing.assert_close(p.grad, gradients[n], atol=2e-6, rtol=2e-5)
            else:
                assert p.grad is None
        # An isolated render objective reaches BOTH q3 mappings through frozen downstream modules.
        optimizer.zero_grad(set_to_none=True); torch.manual_seed(73)
        cached_render_step(model, entry.CachedQ3Batches(scene, targets), scene.geometry, 10., task, 0.)
        for prefix in entry.TRAINABLE:
            assert sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                       if n.startswith(prefix) and p.grad is not None) > 0
        optimizer.step()
        assert entry.frozen_digest(model.state_dict()) == frozen == plan['frozen_state_sha256']


def assert_same_checkpoint(a, b):
    for n, value in a['state_dict'].items():
        torch.testing.assert_close(value, b['state_dict'][n], rtol=0, atol=0)
    for key, state in a['training']['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b['training']['optimizer']['state'][key][name], rtol=0, atol=0)
    torch.testing.assert_close(a['training']['rng']['torch'], b['training']['rng']['torch'], rtol=0, atol=0)


def test_end_to_end_midround_resume_frozen_q2_and_delivery(shared, tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('comparison_baselines.run._execute', side_effect=digital_runner):
        source = torch.load(shared/'checkpoints/final/codec.pt', weights_only=True)
        source_hash = entry.frozen_digest(source['state_dict'])
        full, resumed = tmp_path/'full', tmp_path/'resumed'
        result = entry.main(options(shared, full))
        assert result['step'] == 4 and result['q2_unchanged'] and result['actual_side_delivery_completed']
        entry.main(options(shared, resumed), max_updates=1)
        args = options(shared, resumed); args.resume = True
        entry.main(args)
        a = torch.load(full/'checkpoints/final/codec.pt', weights_only=True)
        b = torch.load(resumed/'checkpoints/final/codec.pt', weights_only=True)
        assert_same_checkpoint(a, b)
        assert entry.frozen_digest(a['state_dict']) == source_hash
        for prefix in entry.TRAINABLE:
            assert any(not torch.equal(value, a['state_dict'][n])
                       for n, value in source['state_dict'].items() if n.startswith(prefix))
        assert {int(s['step']) for s in a['training']['optimizer']['state'].values()} == {4}
        report = entry.read(full/'review_summary.json')
        assert report['q2_unchanged'] and all(c['passed'] for c in report['q2_checks'])
        assert len(report['q3_changes']) == 6
        q2_changes = [c['absolute_metric_changes'] for c in report['q2_checks']]
        assert all(max(c.values()) < 1e-7 for c in q2_changes)
        for r in entry.read(full/'actual_delivery/truck/summary.json')['trials']:
            match = next(x for x in report['rows'] if x['scene'] == 'truck' and x['endpoint'] == 'final'
                         and x['path'] == 'awgn' and x['layout'] == r['method'])
            assert r['total_channel_uses'] == match['total_channel_uses']
            assert r['metrics']['source_psnr'] == pytest.approx(match['source_psnr'], abs=1e-5)
            assert r['comparison_accounting']['link']['n'] == 6144
        assert not list(full.rglob('received.npy'))
        assert (full/'review.tar.gz').is_file()
        args = options(shared, full); args.resume = True
        with patch('comparison_baselines.run._execute', side_effect=AssertionError('reran delivery')):
            entry.main(args)


def test_monitor_interruption_resumes_without_duplicate_rows(shared, tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('comparison_baselines.run._execute', side_effect=digital_runner):
        full, resumed = tmp_path/'full', tmp_path/'resumed'
        entry.main(options(shared, full))
        actual = entry.evaluate_matrix

        def fail(model, scene, args, out, label, **kw):
            result = actual(model, scene, args, out, label, **kw)
            if label == 'monitor_000001' and scene.name == 'train':
                raise RuntimeError('monitor interrupted')
            return result

        with patch.object(entry, 'evaluate_matrix', side_effect=fail):
            with pytest.raises(RuntimeError, match='monitor interrupted'):
                entry.main(options(shared, resumed))
        args = options(shared, resumed); args.resume = True
        entry.main(args)
        a = torch.load(full/'checkpoints/final/codec.pt', weights_only=True)
        b = torch.load(resumed/'checkpoints/final/codec.pt', weights_only=True)
        assert_same_checkpoint(a, b)
        records = [json.loads(line) for line in (resumed/'metrics/validation.jsonl').read_text().splitlines()]
        assert len(records) == len({(r['step'], r['scene']) for r in records}) == 6


def test_reject_incompatible_scope_and_changed_resume_settings(shared, tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('comparison_baselines.run._execute', side_effect=digital_runner):
        args = options(shared, tmp_path/'run')
        args.scenes = ['drjohnson']
        with pytest.raises(ValueError, match='original split'):
            entry.inputs(args)
        args = options(shared, tmp_path/'run')
        entry.main(args, max_updates=1)
        args.resume = True; args.lr *= 2
        with pytest.raises(ValueError, match='same source'):
            entry.main(args)


def test_launcher_defaults_and_cli_override(shared, tmp_path):
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.is_file():
        pytest.skip('Git Bash integration on Windows')
    script = Path('scripts/test_successive_q3_refinement.sh').resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', PYTHON_BIN=Path(sys.executable).as_posix(),
               DIGITAL_PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8='1', RESUME='0')
    command = [str(bash), script.as_posix(), shared.as_posix(), (tmp_path/'out').as_posix(),
               '--heldout-scenes', 'drjohnson', '--dry-run', '--no-lpips']
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding='utf8', timeout=45)
    assert result.returncode == 0, result.stdout+result.stderr
    plan = json.loads(result.stdout[result.stdout.index('{'):])
    assert plan['total_updates'] == 3*250
    assert plan['settings']['lr'] == 1e-5 and plan['settings']['latent_weight'] == 1.
    assert plan['actual_delivery_scope']['layouts'] == ['2', '3']
    assert not (tmp_path/'out').exists()
