import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc import successive_mixed_refinement as entry
from gaussian_jscc import successive_tier_refinement as sequential
from gaussian_jscc.modular_training import render_step
from gaussian_jscc.multisnr_training import cached_render_step
from gaussian_jscc.render_objective import MultiViewRenderTask
from test_successive_multisnr import shared, digital_runner
from test_successive_joint import cameras
from test_render_first import synthetic_render
from test_successive_q3_refinement import load_model_scene, assert_same_checkpoint
from test_successive_tier_refinement import options


def test_mixed_map_replay_resample_padding_and_rng():
    ids = torch.arange(256).reshape(4, 64)
    ids[-1, -5:] = -1
    scene = SimpleNamespace(name='truck', groups=[torch.randn(4, 64, 8)], group_ids=[ids])
    targets = [torch.randn(4, 64, 48)]
    rng = torch.get_rng_state().clone()
    batches = entry.CachedMixedBatches(scene, targets, 42, 1)
    x, q = next(iter(batches))
    assert torch.equal(q, next(iter(batches))[1])
    assert torch.equal(q, next(iter(entry.CachedMixedBatches(scene, targets, 42, 1)))[1])
    assert not torch.equal(q, next(iter(entry.CachedMixedBatches(scene, targets, 42, 2)))[1])
    assert torch.equal(torch.get_rng_state(), rng)
    assert set(q[ids >= 0].tolist()) == {1, 2, 3}
    assert torch.equal(q == 0, ids < 0)
    torch.testing.assert_close(x[..., :8], scene.groups[0])
    torch.testing.assert_close(x[..., 8:], targets[0])


def test_mixed_cached_render_preserves_full_block_gradients(shared, tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
        args, model, scene, _ = load_model_scene(shared, tmp_path)
        optimizer = entry.configure_optimizer(model, 1, args.lr)
        targets = [b['y'] for b in sequential.cache_teacher(model, scene)]
        batches = entry.CachedMixedBatches(scene, targets, 42, 1)
        qs = [q for _, q in batches]
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
        optimizer.zero_grad(set_to_none=True); torch.manual_seed(73)
        with patch.object(model.semantic_encoder, 'forward', side_effect=AssertionError('cache missed')):
            cached, _ = cached_render_step(model, batches, scene.geometry, 10., task, 1.)
        torch.testing.assert_close(loss, cached, atol=1e-7, rtol=1e-6)
        for n, p in model.named_parameters():
            if n in gradients:
                torch.testing.assert_close(p.grad, gradients[n], atol=2e-6, rtol=2e-5)
            else:
                assert p.grad is None
        for prefix in sequential.ALL_STAGES:
            assert sum(float(g.abs().sum()) for n, g in gradients.items() if n.startswith(prefix)) > 0
        frozen = sequential.base.frozen_digest(model.state_dict(), sequential.ALL_STAGES)
        optimizer.step()
        assert sequential.base.frozen_digest(model.state_dict(), sequential.ALL_STAGES) == frozen


def test_completed_sequential_source_mixed_resume_and_delivery(shared, tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('comparison_baselines.run._execute', side_effect=digital_runner):
        source = tmp_path/'sequential'
        sequential.main(options(shared, source))
        saved, metadata, _, identity = entry.load_source(source)
        assert identity['source_step'] == 12
        with pytest.raises(ValueError, match='completed three-stage'):
            entry.load_source(shared)
        full, resumed = tmp_path/'mixed', tmp_path/'resumed'
        result = entry.main(options(source, full))
        assert result['step'] == 4 and result['scene_completed'] == {'mixed': {'truck': 2, 'train': 2}}
        assert result['frozen_state_verified'] and result['actual_side_delivery_completed']
        assert result['selection'] == 'fixed_final'
        assert 'lower_tiers_unchanged' not in result
        assert entry.main(options(source, resumed), max_updates=1)['status'] == 'interrupted'
        args = options(source, resumed); args.resume = True
        entry.main(args)
        a = torch.load(full/'checkpoints/final/codec.pt', weights_only=True)
        b = torch.load(resumed/'checkpoints/final/codec.pt', weights_only=True)
        assert_same_checkpoint(a, b)
        assert a['training']['progress'] == b['training']['progress']
        assert {int(v['step']) for v in a['training']['optimizer']['state'].values()} == {4}
        assert sequential.base.frozen_digest(a['state_dict'], sequential.ALL_STAGES) == metadata['frozen_state_sha256']
        for prefix in sequential.ALL_STAGES:
            assert any(not torch.equal(v, a['state_dict'][n]) for n, v in saved['state_dict'].items() if n.startswith(prefix))
        before = entry.engine.read(source/'evaluation/final/truck/summary.json')['rows']
        initial = entry.engine.read(full/'evaluation/initial/truck/summary.json')['rows']
        for r, old in zip(initial, before):
            assert r['codec_id'] == old['codec_id']
            assert r['seed'] == old['seed'] and r['views'] == old['views']
            assert r['total_channel_uses'] == old['total_channel_uses']
        records = entry.engine.read(full/'training.json')['scene_records']
        assert records == metadata['scene_records']
        validation = [json.loads(line) for line in (resumed/'metrics/validation.jsonl').read_text().splitlines()]
        assert len(validation) == 3*2*4
        assert {r['layout'] for r in validation} == {'1', '2', '3', 'mixed'}
        report = entry.engine.read(full/'review_summary.json')
        assert len(report['rows']) == 2*3*2*4
        assert 'lower_tiers_unchanged' not in report
        actual = entry.engine.read(full/'actual_delivery/truck/summary.json')['trials']
        assert len(actual) == 4
        for r in actual:
            row = next(x for x in report['rows'] if x['endpoint'] == 'final' and x['scene'] == 'truck'
                       and x['path'] == 'awgn' and x['layout'] == r['method'])
            assert r['total_channel_uses'] == row['total_channel_uses']
            assert r['metrics']['source_psnr'] == pytest.approx(row['source_psnr'], abs=1e-5)
        assert (full/'review.tar.gz').exists()
        args = options(source, tmp_path/'dry'); args.dry_run = True
        assert entry.main(args)['total_updates'] == 4
        args.seed += 1
        with pytest.raises(ValueError, match='source validation-view count and seed'):
            entry.main(args)
